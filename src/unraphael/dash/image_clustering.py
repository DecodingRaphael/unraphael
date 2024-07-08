"""Module for clustering of images based on structural similarity, including
alignment of images based on the mean of the images and equalization of 
contrast, sharpness, and brightness."""


# Copyright (c) 2016, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import ssim.ssimlib as pyssim
from skimage.metrics import structural_similarity as ssim
from skimage import color, transform
from skimage.transform import resize

from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans, DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA

from pystackreg import StackReg

from clustimage import Clustimage


SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
NUM_THREADS = 8

def align_images_to_mean(
    images: dict[str, np.ndarray],
    *,
    #selected_option: str,
    motion_model: str,
    feature_method: str,
    target_size: tuple = SIM_IMAGE_SIZE
) -> dict[str, np.ndarray]:
   
    
    """
    Aligns a set of images to the mean image.

    Args:
        images (dict[str, np.ndarray]): Dictionary of images.
        selected_option (str): Selected alignment option.
        motion_model (str): Motion model for alignment.
        feature_method (str, optional): Feature method for alignment. Defaults to 'mean'.
        target_size (tuple, optional): Target size to resize the images to. Defaults to (640, 480).

    Returns:
        dict[str, np.ndarray]: Dictionary of aligned images.
    """        
    
    def resize_image(image, target_size):
        return transform.resize(image, target_size, anti_aliasing=True, preserve_range=True).astype(image.dtype)
    
    def ensure_grayscale(image):
        if image.ndim == 3: # if in color, make grayscale
            return color.rgb2gray(image)
        return image
    
    resized_images = {name: resize_image(ensure_grayscale(image), target_size) for name, image in images.items()}
    image_stack = np.stack(list(resized_images.values()), axis=0)
    
    if image_stack.ndim != 3:
        raise ValueError("Image stack must have three dimensions (num_images, height, width).")
    
    #if selected_option == 'pyStackReg':
    #    sr = None        
    if motion_model == 'translation':            
        sr = StackReg(StackReg.TRANSLATION)
    elif motion_model == 'rigid body':
        sr = StackReg(StackReg.RIGID_BODY)
    elif motion_model == 'scaled rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)
    elif motion_model == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif motion_model == 'bilinear':            
        sr = StackReg(StackReg.BILINEAR)
          
    #if sr is None:
    #        raise ValueError("Invalid motion model selected.")
          
    if feature_method == 'to first image':            
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = "first")
    elif feature_method == 'to mean image':            
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = 'mean')
    elif feature_method == 'each image to the previous (already registered) one':
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = 'previous')
    else:
        raise ValueError("Invalid feature method selected.")
        
    aligned_images = {name: aligned_images_stack[i] for i, name in enumerate(images.keys())}
    print("##") 
    type(aligned_images)    
    return aligned_images    
    
# Once the edge features are computed, the standard deviation is calculated for
# every individual edge feature obtained from an image. The standard deviation serves as
# an effective metric to quantify the variability and intensity of edge features in the image.

def calculate_canny_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return np.std(edges)

def calculate_sobel_edges(img):    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.std(sobelx), np.std(sobely)

def calculate_laplacian_edges(img):    
    laplacian = cv2.Laplacian(img, cv2.CV_64F)    
    return np.std(laplacian)

def calculate_scharr_edges(img):
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)    
    return np.std(scharrx), np.std(scharry)

def calculate_histogram_features(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()

def calculate_features(img):
    canny_edges                    = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y   = calculate_sobel_edges(img)
    laplacian_edges                = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)
    histogram_features             = calculate_histogram_features(img)
          
    # features = np.array([canny_edges, 
    #                      sobel_edges_x, sobel_edges_y, 
    #                      laplacian_edges, 
    #                      scharr_edges_x, scharr_edges_y])
    
    # Replace NaN values with 0 (or another appropriate value)
    features = np.concatenate([
        np.array([
            np.nan_to_num(canny_edges),
            np.nan_to_num(sobel_edges_x),
            np.nan_to_num(sobel_edges_y),
            np.nan_to_num(laplacian_edges),
            np.nan_to_num(scharr_edges_x),
            np.nan_to_num(scharr_edges_y)
        ]),
        histogram_features
    ])
    
    print("Features calculated: ", features)
    return features          

# # Read images in gray
# img1 = cv2.imread('../../../data/interim/aligned_072024/output_0.png')
# img2 = cv2.imread('../../../data/interim/aligned_072024/output_2.png')

# # # Resize the images to the same size
# img1 = cv2.resize(img1, SIM_IMAGE_SIZE)
# img2 = cv2.resize(img2, SIM_IMAGE_SIZE)

# # Convert images to grayscale
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Image', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def get_image_similarity(img1, img2, algorithm = 'SSIM'):
    """ Returns the normalized similarity value (from 0.0 to 1.0) for the provided pair of images. """          
    
    # Ensure the images are of type np.uint8
    img1_gray = img1.astype(np.uint8)
    img2_gray = img2.astype(np.uint8)
    
    print("Initial images")
    print("Image 1 shape:", img1_gray.shape, "dtype:", img1_gray.dtype)
    print("Image 2 shape:", img2_gray.shape, "dtype:", img2_gray.dtype)

    # Create masks to isolate the foreground
    _, mask1 = cv2.threshold(img1_gray, 1, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(img2_gray, 1, 255, cv2.THRESH_BINARY)

    # Ensure the masks are of type np.uint8
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # Apply the masks to isolate the foreground
    i1 = cv2.bitwise_and(img1_gray, img1_gray, mask=mask1)
    i2 = cv2.bitwise_and(img2_gray, img2_gray, mask=mask2)
    
    print("After masking")
    print("Masked Image 1 shape:", i1.shape, "dtype:", i1.dtype)
    print("Masked Image 2 shape:", i2.shape, "dtype:", i2.dtype)
        
    similarity = 0.000

    if algorithm == 'SIFT':
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(i1, None)
        k2, d2 = sift.detectAndCompute(i2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)
        
        good_matches = [m for m, n in matches if m.distance < SIFT_RATIO * n.distance]
        similarity = len(good_matches) / max(len(matches), 1)
    
    elif algorithm == 'CW-SSIM':
        pil_img1 = Image.fromarray(i1)
        pil_img2 = Image.fromarray(i2)
        similarity = pyssim.SSIM(pil_img1).cw_ssim_value(pil_img2)        
                
    elif algorithm == 'SSIM':
        similarity, _ = ssim(i1, i2, full = True)
            
    elif algorithm == 'MSE':
        err = np.sum((i1.astype("float") - i2.astype("float")) ** 2)
        err /= float(i1.shape[0] * i2.shape[1])
        similarity = MSE_NUMERATOR / err if err > 0 else 1.0
    
    elif algorithm == 'Brushstrokes':
        features_i1 = calculate_features(i1)
        
        # Normalize features to get weights              
        weights = features_i1 / np.sum(features_i1)
        features_i2 = calculate_features(i2)
        
        # Replace NaN values with 0 (or another appropriate value)
        features_i2 = np.nan_to_num(features_i2 * weights)
        
        # Compare features between the two images
        difference = np.abs(features_i1 - features_i2)
        similarity = np.mean(difference)
        
        print("Brushstrokes similarity: ", similarity)
                
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Ensure similarity is a valid number
    similarity = np.nan_to_num(similarity)
    
    # Debug prints
    print(f"Algorithm: {algorithm}, Similarity: {similarity}")
    
    return similarity

def build_similarity_matrix(images, algorithm = 'SSIM'):
    """
    For AffinityPropagation, SpectralClustering and DBSCAN one can input similarity 
    matrices of shape (n_samples, n_samples). This function builds such a similarity 
    matrix for the given set of images. 

    Args:
        images (list of numpy.ndarray): A list of images
        algorithm (str, optional): Select image similarity index. Defaults to 'SSIM'.

    Returns:
        numpy.ndarray: The similarity matrix.
    """
    num_images = len(images)
    sm = np.zeros((num_images, num_images), dtype=np.float64)
    np.fill_diagonal(sm, 1.0)

    def compute_similarity(i, j):
        if i != j:
            print(f"Similarity between images {i} and {j} computed.")
            return get_image_similarity(images[i], images[j], algorithm)
        return 1.0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [(i, j, executor.submit(compute_similarity, i, j))
                   for i in range(num_images) for j in range(i + 1, num_images)]
        for i, j, future in futures:
            sm[i, j] = sm[j, i] = future.result()
            
    # Print the similarity matrix with two decimals #TODO: replace with heatmap widget
    for row in sm:
        print(' '.join(f'{val:.2f}' for val in row))        
    return sm

def get_cluster_metrics(X, labels, labels_true = None):
    """
    Calculate cluster evaluation metrics based on the given data and labels.

    Parameters:
    - X: numpy.ndarray
        The input data matrix.
    - labels: numpy.ndarray
        The predicted cluster labels.
    - labels_true: numpy.ndarray, optional
        The true cluster labels (ground truth). Default is None.

    Returns:
    - metrics_dict: dict
        A dictionary containing the calculated cluster evaluation metrics.

    Available metrics:
    - Silhouette coefficient: measures the quality of clustering.
    - Davies-Bouldin index: measures the separation between clusters.
    - Completeness score: measures the completeness of the predicted clusters.
    - Homogeneity score: measures the homogeneity of the predicted clusters.
    """
    np.fill_diagonal(X, 0)
    
    metrics_dict = {}

    if len(set(labels)) > 1:
        metrics_dict['Silhouette coefficient'] = metrics.silhouette_score(X, labels, metric='precomputed')
        metrics_dict['Davies-Bouldin index'] = metrics.davies_bouldin_score(1 - X, labels)   
        
    if labels_true is not None:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)
        
        metrics_dict['V-measure'] = metrics.v_measure_score(labels_true, labels)
        metrics_dict['Adjusted Rand index'] = metrics.adjusted_rand_score(labels_true, labels)
        metrics_dict['Adjusted mutual information'] = metrics.adjusted_mutual_info_score(labels_true, labels)

    return metrics_dict

def determine_optimal_clusters(matrix):
    """
    Determines the optimal number of clusters using the elbow method.

    Args:
        matrix (numpy.ndarray): The input matrix for clustering.

    Returns:
        int: The optimal number of clusters.

    """
    max_clusters = min(len(matrix), 10)
    inertias = []
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters = k, random_state = 42).fit(matrix)
        inertias.append(kmeans.inertia_)
    
    # Plot the inertia to visualize the elbow point
    # plt.figure()
    # plt.plot(range(2, max_clusters+1), inertias, 'bo-')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal Clusters')
    # plt.show()

    # Identify the elbow point
    elbow_point = 2  # default
    if len(inertias) > 1:
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 2  # +2 because the range starts from 2
    return elbow_point


def plot_clusters(images, labels, n_clusters, title = "Clustering Results"):
    """
    Plots the clustering results in 2D space using PCA for dimensionality reduction.

    Parameters:
    - images (list): A list of images to be clustered.
    - labels (list): The cluster labels assigned to each image.
    - n_clusters (int): The number of clusters.
    - title (str): The title of the plot.
    """
    # Flatten the images and reduce to 2D using PCA
    flattened_images = np.array([img.flatten() for img in images.values()])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_images)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap('viridis', n_clusters)
    
    for k in range(n_clusters):
        class_members = (labels == k)
        ax.scatter(reduced_data[class_members, 0], reduced_data[class_members, 1], color=colors(k), marker='.', label=f'Cluster {k}')
    
    ax.set_title(title)
    ax.legend()
    
    return fig
    

def cluster_images(images, algorithm, n_clusters, method, print_metrics = True, labels_true = None):
    """
    Clusters a list of images using different clustering algorithms.

    Parameters:
    - images (list): A list of images to be clustered
    - algorithm (str): The algorithm used to calculate the similarity between images. Default is 'SSIM'
    - n_clusters (int): The number of clusters to create. If None, the optimal number of clusters will be determined automatically. Default is None
    - method (str): The clustering method to use. Options are 'SpectralClustering', 'AffinityPropagation', and 'DBSCAN'. Default is 'SpectralClustering'
    - print_metrics (bool): Whether to print performance metrics for each clustering method. Default is True
    - labels_true (list): The true labels for the images, used for evaluating clustering performance. Default is None

    Returns:
    - labels (list): The cluster labels assigned to each image.
    """
        
    # main input for the clustering algorithm
    matrix = build_similarity_matrix(images, algorithm = algorithm)        
        
    if n_clusters is None and method != 'DBSCAN':
        n_clusters = determine_optimal_clusters(matrix)
    
    # SpectralClustering requires the number of clusters to be specified in advance. It works well for a small 
    # number of clusters, but is not advised for many clusters
    if method == 'SpectralClustering':
        sc = SpectralClustering(n_clusters = n_clusters, random_state = 42, affinity = 'precomputed').fit(matrix)
        sc_metrics = get_cluster_metrics(matrix, sc.labels_, labels_true)

        if print_metrics:
            print("\nPerformance metrics for Spectral Clustering")
            print(f"Number of clusters: {len(set(sc.labels_))}")
            for k, v in sc_metrics.items():
                print(f"{k}: {v:.2f}")
        return sc.labels_, n_clusters
       
    # Affinity propagation is most appropriate for small to medium sized datasets
    elif method == 'AffinityPropagation':
        af = AffinityPropagation(affinity = 'precomputed', random_state = 42).fit(matrix)
        af_metrics = get_cluster_metrics(matrix, af.labels_, labels_true)

        if print_metrics:
            print("\nPerformance metrics for Affinity Propagation Clustering")
            print(f"Number of clusters: {len(set(af.labels_))}")
            for k, v in af_metrics.items():
                print(f"{k}: {v:.2f}")
        return af.labels_, len(set(af.labels_))
        
    elif method == 'DBSCAN':
        db = DBSCAN(metric='precomputed', eps=0.5, min_samples=2).fit(matrix)
        db_labels = db.labels_
        db_metrics = get_cluster_metrics(matrix, db_labels, labels_true)

        if print_metrics:
            print("\nPerformance metrics for DBSCAN Clustering")
            print(f"Number of clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")
            for k, v in db_metrics.items():
                print(f"{k}: {v:.2f}")
        return db_labels, len(set(db_labels)) - (1 if -1 in db_labels else 0)


def preprocess_images(images, target_shape=(128, 128)):
    preprocessed_images = []
    for image in images:        
        resized_image = resize(image, target_shape, anti_aliasing=True)
        preprocessed_images.append(resized_image)
    return np.array(preprocessed_images)

# using the clustimage functionality 
def clustimage_clustering(images):
    """
    Perform image clustering using the clustimage library, aiming to detect natural groups or clusters of 
    images. It uses a multi-step proces of pre-processing the images, extracting the features, and 
    evaluating the optimal number of clusters across the feature space.

    Args:
        images (list): A list of input images.

    Returns:
        dict: A dictionary containing the clustering results. The dictionary has the following keys:
            - 'scatter_plot': The scatter plot of the clustered images.
            - 'dendrogram_plot': The dendrogram plot of the clustering hierarchy.

    """
    imgs = preprocess_images(images)
    flattened_imgs = np.array([img.flatten() for img in imgs])         
    
    # HOG is most likely not the best feature extraction method for this task; see
    # https://erdogant.github.io/clustimage/pages/html/Feature%20Extraction.html
    cl = Clustimage(method = 'pca', params_pca={'n_components':0.95})
            
    imported = cl.import_data(flattened_imgs)
        
    # Preprocessing, feature extraction, embedding and detection of the optimal number of clusters
    results = cl.fit_transform(imported,
                               cluster       = 'agglomerative', # The clustering approaches can be set to agglomerative, kmeans, dbscan and hdbscan
                               evaluate      = 'silhouette',    # Cluster evaluation can be performed based on: Silhouette scores, Davies–Bouldin index, or Derivative method                               metric        = 'euclidean',
                               linkage       = 'ward',
                               min_clust     = 1,
                               max_clust     = 25,
                               cluster_space = 'low') # cluster on the 2-D embedded space
        
    # Collect plots and save them to a dictionary 'figures'
    evaluation_plot = cl.clusteval.plot() # silhoutte versus nr of clusters
    eval_plot       = cl.clusteval.scatter(cl.results['xycoord'])
    scatter_plot    = cl.scatter(zoom=None, dotsize=200, figsize=(25, 15), 
                                 args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})   # tsne plot 
    dendrogram_plot = cl.dendrogram()['ax'].figure  # ensure cl.dendrogram() returns the figure object from 'ax'
    
    figures = {
        'evaluation_plot': evaluation_plot,
        'eval_plot': eval_plot,
        'scatter_plot': scatter_plot,
        'dendrogram_plot': dendrogram_plot
    }
    
    return figures
    
def compute_mean_brightness(images):
    """Compute the mean brightness across a set of images."""
    mean_brightness = 0
    for img in images:
        if len(img.shape) == 3:  # color image
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, _, _ = cv2.split(img_lab)
        else:  # grayscale image
            l_channel = img
        mean_brightness += np.mean(l_channel)
    return mean_brightness / len(images)


def compute_mean_contrast(images):
    """Compute the mean contrast across a set of images."""
    mean_contrast = 0
    for img in images:
        if len(img.shape) == 3:  
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, _, _ = cv2.split(img_lab)
        else:  
            l_channel = img
        mean_contrast += np.std(l_channel)
    return mean_contrast / len(images)


def compute_sharpness(img_gray):
    """Compute the sharpness of a grayscale image using the Sobel operator."""
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.mean(grad)   
    
    
def compute_mean_sharpness(images):
    """Compute the mean sharpness across a set of images."""
    mean_sharpness = 0
    for img in images:
        if len(img.shape) == 3: 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: 
            img_gray = img
        mean_sharpness += compute_sharpness(img_gray)
    return mean_sharpness / len(images)


def normalize_brightness_set(images, mean_brightness):
    """Normalize brightness of all images in the set to the mean brightness."""
    normalized_images = []
    for img in images:
        if len(img.shape) == 3:  # color image
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            current_brightness = np.mean(l_channel)
            print(f"Original brightness: {current_brightness}")
            l_channel = (l_channel * (mean_brightness / current_brightness)).clip(0, 255).astype(np.uint8)
            normalized_img_lab = cv2.merge([l_channel, a_channel, b_channel])
            normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)
            print(f"Normalized brightness: {np.mean(l_channel)}")
        
        else:  # grayscale image
            l_channel = img
            current_brightness = np.mean(l_channel)
            print(f"Original brightness: {current_brightness}")
            normalized_img = (l_channel * (mean_brightness / current_brightness)).clip(0, 255).astype(np.uint8)
            print(f"Normalized brightness: {np.mean(normalized_img)}")
        
        normalized_images.append(normalized_img)
    
    return normalized_images

def normalize_contrast_set(images, mean_contrast):
    """Normalize contrast of all images in the set to the mean contrast."""
    normalized_images = []
    for img in images:
        if len(img.shape) == 3: # color
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            current_contrast = np.std(l_channel)
            print(f"Original contrast: {current_contrast}")
            l_channel = (l_channel * (mean_contrast / current_contrast)).clip(0, 255).astype(np.uint8)
            normalized_img_lab = cv2.merge([l_channel, a_channel, b_channel])
            normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)
            print(f"Normalized contrast: {np.std(l_channel)}")
        
        else:  # grayscale
            l_channel = img
            current_contrast = np.std(l_channel)
            print(f"Original contrast: {current_contrast}")
            normalized_img = (l_channel * (mean_contrast / current_contrast)).clip(0, 255).astype(np.uint8)
            print(f"Normalized contrast: {np.std(normalized_img)}")
        
        normalized_images.append(normalized_img)
    
    return normalized_images


def normalize_sharpness_set(images, target_sharpness):
    """Normalize sharpness of all images in the set to the target sharpness."""
    normalized_images = []
    
    for img in images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        else:
            img_gray = img
        
        original_sharpness = compute_sharpness(img_gray)
        print(f"Original sharpness: {original_sharpness}")

        # Apply unsharp mask
        blurred = cv2.GaussianBlur(img_gray, (0, 0), 3)
        sharpened = cv2.addWeighted(img_gray, 1.5, blurred, -0.5, 0)
        
        # Compute the new sharpness after applying the unsharp mask
        sharpened_sharpness = compute_sharpness(sharpened)
        
        # Scale to match the target sharpness
        scaling_factor = target_sharpness / sharpened_sharpness
        sharpened = (sharpened * scaling_factor).clip(0, 255).astype(np.uint8)
        
        # If the original image was in color, convert the sharpened grayscale back to color
        if len(img.shape) == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = sharpened
            normalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            normalized_img = sharpened
        
        normalized_sharpness = compute_sharpness(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else normalized_img)
        print(f"Normalized sharpness: {normalized_sharpness}")
        
        normalized_images.append(normalized_img)
    
    return normalized_images


def equalize_images(images, brightness=False, contrast=False, sharpness=False):
    """Preprocesses the input images based on the selected enhancement options.

    Parameters:
    - images: The input images in BGR or gray format.
    - brightness: Whether to equalize brightness.
    - contrast: Whether to equalize contrast.
    - sharpness: Whether to equalize sharpness.    

    Returns:
    - images: the images with identical brightnes, cintrast and sharpness
    """
    if brightness:
        mean_brightness = compute_mean_brightness(images)
        images = normalize_brightness_set(images, mean_brightness)

    if contrast:
        mean_contrast = compute_mean_contrast(images)
        images = normalize_contrast_set(images, mean_contrast)

    if sharpness:
        mean_sharpness = compute_mean_sharpness(images)
        images = normalize_sharpness_set(images, mean_sharpness)

    return images

def compute_metrics(images):
    """Computes metrics (mean and standard deviation) for normalized brightness, contrast, and sharpness.

    Parameters:
    - images: The input images in BGR or gray format.

    Returns:
    - metrics: Dictionary containing mean and standard deviation of normalized metrics.
    """
    normalized_brightness = []
    normalized_contrast = []
    normalized_sharpness = []
    
    for img in images:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(img_lab)
        
        # Compute normalized brightness and contrast
        normalized_brightness.append(np.mean(l_channel))
        normalized_contrast.append(np.std(l_channel))
        
        # Compute normalized sharpness
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized_sharpness.append(compute_sharpness(img_gray))
    
    # Compute means and standard deviations, rounded to 2 decimal places
    mean_normalized_brightness = np.round(np.mean(normalized_brightness), 2)
    mean_normalized_contrast = np.round(np.mean(normalized_contrast), 2)
    mean_normalized_sharpness = np.round(np.mean(normalized_sharpness), 2)
    
    sd_normalized_brightness = np.round(np.std(normalized_brightness), 2)
    sd_normalized_contrast = np.round(np.std(normalized_contrast), 2)
    sd_normalized_sharpness = np.round(np.std(normalized_sharpness), 2)
    
    metrics = {
        'mean_normalized_brightness': mean_normalized_brightness,
        'mean_normalized_contrast': mean_normalized_contrast,
        'mean_normalized_sharpness': mean_normalized_sharpness,
        'sd_normalized_brightness': sd_normalized_brightness,
        'sd_normalized_contrast': sd_normalized_contrast,
        'sd_normalized_sharpness': sd_normalized_sharpness
    }
    
    return metrics
