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
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans, DBSCAN
from sklearn import metrics
from pystackreg import StackReg
from skimage import io, color, transform

SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
NUM_THREADS = 8

def align_images_to_mean(
    image: dict[str, np.ndarray],
    *,
    selected_option: str,
    motion_model: str,
    feature_method: str = 'mean',
    #input_dir, 
    target_size = SIM_IMAGE_SIZE
) -> dict[str, dict[str, Any]]:
   
    
    """
    Aligns a set of images to the mean image.

    Args:
        input_dir (str): The directory containing the input images.
        target_size (tuple, optional): The target size to resize the images to. 
        Defaults to (640, 480).

    Returns:
        numpy.ndarray: The aligned images stack.

    Raises:
        None

    """
    
    #jpg_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

    # Resize images to predefined target size
    def resize_image(image, target_size):
        return transform.resize(image, target_size, anti_aliasing=True, preserve_range=True).astype(image.dtype)
    
    resized_images = []

    # Load each JPG file, convert to grayscale, resize it, and add it to the list
    for file_name in image:
        #image_path = os.path.join(input_dir, file_name)
        image = io.imread(file_name)
        gray_image = color.rgb2gray(image)
        resized_image = resize_image(gray_image, target_size)
        resized_images.append(resized_image)
        
    image_stack = np.stack(resized_images, axis=0)

    if selected_option == 'pyStackReg':
        
        # Initialize StackReg and select the transformation type
        if motion_model == 'translation':            
            sr = StackReg(StackReg.TRANSLATION)
        elif motion_model == 'rigid body (translation + rotation)':
            sr = StackReg(StackReg.RIGID_BODY)
        elif motion_model == 'scaled rotation (translation + rotation + scaling)':
            sr = StackReg(StackReg.SCALED_ROTATION)
        elif motion_model == 'affine (translation + rotation + scaling + shearing)':
            sr = StackReg(StackReg.AFFINE)
        elif motion_model == 'bilinear (non-linear transformation; does not preserve straight lines)':            
            sr = StackReg(StackReg.BILINEAR)
            
    if feature_method == 'to first image':
        # Registering the images to the first image
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = "first")
    elif feature_method == 'to mean image':
        # Registering the images to the mean image
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = 'mean')
    elif feature_method == 'each image to the previous (already registered) one':
        aligned_images_stack = sr.register_transform_stack(image_stack, reference = 'previous')
        
    return aligned_images_stack #np.aray

def get_image_similarity(img1, img2, algorithm = 'SSIM'):
    
    """ Returns the normalized similarity value (from 0.0 to 1.0) for the provided 
    pair of images. """
    
    if isinstance(img1, str):
        i1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    else:
        i1 = img1.astype(np.uint8)
    
    if isinstance(img2, str):
        i2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    else:
        i2 = img2.astype(np.uint8)

    i1 = cv2.resize(i1, SIM_IMAGE_SIZE)
    i2 = cv2.resize(i2, SIM_IMAGE_SIZE)

    if algorithm == 'SIFT':
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(i1, None)
        k2, d2 = sift.detectAndCompute(i2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        good_matches = [m for m, n in matches if m.distance < SIFT_RATIO * n.distance]
        similarity = len(good_matches) / max(len(matches), 1)
    
    elif algorithm == 'SSIM':
        similarity, _ = ssim(i1, i2, full=True)
    
    elif algorithm == 'MSE':
        err = np.sum((i1.astype("float") - i2.astype("float")) ** 2)
        err /= float(i1.shape[0] * i2.shape[1])
        similarity = MSE_NUMERATOR / err if err > 0 else 1.0
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return similarity

def build_similarity_matrix(images, algorithm='SSIM'):
    """
    Builds a similarity matrix for a given set of images.

    Args:
        images (numpy.ndarray): An array of images
        algorithm (str, optional): The algorithm to use for computing image similarity. 
        Defaults to 'SSIM'.

    Returns:
        numpy.ndarray: The similarity matrix.

    """
    num_images = images.shape[0]
    sm = np.zeros((num_images, num_images), dtype=np.float64)
    np.fill_diagonal(sm, 1.0)

    def compute_similarity(i, j):
        if i != j:
            return get_image_similarity(images[i], images[j], algorithm)
        return 1.0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [(i, j, executor.submit(compute_similarity, i, j))
                   for i in range(num_images) for j in range(i + 1, num_images)]
        for i, j, future in futures:
            sm[i, j] = sm[j, i] = future.result()
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
    
    # Set diagonal elements of the matrix to zero
    np.fill_diagonal(X, 0)
    
    metrics_dict = {
        'Silhouette coefficient': metrics.silhouette_score(X, labels, metric='precomputed'),
        'Davies-Bouldin index': metrics.davies_bouldin_score(X, labels)
    }
    if labels_true is not None:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)    
    
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
        kmeans = KMeans(n_clusters=k).fit(matrix)
        inertias.append(kmeans.inertia_)
    
    # Plot the inertia to visualize the elbow point
    plt.figure()
    plt.plot(range(2, max_clusters+1), inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal Clusters')
    plt.show()

    # Identify the elbow point
    elbow_point = 2  # default value
    if len(inertias) > 1:
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 2  # +2 because the range starts from 2
    return elbow_point


def cluster_images(images, 
                   algorithm     = 'SSIM', 
                   n_clusters    = None,
                   method        = 'spectral', 
                   print_metrics = True, 
                   labels_true   = None):
    """
    Clusters a list of images using different clustering algorithms.

    Parameters:
    - images (list): A list of images to be clustered.
    - algorithm (str): The algorithm used to calculate the similarity between images. Default is 'SSIM'.
    - n_clusters (int): The number of clusters to create. If None, the optimal number of clusters will be determined automatically. Default is None.
    - method (str): The clustering method to use. Options are 'spectral', 'affinity', and 'dbscan'. Default is 'spectral'.
    - print_metrics (bool): Whether to print performance metrics for each clustering method. Default is True.
    - labels_true (list): The true labels for the images, used for evaluating clustering performance. Default is None.

    Returns:
    - labels (list): The cluster labels assigned to each image.

    """
    
    matrix = build_similarity_matrix(images, algorithm = algorithm)

    if n_clusters is None and method != 'dbscan':
        n_clusters = determine_optimal_clusters(matrix)
    
    if method == 'spectral':
        sc = SpectralClustering(n_clusters = n_clusters, affinity='precomputed').fit(matrix)
        sc_metrics = get_cluster_metrics(matrix, sc.labels_, labels_true)

        if print_metrics:
            print("\nPerformance metrics for Spectral Clustering")
            print(f"Number of clusters: {len(set(sc.labels_))}")
            for k, v in sc_metrics.items():
                print(f"{k}: {v:.2f}")
        return sc.labels_
        
    elif method == 'affinity':
        af = AffinityPropagation(affinity='precomputed').fit(matrix)
        af_metrics = get_cluster_metrics(matrix, af.labels_, labels_true)

        if print_metrics:
            print("\nPerformance metrics for Affinity Propagation Clustering")
            print(f"Number of clusters: {len(set(af.labels_))}")
            for k, v in af_metrics.items():
                print(f"{k}: {v:.2f}")
        return af.labels_    
    
    elif method == 'dbscan':
        db = DBSCAN(metric='precomputed', eps=0.5, min_samples=2).fit(matrix)
        db_labels = db.labels_
        db_metrics = get_cluster_metrics(matrix, db_labels, labels_true)

        if print_metrics:
            print("\nPerformance metrics for DBSCAN Clustering")
            print(f"Number of clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")
            for k, v in db_metrics.items():
                print(f"{k}: {v:.2f}")
        return db_labels
    


#TODO: Implement equalize_images_widget by equalizing accross all images 
def normalize_brightness(template, target):
    """Normalizes the brightness of the target image based on the luminance of
    the template image. This refers to the process of bringing the brightness
    of the target image into alignment with the brightness of the template
    image. This can help ensure consistency in brightness perception between
    the two images, which is particularly useful in applications such as image
    comparison, enhancement, or blending.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.
    .

    Returns:
    - equalized_img: Adjusted target image with equalized brightness

    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to BGR.
    """

    # Convert the template image to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the template image
    l_template, a_template, b_template = cv2.split(template_lab)

    # Convert the target image to LAB color space
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the target image
    l_target, a_target, b_target = cv2.split(target_lab)

    # Adjust the L channel (brightness) of the target image based
    # on the mean brightness of the template
    l_target = (
        (l_target * (np.mean(l_template) / np.mean(l_target))).clip(0, 255).astype(np.uint8)
    )

    # Merge LAB channels back for the adjusted target image
    equalized_img_lab = cv2.merge([l_target, a_target, b_target])

    # Convert the adjusted LAB image back to BGR
    equalized_img = cv2.cvtColor(equalized_img_lab, cv2.COLOR_LAB2BGR)

    ## Using LAB color space
    # we convert the images (template and eq_image) to the LAB color space and calculate
    # the mean brightness from the luminance channel (L) only

    # Calculate the mean of the color images from the luminance channel
    mean_template_lab = np.mean(cv2.split(template_lab)[0])
    mean_eq_image_lab = np.mean(cv2.split(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2LAB))[0])

    # The ratio is computed based on the mean brightness of the L channel for both color images
    ratio_lab = mean_template_lab / mean_eq_image_lab

    ## Using RGB color space
    # We calculate the mean intensity across all color channels (R, G, B) for both images
    # (template and equalized_image), i.e., the ratio is computed based on the mean intensity
    # across all color channels for both images
    mean_template_rgb = np.mean(template)
    mean_eq_image_rgb = np.mean(equalized_img)

    # ratio of the brightness of the images
    ratio_rgb = mean_template_rgb / mean_eq_image_rgb

    # Calculate the mean of the grayscale images
    mean_template_gray = np.mean(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    mean_eq_image_gray = np.mean(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY))
    # Calculate the ratio of the brightness of the grayscale images
    ratio_gray = mean_template_gray / mean_eq_image_gray

    print(f'Brightness ratio (LAB): {ratio_lab}')
    print(f'Brightness ratio (RGB): {ratio_rgb}')
    print(f'Brightness ratio (Grayscale): {ratio_gray}')

    return equalized_img


def normalize_contrast(template, target):
    """Normalize the contrast of the target image to match the contrast of the
    template image.

    Parameters:
    - template: Reference image (template) in BGR or grayscale format.
    - target: Target image to be adjusted in BGR or grayscale format.

    Returns:
    - normalized_img: Target image with contrast normalized to match the template image.
    """
    if len(template.shape) == 3 and len(target.shape) == 3:  # Both images are color
        # Convert images to LAB color space
        template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # Calculate contrast metric (standard deviation) for L channel of both images
        std_template = np.std(template_lab[:, :, 0])
        std_target = np.std(target_lab[:, :, 0])

        # Adjust contrast of target image to match template image
        l_target = (
            (target_lab[:, :, 0] * (std_template / std_target)).clip(0, 255).astype(np.uint8)
        )
        normalized_img_lab = cv2.merge([l_target, target_lab[:, :, 1], target_lab[:, :, 2]])

        # Convert the adjusted LAB image back to BGR
        normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)

        print(f'Contrast value (template): {std_template}')
        print(f'Contrast value (target): {std_target}')
        print(f'Contrast ratio: {std_template / std_target}')
        print(f'Adapted value (target): {np.std(normalized_img_lab[:, :, 0])}')

    else:
        # Both images are grayscale
        # Calculate contrast metric (standard deviation) for grayscale intensity of both images
        std_template = np.std(template)
        std_target = np.std(target)

        # Adjust contrast of target image to match template image
        normalized_img = (target * (std_template / std_target)).clip(0, 255).astype(np.uint8)

        print(f'Contrast value (template): {std_template}')
        print(f'Contrast value (target): {std_target}')
        print(f'Contrast ratio: {std_template / std_target}')
        print(f'Adapted value (target): {np.std(normalized_img_lab[:, :, 0])}')

    return normalized_img


def normalize_sharpness(template, target):
    """Normalize the sharpness of the target image to match the sharpness of
    the template image.

    Parameters
    ----------
    template : ...
        Reference image (template) in BGR or grayscale format.
    target : ...
        Target image to be adjusted in BGR or grayscale format.

    Returns
    -------
    normalized_img : ...
        Target image with sharpness normalized to match the template image.
    """
    if len(template.shape) == 3 and len(target.shape) == 3:
        # Both images are color
        # Convert images to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
        target_gray = target

    # Calculate image gradients for both images
    grad_x_template = cv2.Sobel(template_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_template = cv2.Sobel(template_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_template = cv2.magnitude(grad_x_template, grad_y_template)
    grad_x_target = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_target = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_target = cv2.magnitude(grad_x_target, grad_y_target)

    # Calculate sharpness metric (mean gradient magnitude) for both images
    mean_grad_template = np.mean(grad_template)
    mean_grad_target = np.mean(grad_target)

    # Adjust sharpness of target image to match template image
    normalized_img = (
        (target * (mean_grad_template / mean_grad_target)).clip(0, 255).astype(np.uint8)
    )

    # Print sharpness values
    print(f'Sharpness value (template): {mean_grad_template}')
    print(f'Sharpness value (target): {mean_grad_target}')
    print(f'Sharpness ratio: {mean_grad_template / mean_grad_target}')

    # Calculate sharpness value for the normalized image
    grad_x_normalized = cv2.Sobel(
        cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3
    )
    grad_y_normalized = cv2.Sobel(
        cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3
    )
    grad_normalized = cv2.magnitude(grad_x_normalized, grad_y_normalized)
    mean_grad_normalized = np.mean(grad_normalized)
    print(f'Sharpness value (normalized): {mean_grad_normalized}')

    return normalized_img


def normalize_colors(template, target):
    """Normalize the colors of the target image to match the color distribution
    of the template image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - normalized_img: Target image with colors normalized to match the template image.
    """

    matched = match_histograms(target, template, channel_axis=-1)
    return matched


def get_mean_and_std(x):
    """Calculate the mean and standard deviation of each channel in an image.

    Parameters:
    - x: Input image in BGR color format.

    Returns:
    - mean: Mean values of each channel.
    - std: Standard deviation of each channel.
    """
    mean, std = cv2.meanStdDev(x)
    mean = np.around(mean.flatten(), 2)
    std = np.around(std.flatten(), 2)
    return mean, std


def reinhard_color_transfer(template, target):
    """Perform Reinhard color transfer from the template image to the target
    image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - adjusted_img: Target image with colors adjusted using Reinhard color transfer.
    """
    # Convert images to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Compute mean and standard deviation of each channel in LAB color space
    template_mean, template_std = get_mean_and_std(template_lab)
    target_mean, target_std = get_mean_and_std(target_lab)

    # Apply color transfer
    target_lab = ((target_lab - target_mean) * (template_std / target_std)) + template_mean
    target_lab = np.clip(target_lab, 0, 255)

    # Convert back to BGR color space
    adjusted_img = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return adjusted_img

def equalize_images(
    template: np.ndarray,
    image: np.ndarray,
    *,
    brightness=False,
    contrast=False,
    sharpness=False,
    color=False,
    reinhard=False,
    ):
    """Preprocesses the input image based on the selected enhancement options.

    Parameters:
    - image: The input image in BGR format.
    - brightness: Whether to equalize brightness.
    - contrast: Whether to equalize contrast.
    - sharpness: Whether to equalize sharpness.
    - color: Whether to equalize colors.

    Returns:
    - preprocessed_image: The preprocessed image.
    """
    if brightness:
        image = normalize_brightness(template, image)

    if contrast:
        image = normalize_contrast(template, image)

    if sharpness:
        image = normalize_sharpness(template, image)

    if color:
        image = normalize_colors(template, image)

    if reinhard:
        image = reinhard_color_transfer(template, image)

    return image
