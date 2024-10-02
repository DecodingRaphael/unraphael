"""Module for clustering of images based on structural similarity, including
the equalization of contrast, sharpness, and brightness across the images and
alignment of images to their mean to optimize the clustering process."""


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

from multiprocessing import Pool
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import piq
import ssim.ssimlib as pyssim
import torch
from clusteval import clusteval
from clustimage import Clustimage
from PIL import Image
from pystackreg import StackReg
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage import color, transform
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from sklearn import metrics
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans, SpectralClustering
from sklearn.decomposition import PCA

SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
NUM_THREADS = 8


def align_images_to_mean(
    images: dict[str, np.ndarray],
    *,
    motion_model: Optional[str] = None,
    feature_method: Optional[str] = None,
    target_size: tuple = SIM_IMAGE_SIZE,
) -> dict[str, np.ndarray]:
    """Aligns images based on the selected alignment option and motion model.

    Parameters
    ----------
    images : dict of str, np.ndarray
        Dictionary of images where keys are image names and values are the
        corresponding image arrays.
    motion_model : str, optional
        The motion model to be used for alignment. Defaults to None.
    feature_method : str, optional
        The feature method used for alignment. Defaults to None.
    target_size : tuple, optional
        The target size to resize the images to. Defaults to (640, 480).

    Returns
    -------
    dict of str, np.ndarray
        Dictionary of aligned images with the same keys as the input dictionary.
    """

    def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        return transform.resize(
            image, target_size, anti_aliasing=True, preserve_range=True
        ).astype(image.dtype)

    def ensure_grayscale(image: np.ndarray) -> np.ndarray:
        # make grayscale if in color
        if image.ndim == 3:
            return color.rgb2gray(image)
        return image

    resized_images = {
        name: resize_image(ensure_grayscale(image), target_size)
        for name, image in images.items()
    }
    image_stack = np.stack(list(resized_images.values()), axis=0)

    if image_stack.ndim != 3:
        raise ValueError('Image stack must have three dimensions (num_images, height, width).')

    # Return resized images if no motion model or feature method is selected
    if motion_model is None or feature_method is None:
        return resized_images

    # Select the type of transformation to align the images
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

    # Perform alignment/registering
    if feature_method == 'to first image':
        aligned_images_stack = sr.register_transform_stack(image_stack, reference='first')
    elif feature_method == 'to mean image':
        aligned_images_stack = sr.register_transform_stack(image_stack, reference='mean')
    elif feature_method == 'each image to the previous (already registered) one':
        aligned_images_stack = sr.register_transform_stack(image_stack, reference='previous')
    else:
        raise ValueError('Invalid feature method selected.')

    aligned_images = {name: aligned_images_stack[i] for i, name in enumerate(images.keys())}

    return aligned_images


# Brushstroke Analysis -----------------------------------------------------
# This set of functions analyzes brushstrokes in images by extracting various
# edge detection metrics. The methods implemented here focus
# on identifying and quantifying the characteristics of brushstrokes as they
# manifest in the form of edges in an image.
def calculate_canny_edges(img) -> float:
    """Calculates the standard deviation of the edges detected in the input
    image using the Canny edge detection algorithm.

    Parameters
    ----------
    img : np.ndarray
        Input image in which edges are to be detected. The image should be
        in grayscale format.

    Returns
    -------
    float
        The sd of the detected edges.
    """
    edges = cv2.Canny(img, 100, 200)
    return np.std(edges)


def calculate_sobel_edges(img) -> tuple[float, float]:
    """Computes the standard deviation of the Sobel edges in both x and y
    directions.

    Parameters
    ----------
    img : np.ndarray
        Input image for edge detection.

    Returns
    -------
    tuple[float, float]
        Standard deviations of the Sobel edges in the x and y directions.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.std(sobelx), np.std(sobely)


def calculate_laplacian_edges(img) -> float:
    """Computes the standard deviation of the Laplacian edges in the image.

    Parameters
    ----------
    img : np.ndarray
        Input image for edge detection.

    Returns
    -------
    float
        Standard deviation of the Laplacian edges.
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return np.std(laplacian)


def calculate_scharr_edges(img) -> tuple[float, float]:
    """Computes the standard deviation of the Scharr edges in both x and y
    directions.

    Parameters
    ----------
    img : np.ndarray
        Input image for edge detection.

    Returns
    -------
    tuple[float, float]
        Standard deviations of the Scharr edges in the x and y directions.
    """
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    return np.std(scharrx), np.std(scharry)


def calculate_histogram_features(img) -> np.ndarray:
    """Calculates the histogram features of the input image.

    Parameters
    ----------
    img : np.ndarray
        Input image for histogram calculation.

    Returns
    -------
    np.ndarray
        Flattened histogram of the image.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()


def calculate_features(img) -> np.ndarray:
    """Calculates various features related to brushstroke analysis in the input
    image.

        Parameters
        ----------
        img : numpy.ndarray
            The input image for feature extraction. The image should be a
            single-channel (grayscale) image.

        Returns
        -------
        numpy.ndarray
            A 1D array of features, consisting of:
            - Canny edge standard deviation.
            - Sobel edge standard deviations in the x and y directions.
            - Laplacian edge standard deviation.
            - Scharr edge standard deviations in the x and y directions.
            - Flattened grayscale histogram (256 values).
    Notes
    -----
    - The edge detection features are computed as the standard deviation of
    the edges detected by Canny, Sobel (x and y), Laplacian, and Scharr (x and y) operators.
    - NaN values in the calculated features are replaced with 0 to avoid computational issues.
    """
    canny_edges = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y = calculate_sobel_edges(img)
    laplacian_edges = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)
    histogram_features = calculate_histogram_features(img)

    features = np.concatenate(
        [
            np.array(
                [
                    np.nan_to_num(canny_edges),
                    np.nan_to_num(sobel_edges_x),
                    np.nan_to_num(sobel_edges_y),
                    np.nan_to_num(laplacian_edges),
                    np.nan_to_num(scharr_edges_x),
                    np.nan_to_num(scharr_edges_y),
                ]
            ),
            histogram_features,
        ]
    )
    return features


def calculate_sift_similarity(i1: np.ndarray, i2: np.ndarray) -> float:
    """Calculate similarity using SIFT."""
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(i1, None)
    k2, d2 = sift.detectAndCompute(i2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    good_matches = [m for m, n in matches if m.distance < SIFT_RATIO * n.distance]
    return len(good_matches) / max(len(matches), 1)


def calculate_ssim_similarity(i1: np.ndarray, i2: np.ndarray) -> float:
    """Calculate similarity using Structural Similarity Index."""
    similarity, _ = ssim(i1, i2, full=True)
    return similarity


def calculate_cw_ssim_similarity(i1: np.ndarray, i2: np.ndarray) -> float:
    """Calculate similarity using Complex Wavelet SSIM.

    Strong for handling small geometric distortions in structural
    comparison.
    """
    pil_img1 = Image.fromarray(i1)
    pil_img2 = Image.fromarray(i2)
    return pyssim.SSIM(pil_img1).cw_ssim_value(pil_img2)


def calculate_iw_ssim_similarity(i1_torch: torch.Tensor, i2_torch: torch.Tensor) -> float:
    """Calculate similarity using Information-Weighted SSIM.

    This metric gives more weight to important structural regions of the
    image based on image information content.
    """
    iw_ssim_similarity = piq.information_weighted_ssim(i1_torch, i2_torch, data_range=1.0)
    return iw_ssim_similarity


def calculate_fsim_similarity(i1_torch: torch.Tensor, i2_torch: torch.Tensor) -> float:
    """Calculate similarity using Feature Similarity Index."""
    fsim_similarity = piq.fsim(
        i1_torch, i2_torch, data_range=1.0, reduction='none', chromatic=False
    ).item()
    return fsim_similarity


def calculate_mse_similarity(i1: np.ndarray, i2: np.ndarray) -> float:
    """Calculate similarity using Mean Squared Error."""
    err = np.sum((i1.astype('float') - i2.astype('float')) ** 2)
    err /= float(i1.shape[0] * i2.shape[1])
    return MSE_NUMERATOR / err if err > 0 else 1.0


def calculate_brushstroke_similarity(i1: np.ndarray, i2: np.ndarray) -> float:
    """Calculate similarity based on brushstroke features."""
    features_i1 = calculate_features(i1)

    # Normalize features to get weights
    weights = features_i1 / np.sum(features_i1)
    features_i2 = calculate_features(i2)

    # Replace NaN values with 0
    features_i2 = np.nan_to_num(features_i2 * weights)

    # Compare features
    difference = np.abs(features_i1 - features_i2)
    return np.mean(difference)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess the input image by converting it to uint8, creating a binary
    mask, and applying the mask to isolate the foreground.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to preprocess.

    Returns
    -------
    np.ndarray
        The preprocessed image with foreground isolated.
    """
    # Ensure the image is of type np.uint8
    img_gray = img.astype(np.uint8)

    # Create a mask to isolate the foreground
    _, mask = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    # Apply the mask to isolate the foreground
    img_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)

    return img_masked


def to_torch(img: np.ndarray) -> torch.Tensor:
    """Converts a NumPy array to a Torch tensor and normalizes it."""
    if img.ndim == 2:  # Grayscale case
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
    torch_img = torch.from_numpy(img).permute(2, 0, 1)[None, ...] / 255.0  # Normalize to [0, 1]
    return torch_img


def get_image_similarity(img1: np.ndarray, img2: np.ndarray, algorithm: str = 'SSIM') -> float:
    """Returns the normalized similarity value (from 0.0 to 1.0) for the
    provided pair of images, using the specified algorithm.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first input image for comparison.
    img2 : numpy.ndarray
        The second input image for comparison.
    algorithm : str, optional
        The algorithm used to calculate the similarity between the images.

    Returns
    -------
    float
        The normalized similarity value between the two images.
    Raises
    ------
    ValueError
        If an unsupported algorithm is provided.
    """

    # Preprocess both images
    i1 = preprocess(img1)
    i2 = preprocess(img2)

    # Convert images to torch tensors for torch-based algorithms
    i1_torch = to_torch(i1)
    i2_torch = to_torch(i2)

    # Ensure both images are the same size for torch-based methods
    i1_torch = torch.nn.functional.interpolate(
        i1_torch, size=i2_torch.size()[2:], mode='bilinear', align_corners=False
    )

    if algorithm == 'SIFT':
        return calculate_sift_similarity(i1, i2)

    elif algorithm == 'SSIM':
        return calculate_ssim_similarity(i1, i2)

    elif algorithm == 'CW-SSIM':
        return calculate_cw_ssim_similarity(i1, i2)

    elif algorithm == 'IW-SSIM':
        return calculate_iw_ssim_similarity(i1_torch, i2_torch)

    elif algorithm == 'FSIM':
        return calculate_fsim_similarity(i1_torch, i2_torch)

    elif algorithm == 'MSE':
        return calculate_mse_similarity(i1, i2)

    elif algorithm == 'Brushstrokes':
        return calculate_brushstroke_similarity(i1, i2)

    else:
        raise ValueError(f'Unsupported algorithm: {algorithm}')


def compute_similarity(args):
    i, j, images, algorithm = args
    if i != j:
        return i, j, get_image_similarity(images[i], images[j], algorithm)
    return i, j, 1.0


def build_similarity_matrix(
    images: list[np.ndarray], algorithm: str = 'SSIM', fill_diagonal_value: float = 0.0
) -> np.ndarray:
    """Builds a similarity matrix for a set of images.

    For AffinityPropagation, SpectralClustering, and DBSCAN, one can input
    similarity matrices of shape (n_samples, n_samples). This function builds
    such a similarity matrix for the given set of n images.

    Args:
        images (list[np.ndarray]): A list of images.
        algorithm (str, optional): Select image similarity index. Defaults to 'SSIM'.
        fill_diagonal_value (float, optional): Value to fill the diagonal with,
        typically 1 for perfect self-similarity.

    Returns:
        np.ndarray: The similarity matrix of shape (n_samples, n_samples).
    """

    num_images = len(images)
    sm = np.zeros((num_images, num_images), dtype=np.float64)
    np.fill_diagonal(sm, fill_diagonal_value)

    # Prepare arguments for multiprocessing
    args = [
        (i, j, images, algorithm) for i in range(num_images) for j in range(i + 1, num_images)
    ]

    # Use multiprocessing to compute similarities
    with Pool() as pool:
        results = pool.map(compute_similarity, args)

    # Fill the similarity matrix with the results
    for i, j, similarity in results:
        sm[i, j] = sm[j, i] = similarity

    return sm


def get_cluster_metrics(
    X: np.ndarray, labels: np.ndarray, labels_true: Optional[np.ndarray] = None
) -> dict[str, float]:
    """Calculate cluster evaluation metrics based on the given data and labels.

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

    metrics_dict = {}

    if len(set(labels)) > 1:
        metrics_dict['Silhouette coefficient'] = metrics.silhouette_score(
            X, labels, metric='precomputed'
        )
        metrics_dict['Davies-Bouldin index'] = metrics.davies_bouldin_score(1 - X, labels)

    if labels_true is not None:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)

        metrics_dict['V-measure'] = metrics.v_measure_score(labels_true, labels)
        metrics_dict['Adjusted Rand index'] = metrics.adjusted_rand_score(labels_true, labels)
        metrics_dict['Adjusted mutual information'] = metrics.adjusted_mutual_info_score(
            labels_true, labels
        )

    return metrics_dict


def determine_optimal_clusters(
    matrix: np.ndarray,
    method: str = 'elbow',
    cluster_method: str = 'kmeans',
    metric: str = 'euclidean',
    linkage: str = 'ward',
    min_clust: int = 2,
    max_clust: int = 10,
) -> int:
    """Determines the optimal number of clusters using various evaluation
    methods.

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix for clustering.
    method : str
        The cluster evaluation method. Options are 'elbow', 'silhouette',
        'dbindex', 'derivative'.
    cluster_method : str
        The clustering method to use. Options are 'kmeans', 'agglomerative',
        'dbscan', 'hdbscan'.
    metric : str
        The distance metric to use.
    linkage : str
        The linkage method for agglomerative clustering.
    min_clust : int
        The minimum number of clusters to evaluate.
    max_clust : int
        The maximum number of clusters to evaluate.

    Returns
    -------
    int
        The optimal number of clusters.
    """

    if method == 'elbow':
        inertias = []
        for k in range(min_clust, max_clust + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(matrix)
            inertias.append(kmeans.inertia_)

        # Calculate the second derivative to find the elbow point
        if len(inertias) > 2:
            diffs = np.diff(inertias, 2)
            optimal_clusters = (
                np.argmin(diffs) + min_clust + 1
            )  # +1 due to the way np.diff works
        else:
            optimal_clusters = (
                min_clust + 1
            )  # Fallback if not enough points for second derivative
    else:
        ce = clusteval(
            cluster=cluster_method,
            evaluate=method,
            metric=metric,
            linkage=linkage,
            min_clust=min_clust,
            max_clust=max_clust,
        )

        results = ce.fit(matrix)
        print(f'ClustEval results: {results}')

        if method == 'silhouette':
            # Access silhouette scores and corresponding cluster numbers
            silhouette_scores = results['score']['score']
            cluster_numbers = results['score']['clusters']

            # Find the index of the maximum silhouette score
            optimal_index = np.argmax(silhouette_scores)
            optimal_clusters = cluster_numbers[optimal_index]

            print(f'Optimal_clusters: {optimal_clusters}')

        # TODO: implement DBindex and Derivative method

    return optimal_clusters


def plot_clusters(
    images: dict, labels: np.ndarray, n_clusters: int, title: str = 'Clustering results'
) -> plt.Figure:
    """Plots the clustering results in 2D space using PCA.

    Parameters
    ----------
    images : list
        A list of images to be clustered.
    labels : list
        The cluster labels assigned to each image.
    n_clusters : int
        The number of clusters.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    # Flatten the images and reduce to 2D using PCA
    flattened_images = np.array([img.flatten() for img in images.values()])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_images)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap('viridis', n_clusters)

    for k in range(n_clusters):
        class_members = labels == k
        ax.scatter(
            reduced_data[class_members, 0],
            reduced_data[class_members, 1],
            color=colors(k),
            marker='.',
            label=f'Cluster {k}',
        )

    ax.set_title(title)
    ax.legend()

    return fig


def plot_dendrogram(
    images: dict, labels: np.ndarray, method: str = 'ward', title: str = 'Dendrogram'
) -> plt.Figure:
    """Plots a dendrogram for the clustering results.

    Parameters
    ----------
    images : list
        A list of images to be clustered.
    method : str, optional
        The linkage method to be used for the hierarchical clustering.
        Default is 'ward'.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    # Flatten the images for clustering
    flattened_images = np.array([img.flatten() for img in images.values()])

    # Compute the linkage matrix
    linked = linkage(flattened_images, method=method)

    # Plot the dendrogram
    fig, ax = plt.subplots(figsize=(10, 8))
    dendrogram(linked, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Distance')

    return fig


def cluster_images(
    images: list[np.ndarray],
    algorithm: str,
    n_clusters: int,
    method: str,
    print_metrics: bool = True,
    labels_true: np.ndarray = None,
) -> tuple[np.ndarray, int]:
    """Clusters a list of images using different clustering algorithms.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        A dictionary of images to be clustered.
    algorithm : str, optional
        The algorithm used to calculate the similarity between images.
        Default is 'SSIM'.
    n_clusters : int, optional
        The number of clusters to create. If None, the optimal number of
        clusters will be determined automatically. Default is None.
    method : str, optional
        The clustering method to use. Options are 'SpectralClustering',
        'AffinityPropagation', and 'DBSCAN'. Default is 'SpectralClustering'.
    print_metrics : bool, optional
        Whether to print performance metrics for each clustering method.
        Default is True.
    labels_true : list, optional
        The true labels for the images, used for evaluating clustering
        performance. Default is None.

    Returns
    -------
    list
        The cluster labels assigned to each image.
    """

    # main input for the clustering algorithm
    matrix = build_similarity_matrix(images, algorithm=algorithm)

    if n_clusters is None and method != 'DBSCAN':
        n_clusters = determine_optimal_clusters(matrix)

    # SpectralClustering requires the number of clusters to be specified in advance. It
    # works well for a small number of clusters, but is not advised for many clusters
    if method == 'SpectralClustering':
        sc = SpectralClustering(
            n_clusters=n_clusters, random_state=42, affinity='precomputed'
        ).fit(matrix)
        sc_metrics = get_cluster_metrics(matrix, sc.labels_, labels_true)

        if print_metrics:
            print('\nPerformance metrics for Spectral Clustering')
            print(f'Number of clusters: {len(set(sc.labels_))}')
            for k, v in sc_metrics.items():
                print(f'{k}: {v:.2f}')
        return sc.labels_, n_clusters

    # Affinity propagation is also appropriate for small to medium sized datasets
    elif method == 'AffinityPropagation':
        af = AffinityPropagation(affinity='precomputed', random_state=42).fit(matrix)
        af_metrics = get_cluster_metrics(matrix, af.labels_, labels_true)

        if print_metrics:
            print('\nPerformance metrics for Affinity Propagation Clustering')
            print(f'Number of clusters: {len(set(af.labels_))}')
            for k, v in af_metrics.items():
                print(f'{k}: {v:.2f}')
        return af.labels_, len(set(af.labels_))

    elif method == 'DBSCAN':
        db = DBSCAN(metric='precomputed', eps=0.5, min_samples=2).fit(matrix)
        db_labels = db.labels_
        db_metrics = get_cluster_metrics(matrix, db_labels, labels_true)

        if print_metrics:
            print('\nPerformance metrics for DBSCAN Clustering')
            print(f'Number of clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}')
            for k, v in db_metrics.items():
                print(f'{k}: {v:.2f}')
        return db_labels, len(set(db_labels)) - (1 if -1 in db_labels else 0)

    # Default return for unsupported methods
    else:
        raise ValueError(f'Unsupported clustering method: {method}')


def preprocess_images(images: list, target_shape: tuple = (128, 128)) -> np.ndarray:
    """Preprocesses a list of images by resizing them to a target shape.

    Parameters
    ----------
    images : list
        A list of input images.
    target_shape : tuple, optional
        The target shape to resize the images to. Defaults to (128, 128).

    Returns
    -------
    numpy.ndarray
        An array of preprocessed images.
    """
    preprocessed_images = []
    for image in images:
        resized_image = resize(image, target_shape, anti_aliasing=True)
        preprocessed_images.append(resized_image)
    return np.array(preprocessed_images)


def clustimage_clustering(
    images: list, method: str, evaluation: str, linkage_type: str
) -> dict:
    """Perform image clustering using the clustimage library, aiming to detect
    natural groups or clusters of images. It uses a multi-step proces of pre-
    processing the images, extracting the features, and evaluating the optimal
    number of clusters across the feature.

    - clustering approaches can be set to agglomerative, kmeans, dbscan and hdbscan.
    - cluster evaluation can be performed based on Silhouette scores,
    Davies–Bouldin index, or Derivative method


    Args:
        images (list): A list of input images.

    Returns:
        dict: A dictionary containing the clustering results. The dictionary has the
        following keys:
            - 'scatter_plot': The scatter plot of the clustered images.
            - 'dendrogram_plot': The dendrogram plot of the clustering hierarchy.
    """
    imgs = preprocess_images(images)
    flattened_imgs = np.array([img.flatten() for img in imgs])

    # HOG is most likely not the best feature extraction method for this task; see
    # https://erdogant.github.io/clustimage/pages/html/Feature%20Extraction.html.
    # Therefore, we use PCA for dimensionality reduction
    cl = Clustimage(method='pca', params_pca={'n_components': 0.95})

    imported = cl.import_data(flattened_imgs)

    # Preprocessing, feature extraction, embedding and detection of the
    # optimal number of clusters
    results = cl.fit_transform(
        imported,
        cluster=method,
        evaluate=evaluation,
        metric='euclidean',
        linkage=linkage_type,
        min_clust=1,
        max_clust=25,
        cluster_space='low',
    )  # cluster on the 2-D embedded space

    print(results)

    # Collect plots and save them to a dictionary 'figures'
    evaluation_plot = cl.clusteval.plot()  # silhoutte versus nr of clusters
    eval_plot = cl.clusteval.scatter(cl.results['xycoord'])
    scatter_plot = cl.scatter(
        zoom=None,
        dotsize=200,
        figsize=(25, 15),
        args_scatter={'fontsize': 24, 'gradient': '#FFFFFF', 'cmap': 'Set2', 'legend': True},
    )  # tsne plot
    dendrogram_plot = cl.dendrogram()[
        'ax'
    ].figure  # ensure cl.dendrogram() returns the figure object from 'ax'

    figures = {
        'evaluation_plot': evaluation_plot,
        'eval_plot': eval_plot,
        'scatter_plot': scatter_plot,
        'dendrogram_plot': dendrogram_plot,
    }

    return figures


# Equalization of brightness, contrast and sharpness ----------------------
def compute_mean_brightness(images: list[np.ndarray]) -> float:
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


def compute_mean_contrast(images: list[np.ndarray]) -> float:
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


def compute_sharpness(img_gray: np.ndarray) -> float:
    """Compute the sharpness of a grayscale image using the Sobel operator."""

    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.mean(grad)


def compute_mean_sharpness(images: list[np.ndarray]) -> float:
    """Compute the mean sharpness across a set of images."""

    mean_sharpness = 0.0
    for img in images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        mean_sharpness += compute_sharpness(img_gray)
    return mean_sharpness / len(images)


def normalize_brightness_set(
    images: list[np.ndarray], mean_brightness: float
) -> list[np.ndarray]:
    """Normalize brightness of all images in the set to the mean brightness."""

    normalized_images = []
    for img in images:
        if len(img.shape) == 3:  # color image
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            current_brightness = np.mean(l_channel)
            print(f'Original brightness: {current_brightness}')
            l_channel = (
                (l_channel * (mean_brightness / current_brightness))
                .clip(0, 255)
                .astype(np.uint8)
            )
            normalized_img_lab = cv2.merge([l_channel, a_channel, b_channel])
            normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)
            print(f'Normalized brightness: {np.mean(l_channel)}')

        else:  # grayscale image
            l_channel = img
            current_brightness = np.mean(l_channel)
            print(f'Original brightness: {current_brightness}')
            normalized_img = (
                (l_channel * (mean_brightness / current_brightness))
                .clip(0, 255)
                .astype(np.uint8)
            )
            print(f'Normalized brightness: {np.mean(normalized_img)}')

        normalized_images.append(normalized_img)

    return normalized_images


def normalize_contrast_set(images: list[np.ndarray], mean_contrast: float) -> list[np.ndarray]:
    """Normalize contrast of all images in the set to the mean contrast."""

    normalized_images = []
    for img in images:
        if len(img.shape) == 3:  # color
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            current_contrast = np.std(l_channel)
            print(f'Original contrast: {current_contrast}')
            l_channel = (
                (l_channel * (mean_contrast / current_contrast)).clip(0, 255).astype(np.uint8)
            )
            normalized_img_lab = cv2.merge([l_channel, a_channel, b_channel])
            normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)
            print(f'Normalized contrast: {np.std(l_channel)}')

        else:  # grayscale
            l_channel = img
            current_contrast = np.std(l_channel)
            print(f'Original contrast: {current_contrast}')
            normalized_img = (
                (l_channel * (mean_contrast / current_contrast)).clip(0, 255).astype(np.uint8)
            )
            print(f'Normalized contrast: {np.std(normalized_img)}')

        normalized_images.append(normalized_img)

    return normalized_images


def normalize_sharpness_set(
    images: list[np.ndarray], target_sharpness: float
) -> list[np.ndarray]:
    """Normalize sharpness of all images in the set to the target sharpness."""

    normalized_images = []

    for img in images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        else:
            img_gray = img

        original_sharpness = compute_sharpness(img_gray)
        print(f'Original sharpness: {original_sharpness}')

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
            img_yuv[:, :, 0] = sharpened
            normalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            normalized_img = sharpened

        normalized_sharpness = compute_sharpness(
            cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 3
            else normalized_img
        )
        print(f'Normalized sharpness: {normalized_sharpness}')

        normalized_images.append(normalized_img)

    return normalized_images


def equalize_images(
    images: list[np.ndarray],
    brightness: bool = False,
    contrast: bool = False,
    sharpness: bool = False,
) -> list[np.ndarray]:
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


def compute_metrics(images: list[np.ndarray]) -> dict[str, float]:
    """Computes metrics (mean and standard deviation) for normalized
    brightness, contrast, and sharpness.

    Parameters:
    - images: The input images in BGR or gray format.

    Returns:
    - metrics: Dictionary containing mean and standard deviation of
    normalized metrics.
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
        'sd_normalized_sharpness': sd_normalized_sharpness,
    }

    return metrics
