"""Module for clustering of images based on:
- structural similarity
- shape/ contour similarity.

Includes the equalization of contrast, sharpness, and brightness
across the images and alignment of images to their mean to optimize
the clustering process."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import piq
import ssim.ssimlib as pyssim
import streamlit as st
import torch
from clusteval import clusteval
from PIL import Image
from pystackreg import StackReg
from rembg import remove
from scatterd import scatterd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from skimage import color, transform
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from sklearn import metrics
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SIM_IMAGE_SIZE = (640, 480)
SIFT_RATIO = 0.7
MSE_NUMERATOR = 1000.0
NUM_THREADS = 8


def show_images_widget(
    images: dict[str, np.ndarray],
    *,
    n_cols: int = 4,
    key: str = 'show_images',
    message: str = 'Select image',
) -> None | str:
    """Widget to show images with given number of columns."""

    col1, col2 = st.columns(2)
    n_cols = col1.number_input(
        'Number of columns for display', value=4, min_value=1, step=1, key=f'{key}_cols'
    )

    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]

        col.image(im, use_column_width=True, caption=name, clamp=True)

    return None


# Equalization of brightness, contrast and sharpness
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
    """Normalize sharpness of all images in the set to the sharpness of the
    target."""
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
    - images: the images with identical brightnes, contrast and sharpness
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

    # Perform alignment/registration based on the selected feature method
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


# Brushstroke Analysis
# This set of functions analyzes brushstrokes in images by extracting various
# edge detection metrics. The methods implemented here focus on identifying
# and quantifying the characteristics of brushstrokes as they  manifest in
# the form of edges in an image. Adapted from Ugail, H, Stork, DG, Edwards,
# H, Seward, SC & Brooke, C. (2023). Deep transfer learning for visual analysis
# and attribution of paintings by Raphael. Heritage Science 11(1), 268.
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
    """Convert a NumPy array to a Torch tensor and normalize it."""
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
    Adapted from https://github.com/llvll/imgcluster

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
    - Calinski-Harabasz index: measures cluster dispersion.
    - Completeness score: measures the completeness of the predicted clusters.
    - Homogeneity score: measures the homogeneity of the predicted clusters.
    """

    metrics_dict = {}

    # Interpretation:
    # Silhouette: Higher is better (range: -1 to 1)
    # Davies-Bouldin: Lower is better (≥ 0)
    # Calinski-Harabasz: Higher is better (≥ 0)

    if len(set(labels)) > 1:
        metrics_dict['Silhouette coefficient'] = silhouette_score(
            X, labels, metric='precomputed'
        )
        # 1 - X transforms this similarity matrix into a dissimilarity matrix, which is required
        # for the Davies-Bouldin index to calculate meaningful results
        metrics_dict['Davies-Bouldin index'] = davies_bouldin_score(1 - X, labels)
        metrics_dict['Calinski-Harabasz index'] = calinski_harabasz_score(X, labels)
    if labels_true is not None:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)

    return metrics_dict


def determine_optimal_clusters(
    matrix: np.ndarray,
    method: str = 'elbow',
    cluster_method: str = 'agglomerative',
    metric: str = 'euclidean',
    linkage: str = 'ward',
    min_clust: int = 2,
    max_clust: int = 10,
) -> int:
    """Determines the optimal number of clusters using the elbow or silhoutte
    methods. To be used in case of similarity matrix as imput for the cluster
    process.

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix for clustering.
    method : str
        The cluster evaluation method. Options are 'elbow', 'silhouette'
    cluster_method : str
        The clustering method to use. Options are 'kmeans', 'agglomerative'
    metric : str
        The distance metric to use.
    linkage : str
        The linkage method for agglomerative clustering.
    min_clust : int
        The minimum number of clusters to evaluate.
    max_clust : int
        The maximum number of clusters to evaluate.

    Returns
        The optimal number of clusters.
    """

    optimal_clusters = min_clust

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

    elif method == 'silhouette':
        ce = clusteval(
            cluster=cluster_method,
            evaluate=method,
            metric=metric,
            linkage=linkage,
            min_clust=min_clust,
            max_clust=max_clust,
        )

        results = ce.fit(matrix)
        scores = results['score']['score']
        cluster_numbers = results['score']['clusters']

        # Find the index of the maximum silhouette score
        optimal_index = np.argmax(scores)
        optimal_clusters = cluster_numbers[optimal_index]

    return optimal_clusters


def plot_scatter(features):
    scatterd(features[:, 0], features[:, 1])
    fig = plt.gcf()
    st.pyplot(fig)


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


def matrix_based_clustering(
    images: list[np.ndarray],
    algorithm: str,
    n_clusters: int,
    method: str,
    labels_true: np.ndarray = None,
) -> tuple[np.ndarray, dict, int]:
    """Clusters a list of images using different clustering algorithms
    based on similarity matrix of specified metric. Adapted from
    https://github.com/llvll/imgcluster

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
    labels_true : list, optional
        The true labels for the images, used for evaluating clustering
        performance. Default is None.

    Returns
    -------
    list
        The cluster labels assigned to each image.

    Note:
    using the metric='precomputed' parameter ensures that the input is not
    raw data points but rather a similarity matrix.
    """

    # main input for the clustering algorithm
    matrix = build_similarity_matrix(images, algorithm=algorithm)

    # Determine number of clusters
    if n_clusters is None and method != 'DBSCAN':
        n_clusters = determine_optimal_clusters(matrix)

    metrics = {}

    if method == 'SpectralClustering':
        sc = SpectralClustering(
            n_clusters=n_clusters, random_state=42, affinity='precomputed'
        ).fit(matrix)
        metrics = get_cluster_metrics(matrix, sc.labels_, labels_true)
        return sc.labels_, metrics, n_clusters

    elif method == 'AffinityPropagation':
        af = AffinityPropagation(affinity='precomputed', random_state=42).fit(matrix)
        metrics = get_cluster_metrics(matrix, af.labels_, labels_true)
        return af.labels_, metrics, len(set(af.labels_))

    elif method == 'DBSCAN':
        db = DBSCAN(metric='precomputed', eps=0.3, min_samples=2).fit(matrix)
        db_labels = db.labels_
        metrics = get_cluster_metrics(matrix, db_labels, labels_true)
        num_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        return db_labels, metrics, num_clusters

    else:
        raise ValueError(f'Unsupported clustering method: {method}')


def preprocess_images(images: list, target_shape: tuple = (128, 128)) -> np.ndarray:
    """Preprocesses a list of images by resizing and flattening them.

    Parameters
    ----------
    images : list
        A list of input images.
    target_shape : tuple, optional
        The target shape to resize the images to. Defaults to (128, 128).

    Returns
    -------
    numpy.ndarray
        A 2D array where each row corresponds to a flattened preprocessed image.
    """
    preprocessed_images = []
    for image in images:
        resized_image = resize(image, target_shape, anti_aliasing=True)
        flattened_image = resized_image.flatten()
        preprocessed_images.append(flattened_image)

    return np.array(preprocessed_images)


def feature_based_clustering(
    features,
    cluster_method,
    cluster_evaluation,
    cluster_linkage,
    name_dict,
    is_similarity_matrix=False,
):
    """Performs clustering on the given features using the specified method and
    evaluates the results.

    Args:
        features (np.ndarray): The features to be clustered.
        cluster_method (str): The clustering algorithm to use ('kmeans',
        'agglomerative', 'dbscan').
        cluster_evaluation (str): The method used for cluster evaluation
        ('silhouette', 'dbindex', 'derivative').
        cluster_linkage (str): The linkage method to use for agglomerative clustering.
        name_dict (dict): A dictionary containing names (image names or feature identifiers)
        for plotting.
        is_similarity_matrix (bool): Whether the input features are a similarity matrix
        (defaults to False).


    Returns:
        tuple: A tuple containing the number of clusters found (int) and cluster labels
        (np.ndarray or None).
    """
    n_clusters = None
    ce = clusteval(cluster=cluster_method, evaluate=cluster_evaluation, linkage=cluster_linkage)

    results = ce.fit(features)

    # Check if results is not None and contains 'labx'
    if results is not None and 'labx' in results:
        cluster_labels = results['labx']
    else:
        st.error('No cluster labels found. Check your clustering method and parameters.')
        return None, None  # Early exit

    # Extract optimal number of clusters
    if cluster_method in ['kmeans', 'agglomerative']:
        scores = results['score']['score']
        cluster_numbers = results['score']['clusters']

        # Find the index of the maximum silhouette score
        optimal_index = np.argmax(scores)
        n_clusters = cluster_numbers[optimal_index]

    elif cluster_method == 'dbscan':
        unique_labels = set(cluster_labels) if cluster_labels is not None else set()
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise

    # Generate and display plots
    dendrogram_plot = ce.dendrogram(X=results, linkage=cluster_linkage)['ax'].figure
    st.subheader('Dendrogram plot')
    st.pyplot(dendrogram_plot)

    silhouette_plot, ax, _ = ce.plot_silhouette(
        X=features,
        dot_size=200,
        jitter=0.01,
        cmap='Set2',
        savefig={'fname': None, 'format': 'png', 'dpi': 100},
    )

    st.subheader('Silhouette plot')
    st.pyplot(fig=silhouette_plot)

    # Plot the clusters
    pca_clusters = plot_pca_mds_scatter(
        data=features,
        labels=cluster_labels,
        contours_dict=name_dict,
        is_similarity_matrix=is_similarity_matrix,
        title='PCA dimensions',
    )
    st.subheader('Scatterplot')
    st.pyplot(pca_clusters)

    # Show metrics
    st.subheader('Performance metrics')
    col1, col2 = st.columns(2)
    col1.metric('Number of clusters found:', n_clusters)

    # Calculate and display evaluation metrics
    if cluster_method in ['kmeans', 'agglomerative']:
        if n_clusters is not None and n_clusters > 1:
            silhouette_avg = silhouette_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            ch_index = calinski_harabasz_score(features, cluster_labels)

            col1.metric(label='Silhouette Score', value=f'{silhouette_avg:.2f}')
            col2.metric(label='Davies Bouldin Score', value=f'{davies_bouldin:.2f}')
            col2.metric(label='Calinski Harabasz Score', value=f'{ch_index:.2f}')

    elif cluster_method == 'dbscan':
        # Exclude noise points (-1)
        if len(set(cluster_labels)) > 1:
            valid_labels = cluster_labels[cluster_labels != -1]
            if len(set(valid_labels)) > 1:  # Ensure there's more than 1 cluster
                silhouette_avg = silhouette_score(features[cluster_labels != -1], valid_labels)
                davies_bouldin = davies_bouldin_score(
                    features[cluster_labels != -1], valid_labels
                )
                ch_index = calinski_harabasz_score(features[cluster_labels != -1], valid_labels)

                col1.metric(label='Silhouette Score', value=f'{silhouette_avg:.2f}')
                col2.metric(label='Davies Bouldin Score', value=f'{davies_bouldin:.2f}')
                col2.metric(label='Calinski Harabasz Score', value=f'{ch_index:.2f}')
            else:
                col1.metric(label='Silhouette Score could not be computed')
                col2.metric(label='Davies Bouldin Score could not be computed')
                col2.metric(label='Calinski Harabasz Score could not be computed')

    return n_clusters, cluster_labels


def extract_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Extract the foreground mask using rembg."""
    return remove(image, mask=True)


def extract_outer_contour_from_mask(
    mask: np.ndarray, min_area: int = 25
) -> Optional[np.ndarray]:
    """Extract the outer contour from the mask.

    Args:
        mask (np.ndarray): Input binary mask (BGR or grayscale).
        min_area (int): Minimum area threshold for contours to be considered.

    Returns:
        Optional[np.ndarray]: The largest contour as a NumPy array if found,
                              otherwise None.
    """
    # Convert the mask to grayscale if it's in BGR format
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray_mask, 1, 255, cv2.THRESH_BINARY
    )  # threshold to get binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Return the largest contour if available
    if filtered_contours:
        return max(filtered_contours, key=cv2.contourArea)

    return None


def extract_outer_contours_from_aligned_images(aligned_images: dict[str, np.ndarray]) -> dict:
    """Extract outer contours from the aligned images using rembg for.

    Parameters
    ----------
    aligned_images : dict[str, np.ndarray]
        A dictionary where the keys are image names and the values are
        the corresponding images in np.ndarray format.
    Returns
    -------
    dict
        A dictionary where the keys are the image names and the values
        are the extracted outer contours.
    """
    contours_dict = {}

    for name, image in aligned_images.items():
        if not isinstance(image, np.ndarray):
            st.error(f'Image "{name}" is not a valid np.ndarray.')
            continue

        # Ensure the image is in RGB format for rembg
        if image.shape[-1] == 3:  # Assuming the input is BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image  # Already in RGB format

        mask = extract_foreground_mask(image_rgb)
        outer_contour = extract_outer_contour_from_mask(mask)

        # Use image name as the key
        contours_dict[name] = outer_contour

    return contours_dict


def visualize_outer_contours(
    aligned_images: dict[str, np.ndarray], contours_dict: dict
) -> None:
    """
    Visualize the outer contours for the aligned images on a black background.
    Args:
        aligned_images (dict[str, np.ndarray]): A dictionary where keys are image
        names and values are the aligned images as numpy arrays.
        contours_dict (dict): A dictionary where keys are image names and values
        are the corresponding outer contours.
    Returns:
        None: This function does not return any value. It displays the images with
        outer contours using a widget.
    Inline Documentation:
        - contour_images: Dictionary to store images with only contours.
    """
    contour_images = {}  # Dictionary to store images with only contours

    for name, image in aligned_images.items():
        outer_contour = contours_dict.get(name)

        if outer_contour is not None:
            # Create a blank canvas (black background) with the same size as the original image
            contour_canvas = np.zeros_like(image)

            # Draw contours on the blank canvas
            cv2.drawContours(contour_canvas, [outer_contour], -1, (255, 255, 255), 2)

            contour_images[name] = contour_canvas
        else:
            st.write(f'No valid contour found for {name}')

    if contour_images:
        show_images_widget(
            contour_images, n_cols=4, key='contour_images', message='Contour Images Only'
        )


def visualize_clusters(
    labels, image_names, image_list, name_dict, title='Cluster visualization'
):
    """Helper function to visualize clusters of images."""
    if labels is not None:
        st.subheader(title)
        num_clusters = len(set(labels))

        for n in range(num_clusters):
            cluster_label = n + 1
            st.write(f'\n --- Images from cluster #{cluster_label} ---')

            cluster_indices = np.argwhere(labels == n).flatten()
            cluster_images_dict = {image_names[i]: image_list[i] for i in cluster_indices}

            # Use the provided dictionary (name_dict) for visualizing the images
            show_images_widget(
                cluster_images_dict,
                key=f'cluster_{cluster_label}_images',
                message=f'Images from Cluster #{cluster_label}',
            )


def compute_fourier_descriptors(contour: np.ndarray, num_coeff: int = 10) -> np.ndarray:
    """Compute the Fourier descriptors for a given contour, returning a
    specified number of coefficients."""
    contour_array = contour[:, 0, :]
    complex_contour = contour_array[:, 0] + 1j * contour_array[:, 1]
    fourier_result = np.fft.fft(complex_contour)
    descriptors = np.abs(fourier_result)
    descriptors = descriptors[:num_coeff] / np.abs(descriptors[0])
    return np.pad(descriptors, (0, max(0, num_coeff - len(descriptors))), 'constant')


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    """Compute the log-transformed Hu moments of a given contour for scale
    invariance."""
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return np.log(np.abs(hu_moments))  # Log transform for scale invariance


def compute_hog_features(contour: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Compute HOG features for a given contour and pad/truncate to a desired
    length."""
    blank_image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(blank_image, [contour], -1, 255, 1)
    features, _ = hog(
        blank_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True
    )
    desired_length = 100  # Example length
    return np.pad(features, (0, max(0, desired_length - len(features))), 'constant')[
        :desired_length
    ]


def resample_contour(contour: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resamples a given contour to a specified number of points using linear
    interpolation."""
    x, y = contour[:, 0, 0], contour[:, 0, 1]
    cumulative_lengths = np.cumsum(
        np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
    )
    # Normalize to [0, 1]
    cumulative_lengths /= cumulative_lengths[-1]
    # Create interpolation functions
    interp_x = interp1d(cumulative_lengths, x, kind='linear')
    interp_y = interp1d(cumulative_lengths, y, kind='linear')
    # Generate new cumulative lengths
    new_cumulative_lengths = np.linspace(0, 1, num_points)
    new_x = interp_x(new_cumulative_lengths)
    new_y = interp_y(new_cumulative_lengths)
    return np.stack([new_x, new_y], axis=1)


def compute_aspect_ratio(contour: np.ndarray) -> float:
    """Computes the aspect ratio of the contour's bounding rectangle."""
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0


def compute_contour_length(contour: np.ndarray) -> float:
    """Computes the arc length of the contour."""
    return cv2.arcLength(contour, closed=True)


def compute_centroid_distance(contour: np.ndarray) -> float:
    """Computes the average distance of contour points from the centroid."""
    moments = cv2.moments(contour)
    cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
    cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
    distances = np.sqrt((contour[:, 0, 0] - cx) ** 2 + (contour[:, 0, 1] - cy) ** 2)
    return np.mean(distances)


def compute_procrustes_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """Compute the Procrustes distance between two contours after resampling
    them."""
    resampled_contour1 = resample_contour(contour1)
    resampled_contour2 = resample_contour(contour2)
    mtx1, mtx2, disparity = procrustes(resampled_contour1, resampled_contour2)
    return disparity


def compute_hausdorff_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """Compute the Hausdorff distance between two contours represented as numpy
    arrays."""
    return max(
        directed_hausdorff(contour1[:, 0, :], contour2[:, 0, :])[0],
        directed_hausdorff(contour2[:, 0, :], contour1[:, 0, :])[0],
    )


def extract_and_scale_features(
    contours_dict: dict, selected_features: list, image_shape: tuple
) -> tuple[np.ndarray, None]:
    """Extract and scale several features from contours.

    Parameters:
        contours_dict (dict): Dictionary of contours where keys are image names
        and values are np.ndarrays of contours.
        selected_features (list): List of features to include in the output.
        image_shape (tuple): Shape of the image for HOG feature extraction.

    Returns:
        tuple: A tuple containing:
            - combined_features (np.ndarray): The scaled feature array.
            - None (since scalers are not used in this case).
    """

    def compute_feature_matrix(contours_list, feature_func):
        matrix = np.zeros((len(contours_list), len(contours_list)))
        for i, contour_i in enumerate(contours_list):
            for j, contour_j in enumerate(contours_list):
                if i != j:
                    matrix[i, j] = feature_func(contour_i, contour_j)
        return matrix

    features_by_type: dict[str, list[list[float]]] = {ft: [] for ft in selected_features}
    contours_list = list(contours_dict.values())

    for idx, contour in enumerate(contours_list):
        if 'fd' in selected_features:
            features_by_type['fd'].append(compute_fourier_descriptors(contour))
        if 'hu' in selected_features:
            features_by_type['hu'].append(compute_hu_moments(contour))
        if 'hog' in selected_features:
            features_by_type['hog'].append(compute_hog_features(contour, image_shape))
        if 'aspect_ratio' in selected_features:
            features_by_type['aspect_ratio'].append([compute_aspect_ratio(contour)])
        if 'contour_length' in selected_features:
            features_by_type['contour_length'].append([compute_contour_length(contour)])
        if 'centroid_distance' in selected_features:
            features_by_type['centroid_distance'].append([compute_centroid_distance(contour)])
        if 'hd' in selected_features:
            hausdorff_matrix = compute_feature_matrix(contours_list, compute_hausdorff_distance)
            avg_hd = np.mean(hausdorff_matrix[idx, :][hausdorff_matrix[idx, :] > 0])
            features_by_type['hd'].append([avg_hd])
        if 'procrustes' in selected_features:
            procrustes_matrix = compute_feature_matrix(
                contours_list, compute_procrustes_distance
            )
            avg_procrustes = np.mean(procrustes_matrix[idx, :][procrustes_matrix[idx, :] > 0])
            features_by_type['procrustes'].append([avg_procrustes])

    # Scale each feature type separately
    scaled_features = []
    for ft in selected_features:
        if features_by_type[ft]:
            ft_array = np.array(features_by_type[ft])
            scaled_ft = StandardScaler().fit_transform(ft_array)
            scaled_features.append(scaled_ft)

    combined_features = np.hstack(scaled_features) if scaled_features else np.array([])

    # Reduce the amount of dimensions in the feature vector with PCA
    combined_features = reduce_dimensions(combined_features, n_components=2)
    return combined_features, None


def reduce_dimensions(features: np.ndarray, n_components: int) -> np.ndarray:
    """Reduces the dimensions of the feature vector using PCA.

    Parameters:
      features (np.ndarray): The combined feature array.
      n_components (int): The number of principal components to keep.
      Returns:
          np.ndarray: The transformed feature array with reduced dimensions.
    """
    pca = PCA(n_components=n_components, random_state=22)
    pca.fit(features)
    reduced_features = pca.transform(features)
    return reduced_features


# def plot_pca_scatter(
#     features: np.ndarray,
#     labels: np.ndarray,
#     contours_dict: dict,
#     title: str = 'DBSCAN Clustering of Image Contours',
# ) -> plt.Figure:
#     """Plot clusters of image contours based on extracted features.

#     Parameters:
#         features (np.ndarray): The feature array used for clustering.
#         labels (np.ndarray): The labels resulting from clustering (e.g., from DBSCAN).
#         contours_dict (dict): Dictionary of contours where keys are image names
#         and values are np.ndarrays of contours.
#         title (str): Title of the plot.

#     Returns:
#         plt.Figure: The figure containing the scatter plot.
#     """
#     # Reduce dimensions using PCA
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)
#     print('PCA Explained Variance Ratio:', pca.explained_variance_ratio_)

#     unique_labels = set(labels)

#     # Create a figure for plotting
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Plot each cluster
#     for label in unique_labels:
#         if label == -1:
#             color = 'k'  # Noise points
#             marker = 'x'
#         else:
#             color = plt.cm.jet(float(label) / (max(unique_labels) + 1))
#             marker = 'o'

#         members = labels == label
#         ax.scatter(
#             reduced_features[members, 0], reduced_features[members, 1], c=color, marker=marker
#         )

#     # Annotate each point with the corresponding image name
#     for i, (x, y) in enumerate(reduced_features):
#         # Use the key of contours_dict to get the image name
#         image_name = list(contours_dict.keys())[
#             i
#         ]  # Get the image name corresponding to the index
#         ax.annotate(
#             image_name,
#             (x, y),
#             xytext=(5, 2),
#             textcoords='offset points',
#             ha='left',
#             va='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#             fontsize=8,
#         )

#     ax.set_title(title)
#     ax.set_xlabel('PCA 1')
#     ax.set_ylabel('PCA 2')
#     ax.grid(True)

#     return fig


def plot_pca_mds_scatter(
    data: np.ndarray,  # image features or similarity matrix
    labels: np.ndarray,
    contours_dict: dict,
    is_similarity_matrix: bool = False,
    title: str = 'Clustering of Images',
) -> plt.Figure:
    """Plot clusters of image contours, feature vectors, or similarity matrix
    data based on dimensionality reduction.

    This function can visualize:
    - Clusters of image contours based on extracted features (e.g., Fourier descriptors,
    Hu moments).
    - Clusters of features directly if they are passed as input.
    - Clusters based on similarity matrices (e.g., similarity between images or contours).
    Dimensionality reduction is performed using PCA (Principal Component Analysis) or
    MDS (Multidimensional Scaling) depending on the input type.

    Parameters:
        data (np.ndarray): The feature array, similarity matrix, or raw image data
        used for clustering.
        labels (np.ndarray): The labels resulting from clustering (e.g., from DBSCAN,
        KMeans).
        contours_dict (dict): Dictionary of image names or feature identifiers mapped
        to data points or contours.
        is_similarity_matrix (bool): Flag to indicate whether the input is a similarity
        matrix (default: False).
        title (str): Title of the plot (default: 'Clustering Visualization').

    Returns:
        plt.Figure: The figure containing the scatter plot of clusters with
        PCA or MDS dimensions.
    """
    if is_similarity_matrix:
        # If input is a similarity matrix, apply MDS
        # dissimilarity=='precomputed' -> the input should be the dissimilarity matrix
        mds = MDS(n_components=2, dissimilarity='precomputed')
        reduced_features = mds.fit_transform(data)
    else:
        # If input is feature data, apply PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(data)  # (n_samples, n_features)
        print('PCA Explained Variance Ratio:', pca.explained_variance_ratio_)

    unique_labels = set(labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            color = 'k'  # Noise points
            marker = 'x'
        else:
            color = plt.cm.jet(float(label) / (max(unique_labels) + 1))
            marker = 'o'

        members = labels == label
        ax.scatter(
            reduced_features[members, 0], reduced_features[members, 1], c=color, marker=marker
        )

    # Annotate each point with the corresponding image name
    for i, (x, y) in enumerate(reduced_features):
        # Use the key of contours_dict to get the image name
        image_name = list(contours_dict.keys())[i]
        ax.annotate(
            image_name,
            (x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            fontsize=8,
        )

    ax.set_title(f'{title}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True)

    return fig