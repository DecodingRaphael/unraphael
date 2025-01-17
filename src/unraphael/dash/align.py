"""Module for alignment of figures and objects in paintings, including
equalization of selected to-be-aligned images to a baseline image in terms of
contrast, sharpness, brightness, and colour."""

from __future__ import annotations

import math
from typing import Callable, Tuple

import cv2
import diplib as dip
import numpy as np
from numpy.fft import fft2, ifft2
from skimage.color import rgb2gray

from unraphael.types import ImageType


def detect_and_compute_features(
    image_gray: np.ndarray, method: str, maxFeatures: int
) -> Tuple[list, np.ndarray]:
    """Detects and computes features in the image."""
    if method == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif method == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif method == 'ORB':
        feature_detector = cv2.ORB_create(maxFeatures)
    else:
        raise ValueError("Method must be 'SIFT', 'ORB', or 'SURF'")

    return feature_detector.detectAndCompute(image_gray, None)


def match_features(descsA: np.ndarray, descsB: np.ndarray, method: str) -> list:
    """Matches features between two sets of descriptors."""
    if method == 'ORB':
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    else:
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    matches = matcher.match(descsA, descsB, None)
    return sorted(matches, key=lambda x: x.distance)


def compute_homography(matches: list, kpsA: list, kpsB: list, keepPercent: float) -> np.ndarray:
    """Computes the homography matrix."""
    if len(matches) < 4:
        raise ValueError('Not enough matches between images.')

    ptsA = np.zeros((len(matches), 2), dtype='float')
    ptsB = np.zeros((len(matches), 2), dtype='float')

    for i, m in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    return cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)[0]


def apply_homography(
    target: np.ndarray, H: np.ndarray, template_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Applies the homography matrix to the target image."""
    h, w, c = template_shape
    return cv2.warpPerspective(
        target, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def homography_matrix(
    image: ImageType,
    base_image: ImageType,
    *,
    method: str = 'ORB',
    maxFeatures: int = 50000,
    keepPercent: float = 0.15,
) -> np.ndarray:
    """Computes the homography matrix between an input image and a base image
    using feature matching. This matrix is then used for animating or
    visualizing how the images align over a sequence of frames.

    Parameters
    ----------
    image : ImageType
        The input image to be aligned.
    base_image : ImageType
        The template image to align the input image with.
    method : str, optional
        The feature detection method to use ('SIFT', 'ORB', 'SURF').
        Default is 'ORB'.
    maxFeatures : int, optional
        The maximum number of features to detect and extract
        using ORB. Default is 500.
    keepPercent : float, optional
        The percentage of top matches to keep. Default is 0.2.

    Returns
    -------
    H : np.ndarray
        The homography matrix.
    """
    target = image.data
    template = base_image.data

    templateGray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    imageGray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    # Detect and compute features
    kpsA, descsA = detect_and_compute_features(imageGray, method, maxFeatures)
    kpsB, descsB = detect_and_compute_features(templateGray, method, maxFeatures)

    # Match features
    matches = match_features(descsA, descsB, method)

    # Compute homography matrix
    H = compute_homography(matches, kpsA, kpsB, keepPercent)

    return H


def feature_align(
    image: ImageType,
    base_image: ImageType,
    *,
    method: str = 'ORB',
    maxFeatures: int = 50000,
    keepPercent: float = 0.15,
) -> ImageType:
    """Aligns an input image with a template image using feature matching and
    homography transformation, rather than a correlation based method on the
    whole image to search for these values.

    Parameters
    ----------
    image : ImageType
        The input image to be aligned.
    base_image : ImageType
        The template image to align the input image with.
    method : str, optional
        The feature detection method to use ('SIFT', 'ORB', 'SURF').
        Default is 'ORB'.
    maxFeatures : int, optional
        The maximum number of features to detect and extract
        using ORB. Default is 500.
    keepPercent : float, optional
        The percentage of top matches to keep. Default is 0.2.

    Returns
    -------
    out : ImageType
        The aligned image.
    """
    target = image.data
    template = base_image.data

    templateGray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    imageGray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    if method == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif method == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif method == 'ORB':
        feature_detector = cv2.ORB_create(maxFeatures)
    else:
        raise ValueError("Method must be 'SIFT', 'ORB', or 'SURF'")

    (kpsA, descsA) = feature_detector.detectAndCompute(imageGray, None)
    (kpsB, descsB) = feature_detector.detectAndCompute(templateGray, None)

    # match features
    if method == 'ORB':
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    else:
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance: the smaller the distance,
    # the "more similar" the features are
    matches = sorted(matches, key=lambda x: x.distance)

    # keep top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    if len(matches) < 4:
        raise ValueError('Not enough matches between images.')

    # coordinates to compute homography matrix
    ptsA = np.zeros((len(matches), 2), dtype='float')
    ptsB = np.zeros((len(matches), 2), dtype='float')

    for i, m in enumerate(matches):
        # the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    ## derive rotation angle between figures from the homography matrix
    angle = -math.atan2(H[0, 1], H[0, 0]) * 180 / math.pi

    # apply the homography matrix to align the images, including the rotation
    h, w, c = template.shape
    aligned = cv2.warpPerspective(
        target, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    out = image.replace(data=aligned)
    out.metrics.update(angle=angle)

    return out


def ecc_align(
    image: ImageType,
    base_image: ImageType,
    *,
    mode: None | str = None,
) -> ImageType:
    """Aligns two images using the ECC (Enhanced Correlation Coefficient)
    algorithm. the ECC methodology can compensate for both shifts, shifts +
    rotations (euclidean), shifts + rotation + shear (affine), or homographic
    (3D) transformations of one image to the next.

    Parameters
    ----------
    image : ImageType
        The image to be warped to match the template
    base_image : ImageType
        The template image
    mode : str
        Warp mode, must be one of translation, euclidian, affine, homography (default)

    Returns
    -------
    out : ImageType
        The aligned image.
    """
    target = image.data
    template = base_image.data

    target_size = (template.shape[1], template.shape[0])
    target = cv2.resize(target, target_size)

    if not mode:
        warp_mode = cv2.MOTION_HOMOGRAPHY
    if mode == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
    elif mode == 'euclidian':
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif mode == 'affine':
        warp_mode = cv2.MOTION_AFFINE
    elif mode == 'homography':
        warp_mode = cv2.MOTION_HOMOGRAPHY
    else:
        raise ValueError(f'Invalid warp mode: {warp_mode}')

    # Ensure both images are resized to the same dimensions
    target_size = (
        min(template.shape[1], target.shape[1]),
        min(template.shape[0], target.shape[0]),
    )

    template_resized = cv2.resize(template, target_size)
    target_resized = cv2.resize(target, target_size)

    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_RGB2GRAY)  # template
    target_gray = cv2.cvtColor(target_resized, cv2.COLOR_RGB2GRAY)  # target to be aligned

    sz = template_resized.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 5000

    # Specify the threshold of the increment in the correlation
    # coefficient between two iterations
    termination_eps = 1e-8

    # Define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            template_gray,
            target_gray,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=1,
        )
    except cv2.error as e:
        raise RuntimeError(f'Error during ECC alignment: {e}')

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        target_aligned = cv2.warpPerspective(
            target_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        target_aligned = cv2.warpAffine(
            target_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    row1_col0 = warp_matrix[0, 1]
    angle = math.degrees(math.asin(row1_col0))

    out = image.replace(data=target_aligned)
    out.metrics.update(angle=angle)

    return out


def fourier_mellin_transform_match(
    image: ImageType,
    base_image: ImageType,
    *,
    corr_method: str | None = None,
) -> ImageType:
    """Apply Fourier-Mellin transform to match one image to another.

    Notes:
        - The function applies the Fourier-Mellin transform to match the image to the template.
        - The images are converted to grayscale before applying the transform.
        - The resulting aligned image is returned as a numpy array.

    Parameters
    ----------
    image : ImageType
        The image to align.
    base_image : ImageType
        The template image.

    Returns
    -------
    out : ImageType
        The aligned image.
    """
    if corr_method:
        kwargs = {'correlationMethod': corr_method}
    else:
        kwargs = {}

    target = image.data
    template = base_image.data

    target_size = (template.shape[1], template.shape[0])
    target = cv2.resize(target, target_size)

    img2 = dip.Image(target)

    img1_gray = dip.Image(rgb2gray(template))
    img2_gray = dip.Image(rgb2gray(target))

    out = dip.Image()
    matrix = dip.FourierMellinMatch2D(img1_gray, img2_gray, out=out, **kwargs)

    # Extract elements from the matrix
    m11 = matrix[0]
    m12 = matrix[1]

    # Extract scaling factors
    s_x = matrix[0]  # Scaling factor for x-axis
    s_y = matrix[3]  # Scaling factor for y-axis

    # Calculate average scaling factor
    scaling_factor = (s_x + s_y) / 2

    angle = -math.atan2(m12, m11) * 180 / math.pi

    # Apply the affine transformation using the transformation
    # matrix to align img2 to img1 (template)
    if img2.TensorElements() > 1:
        # If img2 is a color image
        aligned_channels = []
        # For each color channel
        for i in range(img2.TensorElements()):
            img2_channel = img2(i)
            moved_channel = dip.Image()
            dip.AffineTransform(img2_channel, out=moved_channel, matrix=matrix)
            aligned_channels.append(np.asarray(moved_channel))

        aligned_image = np.stack(aligned_channels, axis=-1)
    else:
        # If img2 is grayscale
        moved_img = dip.Image()
        dip.AffineTransform(img2, out=moved_img, matrix=matrix)
        aligned_image = np.asarray(moved_img)

    out = image.replace(data=aligned_image)
    out.metrics.update(
        angle=angle,
        scaling_factor=scaling_factor,
    )

    return out


def align_images_with_translation(image: ImageType, base_image: ImageType) -> ImageType:
    """Aligns two images with FFT phase correlation by finding the translation
    offset that minimizes the difference between them.

    Parameters
    ----------
    image : ImageType
        Image to align (RGB or grayscale)
    base_image : ImageType
        Template image (RGB or grayscale)

    Returns
    -------
    out : ImageType
        The second image aligned with the template
    """
    target = image.data
    template = base_image.data

    target_size = (template.shape[1], template.shape[0])
    target = cv2.resize(target, target_size)

    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    if len(target.shape) == 3:
        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    # FFT phase correlation
    shape = template.shape
    f0 = fft2(template)
    f1 = fft2(target)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))

    # Find peak in cross-correlation
    t0, t1 = np.unravel_index(np.argmax(ir), shape)

    # Adjust shifts if they are larger than half the image size
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]

    rows, cols = target.shape[:2]
    translation_matrix = np.float32([[1, 0, t1], [0, 1, t0]])
    aligned_image = cv2.warpAffine(target, translation_matrix, (cols, rows))

    out = image.replace(data=aligned_image)

    return out


def rotation_align(image: ImageType, base_image: ImageType) -> ImageType:
    """Aligns two images by finding the rotation angle that minimizes the
    difference between them.

    Parameters
    ----------
    image : ImageType
        The image to be aligned
    base_image : ImageType
        The image template

    Returns
    -------
    out : ImageType
        The rotated version of the second image, aligned with the first image
    """
    target = image.data
    template = base_image.data

    target_size = (template.shape[1], template.shape[0])
    target = cv2.resize(target, target_size)

    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    height, width = image_gray.shape[0:2]

    values = np.ones(360)

    # Find the rotation angle that minimizes the difference between the images
    for i in range(0, 360):
        rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), i, 1)
        rot = cv2.warpAffine(image_gray, rotationMatrix, (width, height))
        values[i] = np.mean(template_gray - rot)

    angle = np.argmin(values)
    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(target, rotationMatrix, (width, height))

    out = image.replace(data=rotated)
    out.metrics.update(angle=angle)

    return out


def align_image_to_base(
    base_image: ImageType,
    image: ImageType,
    *,
    align_method: str | None,
    motion_model: str,
    feature_method: str = 'ORB',
) -> ImageType:
    """Aligns all images in a folder to a template image using the selected
    alignment method and preprocess options.

    Parameters
    ----------
    base_image : ImageType
        The template image
    image : ImageType
        The image to align
    align_method : str
        The selected alignment method
    motion_model : str
        The selected motion model for ECC alignment
    feature_method : str, optional
        The feature detection method to use ('SIFT', 'ORB', 'SURF'). Default is 'ORB'

    Returns
    -------
    ImageType
        Aligned image
    """
    func: Callable

    if not align_method:
        return image

    elif align_method == 'Feature based alignment':
        func = feature_align
        kwargs = {'method': feature_method}

    elif align_method == 'Enhanced Correlation Coefficient Maximization':
        func = ecc_align
        kwargs = {'mode': motion_model}

    elif align_method == 'Fourier Mellin Transform':
        func = fourier_mellin_transform_match
        kwargs = {'corr_method': motion_model}

    elif align_method == 'FFT phase correlation':
        func = align_images_with_translation
        kwargs = {}

    elif align_method == 'Rotational Alignment':
        func = rotation_align
        kwargs = {}

    else:
        raise ValueError(f'No such method: {align_method}')

    return func(image=image, base_image=base_image, **kwargs)
