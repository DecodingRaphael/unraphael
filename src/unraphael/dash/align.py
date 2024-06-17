"""Module for alignment of figures and objects in paintings, including
equalization of selected to-be-aligned images to a baseline image in terms of
contrast, sharpness, brightness, and colour."""

from __future__ import annotations

import math
from typing import Any

import cv2
import diplib as dip
import numpy as np
from numpy.fft import fft2, ifft2
from skimage.color import rgb2gray


def feature_align(
    image: np.ndarray,
    template: np.ndarray,
    *,
    method: str = 'ORB',
    maxFeatures: int = 50000,
    keepPercent: float = 0.15,
) -> np.ndarray:
    """Aligns an input image with a template image using feature matching and
    homography transformation, rather than a correlation based method on the
    whole image to search for these values.

    Parameters
    ----------
    image : np.ndarray
        The input image to be aligned.
    template : np.ndarray
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
    np.ndarray
        The aligned image.
    """
    templateGray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
    print(f'Rotational degree ORB feature: {angle:.2f}')  # rotation angle, in degrees

    # apply the homography matrix to align the images, including the rotation
    h, w, c = template.shape
    aligned = cv2.warpPerspective(
        image, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    return aligned, angle


def ecc_align(image: np.ndarray, template: np.ndarray, *, mode: None | str = None):
    """Aligns two images using the ECC (Enhanced Correlation Coefficient)
    algorithm. the ECC methodology can compensate for both shifts, shifts +
    rotations (euclidean), shifts + rotation + shear (affine), or homographic
    (3D) transformations of one image to the next.

    Parameters
    ----------
    image : np.ndarray
        The image to be warped to match the template
    template : np.ndarray
        The template image
    mode : str
        Warp mode, must be one of translation, euclidian, affine, homography (default)

    Returns
    -------
    image_aligned : np.ndarray
        The aligned image.
    """
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
        min(template.shape[1], image.shape[1]),
        min(template.shape[0], image.shape[0]),
    )

    template_resized = cv2.resize(template, target_size)
    image_resized = cv2.resize(image, target_size)

    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_RGB2GRAY)  # template
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)  # image to be aligned

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
            image_gray,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=1,
        )
    except cv2.error as e:
        raise RuntimeError(f'Error during ECC alignment: {e}')

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        image_aligned = cv2.warpPerspective(
            image_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        image_aligned = cv2.warpAffine(
            image_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    row1_col0 = warp_matrix[0, 1]
    angle = math.degrees(math.asin(row1_col0))

    return image_aligned, angle


def fourier_mellin_transform_match(
    image: np.ndarray, template: np.ndarray, *, corr_method: str | None = None
):
    """Apply Fourier-Mellin transform to match one image to another.

    Notes:
        - The function applies the Fourier-Mellin transform to match the image to the template.
        - The images are converted to grayscale before applying the transform.
        - The resulting aligned image is returned as a numpy array.

    Parameters
    ----------
    template : ndarray
        The template image.
    image2 : ndarray
        The the image to align.

    Returns
    -------
    np.ndarray:
        The aligned image as a numpy array.
    """
    if corr_method:
        kwargs = {'correlationMethod': corr_method}
    else:
        kwargs = {}

    img2 = dip.Image(image)

    img1_gray = dip.Image(rgb2gray(template))
    img2_gray = dip.Image(rgb2gray(image))

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

    print('Average scaling factor:', scaling_factor)

    angle = -math.atan2(m12, m11) * 180 / math.pi
    print(f'Rotational degree Fourier Mellin Transformation: {angle:.2f}')

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

    return aligned_image, angle


def align_images_with_translation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Aligns two images with FFT phase correlation by finding the translation
    offset that minimizes the difference between them.

    Parameters
    ----------
    image : np.ndarray
        Image to align (RGB or grayscale)
    template : np.ndarray
        Template image (RGB or grayscale)

    Returns:
    - aligned_image: The second image aligned with the template
    """

    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # FFT phase correlation
    shape = template.shape
    f0 = fft2(template)
    f1 = fft2(image)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))

    # Find peak in cross-correlation
    t0, t1 = np.unravel_index(np.argmax(ir), shape)

    # Adjust shifts if they are larger than half the image size
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]

    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, t1], [0, 1, t0]])
    aligned_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return aligned_image


def rotation_align(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Aligns two images by finding the rotation angle that minimizes the
    difference between them.

    Parameters
    ----------
    image : np.ndarray
        The image to be aligned
    template : np.ndarray
        The image template

    Returns
    -------
    The rotated version of the second image, aligned with the first image
    """
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = image_gray.shape[0:2]

    values = np.ones(360)

    # Find the rotation angle that minimizes the difference between the images
    for i in range(0, 360):
        rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), i, 1)
        rot = cv2.warpAffine(image_gray, rotationMatrix, (width, height))
        values[i] = np.mean(template_gray - rot)

    angle = np.argmin(values)
    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(image, rotationMatrix, (width, height))

    return rotated, angle


def align_image_to_base(
    base_image: np.ndarray,
    image: dict[str, np.ndarray],
    *,
    align_method: str | None,
    motion_model: str,
    feature_method: str = 'ORB',
) -> dict[str, Any]:
    """Aligns all images in a folder to a template image using the selected
    alignment method and preprocess options.

    Parameters
    ----------
    base_image : np.ndarray
        The template image
    input_files : list
        The image to align
    align_method : str
        The selected alignment method
    motion_model : str
        The selected motion model for ECC alignment
    feature_method : str, optional
        The feature detection method to use ('SIFT', 'ORB', 'SURF'). Default is 'ORB'

    Returns
    -------
    aligned_images : dict[str, Any]
        List of tuples containing filename and aligned image.
    """

    # equal size between base_image and image, necessary for some alignment methods
    target_size = (base_image.shape[1], base_image.shape[0])
    image_resized = cv2.resize(image, target_size)

    angle = 0

    if not align_method:
        aligned = image
    elif align_method == 'Feature based alignment':
        aligned, angle = feature_align(image=image, template=base_image, method=feature_method)

    elif align_method == 'Enhanced Correlation Coefficient Maximization':
        aligned, angle = ecc_align(image=image_resized, template=base_image, mode=motion_model)

        if aligned is None:
            raise RuntimeError('Error aligning image')

    elif align_method == 'Fourier Mellin Transform':
        aligned, angle = fourier_mellin_transform_match(
            image=image_resized, template=base_image, corr_method=motion_model
        )

    elif align_method == 'FFT phase correlation':
        aligned = align_images_with_translation(image=image_resized, template=base_image)

    elif align_method == 'Rotational Alignment':
        aligned, angle = rotation_align(image=image_resized, template=base_image)

    else:
        raise ValueError(f'No such method: {align_method}')

    return {'image': aligned, 'angle': angle}
