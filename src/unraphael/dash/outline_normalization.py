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
from skimage.exposure import match_histograms


def featureAlign(image, template, method='ORB', maxFeatures=50000, keepPercent=0.15):
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
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


def ecc_align(im1: np.ndarray, im2: np.ndarray, *, mode: None | str = None):
    """Aligns two images using the ECC (Enhanced Correlation Coefficient)
    algorithm. the ECC methodology can compensate for both shifts, shifts +
    rotations (euclidean), shifts + rotation + shear (affine), or homographic
    (3D) transformations of one image to the next.

    Parameters
    ----------
    im1 : np.ndarray
        template
    im2 : np.ndarray
        The image to be warped to match the template
    mode : str
        Warp mode, must be one of translation, euclidian, affine, homography (default)

    Returns
    -------
    im2_aligned : np.ndarray
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
    target_size = (min(im1.shape[1], im2.shape[1]), min(im1.shape[0], im2.shape[0]))

    im1_resized = cv2.resize(im1, target_size)
    im2_resized = cv2.resize(im2, target_size)

    im1_gray = cv2.cvtColor(im1_resized, cv2.COLOR_BGR2GRAY)  # template
    im2_gray = cv2.cvtColor(im2_resized, cv2.COLOR_BGR2GRAY)  # image to be aligned

    sz = im1_resized.shape

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
            im1_gray,
            im2_gray,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=1,
        )
    except cv2.error as e:
        raise RuntimeError(f'Error during ECC alignment: {e}')

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        im2_aligned = cv2.warpPerspective(
            im2_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        im2_aligned = cv2.warpAffine(
            im2_resized,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    row1_col0 = warp_matrix[0, 1]
    angle = math.degrees(math.asin(row1_col0))

    return im2_aligned, angle


def fourier_mellin_transform_match(image1_path, image2_path, corrMethod=None):
    """Apply Fourier-Mellin transform to match one image to another.

    Notes:
        - The function applies the Fourier-Mellin transform to match the image2 to the image1.
        - The images are converted to grayscale before applying the transform.
        - The resulting aligned image is returned as a numpy array.

    Example:
        aligned_image = fourier_mellin_transform_match('template.jpg', 'image.jpg')

    Parameters
    ----------
    image1_path : str
        The file path of the template image.
    image2_path : str
        The file path of the image to align.

    Returns
    -------
    np.ndarray:
        The aligned image as a numpy array.
    """

    img1 = dip.Image(image1_path)
    img2 = dip.Image(image2_path)

    # Convert images to grayscale if they are not already
    img1 = dip.Image(img1.TensorToSpatial())
    img2 = dip.Image(img2.TensorToSpatial())

    # They're gray-scale images, even if the JPEG file has RGB values
    img1_gray = img1(1)  # green channel only
    img2_gray = img2(1)

    # obtain transformation matrix
    out = dip.Image()
    matrix = dip.FourierMellinMatch2D(
        img1_gray, img2_gray, out=out, correlationMethod=corrMethod
    )

    print('########################')
    print(matrix)

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
    if img2.TensorElements() > 1:  # If img2 is a color image
        aligned_channels = []
        for i in range(img2.TensorElements()):  # For each color channel
            img2_channel = img2(i)
            moved_channel = dip.Image()
            dip.AffineTransform(img2_channel, out=moved_channel, matrix=matrix)
            aligned_channels.append(np.asarray(moved_channel))

        # Stack the aligned channels back into a color image
        aligned_image = np.stack(aligned_channels, axis=-1)
    else:  # If img2 is grayscale
        moved_img = dip.Image()
        dip.AffineTransform(img2, out=moved_img, matrix=matrix)
        aligned_image = np.asarray(moved_img)

    return aligned_image, angle


def align_images_with_translation(im0, im1):
    """Aligns two images with FFT phase correlation by finding the translation
    offset that minimizes the difference between them.

    Parameters:
    - im0: template image (BGR or grayscale)
    - im1: image to align (BGR or grayscale)

    Returns:
    - aligned_image: The second image aligned with the template
    """

    def translation(im0, im1):
        """Calculates the translation between two grayscale images using phase
        correlation.

        Parameters:
        - im0: The first input image (BGR or grayscale)
        - im1: The second input image (BGR or grayscale)

        Returns:
        - A list containing the translation values [t0, t1]
        """
        # Ensure images are grayscale
        if len(im0.shape) == 3:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        if len(im1.shape) == 3:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        # FFT phase correlation
        shape = im0.shape
        f0 = fft2(im0)
        f1 = fft2(im1)
        ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))

        # Find peak in cross-correlation
        t0, t1 = np.unravel_index(np.argmax(ir), shape)

        # Adjust shifts if they are larger than half the image size
        if t0 > shape[0] // 2:
            t0 -= shape[0]
        if t1 > shape[1] // 2:
            t1 -= shape[1]

        return [t0, t1]

    def apply_translation(image, translation):
        """Apply translation to an image.

        Parameters:
        - image: The input image (grayscale or BGR)
        - translation: A list containing the translation values [t0, t1]

        Returns:
        - The translated image
        """
        t0, t1 = translation
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, t1], [0, 1, t0]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        return translated_image

    # Calculate translation between images
    translation_values = translation(im0, im1)

    # Apply translation to the image which needs to be aligned
    aligned_image = apply_translation(im1, translation_values)

    return aligned_image


def rotationAlign(im1, im2):
    """Aligns two images by finding the rotation angle that minimizes the
    difference between them.

    Args:
        im1: The first input image template
        im2: The second input image to be aligned

    Returns:
        The rotated version of the second image, aligned with the first image
    """

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    height, width = im1_gray.shape[0:2]

    values = np.ones(360)

    # Find the rotation angle that minimizes the difference between the images
    for i in range(0, 360):
        rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), i, 1)
        rot = cv2.warpAffine(im2_gray, rotationMatrix, (width, height))
        values[i] = np.mean(im1_gray - rot)

    angle = np.argmin(values)
    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(im2, rotationMatrix, (width, height))

    return rotated, angle


def align_image_to_base(
    base_image: np.ndarray,
    image: dict[str, np.ndarray],
    *,
    selected_option: str,
    motion_model: str,
    feature_method: str = 'ORB',
) -> dict[str, dict[str, Any]]:
    """Aligns all images in a folder to a template image using the selected
    alignment method and preprocess options.

    Parameters
    ----------
    base_image : np.ndarray
        The file path of the template image
    input_files : list
        List of file paths of images to be aligned
    selected_option : str
        The selected alignment method
    motion_model : str
        The selected motion model for ECC alignment
    feature_method : str, optional
        The feature detection method to use ('SIFT', 'ORB', 'SURF'). Default is 'ORB'

    Returns
    -------
    aligned_images : dict[str, dict[str, Any]]
        List of tuples containing filename and aligned image.
    """

    # equal size between base_image and image, necessary for some alignment methods
    target_size = (base_image.shape[1], base_image.shape[0])
    resized_image = cv2.resize(image, target_size)

    if selected_option == 'Feature based alignment':
        aligned, angle = featureAlign(image, base_image, method=feature_method)

    elif selected_option == 'Enhanced Correlation Coefficient Maximization':
        aligned, angle = ecc_align(base_image, resized_image, mode=motion_model)

        if aligned is None:
            raise RuntimeError('Error aligning image')

    elif selected_option == 'Fourier Mellin Transform':
        corr_method = motion_model

        aligned, angle = fourier_mellin_transform_match(base_image, resized_image, corr_method)

    elif selected_option == 'FFT phase correlation':
        aligned = align_images_with_translation(base_image, resized_image)

    elif selected_option == 'Rotational Alignment':
        aligned, angle = rotationAlign(base_image, resized_image)

    else:  # default to feature based alignment
        aligned = featureAlign(base_image, image)

    return {'image': aligned, 'angle': angle}


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


def to_grayscale(image):
    """Convert an image to grayscale.

    Parameters:
    - image: The input image in BGR format.

    Returns:
    - grayscale_img: The input image converted to grayscale.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in BGR format
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    elif len(image.shape) == 2:  # Check if the image is already grayscale
        return image
    else:
        raise ValueError('Input image must be in BGR format with three channels or grayscale.')


def equalize_images(
    template: np.ndarray,
    image: np.ndarray,
    *,
    brightness=False,
    contrast=False,
    sharpness=False,
    color=False,
    reinhard=False,
    gray=False,
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
