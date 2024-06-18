"""Module for equalization of figures and objects in paintings to a baseline
image in terms of contrast, sharpness, brightness, and colour."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.exposure import match_histograms

from unraphael.types import ImageType


def normalize_brightness(
    template_d: ImageType,
    image_d: ImageType,
) -> ImageType:
    """Normalizes the brightness of the target image based on the luminance of
    the template image. This refers to the process of bringing the brightness
    of the target image into alignment with the brightness of the template
    image. This can help ensure consistency in brightness perception between
    the two images, which is particularly useful in applications such as image
    comparison, enhancement, or blending.

    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to RGB.

    Parameters
    ----------
    template_d : ImageType
        Reference image (template) in RGB color format.
    image_d : ImageType
        Target image to be adjusted in RGB color format.

    Returns
    -------
    equalized_img : np.ndarray
        Adjusted target image with equalized brightness
    """
    target = image_d.data
    template = template_d.data

    # Convert the template image to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)

    # Split LAB channels of the template image
    l_template, a_template, b_template = cv2.split(template_lab)

    # Convert the target image to LAB color space
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)

    # Split LAB channels of the target image
    l_target, a_target, b_target = cv2.split(target_lab)

    # Adjust the L channel (brightness) of the target image based
    # on the mean brightness of the template
    l_target = (
        (l_target * (np.mean(l_template) / np.mean(l_target))).clip(0, 255).astype(np.uint8)
    )

    # Merge LAB channels back for the adjusted target image
    equalized_img_lab = cv2.merge([l_target, a_target, b_target])

    # Convert the adjusted LAB image back to RGB
    equalized_img = cv2.cvtColor(equalized_img_lab, cv2.COLOR_LAB2RGB)

    ## Using LAB color space
    # we convert the images (template and eq_image) to the LAB color space and calculate
    # the mean brightness from the luminance channel (L) only

    # Calculate the mean of the color images from the luminance channel
    mean_template_lab = np.mean(cv2.split(template_lab)[0])
    mean_eq_image_lab = np.mean(cv2.split(cv2.cvtColor(equalized_img, cv2.COLOR_RGB2LAB))[0])

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
    mean_template_gray = np.mean(cv2.cvtColor(template, cv2.COLOR_RGB2GRAY))
    mean_eq_image_gray = np.mean(cv2.cvtColor(equalized_img, cv2.COLOR_RGB2GRAY))
    # Calculate the ratio of the brightness of the grayscale images
    ratio_gray = mean_template_gray / mean_eq_image_gray

    out = image_d.replace(data=equalized_img)
    out.metrics.update(
        brightness_ratio_gray=ratio_gray,
        brightness_ratio_lab=ratio_lab,
        brightness_ratio_rgb=ratio_rgb,
    )

    return out


def normalize_contrast(
    template_d: ImageType,
    image_d: ImageType,
) -> ImageType:
    """Normalize the contrast of the target image to match the contrast of the
    template image.

    Parameters
    ----------
    template_d : ImageType
        Reference image (template) in RGB or grayscale format.
    image_d : ImageType
        Target image to be adjusted in RGB or grayscale format.

    Returns
    -------
    normalized_img : np.ndarray
        Target image with contrast normalized to match the template image.
    """
    target = image_d.data
    template = template_d.data

    if len(template.shape) == 3 and len(target.shape) == 3:  # Both images are color
        # Convert images to LAB color space
        template_lab = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)

        # Calculate contrast metric (standard deviation) for L channel of both images
        std_template = np.std(template_lab[:, :, 0])
        std_target = np.std(target_lab[:, :, 0])

        # Adjust contrast of target image to match template image
        l_target = (
            (target_lab[:, :, 0] * (std_template / std_target)).clip(0, 255).astype(np.uint8)
        )
        normalized_img_lab = cv2.merge([l_target, target_lab[:, :, 1], target_lab[:, :, 2]])

        # Convert the adjusted LAB image back to RGB
        normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2RGB)
        adapted_value = np.std(normalized_img_lab[:, :, 0])

    else:
        # Both images are grayscale
        # Calculate contrast metric (standard deviation) for grayscale intensity of both images
        std_template = np.std(template)
        std_target = np.std(target)

        # Adjust contrast of target image to match template image
        normalized_img = (target * (std_template / std_target)).clip(0, 255).astype(np.uint8)
        adapted_value = np.std(normalized_img[:, :, 0])

    out = image_d.replace(data=normalized_img)
    out.metrics.update(
        contrast_template=std_template,
        contrast_target=std_target,
        contrast_ratio=std_template / std_target,
        adapted_value=adapted_value,
    )

    return out


def normalize_sharpness(
    template_d: ImageType,
    image_d: ImageType,
) -> ImageType:
    """Normalize the sharpness of the target image to match the sharpness of
    the template image.

    Parameters
    ----------
    template_d : ImageType
        Reference image (template) in RGB or grayscale format.
    image_d : ImageType
        Target image to be adjusted in RGB or grayscale format.

    Returns
    -------
    normalized_img : np.ndarray
        Target image with sharpness normalized to match the template image.
    """
    target = image_d.data
    template = template_d.data

    if len(template.shape) == 3 and len(target.shape) == 3:
        # Both images are color
        # Convert images to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
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

    # Calculate sharpness value for the normalized image
    grad_x_normalized = cv2.Sobel(
        cv2.cvtColor(normalized_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=3
    )
    grad_y_normalized = cv2.Sobel(
        cv2.cvtColor(normalized_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3
    )
    grad_normalized = cv2.magnitude(grad_x_normalized, grad_y_normalized)
    mean_grad_normalized = np.mean(grad_normalized)

    out = image_d.replace(data=normalized_img)
    out.metrics.update(
        sharpness_template=mean_grad_template,
        sharpness_target=mean_grad_target,
        sharpness_ratio=mean_grad_template / mean_grad_target,
        sharpness_normalized=mean_grad_normalized,
    )

    return out


def normalize_colors(
    template_d: ImageType,
    image_d: ImageType,
) -> ImageType:
    """Normalize the colors of the target image to match the color distribution
    of the template image.

    Parameters
    ----------
    template_d : ImageType
        Reference image (template) in RGB color format.
    image_d : ImageType
        Target image to be adjusted in RGB color format.

    Returns
    -------
    normalized_img : np.ndarray
        Target image with colors normalized to match the template image.
    """
    target = image_d.data
    template = template_d.data

    matched = match_histograms(target, template, channel_axis=-1)

    out = image_d.replace(data=matched)
    return out


def reinhard_color_transfer(
    template_d: ImageType,
    image_d: ImageType,
) -> ImageType:
    """Perform Reinhard color transfer from the template image to the target
    image.

    Parameters
    ----------
    template_d : ImageType
        Reference image (template) in RGB color format.
    image_d : ImageType
        Target image to be adjusted in RGB color format.

    Returns
    -------
    adjusted_img : np.ndarray
        Target image with colors adjusted using Reinhard color transfer.
    """
    target = image_d.data
    template = template_d.data

    # Convert images to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)

    # Compute mean and standard deviation of each channel in LAB color space
    template_mean = template_lab.mean((0, 1))
    template_std = template_lab.std((0, 1))
    target_mean = target_lab.mean((0, 1))
    target_std = target_lab.std((0, 1))

    # Apply color transfer
    target_lab = ((target_lab - target_mean) * (template_std / target_std)) + template_mean
    target_lab = np.clip(target_lab, 0, 255)

    # Convert back to RGB color space
    adjusted_img = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    out = image_d.replace(data=adjusted_img)
    return out


def equalize_image_with_base(
    base_image: ImageType,
    image_d: ImageType,
    *,
    brightness: bool = False,
    contrast: bool = False,
    sharpness: bool = False,
    color: bool = False,
    reinhard: bool = False,
) -> ImageType:
    """Preprocesses the input image based on the selected enhancement options.

    Parameters
    ----------
    base_image : np.ndarray
        The base image to equalize to
    image : np.ndarray
        The input image in RGB format.
    brightness : bool
        Whether to equalize brightness.
    contrast : bool
        Whether to equalize contrast.
    sharpness : bool
        Whether to equalize sharpness.
    color : bool
        Whether to equalize colors.
    reinhard : bool
        Whether to perform Reinhard color transfer

    Returns
    -------
    preprocessed_image_d : ImageType
        The equalized image
    """
    if brightness:
        image_d = normalize_brightness(template_d=base_image, image_d=image_d)

    if contrast:
        image_d = normalize_contrast(template_d=base_image, image_d=image_d)

    if sharpness:
        image_d = normalize_sharpness(template_d=base_image, image_d=image_d)

    if color:
        image_d = normalize_colors(template_d=base_image, image_d=image_d)

    if reinhard:
        image_d = reinhard_color_transfer(template_d=base_image, image_d=image_d)

    return image_d
