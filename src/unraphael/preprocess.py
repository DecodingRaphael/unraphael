from __future__ import annotations

import cv2
import numpy as np
import rembg


def apply_mask(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to original image."""
    return cv2.bitwise_and(original_image, original_image, mask=mask)


def process_image(
    image: np.array,
    *,
    bilateral_strength: int,
    clahe_clip_limit: float,
    clahe_tiles: int,
    sigma_sharpness: float,
    contrast: float,
    brightness: int,
    sharpening_kernel_size: int,
    saturation_factor: float,
) -> np.ndarray:
    """Process the uploaded image with user-defined parameters.

    Parameters
    ----------
    image : np.array
        Input image
    bilateral_strength : int
        Apply bilateral blur filter to each color channel
    clahe_clip_limit : float
        Clip limit for histogram equalization
    clahe_tiles : int
        Number of tiles for histogram equalization
    contrast : float
      Change contrast
    brightness : int
        Change brightness
    sigma_sharpness : float
    sharpening_kernel_size : int
        Sharpen image with this kernel size
    saturation_factor : float
        Multiply the saturation channel by user-defined factor

    Returns
    -------
    image : np.ndarray
        Processed image
    """
    # Check if the image is grayscale and convert it to 3 channels if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Split PIL image into its individual color channels
    blue, green, red = cv2.split(image)

    # Apply bilateral blur filter to each color channel with user-defined 'bilateral_strength'
    blue_blur = cv2.bilateralFilter(blue, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)
    green_blur = cv2.bilateralFilter(green, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)
    red_blur = cv2.bilateralFilter(red, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)

    # Create CLAHE object with user-defined clip limit
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tiles, clahe_tiles))

    # Adjust histogram and contrast for each color channel using CLAHE
    blue_eq = clahe.apply(blue_blur)
    green_eq = clahe.apply(green_blur)
    red_eq = clahe.apply(red_blur)

    # Merge the color channels back into a single RGB image
    output_img = cv2.merge((blue_eq, green_eq, red_eq))

    # Color saturation: convert image from BGR color space to HSV (Hue, Saturation, Value)
    # color space
    hsv_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)

    # Multiply the saturation channel by user-defined 'saturation_factor'
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 1, 254).astype(
        np.uint8
    )

    # Convert image back to BGR color space
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Create user-defined 'sharpening_kernel_size'
    kernel = np.ones((sharpening_kernel_size, sharpening_kernel_size), np.float32) * -1
    kernel[sharpening_kernel_size // 2, sharpening_kernel_size // 2] = sharpening_kernel_size**2

    # Apply sharpening kernel to image using filter2D
    processed_image = cv2.filter2D(result_image, -1, kernel)

    # Alpha controls contrast and beta controls brightness
    processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast, beta=brightness)

    # Additional sharpening: Create the sharpening kernel and apply it to the image
    custom_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Sharpen the image
    processed_image = cv2.filter2D(processed_image, -1, custom_kernel)

    # Apply Gaussian blur to the image with user-defined 'sigma_sharpness'
    processed_image = cv2.GaussianBlur(processed_image, (0, 0), sigma_sharpness)

    return processed_image


def remove_background(image: np.ndarray, mask_process: bool = False, **kwargs) -> np.ndarray:
    """Remove background using rembg.

    Parameters
    ----------
    image : np.ndarray
        Input image
    mask_process : bool
        Set `only_mask` and `post_process_mask` to True
    **kwargs
        These parameters are passed to rembg.remove

    Returns
    -------
    image : np.ndarray
        Image with background removed
    """
    if mask_process:
        kwargs['only_mask'] = True
        kwargs['post_process_mask'] = True

    image = rembg.remove(image, **kwargs)

    return image
