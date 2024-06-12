from __future__ import annotations

import numpy as np
import rembg
from skimage import color, img_as_ubyte
from skimage.color import hsv2rgb, rgb2hsv
from skimage.exposure import adjust_gamma, equalize_adapthist
from skimage.filters import rank, unsharp_mask
from skimage.morphology import disk


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to original image."""
    return np.dstack([image, mask])


def process_image(
    image: np.array,
    *,
    bilateral_strength: int,
    clahe_clip_limit: float,
    clahe_tiles: int,
    sigma_sharpness: float,
    gamma: float,
    gain: int,
    sharpening_radius: int,
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
    gamma : float
        Gamma correction
    gain : int
        Gain correction
    sharpening_radius : int
        Sharpen image with this radius
    saturation_factor : float
        Multiply the saturation channel by user-defined factor

    Returns
    -------
    image : np.ndarray
        Processed image
    """
    # convert grayscale to 3 channels if necessary
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    # Split image into its individual color channels
    channels = [
        image[:, :, 0],
        image[:, :, 1],
        image[:, :, 2],
    ]

    # Apply bilateral blur filter to each color channel
    channels = [
        rank.mean_bilateral(channel, footprint=disk(bilateral_strength), s0=55, s1=55)
        for channel in channels
    ]

    # Contrast limited adabtive histogram equalization
    kernel_size = tuple([max(s // clahe_tiles, 1) for s in image.shape[:2]])
    channels = [
        equalize_adapthist(channel, clip_limit=clahe_clip_limit, kernel_size=kernel_size)
        for channel in channels
    ]

    image = np.dstack(channels)

    hsv_image = rgb2hsv(image)

    # Multiply the saturation channel by user-defined 'saturation_factor'
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 1)

    image = hsv2rgb(hsv_image)

    # Apply sharpening to image using unsharp mask method
    image = unsharp_mask(image, radius=sharpening_radius, amount=1)

    # Gamma and gain correction
    image = adjust_gamma(image, gamma=gamma, gain=gain)

    return image


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

    image = img_as_ubyte(image)

    image = rembg.remove(image, **kwargs)

    return image
