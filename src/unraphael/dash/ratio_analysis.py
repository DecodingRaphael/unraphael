from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove
from skimage import measure

from unraphael.io import load_images, load_images_from_drc

if TYPE_CHECKING:
    pass

_load_images = st.cache_data(load_images)
_load_images_from_drc = st.cache_data(load_images_from_drc)


def get_image_size_resolution(image_array: np.ndarray, name=None) -> Tuple[int, int, Tuple[float, float], float, float]:
    """Get the height, width, and resolution of an image from an in-memory
    NumPy array."""
    try:
        # Convert NumPy array to a BytesIO object
        image_bytes = io.BytesIO()
        # Save the NumPy array (image) into the BytesIO buffer as a PNG
        Image.fromarray(image_array).save(image_bytes, format='PNG')
        image_bytes.seek(0)  # Reset the buffer's position to the beginning

        # Open the image from the BytesIO buffer
        with Image.open(image_bytes) as img:
            # Get dimensions (PIL's size returns width, height, so reverse for consistency)
            height_pixels, width_pixels = img.size[::-1]

            # Method 1: Check basic PIL info dictionary
            dpi = img.info.get('dpi')

            # Method 2: Try to get resolution from JFIF header
            if not dpi and hasattr(img, 'applist'):
                for segment, content in img.applist:
                    if segment == 'APP0' and content.startswith(b'JFIF'):
                        try:
                            unit, x_density, y_density = struct.unpack('>BHH', content[7:12])
                            if unit == 1:  # dots per inch
                                dpi = (x_density, y_density)
                            elif unit == 2:  # dots per cm
                                dpi = (x_density * 2.54, y_density * 2.54)
                        except struct.error:
                            pass

            # Method 3: Try to get resolution from EXIF data
            if not dpi:
                try:
                    exif = img.getexif()
                    if exif:
                        x_resolution = exif.get(282)  # XResolution
                        y_resolution = exif.get(283)  # YResolution
                        resolution_unit = exif.get(296, 2)  # ResolutionUnit, default to inches

                        if x_resolution and y_resolution:
                            x_dpi = float(x_resolution[0]) / float(x_resolution[1])
                            y_dpi = float(y_resolution[0]) / float(y_resolution[1])

                            if resolution_unit == 3:  # If resolution is in cm
                                x_dpi *= 2.54
                                y_dpi *= 2.54

                            dpi = (x_dpi, y_dpi)
                except Exception:
                    pass

            # Method 4: Try to get resolution from PhotoShop metadata
            if not dpi and 'photoshop' in img.info:
                photoshop = img.info['photoshop']
                try:
                    if isinstance(photoshop, bytes):
                        pos = photoshop.find(b'\x03\xed')
                        if pos >= 0:
                            x_res = struct.unpack('>I', photoshop[pos + 4 : pos + 8])[0]
                            y_res = struct.unpack('>I', photoshop[pos + 8 : pos + 12])[0]
                            dpi = (x_res * 2.54, y_res * 2.54)
                except Exception:
                    pass

            # Default DPI if no resolution found
            if not dpi or dpi == (0, 0):
                if name:
                    print(f'DPI information not found in any metadata for image {name}. ' 'Using default.')
                dpi = (96.0, 96.0)  # Common default DPI

            dpi_x, dpi_y = dpi
            dpi_x = dpi_x if dpi_x != 0 else 96.0
            dpi_y = dpi_y if dpi_y != 0 else 96.0

            # Calculate physical sizes in inches
            height_inches = height_pixels / dpi_y
            width_inches = width_pixels / dpi_x

            return (height_pixels, width_pixels, (dpi_x, dpi_y), height_inches, width_inches)

    except Exception as e:
        print(f'Error occurred while processing image: {e}')
        raise


def calculate_corrected_area(image: np.ndarray, real_size_cm: list[float], dpi: float, tolerance: float = 0.05) -> Optional[float]:
    """Calculate the corrected area of an image based on real physical
    dimensions and DPI.

    Parameters:
    ----------
    image : np.ndarray
        The image data
    real_size_cm : list[float]
        Physical dimensions [height, width] in centimeters
    dpi : float
        The calculated true DPI of the image
    tolerance : float
        Maximum allowed difference between height and width ratios (default 5%)

    Returns:
    -------
    Optional[float]
        The corrected area in square inches, or None if calculation fails
    """
    try:
        # Print input parameters
        print('\nInput Parameters:')
        print(f'Dimensions of real painting (cm): {real_size_cm[0]:.2f} x {real_size_cm[1]:.2f}')
        print(f'DPI: {dpi:.2f}')

        # Create mask of the main figure
        mask = remove(image, only_mask=True)

        # Protect against mask processing failures
        if mask is None or mask.size == 0:
            st.error('Failed to create mask')
            return None

        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        if not regions:
            st.error('No figure detected in the image.')
            return None

        # Get the largest connected component (main figure)
        largest_region = max(regions, key=lambda r: r.area)
        area_pixels = largest_region.area

        # Get image dimensions in pixels
        img_height, img_width = image.shape[:2]
        print(f'Image dimensions (pixels): {img_height} x {img_width}')

        # Calculate physical dimensions from pixels and DPI
        photo_height_inches = img_height / dpi
        photo_width_inches = img_width / dpi
        print('\nPhoto dimensions (inches): {photo_height_inches:.2f} x {photo_width_inches:.2f}')

        # Convert real dimensions to inches
        real_height_inches = real_size_cm[0] / 2.54
        real_width_inches = real_size_cm[1] / 2.54
        print('Real dimensions (inches): {real_height_inches:.2f} x {real_width_inches:.2f}')

        # Calculate scaling ratios
        height_ratio = real_height_inches / photo_height_inches
        width_ratio = real_width_inches / photo_width_inches
        print('\nScaling Ratios:')
        print(f'Height ratio (real/photo): {height_ratio:.4f}')
        print(f'Width ratio (real/photo): {width_ratio:.4f}')

        # Check if scaling ratios are consistent within tolerance
        ratio_diff = abs(height_ratio - width_ratio) / min(height_ratio, width_ratio)
        print(f'Ratio difference: {ratio_diff:.2%}')

        # If scaling ratios are too different, use the less extreme ratio
        if ratio_diff > tolerance:
            st.warning(f'Inconsistent scaling detected with a {ratio_diff:.2%} ' 'difference between height and width ratios. Using more ' 'conservative scaling.')
            # Use the ratio closer to 1.0 to minimize distortion
            if abs(height_ratio - 1.0) < abs(width_ratio - 1.0):
                scaling_factor = height_ratio
            else:
                scaling_factor = width_ratio
        else:
            # Use average of height and width ratios for scaling
            scaling_factor = (height_ratio + width_ratio) / 2

        print(f'Final scaling factor: {scaling_factor:.4f}')

        # Calculate area in square inches using the true DPI and applying scaling
        raw_area = area_pixels / (dpi**2)
        corrected_area = raw_area * scaling_factor
        print('\nArea Calculations:')
        print(f'Raw area (sq inches): {raw_area:.2f}')
        print(f'Corrected area (sq inches): {corrected_area:.2f}')

        return corrected_area

    except Exception as e:
        st.error(f'Error calculating area: {e}')
        print(f'Error in calculation: {str(e)}')
        import traceback

        traceback.print_exc()
        return None


# def calculate_corrected_area(
#     image: np.ndarray,
#     real_size_cm: list[float],
#     dpi: float,
#     tolerance: float = 0.05
# ) -> Optional[float]:
#     """Calculate the corrected area of an image based on real physical dimensions and DPI.

#     Parameters:
#     ----------
#     image : np.ndarray
#         The image data
#     real_size_cm : list[float]
#         Physical dimensions [height, width] in centimeters
#     dpi : float
#         The calculated true DPI of the image
#     tolerance : float
#         Maximum allowed difference between height and width ratios (default 5%)

#     Returns:
#     -------
#     Optional[float]
#         The corrected area in square inches, or None if calculation fails
#     """
#     try:
#         # Print input parameters
#         print("\nInput Parameters:")
#         print(f"Real dimensions (cm): {real_size_cm[0]:.2f} x {real_size_cm[1]:.2f}")
#         print(f"DPI: {dpi:.2f}")

#         # Create mask of the main figure
#         mask = remove(image, only_mask=True)
#         labeled_mask = measure.label(mask)
#         regions = measure.regionprops(labeled_mask)

#         if not regions:
#             st.error('No figure detected in the image.')
#             return None

#         # Get the largest connected component (main figure)
#         largest_region = max(regions, key=lambda r: r.area)
#         area_pixels = largest_region.area
#         print(f"\nMask Area (pixels): {area_pixels}")

#         # Get image dimensions in pixels
#         img_height, img_width = image.shape[:2]
#         print(f"Image dimensions (pixels): {img_height} x {img_width}")

#         # Calculate physical dimensions from pixels and DPI
#         photo_height_inches = img_height / dpi
#         photo_width_inches = img_width / dpi
#         print(f"\nPhoto dimensions (inches): {photo_height_inches:.2f} x
#         {photo_width_inches:.2f}")

#         # Convert real dimensions to inches
#         real_height_inches = real_size_cm[0] / 2.54
#         real_width_inches = real_size_cm[1] / 2.54
#         print(f"Real dimensions (inches): {real_height_inches:.2f} x
#         {real_width_inches:.2f}")

#         # Calculate scaling ratios
#         height_ratio = real_height_inches / photo_height_inches
#         width_ratio = real_width_inches / photo_width_inches
#         print(f"\nScaling Ratios:")
#         print(f"Height ratio (real/photo): {height_ratio:.4f}")
#         print(f"Width ratio (real/photo): {width_ratio:.4f}")

#         # Check if scaling ratios are consistent within tolerance
#         ratio_diff = abs(height_ratio - width_ratio) / min(height_ratio, width_ratio)
#         print(f"Ratio difference: {ratio_diff:.2%}")

#         if ratio_diff > tolerance:
#             st.warning(f'Inconsistent scaling detected: {ratio_diff:.1%} difference between
#         height and width ratios')

#         # Use average of height and width ratios for scaling
#         scaling_factor = (height_ratio + width_ratio) / 2
#         print(f"Final scaling factor (average): {scaling_factor:.4f}")

#         # Calculate area in square inches using the true DPI and applying scaling
#         raw_area = area_pixels / (dpi ** 2)
#         area_inches = raw_area * scaling_factor
#         print(f"\nArea Calculations:")
#         print(f"Raw area (sq inches): {raw_area:.2f}")
#         print(f"Corrected area (sq inches): {area_inches:.2f}")

#         return area_inches

#     except Exception as e:
#         st.error(f'Error calculating area: {e}')
#         print(f"Error in calculation: {str(e)}")
#         return None
