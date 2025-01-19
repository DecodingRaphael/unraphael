from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove
from skimage import img_as_ubyte, measure

from unraphael.io import load_images, load_images_from_drc
from unraphael.locations import image_directory
from unraphael.types import ImageType

if TYPE_CHECKING:
    pass

_load_images = st.cache_data(load_images)
_load_images_from_drc = st.cache_data(load_images_from_drc)


def compute_size_in_cm(pixels: int, dpi: int) -> Optional[float]:
    """Compute the size in centimeters based on pixels and DPI."""
    if dpi == 0:
        raise ValueError('DPI cannot be zero.')
    inches = pixels / dpi
    cm = inches * 2.54
    return cm


def create_mask(image):
    binary_mask = remove(image, only_mask=True)
    return binary_mask


def get_image_size_resolution(image_file, name=None) -> Tuple[int, int, Tuple[float, float]]:
    """Get the height, width, and resolution of an image from an in-memory
    file.

    This function attempts multiple methods to extract DPI information:
    1. Basic PIL info dictionary
    2. JFIF header parsing
    3. EXIF data
    4. PhotoShop resolution data

    Parameters:
    - image_file (BytesIO): The uploaded image file.
    - name (str): Optional. The name of the image file for logging and warning purposes.

    Returns:
    - Tuple[int, int, Tuple[float, float]]: The height, width, and resolution (DPI) of
    the image.
    """
    try:
        with Image.open(image_file) as img:
            # Get dimensions (PIL's size returns width, height, so reverse for consistency)
            height_pixels, width_pixels = img.size[::-1]

            # Method 1: Check basic PIL info dictionary
            dpi = img.info.get('dpi')

            # Method 2: Try to get resolution from JFIF header
            if not dpi and hasattr(img, 'applist'):
                for segment, content in img.applist:
                    if segment == 'APP0' and content.startswith(b'JFIF'):
                        # JFIF header format: 'JFIF\x00\x01\x01\x00\x48\x00\x48\x00'
                        # Resolution units and values are at offset 7
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
                    # PhotoShop stores resolution in pixels/cm
                    if isinstance(photoshop, bytes):
                        # Find resolution info block (ID: 0x03ED)
                        pos = photoshop.find(b'\x03\xed')
                        if pos >= 0:
                            x_res = struct.unpack('>I', photoshop[pos + 4 : pos + 8])[0]
                            y_res = struct.unpack('>I', photoshop[pos + 8 : pos + 12])[0]
                            # Convert to DPI (1 inch = 2.54 cm)
                            dpi = (x_res * 2.54, y_res * 2.54)
                except Exception:
                    pass

            # If no DPI information found in any method, use default
            if not dpi or dpi == (0, 0):
                if name:
                    print(
                        f'DPI information not found in any metadata for image {name}. '
                        'Using default.'
                    )
                dpi = (96.0, 96.0)  # Common default DPI

            return height_pixels, width_pixels, dpi

    except Exception as e:
        if name:
            print(f'Error processing image {name}: {str(e)}')
        return 0, 0, (0.0, 0.0)


# def get_image_size_resolution(image_file, name=None) -> Tuple[int, int, Tuple[float, float]]:
#     """Get the height, width, and resolution of an image from an in-memory
#     file.

#     Parameters:
#     - image_file (BytesIO): The uploaded image file.
#     - name (str): Optional. The name of the image file for logging and warning purposes.

#     Returns:
#     - Tuple[int, int, Tuple[float, float]]: The height, width, and resolution of the image.
#     """
#     try:
#         with Image.open(image_file) as img:
#             # Note: PIL's img.size returns (width, height), so we reverse it here
#             height_pixels, width_pixels = img.size[::-1]

#             # Attempt to get DPI from the image metadata
#             dpi = img.info.get('dpi', (None, None))

#             # Handle cases where DPI might not be available or is set to default values
#             if dpi == (None, None) or dpi == (0, 0):
#                 if name:
#                     st.warning(
#                         f'DPI information not found or default value detected '
#                          'for image {name}.'
#                     )
#                 dpi = (96.0, 96.0)  # Common fallback DPI

#             return height_pixels, width_pixels, dpi  # Return (height, width)
#     except Exception as e:
#         st.error(f'Error processing image: {e}')
#         return 0, 0, (0.0, 0.0)


def pixels_to_cm(pixels, dpi):
    if dpi != 0:
        inches = pixels / dpi
        cm = inches * 2.54
        return cm
    else:
        return None


def process_image(image_file) -> tuple:
    """Process a single image file and return an ImageType object and its
    metrics."""

    # Read image data into memory
    image_bytes = image_file.read()
    image_data = imageio.imread(io.BytesIO(image_bytes))

    # Get the name from the original file
    name, _ = image_file.name.rsplit('.', 1)

    # Extract image size and DPI
    height, width, resolution = get_image_size_resolution(io.BytesIO(image_bytes), name=name)

    # Calculate size in centimeters
    height_cm = pixels_to_cm(height, resolution[1])
    width_cm = pixels_to_cm(width, resolution[0])

    # Create metrics dictionary
    metrics = {
        'height': height,
        'width': width,
        'dpi': resolution,
        'height_cm': height_cm,
        'width_cm': width_cm,
    }

    # Return the ImageType and metrics separately
    return ImageType(data=image_data, name=name), metrics


def calculate_corrected_area(
    image: np.ndarray,
    real_size_cm: list[float],
    # photo_size_cm: list[float],
    dpi: float,
) -> Optional[float]:
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

    Returns:
    -------
    Optional[float]
        The corrected area in square inches, or None if calculation fails
    """
    try:
        # Create mask of the main figure
        mask = remove(image, only_mask=True)
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        if not regions:
            st.error('No figure detected in the image.')
            return None

        # Get the largest connected component (main figure)
        largest_region = max(regions, key=lambda r: r.area)
        area_pixels = largest_region.area

        # Convert physical dimensions from cm to inches
        # real_height_inches = real_size_cm[0] / 2.54
        # real_width_inches = real_size_cm[1] / 2.54

        # Calculate the real and photo areas in inches
        # real_area_inches = (real_size_cm[0] / 2.54) * (real_size_cm[1] / 2.54)
        # photo_area_inches = (photo_size_cm[0] / 2.54) * (photo_size_cm[1] / 2.54)

        # The scaling factor is the ratio of the real-world painting area to
        # the photo area, which adjusts for any size differences between
        # the photo and the actual painting
        # scaling_factor = real_area_inches / photo_area_inches

        # The corrected_area calculated by this function represents
        # the area of the region of interest (the largest connected component)
        # in the original painting, but it is derived from the corresponding
        # area in the digital photo
        # area_inches = (area_pixels / (dpi**2)) * scaling_factor

        # Calculate area in square inches using the true DPI
        area_inches = area_pixels / (dpi**2)

        return area_inches

    except Exception as e:
        st.error(f'Error calculating area: {e}')
        return None


def load_images_widget(as_ubyte: bool = False, **loader_kwargs) -> tuple[list[ImageType], dict]:
    """Widget to load images and return their metrics."""

    load_example = st.sidebar.checkbox('Load example', value=False, key='load_example')
    uploaded_files = st.file_uploader('Upload Images', accept_multiple_files=True)

    images = []
    image_metrics = {}

    if load_example:
        loaded_images = _load_images_from_drc(image_directory, **loader_kwargs)

        for name, data in loaded_images.items():
            # Assuming data is image data and name is the image name
            image = ImageType(name=name, data=data)
            images.append(image)

            # Extract metrics
            height, width, resolution = get_image_size_resolution(io.BytesIO(data))
            height_cm = pixels_to_cm(height, resolution[1])
            width_cm = pixels_to_cm(width, resolution[0])
            metrics = {
                'height': height,
                'width': width,
                'dpi': resolution,
                'height_cm': height_cm,
                'width_cm': width_cm,
            }
            image_metrics[name] = metrics
    else:
        if not uploaded_files:
            st.info('Upload images to continue')
            st.stop()

        for file in uploaded_files:
            image, metrics = process_image(file)
            images.append(image)
            image_metrics[image.name] = metrics

    if not images:
        raise ValueError('No images were loaded')

    if as_ubyte:
        images = [image.apply(img_as_ubyte) for image in images]

    return images, image_metrics


def show_masks(
    images: list[ImageType],
    *,
    n_cols: int = 4,
    key: str = 'show_images',
    message: str = 'Select image',
    display_masks: bool = True,  # Add parameter to choose between showing images or masks
) -> None | ImageType:
    """Widget to show images or their masks with a given number of columns."""
    col1, col2 = st.columns(2)
    n_cols = col1.number_input(
        'Number of columns for display', value=8, min_value=1, step=1, key=f'{key}_cols'
    )
    options = [None] + [image.name for image in images]
    selected = col2.selectbox(message, options=options, key=f'{key}_sel')
    selected_image = None

    cols = st.columns(n_cols)

    for i, image in enumerate(images):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]

        if image.name == selected:
            selected_image = image

        if display_masks:
            # Create and display the mask
            mask = create_mask(image.data)
            col.image(
                mask, use_column_width=True, caption=f'Mask of {image.name}', channels='GRAY'
            )
        else:
            # Display the image itself
            col.image(image.data, use_column_width=True, caption=image.name)

    return selected_image
