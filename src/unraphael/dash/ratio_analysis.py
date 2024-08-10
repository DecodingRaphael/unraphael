from __future__ import annotations

import io
from typing import TYPE_CHECKING, Tuple

import imageio.v3 as imageio
import streamlit as st
from PIL import Image
from skimage import img_as_ubyte

from unraphael.io import load_images, load_images_from_drc, resize_to_width
from unraphael.locations import image_directory
from unraphael.types import ImageType

if TYPE_CHECKING:
    pass

_load_images = st.cache_data(load_images)
_load_images_from_drc = st.cache_data(load_images_from_drc)


def get_image_size_resolution(image_file, name=None) -> Tuple[int, int, Tuple[float, float]]:
    """Get the width, height, and resolution of an image from an in-memory
    file.

    Parameters:
    - image_file (BytesIO): The uploaded image file.
    - name (str): Optional. The name of the image file for logging and warning purposes.

    Returns:
    - Tuple[int, int, Tuple[float, float]]: The width, height, and resolution of the image.
    """
    try:
        with Image.open(image_file) as img:
            width_pixels, height_pixels = img.size

            # Attempt to get DPI from the image metadata
            dpi = img.info.get('dpi', (None, None))

            # Handle cases where DPI might not be available or is set to default values
            if dpi == (None, None) or dpi == (0, 0):
                if name:
                    st.warning(
                        f'DPI information not found or default value detected for image {name}.'
                    )
                dpi = (72.0, 72.0)  # Common fallback DPI

            return width_pixels, height_pixels, dpi
    except Exception as e:
        st.error(f'Error processing image: {e}')
        return 0, 0, (0.0, 0.0)


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

    # Extract image size and DPI using the modified get_image_size_resolution function
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


def load_images_widget(as_ubyte: bool = False, **loader_kwargs) -> tuple[list[ImageType], dict]:
    """Widget to load images and return their metrics."""

    load_example = st.sidebar.checkbox('Load example', value=False, key='load_example')
    uploaded_files = st.file_uploader('Upload Images', accept_multiple_files=True)

    images = []
    image_metrics = {}

    if load_example:
        images = _load_images_from_drc(image_directory, **loader_kwargs)
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

    images = equalize_width_widget(images)

    if as_ubyte:
        images = [image.apply(img_as_ubyte) for image in images]

    return images, image_metrics


def equalize_width_widget(images: list[ImageType]) -> list[ImageType]:
    """This widget equalizes the width of the images."""
    enabled = st.checkbox('Equalize width', value=True)

    width = st.number_input(
        'Width',
        value=240,
        step=10,
        key='width',
        disabled=(not enabled),
    )

    if enabled:
        return [image.apply(resize_to_width, width=width) for image in images]

    return images
