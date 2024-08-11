from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import streamlit as st
from align import align_image_to_base
from equalize import equalize_image_with_base
from ratio_analysis import load_images_widget, show_aligned_masks_widget
from rembg import remove
from skimage import measure
from styling import set_custom_css
from widgets import show_images_widget

from unraphael.types import ImageType

_align_image_to_base = st.cache_data(align_image_to_base)
_equalize_image_with_base = st.cache_data(equalize_image_with_base)

# Helper functions


def compute_size_in_cm(pixels, dpi):
    """Compute the size in centimeters based on pixels and DPI."""
    if dpi != 0:
        inches = pixels / dpi
        cm = inches * 2.54
        return cm
    else:
        return None


def create_mask(image):
    binary_mask = remove(image, only_mask=True)
    return binary_mask


def calculate_corrected_area(image, real_size_cm, photo_size_cm, dpi):
    """Calculate the corrected area of an image based on the real and photo
    sizes."""

    photo_size_cm = photo_size_cm

    # Create mask and find the largest connected component
    largest_contour_mask = create_mask(np.array(image))
    labeled_image = measure.label(largest_contour_mask)
    regions = measure.regionprops(labeled_image)

    if not regions:
        st.error('No connected components found in the image.')
        return None

    largest_region = max(regions, key=lambda r: r.area)
    area_pixels = largest_region.area

    # Calculate the real and photo areas in inches
    real_area_inches = (real_size_cm[0] / 2.54) * (real_size_cm[1] / 2.54)
    photo_area_inches = (photo_size_cm[0] / 2.54) * (photo_size_cm[1] / 2.54)

    # The scaling factor is the ratio of the real-world painting area to
    # the photo area, which adjusts for any size differences between
    # the photo and the actual painting
    scaling_factor = real_area_inches / photo_area_inches

    # Debug: Print sizes and ratios
    print(f'Real area (inches^2): {real_area_inches}')
    print(f'Photo area (inches^2): {photo_area_inches}')
    print(f'Scaling factor: {scaling_factor}')

    # The corrected_area calculated by this function is intended
    # to represent the area of the region of interest (the largest
    # connected component) in the original painting, but it is
    # derived from the corresponding area in the digital photo
    corrected_area = area_pixels / (dpi**2) * scaling_factor

    return corrected_area


def equalize_images_widget(*, base_image: np.ndarray, images: dict[str, np.ndarray]):
    """This widget helps with equalizing images."""
    st.subheader('Equalization parameters')

    brightness = st.checkbox('Equalize brightness', value=False)
    contrast = st.checkbox('Equalize contrast', value=False)
    sharpness = st.checkbox('Equalize sharpness', value=False)
    color = st.checkbox('Equalize colors', value=False)
    reinhard = st.checkbox('Reinhard color transfer', value=False)

    preprocess_options = {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'color': color,
        'reinhard': reinhard,
    }

    return [
        _equalize_image_with_base(base_image=base_image, image=image, **preprocess_options)
        for image in images
    ]


def align_images_widget(*, base_image: ImageType, images: list[ImageType]) -> list[ImageType]:
    """This widget helps with aligning images."""
    st.subheader('Alignment parameters')

    options = [
        None,
        'Feature based alignment',
        'Enhanced Correlation Coefficient Maximization',
        'Fourier Mellin Transform',
        'FFT phase correlation',
        'Rotational Alignment',
        'User-provided keypoints (from pose estimation)',
    ]

    align_method = st.selectbox(
        'Alignment procedure:',
        options,
        help=(
            '**Feature based alignment**: Aligns images based on detected features using '
            'algorithms like SIFT, SURF, or ORB.'
            '\n\n**Enhanced Correlation Coefficient Maximization**: Estimates the '
            'he parameters of a geometric transformation between two images by '
            'maximizing the correlation coefficient.'
            '\n\n**Fourier Mellin Transform**: Uses the Fourier Mellin Transform to align '
            'images based on their frequency content.'
            '\n\n**FFT phase correlation**: Aligns images by computing '
            'the phase correlation between their Fourier transforms.'
            '\n\n**Rotational Alignment**: Aligns images by rotating them to a common '
            'orientation.'
        ),
    )

    if align_method == 'Feature based alignment':
        motion_model = st.selectbox(
            'Algorithm:',
            ['SIFT', 'SURF', 'ORB'],
        )
    elif align_method == 'Enhanced Correlation Coefficient Maximization':
        motion_model = st.selectbox(
            'Motion model:',
            ['translation', 'euclidian', 'affine', 'homography'],
            help=(
                'The motion model defines the transformation between the base '
                'image and the input images. Translation is the simplest model, '
                'while homography is the most complex.'
            ),
        )
    elif align_method == 'Fourier Mellin Transform':
        motion_model = st.selectbox(
            'Normalization method for cross correlation',
            [None, 'normalize', 'phase'],
            help=(
                'The normalization applied in the cross correlation. If `None`, '
                'the cross correlation is not normalized. If `normalize`, the '
                'cross correlation is normalized by the product of the magnitudes of the '
                'Fourier transforms of the images. If `phase`, the cross '
                'correlation is normalized by the product of the magnitudes and phases '
                'of the Fourier transforms of the images.'
            ),
        )
    else:
        motion_model = None

    res = []

    progress = st.progress(0, text='Aligning...')

    for i, image in enumerate(images):
        progress.progress((i + 1) / len(images), f'Aligning {image.name}...')
        res.append(
            _align_image_to_base(
                base_image=base_image,
                image=image,
                align_method=align_method,
                motion_model=motion_model,
            )
        )

    return res


def main():
    set_custom_css()

    st.title('Ratio analysis')

    with st.sidebar:
        images, image_metrics = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.stop()

    # Overview of image sizes and DPI
    st.subheader('Overview of Imported Images')

    for image in images:
        metrics = image_metrics[image.name]
        size_pixels = metrics.get('height'), metrics.get('width')
        dpi = metrics.get('dpi')
        size_cm = metrics.get('height_cm'), metrics.get('width_cm')

        st.write(f'**Image Name**: {image.name}')

        st.write(f'**Size (Height x Width)**: {size_pixels[0]} x {size_pixels[1]} pixels')
        st.write(f'**Size (Height x Width)**: {size_cm[0]:.2f} x {size_cm[1]:.2f} cm')
        st.write(f'**DPI**: {dpi[0]} x {dpi[1]}')
        st.write('---')

    st.subheader('Select base image')
    base_image = show_images_widget(
        images, key='original images', message='Select base image for alignment'
    )

    if not base_image:
        st.stop()

    # Equalize and align images
    col1, col2 = st.columns(2)

    with col1:
        images = equalize_images_widget(base_image=base_image, images=images)

    # creates aligned images which are now in similar size and gray scale
    with col2:
        images = align_images_widget(base_image=base_image, images=images)

    # show_images_widget(images, message='The aligned images')
    # Display the masks of the aligned images
    show_aligned_masks_widget(
        images, message='The masks of the aligned images', display_masks=True
    )

    # Upload excel file containing real sizes of paintings
    st.header('Upload Real Sizes Excel File')
    uploaded_excel = st.file_uploader('Choose an Excel file', type=['xlsx'])

    if images and uploaded_excel:
        real_sizes_df = pd.read_excel(uploaded_excel, header=0)

        st.write('Information on painting sizes:')
        st.write(real_sizes_df)

        if len(images) != len(real_sizes_df):
            st.error('The number of images and rows in the Excel file must match.')
            st.stop()

        # Add a slider to select the atol value
        atol_value = st.sidebar.slider(
            'Set absolute tolerance (atol) for area comparison:',
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help='Adjust the tolerance level for comparing the corrected areas.',
        )

        corrected_areas = []

        for i, uploaded_file in enumerate(images):
            image_data = uploaded_file.data
            image_name = uploaded_file.name

            # Retrieve sizes from the paintings
            real_size_cm = real_sizes_df.iloc[i, 1:3].tolist()

            # Retrieve dpi's from the photos
            dpi = image_metrics[image_name]['dpi'][0]  # Assuming square DPI, use dpi[0]

            # Compute the new size of the aligned image in cm
            height_pixels, width_pixels = image_data.shape[:2]
            photo_size_cm = [
                compute_size_in_cm(height_pixels, dpi),
                compute_size_in_cm(width_pixels, dpi),
            ]

            print(f'Photo size (cm): {photo_size_cm}')

            if None in photo_size_cm:
                st.error(f'Could not compute size for image {image_name}.')
                continue

            corrected_area = calculate_corrected_area(
                image_data, real_size_cm, photo_size_cm, dpi
            )
            corrected_areas.append((uploaded_file.name, corrected_area))

        # Generate all possible pairs between two paintings for comparison
        combinations = list(itertools.combinations(corrected_areas, 2))

        st.subheader('Results')

        # Compare each combination of images
        for (name1, area1), (name2, area2) in combinations:
            if area1 is not None and area2 is not None:
                area_ratio = area1 / area2
                st.write(f'Comparing {name1} and {name2}:')
                st.write(f'Corrected Area 1: {area1}')
                st.write(f'Corrected Area 2: {area2}')
                st.write(f'Ratio of Corrected Areas: {area_ratio}')

                # absolute tolerance of 5% for area ratio
                if np.isclose(area_ratio, 1.0, atol=atol_value):
                    st.success('The areas are very close to being equal.')
                else:
                    st.warning('The areas are not equal.')
    else:
        st.error('Please upload images and an excel file to continue.')


if __name__ == '__main__':
    main()
