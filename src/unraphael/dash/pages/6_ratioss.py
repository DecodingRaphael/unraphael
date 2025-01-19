from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from ratio_analysis import (
    calculate_corrected_area,
    compute_size_in_cm,
    load_images_widget,
    show_masks,
)
from styling import set_custom_css


def main():
    set_custom_css()

    st.title('Ratio analysis')

    with st.sidebar:
        images, image_metrics = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.stop()

    st.subheader('The masks of the images')

    show_masks(
        images,
        display_masks=True,
    )

    st.subheader('Information on sizes and DPI of images')

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

    # Upload excel file containing real sizes of paintings
    st.header('Upload information on real sizes of paintings')
    uploaded_excel = st.file_uploader('Choose an Excel file', type=['xlsx'])

    if images and uploaded_excel:
        try:
            real_sizes_df = pd.read_excel(uploaded_excel, header=0)
        except Exception as e:
            st.error(f'Error reading Excel file: {e}')
            st.stop()

        st.write('Information on painting sizes:')
        st.write(real_sizes_df)

        if len(images) != len(real_sizes_df):
            st.error('The number of images and rows in the Excel file must match.')
            st.stop()

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
            dpi = image_metrics[image_name]['dpi'][0]

            # Compute the photo size in centimeters
            height_pixels, width_pixels = image_data.shape[:2]
            photo_size_cm = [
                compute_size_in_cm(height_pixels, dpi),
                compute_size_in_cm(width_pixels, dpi),
            ]

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

        # Prepare data for heatmap
        image_names = [name for name, _ in corrected_areas]
        heatmap_data = np.zeros((len(image_names), len(image_names)))

        # Compare each combination of images
        for (name1, area1), (name2, area2) in combinations:
            if area1 is not None and area2 is not None:
                area_ratio = area1 / area2
                i = image_names.index(name1)
                j = image_names.index(name2)
                heatmap_data[i, j] = area_ratio
                heatmap_data[j, i] = area_ratio

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            xticklabels=image_names,
            yticklabels=image_names,
            cmap='coolwarm',
            cbar_kws={'label': 'Area Ratio'},
        )
        plt.title('Heatmap of area ratios')
        plt.xlabel('Image Names')
        plt.ylabel('Image Names')

        st.pyplot(fig)

        # Compare each combination of images
        for (name1, area1), (name2, area2) in combinations:
            if area1 is not None and area2 is not None:
                area_ratio = area1 / area2
                st.write(f'Comparing {name1} and {name2}:')
                st.write(f'Corrected area 1: {area1:.2f}')  # add unit
                st.write(f'Corrected area 2: {area2:.2f}')  # add unit
                st.write(f'Ratio of corrected areas: {area_ratio:.2f}')

                # absolute tolerance of 5% for area ratio
                if np.isclose(area_ratio, 1.0, atol=atol_value):
                    st.success(
                        'The areas of the main figures in the real paintings are '
                        'very close to being equal.'
                    )
                else:
                    st.warning('The areas are not equal.')
    else:
        st.error('Please upload images and an excel file to continue.')


if __name__ == '__main__':
    main()
