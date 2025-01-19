from __future__ import annotations

import itertools

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from ratio_analysis import calculate_corrected_area
from rembg import remove


def main():
    st.title('Painting Analysis')

    # First, get the Excel file with real dimensions
    st.sidebar.header('Upload painting dimensions')
    uploaded_excel = st.sidebar.file_uploader(
        'Choose Excel file with real dimensions', type=['xlsx']
    )

    if uploaded_excel:
        try:
            real_sizes_df = pd.read_excel(uploaded_excel, header=0)
            st.sidebar.success('Excel file loaded successfully')
            st.sidebar.dataframe(real_sizes_df)
        except Exception as e:
            st.sidebar.error(f'Error reading Excel file: {e}')
            st.stop()
    else:
        st.sidebar.warning('Please upload Excel file with painting dimensions')
        real_sizes_df = None

    # Now load images
    with st.sidebar:
        uploaded_files = st.file_uploader('Upload Images', accept_multiple_files=True)

    if not uploaded_files:
        st.info('Please upload images to continue')
        st.stop()

    if real_sizes_df is None or len(uploaded_files) != len(real_sizes_df):
        st.error('Number of images must match number of rows in Excel file')
        st.stop()

    # Process images with real dimensions
    images = []
    image_metrics = {}

    for i, file in enumerate(uploaded_files):
        # Get real dimensions from each painting from the imported Excel file
        real_height_cm = real_sizes_df.iloc[i, 1]  # height
        real_width_cm = real_sizes_df.iloc[i, 2]  # width

        if real_height_cm <= 0 or real_width_cm <= 0:
            st.error(f'Invalid real dimensions for file {file.name}. Skipping.')
            continue

        # Create a copy of the file pointer for each operation
        # file_copy = io.BytesIO(file.read())
        file.seek(0)  # Reset file pointer for next read

        # Read image data
        image_data = imageio.imread(file)
        name, _ = file.name.rsplit('.', 1)

        # Get image dimensions in pixels
        height_pixels, width_pixels = image_data.shape[:2]

        # Convert cm to inches
        physical_height_inches = real_height_cm / 2.54
        physical_width_inches = real_width_cm / 2.54

        # Calculate DPI
        dpi_v = height_pixels / physical_height_inches
        dpi_h = width_pixels / physical_width_inches

        # Use average DPI if they're close, otherwise warn about distortion
        if abs(dpi_v - dpi_h) > dpi_h * 0.05:  # 5% threshold
            st.warning(f'Image {name} appears to be distorted (non-uniform scaling)')

        dpi = (dpi_h + dpi_v) / 2  # Use average DPI

        # Create metrics
        metrics = {
            'height': height_pixels,
            'width': width_pixels,
            'dpi': dpi,
            'height_cm': real_height_cm,
            'width_cm': real_width_cm,
        }

        # Append to our collections
        images.append({'data': image_data, 'name': name})
        image_metrics[name] = metrics

    # Display image information
    st.subheader('Image Information')
    for image in images:
        metrics = image_metrics[image['name']]
        st.write(f'**Image Name**: {image["name"]}')
        st.write(f'**Pixel Dimensions**: {metrics["height"]} x {metrics["width"]} pixels')
        st.write(
            f'**Physical Size**: {metrics["height_cm"]:.2f} x {metrics["width_cm"]:.2f} cm'
        )
        st.write(f'**Calculated DPI**: {metrics["dpi"]:.1f}')
        st.write('---')

    # Display masks
    st.subheader('Figure Masks')
    cols = st.columns(min(3, len(images)))
    for idx, image in enumerate(images):
        mask = remove(image['data'], only_mask=True)
        cols[idx % 3].image(mask, caption=f'Mask: {image["name"]}', use_column_width=True)

    # Calculate areas
    st.subheader('Area Analysis')

    atol_value = st.slider(
        'Set tolerance for area comparison',
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help='Adjust the tolerance level for comparing areas (5% = 0.05)',
    )

    # Calculate corrected areas
    corrected_areas = []
    for i, image in enumerate(images):
        real_size_cm = [real_sizes_df.iloc[i, 1], real_sizes_df.iloc[i, 2]]
        corrected_area = calculate_corrected_area(
            image['data'], real_size_cm, image_metrics[image['name']]['dpi']
        )
        if corrected_area is not None:
            corrected_areas.append((image['name'], corrected_area))

    # Visualization
    if len(corrected_areas) > 1:
        st.subheader('Area Comparisons')

        # Prepare data for heatmap
        image_names = [name for name, _ in corrected_areas]
        n_images = len(image_names)
        heatmap_data = np.zeros((n_images, n_images))

        # Calculate all pairwise ratios
        for i, (name1, area1) in enumerate(corrected_areas):
            for j, (name2, area2) in enumerate(corrected_areas):
                ratio = area1 / area2 if area2 != 0 else 0
                heatmap_data[i, j] = ratio

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            xticklabels=image_names,
            yticklabels=image_names,
            cmap='coolwarm',
            center=1.0,
            vmin=0.5,
            vmax=1.5,
            cbar_kws={'label': 'Area Ratio'},
        )
        plt.title('Area Ratios Between Paintings')
        plt.xlabel('Reference Painting')
        plt.ylabel('Comparison Painting')
        st.pyplot(fig)

        # Detailed comparisons
        st.subheader('Detailed Comparisons')
        combinations = list(itertools.combinations(corrected_areas, 2))
        for (name1, area1), (name2, area2) in combinations:
            area_ratio = area1 / area2
            st.write(f'**Comparing {name1} and {name2}:**')
            col1, col2 = st.columns(2)
            with col1:
                st.write(f'Area of {name1}: {area1:.2f} sq inches')
                st.write(f'Area of {name2}: {area2:.2f} sq inches')
            with col2:
                st.write(f'Ratio: {area_ratio:.2f}')
                if np.isclose(area_ratio, 1.0, atol=atol_value):
                    st.success('Areas are approximately equal')
                else:
                    st.warning('Areas differ significantly')
            st.write('---')


if __name__ == '__main__':
    main()
