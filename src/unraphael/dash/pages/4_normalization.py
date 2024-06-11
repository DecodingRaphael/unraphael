# run with: streamlit run normalization_app.py --server.enableXsrfProtection false

# Import libraries
from __future__ import annotations

import io

import cv2
import imageio
import numpy as np
import streamlit as st
from outline_normalization import align_all_selected_images_to_template
from PIL import Image
from streamlit_image_comparison import image_comparison
from widgets import load_config_widget, load_images_widget, show_images_widget

st.set_page_config(layout='wide', page_title='')


def image_downloads_widget(*, images: dict[str, np.ndarray]):
    """This widget takes a dict of images and shows them with download
    buttons."""

    st.title('Save Aligned Images to Disk')

    cols = st.columns(len(images))

    for col, key in zip(cols, images):
        image = images[key]

        height, width = image.shape[:2]

        # Remove '.jpg' or '.png' from filename if present
        if '.jpg' in key:
            filename = key.replace('.jpg', '.png')
        elif '.png' in key:
            filename = key.replace('.png', '.png')  # Just to ensure it's .png
        else:
            filename = f'{key}.png'  # If neither '.jpg' nor '.png' is present

        img_bytes = io.BytesIO()
        imageio.imwrite(img_bytes, image, format='png')
        img_bytes.seek(0)

        col.download_button(
            label=f' {filename} ({width}x{height})',
            data=img_bytes,
            file_name=filename,
            mime='image/png',
            key=filename,
        )


def main():
    st.title('Image normalization')
    st.write('For a selected image, normalize and align all other images')
    st.markdown('---')

    with st.sidebar:
        load_config_widget()

        uploaded_files = load_images_widget(as_gray=False)

    processed_images = []

    if not uploaded_files:
        st.stop()

    with st.expander('Parameters for equalizing images', expanded=False):
        st.write(
            'The processed image is shown with a preset of parameters. Use the sliders to '
            'explore the effects of image filters, or to refine the adjustment.'
        )

        col1, col2 = st.columns(2)

        brightness = col1.checkbox('Equalize brightness', value=False)
        contrast = col1.checkbox('Equalize contrast', value=False)
        sharpness = col1.checkbox('Equalize sharpness', value=False)
        color = col2.checkbox('Equalize colors', value=False)
        reinhard = col2.checkbox('Reinhard color transfer', value=False)

    preprocess_options = {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'color': color,
        'reinhard': reinhard,
    }

    with st.expander('Parameters for aligning images', expanded=False):
        st.write(
            (
                'The following methods are used for image registration and alignment. '
                'Depending on your specific alignment requirements and computational '
                'constraints, you may choose one method over the other. Example usage '
                'scenarios and comparative analysis can help you choose the most suitable '
                'alignment technique for your specific requirements.'
            )
        )

        st.write(
            (
                '- **Feature-based Alignment (ORB, SIFT or SURF)**: '
                'Utilizes feature detection and matching for estimating '
                'translation, rotation, shear, and scaling. Suitable for images with '
                'distinct features and complex transformations. Note that keypoint '
                'matching may fail with poor feature detection.'
            )
        )

        st.write(
            (
                '- **Enhanced Correlation Coefficient (ECC) Maximization**: '
                'Identifies the geometric transformation that maximizes the correlation '
                'coefficient between two images. It can handle translation, rotation, and '
                'scaling, especially accurate for small to moderate transformations, '
                'and robust to noise and varying illumination.'
            )
        )

        st.write(
            (
                '- **Fast Fourier Transform (FFT) Phase Correlation Method**: '
                'Primarily designed for translational shifts. For rotation, consider '
                'alternate methods like log-polar transform or feature matching. Efficient '
                'for translational alignment but may not handle rotation or '
                'scaling effectively.'
            )
        )

        st.write(
            (
                '- **Fourier Mellin Transform (FMT) Method**: Logarithm of the Fourier '
                'magnitude of an image followed by another Fourier transform to obtain a '
                'log-polar transform. Rotation and scale invariant but computationally '
                'intensive compared to other methods.'
            )
        )

        st.write(
            (
                '- **Rotation Alignment Method**: Aligns images by finding the '
                'optimal rotation to minimize the difference between them. Suited when '
                'rotation is the primary misalignment source and computational cost '
                'is not a major concern.'
            )
        )

        st.write(
            (
                '- **User-provided keypoints** (from pose estimation): '
                'Aligns images based on user-provided keypoints obtained from pose estimation.'
            )
        )

    col3, col4 = st.columns(2)

    options = [
        'Feature based alignment',
        'Enhanced Correlation Coefficient Maximization',
        'Fourier Mellin Transform',
        'FFT phase correlation',
        'Rotational Alignment',
        'User-provided keypoints (from pose estimation)',
    ]

    selected_option = col3.selectbox(
        'Alignment procedure:',
        options,
        help=(
            '**Feature based alignment**: Aligns images based on detected features using '
            'algorithms like SIFT, SURF, or ORB.'
            '\n**Enhanced Correlation Coefficient Maximization**: Estimates the parameters '
            'of a geometric transformation between two images by maximizing the correlation '
            'coefficient.'
            '\n**Fourier Mellin Transform**: Uses the Fourier Mellin Transform to align '
            'images based on their frequency content.'
            '\n**FFT phase correlation**: Aligns images by computing the phase correlation '
            'between their Fourier transforms.'
            '\n**Rotational Alignment**: Aligns images by rotating them to a common '
            'orientation.'
        ),
    )

    motion_model = None

    if selected_option == 'Feature based alignment':
        motion_model = col4.selectbox(
            'Algorithm:',
            ['SIFT', 'SURF', 'ORB'],
        )

    if selected_option == 'Enhanced Correlation Coefficient Maximization':
        motion_model = col4.selectbox(
            'Motion model:',
            ['translation', 'euclidian', 'affine', 'homography'],
            help=(
                'The motion model defines the transformation between the base image and the '
                'input images. Translation is the simplest model, while homography is the most '
                'complex.'
            ),
        )

    if selected_option == 'Fourier Mellin Transform':
        motion_model = col4.selectbox(
            'Normalization method for cross correlation',
            [None, 'normalize', 'phase'],
            help=(
                'The normalization applied in the cross correlation. If `None` is selected, '
                ' the cross correlation is not normalized. If `normalize` is selected, the '
                'cross correlation is normalized by the product of the magnitudes of the '
                ' Fourier transforms of the images. If `phase` is selected, the cross '
                ' correlation is normalized by the product of the magnitudes and phases '
                'of the Fourier transforms of the images.'
            ),
        )

    scol1, scol2 = st.columns(2)
    fcol1, fcol2 = st.columns(2)

    selected = show_images_widget(uploaded_files)

    # todo: default to None to in show_images_widget
    if selected:
        base_image = uploaded_files.pop(selected)
        other_images = uploaded_files

    # equalizing brightness,contrast, sharpness,and/or color
    processed_images = align_all_selected_images_to_template(
        base_image=base_image,
        input_images=other_images,
        selected_option=selected_option,
        motion_model=motion_model,
        preprocess_options=preprocess_options,
    )

    if not processed_images:
        st.stop()

    fcol2.write('Aligned image:')

    base_image = Image.open(st.session_state['disp_img'])

    image_list = {name: image for name, image, _ in processed_images}
    show_images_widget(image_list)

    for filename, aligned_image, angle in processed_images:
        aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        image_comparison(img1=base_image, img2=aligned_image_rgb)

    image_downloads_widget(images=image_list)


# run main function
if __name__ == '__main__':
    main()
