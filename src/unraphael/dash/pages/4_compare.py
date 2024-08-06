from __future__ import annotations

import imageio.v3 as imageio
import numpy as np
import streamlit as st
from align import align_image_to_base
from equalize import equalize_image_with_base
from streamlit_image_comparison import image_comparison
from styling import set_custom_css
from widgets import load_images_widget, show_images_widget

from unraphael.types import ImageType

# import streamlit.components.v1 as components
# from matplotlib import pyplot as plt, animation
# import matplotlib.pyplot as plt
# import cv2

_align_image_to_base = st.cache_data(align_image_to_base)
_equalize_image_with_base = st.cache_data(equalize_image_with_base)


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


def display_two(base_image, images):
    col1, col2 = st.columns(2)

    if 'count' not in st.session_state:
        st.session_state.count = 0

    if 'images' not in st.session_state:
        st.session_state.images = images  # Directly assign the list

    def display_image():
        try:
            image = st.session_state.images[st.session_state.count].data
            col1.image(base_image, caption='Base Image', use_column_width=True)
            col2.image(
                image, caption=f'Image {st.session_state.count + 1}', use_column_width=True
            )
        except IndexError as e:
            st.error(f'Error displaying image: {e}')

    def next_image():
        if st.session_state.count + 1 >= len(st.session_state.images):
            st.session_state.count = 0
        else:
            st.session_state.count += 1

    def previous_image():
        if st.session_state.count > 0:
            st.session_state.count -= 1

    if col1.button('⏮️ Previous', on_click=previous_image):
        pass

    if col2.button('Next ⏭️', on_click=next_image):
        pass

    with col2:
        display_image()


def alignment_help_widget():
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


# def animate_images(img_aligned, baseline, num_frames=100):
#     fig, ax = plt.subplots()
#     blended_image = cv2.addWeighted(img_aligned, 1, baseline, 0, 0)
#     im = ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB),
#       extent=[0, blended_image.shape[1], 0, blended_image.shape[0]])

#     def update(frame):
#         alpha = frame / num_frames
#         blended_image = cv2.addWeighted(img_aligned, 1 - alpha, baseline, alpha, 0)
#         im.set_array(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
#         ax.set_title(f'Frame {frame + 1}/{num_frames}')

#     ani = animation.FuncAnimation(fig, update, frames=num_frames,
#           interval=25, repeat=True, repeat_delay=1000)
#     return ani.to_jshtml()


def comparison_widget(
    base_image: ImageType,
    images: list[ImageType],
):
    """Widget to compare processed images."""
    st.subheader('Comparison')

    col1, col2 = st.columns((0.20, 0.80))

    if 'count_comp' not in st.session_state:
        st.session_state.count_comp = 0

    def display_image():
        try:
            image = images[st.session_state.count_comp]
            for key, value in image.metrics.items():
                col1.metric(key, f'{value:.2f}')

            col1.download_button(
                label='Download left',
                data=imageio.imwrite('<bytes>', base_image.data, extension='.png'),
                file_name=base_image.name + '.png',
                key=base_image.name,
            )

            col1.download_button(
                label='Download right',
                data=imageio.imwrite('<bytes>', image.data, extension='.png'),
                file_name=image.name + '.png',
                key=image.name,
            )

            image_comparison(
                img1=base_image.data,
                label1=base_image.name,
                img2=image.data,
                label2=image.name,
                width=450,
            )

            # animate_images(
            #     image.data,
            #     base_image.data,
            # )

        except IndexError as e:
            st.error(f'Error displaying image: {e}')

    def next_image():
        if st.session_state.count_comp + 1 >= len(images):
            st.session_state.count_comp = 0
        else:
            st.session_state.count_comp += 1

    def previous_image():
        if st.session_state.count_comp > 0:
            st.session_state.count_comp -= 1

    if col1.button('⏮️ Previous', on_click=previous_image):
        pass

    if col2.button('Next ⏭️', on_click=next_image):
        pass

    with col2:
        display_image()


def main():
    set_custom_css()

    st.title('Image alignment')
    st.write('For a selected image, normalize and align all other images')

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.stop()

    st.subheader('Select base image')
    base_image = show_images_widget(images, message='Select base image for alignment')

    if not base_image:
        st.stop()

    images = [image for image in images if image.name != base_image.name]

    col1, col2 = st.columns(2)

    with col1:
        images = equalize_images_widget(base_image=base_image, images=images)

    with col2:
        images = align_images_widget(base_image=base_image, images=images)

    # Update session state with the aligned images
    st.session_state.images = images

    with st.expander('Help for parameters for aligning images', expanded=False):
        alignment_help_widget()

    # Add a radio button to select the widget to display
    option = st.radio('Select Display Option', ('Compare with slider', 'Alongside each other'))

    if option == 'Compare with slider':
        comparison_widget(base_image=base_image, images=images)
    else:
        display_two(base_image=base_image.data, images=st.session_state.images)


if __name__ == '__main__':
    main()
