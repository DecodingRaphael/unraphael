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

    return {
        name: _equalize_image_with_base(base_image, image, **preprocess_options)
        for name, image in images.items()
    }


def align_images_widget(
    *, base_image: ImageType, images: dict[str, ImageType]
) -> dict[str, ImageType]:
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

    res = {}

    progress = st.progress(0, text='Aligning...')

    for i, (name, image_d) in enumerate(images.items()):
        progress.progress(i / len(images), f'Aligning {name}...')
        res[name] = _align_image_to_base(
            base_image=base_image,
            image_d=image_d,
            align_method=align_method,
            motion_model=motion_model,
        )

    return res


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


def comparison_widget(
    base_image: ImageType,
    images: dict[str, ImageType],
):
    """Widget to compare processed images."""
    st.subheader('Comparison')

    col1, col2 = st.columns((0.3, 0.7))

    image_name = col1.selectbox('Pick image', options=tuple(images.keys()))
    image_d = images[image_name]

    with col1:
        for key, value in image_d.metrics.items():
            st.metric(key, f'{value:.2f}')

        st.download_button(
            label='Download left',
            data=imageio.imwrite('<bytes>', base_image.data, extension='.png'),
            file_name=image_d.name + '.png',
            key=image_d.name,
        )

        st.download_button(
            label='Download right',
            data=imageio.imwrite('<bytes>', image_d.data, extension='.png'),
            file_name=base_image.name + '.png',
            key=base_image.name,
        )

    with col2:
        image_comparison(
            img1=base_image.data,
            label1=base_image.name,
            img2=image_d.data,
            label2=image_d.name,
            width=450,
        )


def main():
    set_custom_css()

    st.title('Image alignment')
    st.write('For a selected image, normalize and align all other images')

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.stop()

    st.subheader('Select base image')
    base_name = show_images_widget(images, message='Select base image for alignment')

    if not base_name:
        st.stop()

    images = {name: ImageType(data=data, name=name) for name, data in images.items()}

    base_image = images[base_name]
    images = {name: image for name, image in images.items() if name != base_name}

    col1, col2 = st.columns(2)

    with col1:
        images = equalize_images_widget(base_image=base_image, images=images)

    with col2:
        images = align_images_widget(base_image=base_image, images=images)

    with st.expander('Help for parameters for aligning images', expanded=False):
        alignment_help_widget()

    comparison_widget(
        base_image=base_image,
        images=images,
    )


if __name__ == '__main__':
    main()
