from __future__ import annotations

import cv2
import imageio.v3 as imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from align import align_image_to_base, homography_matrix
from equalize import equalize_image_with_base
from rembg import remove
from skimage import img_as_ubyte, transform
from streamlit_image_comparison import image_comparison
from styling import set_custom_css
from widgets import load_images_widget, show_images_widget

from unraphael.types import ImageType

_align_image_to_base = st.cache_data(align_image_to_base)
_equalize_image_with_base = st.cache_data(equalize_image_with_base)


# Define the function to extract contours
def extract_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Extract the foreground mask using rembg."""
    return remove(image, mask=True)


def extract_outer_contour_from_mask(mask: np.ndarray, min_area: int = 25) -> list:
    """Extract the outer contour from the mask."""
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray_mask, 1, 255, cv2.THRESH_BINARY
    )  # threshold to get binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours


def overlay_contours(image1, image2, name1, name2, contours1, contours2):
    """Overlay the contours from two images and return the combined image."""
    # Create a black canvas with the same size as the images
    h, w = image1.shape[:2]
    contour_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Define the colors for each contour
    color1 = (255, 0, 0)  # Red
    color2 = (0, 0, 255)  # Blue

    cv2.drawContours(contour_overlay, contours1, -1, color1, thickness=2)
    cv2.drawContours(contour_overlay, contours2, -1, color2, thickness=2)

    return contour_overlay, color1, color2


def warp_image_skimage(img, H, output_shape):
    """Warp the image using the given transformation matrix."""
    warped_image = transform.warp(
        img, inverse_map=H, output_shape=output_shape, mode='constant', cval=0
    )
    return img_as_ubyte(warped_image)


def blend_images_skimage(img1, img2, alpha):
    """Blend two images with given alpha."""
    blended = (1 - alpha) * img1 + alpha * img2
    return blended.astype(np.uint8)


def animate_images(img1, img2, H, num_frames=50):
    """Create an animation of blending two images with a given warp matrix."""
    h, w = img2.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    fig.patch.set_alpha(0)

    ax.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_aspect('equal', adjustable='box')

    warped_image = warp_image_skimage(img1, H, (h, w))
    im = ax.imshow(warped_image, aspect='auto')

    def update(frame):
        alpha = frame / num_frames
        warped_image = warp_image_skimage(img1, H, (h, w))
        blended_image = blend_images_skimage(warped_image, img2, alpha)
        im.set_array(blended_image)
        ax.set_title(f'Frame {frame + 1}/{num_frames}', fontsize=16, color='white', pad=20)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=25, repeat=True, repeat_delay=1000
    )
    return ani


def equalize_images_widget(
    *, base_image: np.ndarray, images: dict[str, np.ndarray]
) -> list[np.ndarray]:
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
    """This widget helps with aligning images.

    Parameters:
    - base_image: The base image to which other images will be aligned
    - images: List of images to be aligned

    Returns:
    - List of aligned images
    """

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


def display_images_widget(
    base_image: np.ndarray,
    images: list[ImageType],
):
    """Generalized widget to display and navigate through images with various
    options.

    Parameters:
    - base_image: The base image for comparison or alignment
    - images: List of images to display or compare
    """
    st.subheader('Image comparison')

    col1, col2, col3 = st.columns((2, 1, 1))

    display_mode = col1.radio(
        'Select Display Option',
        options=('slider', 'side-by-side', 'animation', 'contour comparison'),
        # captions=('Compare with slider', 'Alongside each other'),
        format_func=str.capitalize,
        horizontal=True,
    )

    if 'count' not in st.session_state:
        st.session_state.count = 0

    def next_image():
        st.session_state.count = (st.session_state.count + 1) % len(images)

    def previous_image():
        st.session_state.count = (st.session_state.count - 1) % len(images)

    col2.button('⏮️ Previous', on_click=previous_image, use_container_width=True, type='primary')
    col3.button('⏭️ Next', on_click=next_image, use_container_width=True, type='primary')

    image = images[st.session_state.count]

    if display_mode == 'slider':
        col1, col2 = st.columns((0.20, 0.80))

        for key, value in image.metrics.items():
            col1.metric(key, f'{value:.2f}')

        with col2:
            image_comparison(
                img1=base_image.data,
                label1=base_image.name,
                img2=image.data,
                label2=image.name,
                width=850,
            )

    elif display_mode == 'side-by-side':
        col1, col2 = st.columns(2)
        col1.image(base_image.data, caption=base_image.name, use_column_width=True)
        col2.image(image.data, caption=image.name, use_column_width=True)

    elif display_mode == 'animation':
        homo_matrix = homography_matrix(image=image, base_image=base_image)

        ani = animate_images(base_image.data, image.data, homo_matrix)
        html_animation = ani.to_jshtml()
        st.components.v1.html(html_animation, height=1500)

    elif display_mode == 'contour comparison':
        mask1 = extract_foreground_mask(base_image.data)
        mask2 = extract_foreground_mask(image.data)

        contours1 = extract_outer_contour_from_mask(mask1)
        contours2 = extract_outer_contour_from_mask(mask2)

        contour_overlay, color1, color2 = overlay_contours(
            base_image.data, image.data, base_image.name, image.name, contours1, contours2
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<span style="color: rgb{color1}; font-size: 20px;">{base_image.name}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span style="color: rgb{color2}; font-size: 20px;">{image.name}</span>',
                unsafe_allow_html=True,
            )

        col1.image(contour_overlay, caption='Contour Comparison', use_column_width=True)

    col1, col2 = st.columns(2)

    col1.download_button(
        label=f'Download {base_image.name}.png',
        data=imageio.imwrite('<bytes>', base_image.data, extension='.png'),
        file_name=base_image.name + '.png',
        key=base_image.name,
    )
    col2.download_button(
        label=f'Download {image.name}.png',
        data=imageio.imwrite('<bytes>', image.data, extension='.png'),
        file_name=image.name + '.png',
        key=image.name,
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
    base_image = show_images_widget(images, message='Select base image for alignment')

    if not base_image:
        st.stop()

    images = [image for image in images if image.name != base_image.name]

    col1, col2 = st.columns(2)

    with col1:
        images = equalize_images_widget(base_image=base_image, images=images)

    with col2:
        images = align_images_widget(base_image=base_image, images=images)

    with st.expander('Help for parameters for aligning images', expanded=False):
        alignment_help_widget()

    display_images_widget(base_image=base_image, images=images)


if __name__ == '__main__':
    main()
