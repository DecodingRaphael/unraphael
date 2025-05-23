from __future__ import annotations

import streamlit as st
from styling import set_custom_css
from widgets import image_downloads_widget, load_image_widget, load_images_widget

from unraphael.preprocess import apply_mask, process_image, remove_background
from unraphael.types import ImageType

_process_image = st.cache_data(process_image)
_apply_mask = st.cache_data(apply_mask)
_remove_background = st.cache_data(remove_background)


def preprocess_image_widget(image: ImageType) -> ImageType:
    """Widget to preprocess image with user input options."""
    st.title(f'Preprocessing {image.name}')
    st.write('The processed image is shown with a preset of parameters. ' 'Use the sliders to explore the effects of image filters, or to' 'refine the adjustment. When you are happy with the result, ' 'download the processed image.')

    col1, col2, col3, col4 = st.columns(4)

    image_params = {}

    # Generate unique keys using a prefix
    unique_prefix = f'preprocess_{image.name}_'

    image_params['bilateral_strength'] = col1.number_input(
        'Bilateral Filter Strength',
        min_value=0,
        max_value=15,
        value=2,
        key=f'{unique_prefix}bilateral',
        help='(default = 2)',
    )
    image_params['saturation_factor'] = col1.number_input(
        'Color Saturation',
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        value=1.1,
        key=f'{unique_prefix}saturation',
        help='(default = 1.1)',
    )
    image_params['clahe_clip_limit'] = col1.number_input(
        'CLAHE Clip Limit',
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.01,
        key=f'{unique_prefix}clahe',
        help='Threshold for contrast limiting (default = 2)',
    )
    image_params['clahe_tiles'] = col1.number_input(
        'CLAHE Tile Grid Size',
        min_value=1,
        max_value=15,
        value=1,
        step=1,
        key=f'{unique_prefix}tiles',
        help='Tile size for local contrast enhancement (default = 8)',
    )

    image_params['sigma_sharpness'] = col2.number_input(
        'Sharpness Sigma',
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.1,
        key=f'{unique_prefix}sharpness',
        help='(default = 0.5)',
    )
    image_params['gamma'] = col2.number_input(
        'Gamma',
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key=f'{unique_prefix}gamma',
        help='(default = 1)',
    )
    image_params['gain'] = col2.number_input(
        'Gain',
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key=f'{unique_prefix}gain',
        help='(default = 1)',
    )
    image_params['sharpening_radius'] = col2.number_input(
        'Sharpening radius',
        min_value=1,
        max_value=20,
        step=2,
        value=3,
        key=f'{unique_prefix}sharpen',
        help='(default = 3)',
    )
    out = image.apply(_process_image, **image_params)
    col3.image(image.data, 'before')
    col4.image(out.data, 'after')
    return out


def remove_background_widget(image: ImageType) -> tuple[ImageType, ImageType]:
    """This widget takes an image and provides the user with some choices to
    remove the background."""
    st.title('Background removal')
    st.write('Change these parameters to tune the background removal.')

    col1, col2, col3, col4 = st.columns(4)

    background_params = {}

    background_params['alpha_matting'] = col1.checkbox(
        'Use Alpha matting',
        value=False,
        help=('Alpha matting is a post processing step that can be used to ' 'improve the quality of the output.'),
    )
    background_params['only_mask'] = col1.checkbox('Keep mask only', value=False)
    background_params['post_process_mask'] = col1.checkbox('Postprocess mask', value=False)

    bgmap = {
        (0, 0, 0, 0): 'Transparent',
        (255, 255, 255, 255): 'White',
        (0, 0, 0, 255): 'Black',
    }

    background_params['bgcolor'] = col1.radio(
        'Background color',
        bgmap.keys(),
        format_func=lambda x: bgmap[x],
        help=('You can use the post_process_mask argument to post process the ' 'mask to get better results.'),
    )

    background_params['bg_threshold'] = col2.slider(
        'Background threshold',
        min_value=0,
        max_value=255,
        value=10,
        key='background',
        help='(default = 10)',
    )
    background_params['fg_threshold'] = col2.slider(
        'Foreground threshold',
        min_value=0,
        max_value=255,
        value=200,
        key='foreground',
        help='(default = 200)',
    )
    background_params['erode_size'] = col2.slider(
        'Erode size',
        min_value=0,
        max_value=25,
        value=10,
        key='erode',
        help='(default = 10)',
    )

    nobg = image.apply(_remove_background, mask_process=False, **background_params)
    mask = image.apply(_remove_background, mask_process=True, **background_params)

    col3.image(mask.data, 'mask')
    col4.image(nobg.data, 'background removed')

    return nobg, mask


def main():
    set_custom_css()

    upload_mode = st.sidebar.radio(
        'Select upload mode:',
        ('Single Image', 'Multiple Images'),
        index=0,
        help='Choose whether to upload one image or multiple images.',
    )

    if upload_mode == 'Single Image':
        image = load_image_widget()

        processed = preprocess_image_widget(image)
        processed_nobg, processed_mask = remove_background_widget(processed)

        images = [
            image.replace(name='original'),
            processed_nobg.replace(name='processed'),
            ImageType(name='extracted', data=_apply_mask(image.data, processed_mask.data)),
        ]

        image_downloads_widget(images=images, basename=image.name)

    elif upload_mode == 'Multiple Images':
        with st.sidebar:
            images = load_images_widget(as_gray=False, as_ubyte=True)

        if not images:
            st.stop()

        for idx, image in enumerate(images):
            processed = preprocess_image_widget(image)

            processed_nobg = ImageType(
                name=f'{image.name}_nobg',
                data=_remove_background(processed.data, mask_process=False),
            )

            processed_images = [
                image.replace(name='original'),
                processed_nobg,
            ]

            image_downloads_widget(images=processed_images, basename=image.name)


if __name__ == '__main__':
    main()
