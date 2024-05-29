from __future__ import annotations

import numpy as np
import streamlit as st
from styling import set_custom_css
from widgets import image_downloads_widget, load_image_widget

from unraphael.preprocess import apply_mask, process_image, remove_background

_process_image = st.cache_data(process_image)
_apply_mask = st.cache_data(process_image)
_remove_background = st.cache_data(remove_background)


def preprocess_image_widget(image: np.ndarray):
    """Widget to preprocess image with user input options."""
    st.title('Preprocessing')
    st.write(
        'The processed image is shown with a preset of parameters. '
        'Use the sliders to explore the effects of image filters, or to'
        'refine the adjustment. When you are happy with the result, '
        'download the processed image.'
    )

    col1, col2, col3, col4 = st.columns(4)

    image_params = {}

    image_params['bilateral_strength'] = col1.slider(
        'Bilateral Filter Strength',
        min_value=0,
        max_value=15,
        value=5,
        key='bilateral',
        help='(default = 5)',
    )
    image_params['saturation_factor'] = col1.slider(
        'Color Saturation',
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        value=1.1,
        key='saturation',
        help='(default = 1.1)',
    )
    image_params['clahe_clip_limit'] = col1.slider(
        'CLAHE Clip Limit',
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.05,
        key='clahe',
        help='Threshold for contrast limiting (default = 2)',
    )
    image_params['clahe_tiles'] = col1.slider(
        'CLAHE Tile Grid Size',
        min_value=2,
        max_value=15,
        value=8,
        step=1,
        key='tiles',
        help='Tile size for local contrast enhancement (default = 8)',
    )

    image_params['sigma_sharpness'] = col2.slider(
        'Sharpness Sigma',
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.1,
        key='sharpness',
        help='(default = 0.5)',
    )
    image_params['contrast'] = col2.slider(
        'Contrast',
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key='contrast',
        help='(default = 1.0)',
    )
    image_params['brightness'] = col2.slider(
        'Brightness',
        min_value=-100,
        max_value=100,
        value=10,
        step=1,
        key='brightness',
        help='(default = 10)',
    )
    image_params['sharpening_kernel_size'] = col2.slider(
        'Sharpening kernel size',
        min_value=1,
        max_value=9,
        step=2,
        value=3,
        key='sharpen',
        help='(default = 3)',
    )
    out = process_image(image, **image_params)
    col3.image(image, 'before')
    col4.image(out, 'after')
    return out


def remove_background_widget(image: np.ndarray) -> np.ndarray:
    """This widget takes an image and provides the user with some choices to
    remove the background."""
    st.title('Background removal')
    st.write('Change these parameters to tune the background removal.')

    col1, col2, col3, col4 = st.columns(4)

    background_params = {}

    background_params['alpha_matting'] = col1.checkbox(
        'Use Alpha matting',
        value=False,
        help=(
            'Alpha matting is a post processing step that can be used to '
            'improve the quality of the output.'
        ),
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
        help=(
            'You can use the post_process_mask argument to post process the '
            'mask to get better results.'
        ),
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

    nobg = remove_background(image, **background_params, mask_process=False)

    mask = remove_background(image, **background_params, mask_process=True)

    col3.image(mask, 'mask')
    col4.image(nobg, 'background removed')

    return nobg, mask


def main():
    set_custom_css()

    name, image = load_image_widget()

    processed = preprocess_image_widget(image)
    processed_nobg, processed_mask = remove_background_widget(processed)

    images = {
        'original': image,
        'processed': processed_nobg,
        'extracted': apply_mask(image, processed_mask),
    }

    image_downloads_widget(basename=name, images=images)


if __name__ == '__main__':
    main()
