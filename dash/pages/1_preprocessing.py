from __future__ import annotations

from io import BytesIO
from typing import Literal
import cv2
import imageio
import numpy as np
import rembg
import streamlit as st
from PIL import Image
from styling import set_custom_css


def apply_mask_and_convert_to_pil(original_image, mask):
    """Apply mask to original image and convert it to PIL format."""
    # Apply mask to original image
    extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    # Convert extracted image to PIL format
    return Image.fromarray(extracted_image)


def process_image(
    input_image: Image,
    *,
    bilateral_strength: int,
    clahe_clip_limit: float,
    clahe_tiles: int,
    sigma_sharpness: float,
    contrast: float,
    brightness: int,
    sharpening_kernel_size: int,
    saturation_factor: float,
) -> np.ndarray:
    """Process the uploaded image with user-defined parameters."""
    # Convert PIL Image to NumPy array
    input_image_np = np.array(input_image)

    # Check if the image is grayscale and convert it to 3 channels if necessary
    if len(input_image_np.shape) == 2:
        input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2BGR)

    # Noise Reduction
    # input_image_np = cv2.GaussianBlur(input_image_np, (5, 5), 0)

    # Split PIL image into its individual color channels
    blue, green, red = cv2.split(input_image_np)

    # Apply bilateral blur filter to each color channel with user-defined 'bilateral_strength'
    blue_blur = cv2.bilateralFilter(
        blue, d=bilateral_strength, sigmaColor=55, sigmaSpace=55
    )
    green_blur = cv2.bilateralFilter(
        green, d=bilateral_strength, sigmaColor=55, sigmaSpace=55
    )
    red_blur = cv2.bilateralFilter(
        red, d=bilateral_strength, sigmaColor=55, sigmaSpace=55
    )

    # Create CLAHE object with user-defined clip limit
    # clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(3, 3))
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=(clahe_tiles, clahe_tiles)
    )

    # Adjust histogram and contrast for each color channel using CLAHE
    blue_eq = clahe.apply(blue_blur)
    green_eq = clahe.apply(green_blur)
    red_eq = clahe.apply(red_blur)

    # Merge the color channels back into a single RGB image
    output_img = cv2.merge((blue_eq, green_eq, red_eq))

    # Color saturation: convert image from BGR color space to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)

    # Multiply the saturation channel by user-defined 'saturation_factor'
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 1, 254).astype(
        np.uint8
    )

    # Convert image back to BGR color space
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Create user-defined 'sharpening_kernel_size'
    kernel = np.ones((sharpening_kernel_size, sharpening_kernel_size), np.float32) * -1
    kernel[sharpening_kernel_size // 2, sharpening_kernel_size // 2] = (
        sharpening_kernel_size**2
    )

    # Apply sharpening kernel to image using filter2D
    processed_image = cv2.filter2D(result_image, -1, kernel)

    # Alpha controls contrast and beta controls brightness
    processed_image = cv2.convertScaleAbs(
        processed_image, alpha=contrast, beta=brightness
    )

    # Additional sharpening: Create the sharpening kernel and apply it to the image
    custom_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Sharpen the image
    processed_image = cv2.filter2D(processed_image, -1, custom_kernel)

    # Apply Gaussian blur to the image with user-defined 'sigma_sharpness'
    processed_image = cv2.GaussianBlur(processed_image, (0, 0), sigma_sharpness)

    return processed_image


def remove_background(
    processed_image: np.ndarray,
    *,
    bg_threshold: int,
    fg_threshold: int,
    erode_size: int,
    alpha_matting: bool,
    only_mask: bool,
    post_process: bool,
    background_color: Literal['Transparent', 'White', 'Black'],
    mask_process: bool = False,
) -> Image:
    bgcolor = {
        'Transparent': (0, 0, 0, 0),
        'White': (255, 255, 255, 255),
        'Black': (0, 0, 0, 255),
    }[background_color]

    if mask_process:
        only_mask = True
        post_process_mask = True

    processed_image = rembg.remove(
        processed_image,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_threshold,
        alpha_matting_background_threshold=bg_threshold,
        alpha_matting_erode_size=erode_size,
        only_mask=only_mask,
        post_process_mask=post_process_mask,
        bgcolor=bgcolor,
    )

    return Image.fromarray(processed_image)


def main():
    set_custom_css()

    st.sidebar.title('Image preprocessing')

    uploaded_file = st.sidebar.file_uploader('Upload Image ', type=['JPG', 'JPEG'])

    if not uploaded_file:
        st.info('Upload file to continue...')
        st.stop()

    st.subheader('Apply Image Filters')
    with st.expander('Click to expand...', expanded=False):
        st.write(
            'The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to'
            'refine the adjustment. When you are happy with the result, download the processed image.'
        )

        col1, col2 = st.columns(2)

        image_params = {}

        image_params['bilateral_strength'] = col1.slider(
            'Bilateral Filter Strength (preset = 5)',
            min_value=0,
            max_value=15,
            value=5,
            key='bilateral',
        )
        image_params['saturation_factor'] = col1.slider(
            'Color Saturation (preset = 1.1)',
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=1.1,
            key='saturation',
        )
        image_params['clahe_clip_limit'] = col1.slider(
            'CLAHE Clip Limit - Threshold for contrast limiting (preset = 2)',
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.05,
            key='clahe',
        )
        image_params['clahe_tiles'] = col1.slider(
            'CLAHE Tile Grid Size - Tile size for local contrast enhancement (preset = 8,8)',
            min_value=2,
            max_value=15,
            value=8,
            step=1,
            key='tiles',
        )

        image_params['sigma_sharpness'] = col2.slider(
            'Sharpness Sigma (preset = 0.5)',
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
            key='sharpness',
        )
        image_params['contrast'] = col2.slider(
            'Contrast (preset = 1.0)',
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key='contrast',
        )
        image_params['brightness'] = col2.slider(
            'Brightness (preset = 10)',
            min_value=-100,
            max_value=100,
            value=10,
            step=1,
            key='brightness',
        )
        image_params['sharpening_kernel_size'] = col2.slider(
            'Sharpening Kernel Size (preset = 3)',
            min_value=1,
            max_value=9,
            step=2,
            value=3,
            key='sharpen',
        )

    st.subheader('Parameters for background removal')
    with st.expander('Click to expand...', expanded=False):
        st.write('Change these parameters to tune the background removal.')

        col1, col2 = st.columns(2)

        background_params = {}

        background_params['alpha_matting'] = col1.checkbox(
            'Use Alpha matting',
            value=False,
            help='Alpha matting is a post processing step that can be used to improve the quality of the output.',
        )
        background_params['only_mask'] = col1.checkbox('Keep mask only', value=False)
        background_params['post_process_mask'] = col1.checkbox(
            'Postprocess mask', value=False
        )
        background_params['background_color'] = col1.radio(
            'Background Color',
            ['Transparent', 'White', 'Black'],
            help='You can use the post_process_mask argument to post process the mask to get better results.',
        )
        background_params['bg_threshold'] = col2.slider(
            'Background Threshold (default = 10)',
            min_value=0,
            max_value=255,
            value=10,
            key='background',
        )
        background_params['fg_threshold'] = col2.slider(
            'Foreground Threshold (default = 200)',
            min_value=0,
            max_value=255,
            value=200,
            key='foreground',
        )
        background_params['erode_size'] = col2.slider(
            'Erode Size (default = 10)',
            min_value=0,
            max_value=25,
            value=10,
            key='erode',
        )

    image = Image.open(uploaded_file)

    processed_image = process_image(image, **image_params)

    processed_image_pil = remove_background(
        processed_image, **background_params, mask_process=False
    )
    processed_mask_pil = remove_background(
        processed_image, **background_params, mask_process=True
    )

    col1, col2 = st.columns(2)
    col1.image(image, caption='Uploaded Image', use_column_width=True)
    col2.image(processed_image_pil, caption='Processed Image', use_column_width=True)

    png_buffer = BytesIO()
    imageio.imwrite(png_buffer, np.array(processed_image_pil), format='PNG')

    extracted_image_pil = apply_mask_and_convert_to_pil(
        np.array(image), np.array(processed_mask_pil)
    )

    extracted_png_buffer = BytesIO()
    imageio.imwrite(extracted_png_buffer, np.array(extracted_image_pil), format='PNG')

    st.markdown('---')

    st.subheader('Download Processed Image')

    col1, col2 = st.columns(2)

    stem, _ = uploaded_file.name.rsplit('.')
    processed_image_name_png = f'{stem}_processed.png'

    col1.download_button(
        label=f'PNG ({processed_image_name_png})',
        data=png_buffer.getvalue(),
        file_name=processed_image_name_png,
        key='processed_image_download_png',
    )

    extracted_image_name_png = f'{stem}_extracted.png'

    col2.download_button(
        label=f'PNG - Extracted ({extracted_image_name_png})',
        data=extracted_png_buffer.getvalue(),
        file_name=extracted_image_name_png,
        key='extracted_image_download_png',
    )


# run main function
if __name__ == '__main__':
    main()
