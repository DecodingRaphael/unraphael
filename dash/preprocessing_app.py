from __future__ import annotations

from io import BytesIO

import cv2
import imageio
import numpy as np
import rembg
import streamlit as st
from PIL import Image

st.set_page_config(layout='wide', page_title='Image Background Remover')


def apply_mask_and_convert_to_pil(original_image, mask):
    """Apply mask to original image and convert it to PIL format."""
    # Apply mask to original image
    extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    # Convert extracted image to PIL format
    return Image.fromarray(extracted_image)


def process_image(
    input_image,
    *,
    bilateral_strength,
    clahe_clip_limit,
    clahe_tiles,
    sigma_sharpness,
    contrast,
    brightness,
    sharpening_kernel_size,
    saturation_factor,
    bg_threshold,
    fg_threshold,
    erode_size,
    alpha_matting,
    mask,
    post_process,
    background_color,
    mask_process=False,
):
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

    # Finally, remove background
    if mask_process:
        processed_image = rembg.remove(
            processed_image,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_size=erode_size,
            only_mask=True,
            post_process_mask=True,
            bgcolor=(0, 0, 0, 0)
            if background_color == 'Transparent'
            else (255, 255, 255, 255)
            if background_color == 'White'
            else (0, 0, 0, 255),
        )
    else:
        processed_image = rembg.remove(
            processed_image,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_size=erode_size,
            only_mask=mask,
            post_process_mask=post_process,
            bgcolor=(0, 0, 0, 0)
            if background_color == 'Transparent'
            else (255, 255, 255, 255)
            if background_color == 'White'
            else (0, 0, 0, 255),
        )

    return Image.fromarray(processed_image)


def main():
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

        bilateral_strength = col1.slider(
            'Bilateral Filter Strength (preset = 5)',
            min_value=0,
            max_value=15,
            value=5,
            key='bilateral',
        )
        saturation_factor = col1.slider(
            'Color Saturation (preset = 1.1)',
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=1.1,
            key='saturation',
        )
        clahe_clip_limit = col1.slider(
            'CLAHE Clip Limit - Threshold for contrast limiting (preset = 2)',
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.05,
            key='clahe',
        )
        clahe_tiles = col1.slider(
            'CLAHE Tile Grid Size - Tile size for local contrast enhancement (preset = 8,8)',
            min_value=2,
            max_value=15,
            value=8,
            step=1,
            key='tiles',
        )

        sigma_sharpness = col2.slider(
            'Sharpness Sigma (preset = 0.5)',
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
            key='sharpness',
        )
        contrast = col2.slider(
            'Contrast (preset = 1.0)',
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key='contrast',
        )
        brightness = col2.slider(
            'Brightness (preset = 10)',
            min_value=-100,
            max_value=100,
            value=10,
            step=1,
            key='brightness',
        )
        sharpening_kernel_size = col2.slider(
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

        alpha_matting = col1.checkbox('Use Alpha matting', value=False)
        # Alpha matting is a post processing step that can be used to improve the quality of the output.
        mask = col1.checkbox('Keep mask only', value=False)
        post_process = col1.checkbox('Postprocess mask', value=False)
        # You can use the post_process_mask argument to post process the mask to get better results.
        background_color = col1.radio(
            'Background Color', ['Transparent', 'White', 'Black']
        )

        bg_threshold = col2.slider(
            'Background Threshold (default = 10)',
            min_value=0,
            max_value=255,
            value=10,
            key='background',
        )
        fg_threshold = col2.slider(
            'Foreground Threshold (default = 200)',
            min_value=0,
            max_value=255,
            value=200,
            key='foreground',
        )
        erode_size = col2.slider(
            'Erode Size (default = 10)',
            min_value=0,
            max_value=25,
            value=10,
            key='erode',
        )

    image = Image.open(uploaded_file)

    processed_image_pil = process_image(
        image,
        bilateral_strength=bilateral_strength,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tiles=clahe_tiles,
        sigma_sharpness=sigma_sharpness,
        contrast=contrast,
        brightness=brightness,
        sharpening_kernel_size=sharpening_kernel_size,
        saturation_factor=saturation_factor,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
        erode_size=erode_size,
        alpha_matting=alpha_matting,
        mask=mask,
        post_process=post_process,
        background_color=background_color,
        mask_process=False,
    )

    processed_mask_pil = process_image(
        image,
        bilateral_strength=bilateral_strength,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tiles=clahe_tiles,
        sigma_sharpness=sigma_sharpness,
        contrast=contrast,
        brightness=brightness,
        sharpening_kernel_size=sharpening_kernel_size,
        saturation_factor=saturation_factor,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
        erode_size=erode_size,
        alpha_matting=alpha_matting,
        mask=mask,
        post_process=post_process,
        background_color=background_color,
        mask_process=True,
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
