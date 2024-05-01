## USE streamlit run preprocessing_app.py --server.enableXsrfProtection false

# Import libraries
from __future__ import annotations

import os
from io import BytesIO

import cv2
import imageio
import numpy as np
import rembg
import streamlit as st
from PIL import Image

st.set_page_config(layout='wide', page_title='Image Background Remover')


# Apply mask to original image and convert it to PIL format
def apply_mask_and_convert_to_pil(original_image, mask):
    # Apply mask to original image
    extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    # Convert extracted image to PIL format
    return Image.fromarray(extracted_image)


# Process the uploaded image with user-defined parameters
def process_image(
    input_image,
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


# Run the streamlit app
def main():
    st.markdown(
        '<div style="text-align: center;"><h1 style="color: orange;">Image preprocessing</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align: center;"><h2 style="font-size: 26px;">For optimal background removal from '
        'an image</h2></div>',
        unsafe_allow_html=True,
    )

    # Draw a dividing line
    st.markdown('---')

    # Step 1: upload image file as jpg/jpeg, include label
    uploaded_file = st.sidebar.file_uploader(
        ' #### :camera: :orange[1. Upload Image] ', type=['JPG', 'JPEG']
    )

    if uploaded_file is not None:
        # Step 2: Slider customization with live updates
        with st.expander('', expanded=True):
            # Set subtitle and short explanation
            st.write('#### :orange[2a. Apply Image Filters]')
            st.write(
                'The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to \
                    refine the adjustment. - When you are happy with the result, download the processed image.'
            )

            # create 2 columns to distribute sliders
            col1, col2 = st.columns(2)

            # Column 1
            bilateral_strength = col1.slider(
                '###### :heavy_check_mark: :blue[Bilateral Filter Strength] (preset = 5)',
                min_value=0,
                max_value=15,
                value=5,
                key='bilateral',
            )
            saturation_factor = col1.slider(
                '###### :heavy_check_mark: :blue[Color Saturation] (preset = 1.1)',
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                value=1.1,
                key='saturation',
            )
            clahe_clip_limit = col1.slider(
                '###### :heavy_check_mark: :blue[CLAHE Clip Limit - Threshold for contrast limiting] (preset = 2)',
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.05,
                key='clahe',
            )
            clahe_tiles = col1.slider(
                '###### :heavy_check_mark: :blue[CLAHE Tile Grid Size - Tile size for local contrast enhancement] (preset = 8,8)',
                min_value=2,
                max_value=15,
                value=8,
                step=1,
                key='tiles',
            )

            # Column 2
            sigma_sharpness = col2.slider(
                '###### :heavy_check_mark: :blue[Sharpness Sigma] (preset = 0.5)',
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                key='sharpness',
            )
            contrast = col2.slider(
                '###### :heavy_check_mark: :blue[Contrast] (preset = 1.0)',
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key='contrast',
            )
            brightness = col2.slider(
                '###### :heavy_check_mark: :blue[Brightness] (preset = 10)',
                min_value=-100,
                max_value=100,
                value=10,
                step=1,
                key='brightness',
            )
            sharpening_kernel_size = col2.slider(
                '###### :heavy_check_mark: :blue[Sharpening Kernel Size] (preset = 3)',
                min_value=1,
                max_value=9,
                step=2,
                value=3,
                key='sharpen',
            )

        # Second block with image filters and explanation
        with st.expander('', expanded=True):
            # Set subtitle and short explanation
            st.write('#### :orange[2b. Parameters for background removal]')
            st.write('Specifics to background removal.')

            # create 2 columns for second block
            col3, col4 = st.columns(2)

            # Draw a dividing line
            st.markdown('---')

            # mask
            alpha_matting = col3.checkbox(
                'Use Alpha matting', value=False
            )  # Alpha matting is a post processing step that can be used to improve the quality of the output.
            mask = col3.checkbox('Keep mask only', value=False)
            post_process = col3.checkbox(
                'Postprocess mask', value=False
            )  # You can use the post_process_mask argument to post process the mask to get better results.
            background_color = col3.radio(
                'Background Color', ['Transparent', 'White', 'Black']
            )

            bg_threshold = col4.slider(
                '###### :heavy_check_mark: :blue[Background Threshold] (default = 10)',
                min_value=0,
                max_value=255,
                value=10,
                key='background',
            )  # [default: 10]
            fg_threshold = col4.slider(
                '###### :heavy_check_mark: :blue[Foreground Threshold] (default = 200)',
                min_value=0,
                max_value=255,
                value=200,
                key='foreground',
            )  # [default: 240]
            erode_size = col4.slider(
                '###### :heavy_check_mark: :blue[Erode Size] (default = 10)',
                min_value=0,
                max_value=25,
                value=10,
                key='erode',
            )  # [default: 10]

        # Step 3: Live preview of image processing
        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            image = Image.open(uploaded_file)

            # Display uploaded image with label
            col1.image(image, caption='Uploaded Image', use_column_width=True)

            # Process image with updated parameters
            processed_image_pil = process_image(
                image,
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
            )

            # Process mask with updated parameters
            processed_mask_pil = process_image(
                image,
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
                mask_process=True,
            )

            # Display resulting image dynamically
            col2.image(
                processed_image_pil, caption='Processed Image', use_column_width=True
            )

            # Get filename and extension using os.path
            original_name, original_extension = os.path.splitext(uploaded_file.name)

            # Construct file names for processed images
            processed_image_name_jpeg = f'{original_name}_processed.jpg'
            processed_image_name_png = f'{original_name}_processed.png'
            extracted_image_name_jpeg = f'{original_name}_extracted.jpg'
            extracted_image_name_png = f'{original_name}_extracted.png'

            # Save processed image in JPEG format
            jpeg_buffer = BytesIO()
            processed_image_pil.convert('RGB').save(jpeg_buffer, format='JPEG')
            jpeg_data = jpeg_buffer.getvalue()

            # Save processed image in PNG format
            png_buffer = BytesIO()
            imageio.imwrite(png_buffer, np.array(processed_image_pil), format='PNG')
            png_data = png_buffer.getvalue()

            # Apply mask to original image
            extracted_image_pil = apply_mask_and_convert_to_pil(
                np.array(image), np.array(processed_mask_pil)
            )

            # Save extracted image in JPEG format
            extracted_jpeg_buffer = BytesIO()
            extracted_image_pil.save(extracted_jpeg_buffer, format='JPEG')
            extracted_jpeg_data = extracted_jpeg_buffer.getvalue()

            # Save extracted image in PNG format
            extracted_png_buffer = BytesIO()
            imageio.imwrite(
                extracted_png_buffer, np.array(extracted_image_pil), format='PNG'
            )
            extracted_png_data = extracted_png_buffer.getvalue()

            st.markdown('---')

            # Step 4: Download Buttons
            # write subtitle and download information
            st.sidebar.markdown('\n')
            st.sidebar.markdown(
                '#### :floppy_disk: :orange[3. Download Processed Image] '
            )
            st.sidebar.write(
                'Download image in compressed JPG or losless PNG file format:'
            )

            # Provide download buttons for both formats
            col1, col2 = st.columns(2)

            # Place download button for jpeg file in column 1
            with col1:
                st.sidebar.download_button(
                    label=f':orange[JPG] ({processed_image_name_jpeg})',
                    data=jpeg_data,
                    file_name=processed_image_name_jpeg,
                    key='processed_image_download_jpeg',
                )
                st.sidebar.download_button(
                    label=f':orange[JPG - Extracted] ({extracted_image_name_jpeg})',
                    data=extracted_jpeg_data,
                    file_name=extracted_image_name_jpeg,
                    key='extracted_image_download_jpeg',
                )

            # Place download button for png file in column 2
            with col2:
                st.sidebar.download_button(
                    label=f':orange[PNG] ({processed_image_name_png})',
                    data=png_data,
                    file_name=processed_image_name_png,
                    key='processed_image_download_png',
                )
                st.sidebar.download_button(
                    label=f':orange[PNG - Extracted] ({extracted_image_name_png})',
                    data=extracted_png_data,
                    file_name=extracted_image_name_png,
                    key='extracted_image_download_png',
                )


# run main function
if __name__ == '__main__':
    main()
