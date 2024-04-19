## USE streamlit run preprocessing_app.py --server.enableXsrfProtection false

# Import libraries
import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import imageio
import os
import cv2
import rembg

st.set_page_config(layout="wide", page_title = "Image Background Remover")

st.sidebar.write("## Upload and download :gear:")

## Preset: Change colors of all slider elements using CSS custom styles
# Set background of min/max values transparent
color_min_max = st.markdown('''
<style>
div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%);
}
</style>
''', unsafe_allow_html=True)

# Set color for cursor and remove any residual green
slider_cursor = st.markdown('''
<style>
div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
    background-color: orange;
    border: 1px solid orange; /* Remove any residual green border */
    box-shadow: none; /* Remove shadow */
}
</style>
''', unsafe_allow_html=True)

# Set color for slider number
slider_number = st.markdown('''
<style>
div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
    color: orange;
}
</style>
''', unsafe_allow_html=True)

# Set color shading for slider
color_shading = f'''
<style>
div.stSlider > div[data-baseweb="slider"] > div > div {{
    background: linear-gradient(to left, orange 0%, 
                                rgba(255, 165, 0, 0.25) 50%, 
                                orange 100%);
}}
</style>
'''
# Apply color shading on slider
color_slider = st.markdown(color_shading, unsafe_allow_html=True)


## 1. Function to process the uploaded image with user-defined parameters
def process_image(input_image, bilateral_strength, clahe_clip_limit, clahe_tiles, 
                  sigma_sharpness, contrast, brightness,
                  sharpening_kernel_size, saturation_factor,bg_threshold, fg_threshold,
                  erode_size, alpha_matting, mask, post_process, background_color):
    
    # Convert PIL Image to NumPy array
    input_image_np = np.array(input_image)
    
    # Check if the image is grayscale and convert it to 3 channels if necessary
    if len(input_image_np.shape) == 2:
        input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2BGR)      
        
    # Split PIL image into its individual color channels
    blue, green, red = cv2.split(input_image_np)
  
    # Apply bilateral blur filter to each color channel with user-defined 'bilateral_strength'
    blue_blur = cv2.bilateralFilter(blue, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)
    green_blur = cv2.bilateralFilter(green, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)
    red_blur = cv2.bilateralFilter(red, d=bilateral_strength, sigmaColor=55, sigmaSpace=55)

    # Create CLAHE object with user-defined clip limit
    #clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(3, 3))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tiles, clahe_tiles))
  
    # Adjust histogram and contrast for each color channel using CLAHE
    blue_eq = clahe.apply(blue_blur)
    green_eq = clahe.apply(green_blur)
    red_eq = clahe.apply(red_blur)

    # Merge the color channels back into a single RGB image
    output_img = cv2.merge((blue_eq, green_eq, red_eq))

    # Color saturation: convert image from BGR color space to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)

    # Multiply the saturation channel by user-defined 'saturation_factor'
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 1, 254).astype(np.uint8)

    # Convert image back to BGR color space
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Create user-defined 'sharpening_kernel_size'
    kernel = np.ones((sharpening_kernel_size, sharpening_kernel_size), np.float32) * -1
    kernel[sharpening_kernel_size//2, sharpening_kernel_size//2] = sharpening_kernel_size**2

    # Apply sharpening kernel to image using filter2D
    processed_image = cv2.filter2D(result_image, -1, kernel)
    
    # --------- added -------
    # contrast and brightness adjustment
    # alpha controls contrast and beta controls brightness
    processed_image = cv2.convertScaleAbs(processed_image, alpha= contrast, beta=brightness)
    
    # Additional sharpening    
    # Create the sharpening kernel and apply it to the image
    custom_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    # Sharpen the image
    processed_image = cv2.filter2D(processed_image, -1, custom_kernel)
    
    # Apply Gaussian blur to the image with user-defined 'sigma_sharpness'
    processed_image = cv2.GaussianBlur(processed_image, (0, 0), sigma_sharpness)
    
    # Finally, remove background
    processed_image = rembg.remove(processed_image,
                                   alpha_matting = alpha_matting,
                                   alpha_matting_foreground_threshold=fg_threshold,
                                   alpha_matting_background_threshold=bg_threshold,
                                   alpha_matting_erode_size = erode_size,
                                   only_mask = mask,
                                   post_process_mask = post_process,
                                   bgcolor = (0, 0, 0, 0) if background_color == "Transparent" else \
                                             (255, 255, 255, 255) if background_color == "White" else \
                                             (0, 0, 0, 255)
                                )
    # --------- added
    return Image.fromarray(processed_image)

## 2. Main function to run the streamlit app
def main():
    # Set title for the App
    #st.markdown("# :orange[Image preprocessing]")
    st.markdown('<div style="text-align: center;"><h1 style="color: orange;">Image preprocessing</h1></div>', unsafe_allow_html=True)
    #st.markdown("## For optimal background removal from an image")   
    st.markdown('<div style="text-align: center;"><h2 style="font-size: 26px;">For optimal background removal from an image</h2></div>', unsafe_allow_html=True)
        
    # Draw a dividing line
    #st.divider()

    # Step 1: upload image file as jpg/jpeg, include label
    uploaded_file = st.sidebar.file_uploader(" #### :camera: :orange[1. Upload Image] ", type=["JPG", "JPEG"])

    if uploaded_file is not None:        
        
        # Step 2: Slider customization with live updates
        
        # First block with sliders
        with st.expander("",expanded=True):
            # Set subtitle and short explanation
            st.write("#### :orange[2a. Apply Image Filters]")
            st.write("The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to \
                    refine the adjustment. - When you are happy with the result, download the processed image.")
                        
            # create 2 columns to distribute sliders
            col1, col2 = st.columns(2)

            # Column 1: create sliders for Bilateral Filter and Saturation
            bilateral_strength = col1.slider("###### :heavy_check_mark: :blue[Bilateral Filter Strength] (preset = 5)", min_value=0, max_value=15, value=5, key='bilateral')
            saturation_factor =  col1.slider("###### :heavy_check_mark: :blue[Color Saturation] (preset = 1.1)", min_value=0.0, max_value=2.0, step=0.05, value=1.1, key='saturation')
            clahe_clip_limit =   col1.slider("###### :heavy_check_mark: :blue[CLAHE Clip Limit - Threshold for contrast limiting] (preset = 2)", min_value=0.0, max_value=5.0, value=2.0, step=0.05, key='clahe')
            clahe_tiles =        col1.slider("###### :heavy_check_mark: :blue[CLAHE Tile Grid Size - Tile size for local contrast enhancement] (preset = 8,8)", min_value=2, max_value=15, value=8, step= 1, key='tiles')            
                        
            # Column 2: create sliders for CLAHE and Sharpening            
            sigma_sharpness = col2.slider("###### :heavy_check_mark: :blue[Sharpness Sigma] (preset = 0.5)", min_value=0.0, max_value=3.0, value=0.5, step=0.1, key='sharpness')
            contrast = col2.slider("###### :heavy_check_mark: :blue[Contrast] (preset = 1.0)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='contrast')
            brightness = col2.slider("###### :heavy_check_mark: :blue[Brightness] (preset = 10)", min_value=-100, max_value=100, value=10, step=1, key='brightness')
            sharpening_kernel_size = col2.slider("###### :heavy_check_mark: :blue[Sharpening Kernel Size] (preset = 3)", min_value=1, max_value=9, step=2, value=3, key='sharpen')
                    
            
        # Second block with image filters and explanation
        with st.expander("",expanded=True):
        
            # Set subtitle and short explanation
            st.write("#### :orange[2b. Parameters for background removal]")
            st.write("Specifics to background removal.")
        
            # create 2 columns for second block
            col3, col4 = st.columns(2)
        
            # Draw a dividing line
            #st.divider()
            
            # mask
            alpha_matting = col3.checkbox("Use Alpha matting", value = False) # Alpha matting is a post processing step that can be used to improve the quality of the output.
            mask = col3.checkbox("Keep mask only", value = False)
            post_process = col3.checkbox("Postprocess mask", value = False)   # You can use the post_process_mask argument to post process the mask to get better results.       
            background_color = col3.radio("Background Color", ["Transparent", "White", "Black"])

            
            bg_threshold = col4.slider('###### :heavy_check_mark: :blue[Background Threshold] (default = 10)', min_value= 0, max_value = 255, value = 10, key = 'background') # [default: 10]
            fg_threshold = col4.slider('###### :heavy_check_mark: :blue[Foreground Threshold] (default = 200)', min_value = 0,max_value = 255, value = 200, key = 'foreground') # [default: 240]
            erode_size   = col4.slider('###### :heavy_check_mark: :blue[Erode Size] (default = 10)', min_value = 0,max_value = 25, value = 10, key = 'erode') # [default: 10]
                        
        
        # Step 3: Live Preview of Image Processing
        if uploaded_file is not None:
            
            col1, col2 = st.columns(2)
            # Read uploaded image
            image = Image.open(uploaded_file)
            
            # Display uploaded image with label
            col1.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image with updated parameters
            processed_image_pil = process_image(image, bilateral_strength, clahe_clip_limit,clahe_tiles,
                                                sigma_sharpness,contrast, brightness ,sharpening_kernel_size, saturation_factor,
                                                bg_threshold, fg_threshold, erode_size, alpha_matting, mask, post_process, background_color)

            # Display resulting image dynamically
            col2.image(processed_image_pil, caption="Processed Image", use_column_width=True)

            # Get filename and extension using os.path
            original_name, original_extension = os.path.splitext(uploaded_file.name)

            # Construct file names for processed images
            processed_image_name_jpeg = f"{original_name}_processed.jpg"
            processed_image_name_png = f"{original_name}_processed.png"
            
            # Convert processed image to RGB (removing alpha channel)
            processed_image_rgb = processed_image_pil.convert("RGB")

            # Save processed image in JPEG format in-memory
            jpeg_buffer = BytesIO()
            processed_image_rgb.save(jpeg_buffer, format="JPEG")
            jpeg_data = jpeg_buffer.getvalue()

            # Save processed image in PNG format in-memory using imageio
            png_buffer = BytesIO()
            imageio.imwrite(png_buffer, np.array(processed_image_pil), format='PNG')
            png_data = png_buffer.getvalue()
            
            # Draw a dividing line
            st.divider()
            
            # Step 4: Download Buttons
            # write subtitle and download information
            st.sidebar.markdown("\n")
            st.sidebar.markdown(f"#### :floppy_disk: :orange[3. Download Processed Image] ")
            st.sidebar.write('Download image in compressed JPG or losless PNG file format:')
            
            # Provide download buttons for both formats
            # create 2 columns to distribute download buttons
            #col1, col2 = st.columns(2)
            # place download button for jpeg file in column 1
            #with col1:
            st.sidebar.download_button(
                    label=f":orange[JPG] ({processed_image_name_jpeg})",
                    data=jpeg_data,
                    file_name=processed_image_name_jpeg,
                    key="processed_image_download_jpeg",
            )            
                        
            # place download button for png file in column 2
            #with col2:
            st.sidebar.download_button(
                    label=f":orange[PNG] ({processed_image_name_png})",
                    data=png_data,
                    file_name=processed_image_name_png,
                    key="processed_image_download_png",
            )

            # Download message
            #st.caption("Preparing Download... Please wait.")
  
# run main function
if __name__ == "__main__":
    main()
