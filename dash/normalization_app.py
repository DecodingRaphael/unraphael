# run with: streamlit run normalization_app.py --server.enableXsrfProtection false

# Import libraries
import math
import os
import sys
import io
import tempfile
import streamlit as st
from streamlit_image_viewer import image_viewer
from PIL import Image
import numpy as np
import imageio
import cv2
from outline_normalization import align_all_selected_images_to_template

sys.path.append('../')

st.set_page_config(layout="wide", page_title = "")

def image_downloads_widget(*, images: dict[str, np.ndarray]):
    """This widget takes a dict of images and shows them with download buttons."""
    
    st.title("Save Images to Disk")

    cols = st.columns(len(images))

    for col, key in zip(cols, images):
        image = images[key]
                
        filename = f'{key}.png'
        
        img_bytes = io.BytesIO()
        imageio.imwrite(img_bytes, image, format='png')
        img_bytes.seek(0)

        col.download_button(
            label=f'({filename})',
            data=img_bytes,
            file_name=filename,
            mime='image/png',
            key=filename,
        )
      
def main():
    st.markdown('<div style="text-align: center;"><h1 style="color: orange;">Image normalization</h1></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 style="font-size: 26px;">For a selected image, align and/ or normalize all other images</h2></div>', unsafe_allow_html=True)                
    st.markdown("---")
        
    uploaded_files = st.sidebar.file_uploader("#### :orange[1. Select the images to normalize and the base image]",
                                               type=["JPG", "JPEG", "PNG"], accept_multiple_files = True)
    
    # Extract all image names
    names = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            names.append(uploaded_file.name)

    #TODO: use buffer to store the uploaded images instead of saving them to disk
    if uploaded_files:
        for uploaded_file in uploaded_files:
            
            path = uploaded_file.name            
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
                                    
    # Initialize processed_images
    processed_images = []
    
    if uploaded_files and len(names) > 0:
                        
        # First block with options
        with st.expander("", expanded = True):
            st.write("#### :orange[2a. Parameters for equalizing images]")
            st.write("The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to \
                    refine the adjustment. - When you are happy with the result, download the processed image.")
            
            col1, col2 = st.columns(2)
            st.markdown("---")
            
            brightness = col1.checkbox("Equalize brightness", value = False)
            contrast   = col1.checkbox("Equalize contrast", value = False)
            sharpness  = col2.checkbox("Equalize sharpness", value = False)
            color      = col2.checkbox("Equalize colors", value = False)
                        
        preprocess_options = {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'color': color
        }
        
        # Second block with options
        with st.expander("", expanded = True):
            st.write("#### :orange[2b. Parameters for aligning images]")
            st.write("Select the alignment procedure to align the images to the base image.")
                    
            col3, col4 = st.columns(2)
            st.markdown("---")
            
            options = ['ORB feature based alignment',
                       'Enhanced Correlation Coefficient Maximization', 
                       'Fourier Mellin Transform',
                       'FFT phase correlation',
                       'Rotational Alignment',
                       'User-provided keypoints (from pose estimation)']

            # Display the dropdown menu
            selected_option = col3.selectbox('Select an option:', options)
            # Initialize motion_model
            motion_model = None
            
            if selected_option == 'Enhanced Correlation Coefficient Maximization':                
                motion_model = col4.selectbox("Select motion model:", ['translation','euclidian','affine','homography'])                
                                              
        # Alignment procedure
        if uploaded_files and len(names) > 0:
            
            scol1 , scol2 = st.columns(2)
            fcol1 , fcol2 = st.columns(2)
                                   
            ch = scol1.button("Select baseline image to align to")
            fs = scol2.button("Align images to baseline image")
            
            #Set selected image as base image
            if ch:
                filename = uploaded_files[np.random.randint(len(uploaded_files))].name.split('/')[-1]
                fcol1.image(Image.open(filename),use_column_width = True)                
                st.session_state["disp_img"] = filename
                st.write(f"Base Image: {filename}")
            
            # Align images to selected base image
            if fs:
                idx = names.index(st.session_state["disp_img"])
                                
                # Remove selected base image from the list of selected images
                base_image = uploaded_files.pop(idx).name.split('/')[-1]
                                
                # Extract the filenames from the list of selected images which will be aligned
                file_names = [file.name.split('/')[-1] for file in uploaded_files]
                               
                processed_images = align_all_selected_images_to_template(
                             base_image_path    = base_image,         # base image
                             input_files        = file_names,         # images to be aligned
                             selected_option    = selected_option,    # alignment procedure
                             motion_model       = motion_model,       # motion model
                             preprocess_options = preprocess_options) # equalizing brightness,contrast, sharpness,and/o rcolor

                if processed_images:
                    
                    stacked_image_paths = []
                    stacked_image_names = []
                    save_aligned_images = {}

                    fcol1.write("")
                    fcol1.write(f"Base Image: {base_image}")
                    fcol2.write("")
                    fcol2.write("Aligned Image:")
                                        
                    base_image = Image.open(st.session_state["disp_img"])

                    for filename, aligned_image, angle in processed_images:
                        # Convert aligned_image to RGB color mode for optimal display
                        aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(aligned_image_rgb)
                        
                        stacked_image = np.hstack([base_image, pil_image])
                        stacked_image_pil = Image.fromarray(stacked_image)
                        
                        # Save all figures to dictionary
                        save_aligned_images[f'aligned_{filename}'] = aligned_image_rgb
                                                                                            
                        # Create a temporary file for the stacked image (neccesary for image_viewer widget)
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                            stacked_image_pil.save(temp_file.name, format="JPEG")
                            stacked_image_paths.append(temp_file.name)
                            stacked_image_names.append(filename)                        
                                        
                    image_viewer(stacked_image_paths, stacked_image_names, ncol=1, nrow=1, image_name_visible=True)                    
                    
                    # Remove temporary stacked images (for display only)
                    for temp_file in stacked_image_paths:
                        os.remove(temp_file)
                        
                    #TODO: add heatmap with rotation angles

                    image_downloads_widget(images = save_aligned_images)

# run main function
if __name__ == "__main__":
    main()