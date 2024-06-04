# run with: streamlit run normalization_app.py --server.enableXsrfProtection false

# Import libraries
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
    
    st.title("Save Aligned Images to Disk")

    cols = st.columns(len(images))

    for col, key in zip(cols, images):
        image = images[key]
                
        # Get image size
        height, width = image.shape[:2]
               
        # Remove '.jpg' or '.png' from filename if present
        if '.jpg' in key:
            filename = key.replace('.jpg', '.png')
        elif '.png' in key:
            filename = key.replace('.png', '.png')  # Just to ensure it's .png
        else:
            filename = f'{key}.png'  # If neither '.jpg' nor '.png' is present
        
        img_bytes = io.BytesIO()
        imageio.imwrite(img_bytes, image, format='png')
        img_bytes.seek(0)

        col.download_button(
            label=f' {filename} ({width}x{height})',
            data=img_bytes,
            file_name=filename,
            mime='image/png',
            key=filename,
        )
      

def main():
    st.markdown('<div style="text-align: center;"><h1 style="color: orange;">Image normalization</h1></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 style="font-size: 26px;">For a selected image, normalize and align all other images</h2></div>', unsafe_allow_html=True)
    st.markdown("---")
        
    uploaded_files = st.sidebar.file_uploader("#### :orange[1. Select the images to normalize and the base image]",
                                               type=["JPG", "JPEG", "PNG"], accept_multiple_files = True)
    
    # Extract all image names
    names = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            names.append(uploaded_file.name)

    #TODO: use buffer to instead of using stored images on disk
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
            reinhard    = col2.checkbox("Reinhard color transfer", value = False)
                        
        preprocess_options = {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'color': color,
            'reinhard': reinhard
        }
        
        # Second block with options
        with st.expander("", expanded = True):
            st.write("#### :orange[2b. Parameters for aligning images]")
            st.write("Select the alignment procedure to align the images to the base image.")
                    
            col3, col4 = st.columns(2)
            st.markdown("---")
            
            options = ['Feature based alignment',
                       'Enhanced Correlation Coefficient Maximization', 
                       'Fourier Mellin Transform',
                       'FFT phase correlation',
                       'Rotational Alignment',
                       'User-provided keypoints (from pose estimation)']
            
            #TODO: add more options for alignment procedures
            # https://pypi.org/project/pystackreg/
            # https://github.com/bnsreenu/python_for_microscopists/blob/master/119_sub_pixel_image_registration.py
            # https://github.com/bnsreenu/python_for_microscopists/blob/master/120_img_registration_methods_in_python.py
            # https://github.com/bnsreenu/python_for_microscopists/blob/master/121_image_registration_using_pystackreg.py
            

            # Display the dropdown menu
            selected_option = col3.selectbox('Select an option:', options,                                             
            help="""**Feature based alignment**: Aligns images based on detected features using algorithms like SIFT, SURF, or ORB.
                    **Enhanced Correlation Coefficient Maximization**: Estimates the parameters of a geometric transformation between two images by maximizing the correlation coefficient.
                    **Fourier Mellin Transform**: Uses the Fourier Mellin Transform to align images based on their frequency content.
                    **FFT phase correlation**: Aligns images by computing the phase correlation between their Fourier transforms.
                    **Rotational Alignment**: Aligns images by rotating them to a common orientation.
                    **User-provided keypoints (from pose estimation)**: Aligns images based on user-provided keypoints obtained from pose estimation.""")
            
            # Initialize motion_model
            motion_model = None
            
            if selected_option == 'Feature based alignment':
                motion_model = col4.selectbox("Select algorithm for feature detection and description:", ['SIFT','SURF','ORB'])
                
            if selected_option == 'Enhanced Correlation Coefficient Maximization':
                motion_model = col4.selectbox("Select motion model:", ['translation','euclidian','affine','homography'],
                help         = ". The motion model defines the transformation between the base image and the input images. Translation is the simplest model, while homography is the most complex.")
                
            if selected_option == 'Fourier Mellin Transform':
                motion_model = col4.selectbox("normalization applied in the cross correlation?", ["don't normalize","normalize","phase"],
                help         = """The normalization applied in the cross correlation. If 'don't normalize' is selected, the cross correlation is not normalized. 
                If 'normalize' is selected, the cross correlation is normalized by the product of the magnitudes of the Fourier transforms of the images. 
                If 'phase' is selected, the cross correlation is normalized by the product of the magnitudes and phases of the Fourier transforms of the images.""")               
                
                                              
        # Alignment procedure
        if uploaded_files and len(names) > 0:
            
            scol1 , scol2 = st.columns(2)
            fcol1 , fcol2 = st.columns(2)
                                   
            set_baseline = scol1.button("Select baseline image to align to")
            align_images = scol2.button("Align images to baseline image")
                        
            if set_baseline:
                filename = uploaded_files[np.random.randint(len(uploaded_files))].name.split('/')[-1]
                width, height = Image.open(filename).size
                
                fcol1.write(f"**Base Image:** {filename} ({width}x{height})")
                fcol1.image(Image.open(filename),use_column_width = True)
                st.session_state["disp_img"] = filename                
                        
            if align_images:
                idx = names.index(st.session_state["disp_img"])
                                
                # Remove baseline image from the list of selected images
                base_image = uploaded_files.pop(idx).name.split('/')[-1]
                
                width, height = Image.open(base_image).size
                                
                # Extract the filenames from the list of selected images which will be aligned
                file_names = [file.name.split('/')[-1] for file in uploaded_files]
                               
                processed_images = align_all_selected_images_to_template(
                             base_image_path    = base_image,         # base image
                             input_files        = file_names,         # images to be aligned
                             selected_option    = selected_option,    # alignment procedure
                             motion_model       = motion_model,       # motion model
                             preprocess_options = preprocess_options) # equalizing brightness,contrast, sharpness,and/or color

                if processed_images:
                    
                    stacked_image_paths = []
                    stacked_image_names = []
                    save_aligned_images = {}

                    fcol1.write("")
                    fcol1.write(f"**Base Image:** {base_image} ({width}x{height})")
                    fcol1.text(f"Original width = {width}px and height = {height}px")
                    fcol2.write("")
                    fcol2.write("**Aligned Image:**")
                                                            
                    base_image = Image.open(st.session_state["disp_img"])
                    
                    for filename, aligned_image, angle in processed_images:
                        # Convert aligned_image to RGB color mode for optimal display
                        fcol2.write(f'Difference in rotational degree for {filename}: {angle:.2f}')
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