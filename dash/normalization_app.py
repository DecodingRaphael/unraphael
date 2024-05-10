# run with: streamlit run normalization_app.py --server.enableXsrfProtection false
# Import libraries
import math
import os
import streamlit as st
from streamlit_image_viewer import image_viewer
from PIL import Image
import numpy as np
import imageio
import cv2
import datetime as dt
import sys
import tempfile
sys.path.append('../')

st.set_page_config(layout="wide", page_title = "Image Background Remover")

# import the function to align images in the directory
#from src.unraphael.modules.comparison_preparation import (align_images, 
#align_all_images_in_folder_to_template, normalize_brightness, normalize_contrast, 
#normalize_sharpness, normalize_colors)

def align_images(image, template, maxFeatures=5000, keepPercent=0.2):
    """
    Aligns an input image with a template image using feature matching and homography transformation.

    Parameters:
        image (numpy.ndarray): The input image to be aligned.
        template (numpy.ndarray): The template image to align the input image with.
        maxFeatures (int, optional): The maximum number of features to detect and extract using ORB. Default is 500.
        keepPercent (float, optional): The percentage of top matches to keep. Default is 0.2.        

    Returns:
        numpy.ndarray: The aligned image.

    """
    # convert both the input image and template to grayscale
    imageGray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # allocate memory for the keypoints (x, y)-coordinates from the top matches
    # we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points.    
    (H, mask) = cv2.findHomography(ptsA, ptsB, method = cv2.RANSAC)
        
    ## derive rotation angle between figures from the homography matrix
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
    print(f'Rotational degree: {theta:.2f}') # rotation angle, in degrees
    #print(theta) 
    
    # apply the homography matrix to align the images, including the rotation
    #(h, w) = template.shape[:2]
    h, w, c = template.shape
    aligned = cv2.warpPerspective(image, H, (w, h),borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))    
    
    return aligned

def align_all_images_in_folder_to_template2(base_image_path, input_files):
   
    # load the base image to which we want to align all the other images
    template = cv2.imread(base_image_path)

    # list to store aligned images
    aligned_images = []

    # loop over all images in the input directory
    for filename in input_files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # image to be aligned            
            image = cv2.imread(filename)
            
            aligned = align_images(image, template)

            # append filename and aligned image to list
            aligned_images.append((filename, aligned))
    return aligned_images

def main():
    st.markdown('<div style="text-align: center;"><h1 style="color: orange;">Image normalization</h1></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 style="font-size: 26px;">For a selected image, align and/ or normalize all other images</h2></div>', unsafe_allow_html=True)                

    st.markdown("---")
    
    # Link to the directory containing the images to normalize, including the base image
    uploaded_files = st.sidebar.file_uploader("#### :camera: :orange[1. Select the images to normalize and the base image]",
                                               type=["JPG", "JPEG", "PNG"], accept_multiple_files = True)    
    
    # Extract the names of all the images in the directory
    names = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Append the filename to the names list
            names.append(uploaded_file.name)           

    if uploaded_files:
        for uploaded_file in uploaded_files:                       
            
            path = uploaded_file.name
            print("Path:", path)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())               
                                
    # Initialize processed_images
    processed_images = []    
    
    if uploaded_files and len(names) > 0:
                        
        # First block with image filters and parameters
        with st.expander("", expanded = True):
            
            # Set subtitle and short explanation
            st.write("#### :orange[2a. Parameters for image equalization]")
            st.write("The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to \
                    refine the adjustment. - When you are happy with the result, download the processed image.")            
                                              
            # create 2 columns within the first block
            col1, col2 = st.columns(2)
            
            # Draw a dividing line
            st.markdown("---")

            # Col1
            brightness = col1.checkbox("Equalize brightness", value = False)
            contrast   = col1.checkbox("Equalize contrast", value = False)
            
            # Col2
            sharpness  = col2.checkbox("Equalize sharpness", value = False)
            color      = col2.checkbox("Equalize colors", value = False)
                 
                        
        # Second block with image filters and parameters
        with st.expander("", expanded = True):
        
            # Set subtitle and short explanation
            st.write("#### :orange[2b. Parameters for image alignnment]")
            st.write("Specifics to procedures TODO.")
        
            # create 2 columns within the second block
            col3, col4 = st.columns(2)
        
            st.markdown("---")
                        
            # Col3
            #bilateral_strength = col3.slider("###### :heavy_check_mark: :blue[Bilateral Filter Strength] (preset = 5)", min_value=0, max_value=15, value=5, key='bilateral')
            #saturation_factor =  col3.slider("###### :heavy_check_mark: :blue[Color Saturation] (preset = 1.1)", min_value=0.0, max_value=2.0, step=0.05, value=1.1, key='saturation')
                        
            # Col4
            #sigma_sharpness = col4.slider("###### :heavy_check_mark: :blue[Sharpness Sigma] (preset = 0.5)", min_value=0.0, max_value=3.0, value=0.5, step=0.1, key='sharpness')
            #contrast = col4.slider("###### :heavy_check_mark: :blue[Contrast] (preset = 1.0)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='contrast')
              
        # Alignment procedure
        if uploaded_files and len(names) > 0:
            
            scol1 , scol2 = st.columns(2)
            fcol1 , fcol2 = st.columns(2)
                                   
            ch = scol1.button("Select baseline image to align to")
            fs = scol2.button("Align images to baseline image")            
            
            if ch:
                filename = uploaded_files[np.random.randint(len(uploaded_files))].name.split('/')[-1]
                               
                # Display the base image
                fcol1.image(Image.open(filename),use_column_width=True)
                
                #Set the selected image as the base image
                st.session_state["disp_img"] = filename                
                st.write(f"Base Image: {filename}")
            
            # Align images to the selected base image
            if fs:                
                idx = names.index(st.session_state["disp_img"])                
                                
                # Remove the selected base image from the list of selected images
                base_image = uploaded_files.pop(idx).name.split('/')[-1]
                                
                # Extract the filenames from the list of selected images which will be aligned
                file_names = [file.name.split('/')[-1] for file in uploaded_files]                
                               
                processed_images = align_all_images_in_folder_to_template2(
                             base_image_path  = base_image,
                             input_files = file_names)
                
                # Create a list to store the file paths
                jpeg_file_paths = []

                for filename, aligned_image in processed_images:
                    # Create a temporary file for the JPEG image
                    with tempfile.NamedTemporaryFile(prefix=filename, delete=False) as aligned_image_jpeg_file:
                        # Get the full path of the temporary file
                        aligned_image_jpeg_file_path = aligned_image_jpeg_file.name
                    
                        # Save aligned_image as JPEG to the temporary file
                        Image.fromarray(aligned_image).convert("RGB").save(aligned_image_jpeg_file_path, format="JPEG")
                    
                        # Append the file path to the list
                        jpeg_file_paths.append(aligned_image_jpeg_file_path)
                
                if processed_images:
                                                                                  
                    # Display the base image
                    fcol1.write("")
                    fcol1.write("")
                    fcol1.write("")
                    fcol1.write("")
                    fcol1.write("")
                    fcol1.write("")                    
                    #fcol1.subheader(f"Base Image: {base_image}")
                    fcol1.write(f"Base Image: {base_image}")
                    fcol1.image(Image.open(st.session_state["disp_img"]), use_column_width=True)
                    
                    
                    with fcol2:                        
                        image_viewer(jpeg_file_paths, ncol=1, nrow=1, image_name_visible=True)
                        #st.image(use_column_width=True)  # Display images with column width
                       
                # Saving section ---------------------------------------------------------------------------
                # Create a directory to save the aligned images
                output_directory = "images_aligned_to_" + base_image.split('.')[0]
                os.makedirs(output_directory, exist_ok=True)
                                        
                # Save aligned images to output directory
                for filename, aligned_image in processed_images:
                    aligned_image_jpeg_path = os.path.join(output_directory, f"aligned_{filename}.jpg")
                    aligned_image_png_path = os.path.join(output_directory, f"aligned_{filename}.png")
                    aligned_image = Image.fromarray(aligned_image)
                    aligned_image.convert("RGB").save(aligned_image_jpeg_path, format="JPEG")
                    imageio.imwrite(aligned_image_png_path, np.array(aligned_image), format='PNG')
                                        
                st.markdown("---")
                            
                # Provide download buttons for both formats
                st.sidebar.markdown("\n")
                st.sidebar.markdown(f"#### :floppy_disk: :orange[3. Download Aligned Images] ")
                st.sidebar.write('In compressed :orange[JPG] or losless :orange[PNG] file format:')
                            
                # Provide download buttons for both formats
                col1, col2 = st.columns(2)
                            
                with col1:
                    for filename, aligned_image in processed_images:
                        aligned_image_jpeg_path = os.path.join(output_directory, f"aligned_{filename}.jpg")
                        with open(aligned_image_jpeg_path, "rb") as f:
                            st.sidebar.download_button(
                            label=f":orange[JPG] (aligned_{filename}.jpg)",
                            data=f,
                            file_name=f"aligned_{filename}.jpg",
                            key=f"aligned_image_download_jpeg_{filename}"
                            )

                with col2:
                    for filename, aligned_image in processed_images:
                        aligned_image_png_path = os.path.join(output_directory, f"aligned_{filename}.png")
                        with open(aligned_image_png_path, "rb") as f:
                            st.sidebar.download_button(
                            label=f":orange[PNG] (aligned_{filename}.png)",
                            data=f,
                            file_name=f"aligned_{filename}.png",
                            key=f"aligned_image_download_png_{filename}"
                            )
                
# run main function
if __name__ == "__main__":
    main()