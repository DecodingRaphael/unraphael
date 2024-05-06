# run with: streamlit run normalization_app.py --server.enableXsrfProtection false
# Import libraries
import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import imageio
import os
import cv2
from datetime import datetime
import sys
sys.path.append('../')

import tempfile

# import the function to align images in the directory
from src.unraphael.modules.comparison_preparation import (align_images, 
align_all_images_in_folder_to_template, normalize_brightness, normalize_contrast, 
normalize_sharpness, normalize_colors)

def align_all_images_in_folder_to_template2(base_image_path, input_files):
    """
    Aligns images in a directory to a selected base image.

    Parameters:
    - base_image_path (str): The file path of the base image.
    - input_files (str): A list of filenames representing the input images to be aligned.

    Returns:
    - aligned_images (list): A list of tuples containing the filename and aligned image.
    """
    # load the base image to which we want to align all the other images
    template = cv2.imread(base_image_path)

    # list to store aligned images
    aligned_images = []

    # loop over all images in the input directory
    for filename in input_files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # load the input image: this is the image we want to align to the base image            
            image = cv2.imread(filename)

            # align the image by applying the alignment function
            aligned = align_images(image, template)

            # append filename and aligned image to the list
            aligned_images.append((filename, aligned))

    return aligned_images


def main():
    st.markdown('<div style="text-align: center;"><h1 style="color: orange;">Image normalization</h1></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 style="font-size: 26px;">For a selected image, align and/ or normalize all other images</h2></div>', unsafe_allow_html=True)                

    # Draw a dividing line
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
       
    # Debugging: Print names array to verify contents
    print("Names of selected images:", names)

    directory_path = None
    if uploaded_files:
        for uploaded_file in uploaded_files:
            
            # Create a temporary directory
            #temp_dir = tempfile.mkdtemp()
            
            path = uploaded_file.name
            print("Path:", path)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
        # Extract the directory path from the temporary directory
        directory_path = path
    
    # Debugging: Print directory path
    print("Directory path:", directory_path)

    # Initialize processed_images
    processed_images = []
    
    # Customization with live updating of the aligned images
    if uploaded_files and len(names) > 0:  # Check if names array is not empty
                        
        # First block with image filters and parameters       
        with st.expander("",expanded = True):
            
            # Set subtitle and short explanation
            st.write("#### :orange[2a. Parameters for image equalization]")
            st.write("The processed image is shown with a preset of parameters. Use the sliders to explore the effects of image filters, or to \
                    refine the adjustment. - When you are happy with the result, download the processed image.")            
            
            #Image size
            # Option to resize image, based on the size of the base image 
            # TODO                       
                        
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
            # - histogram matching   
            # - color quantization using k-Means clustering          
                        
        # Second block with image filters and parameters
        with st.expander("",expanded = True):
        
            # Set subtitle and short explanation
            st.write("#### :orange[2b. Parameters for image alignnment]")
            st.write("Specifics to procedures TODO.")
        
            # create 2 columns within the second block
            col3, col4 = st.columns(2)
        
            # Draw a dividing line
            st.markdown("---")
                        
            # Col3
            #bilateral_strength = col3.slider("###### :heavy_check_mark: :blue[Bilateral Filter Strength] (preset = 5)", min_value=0, max_value=15, value=5, key='bilateral')
            #saturation_factor =  col3.slider("###### :heavy_check_mark: :blue[Color Saturation] (preset = 1.1)", min_value=0.0, max_value=2.0, step=0.05, value=1.1, key='saturation')
                        
            # Col4
            #sigma_sharpness = col4.slider("###### :heavy_check_mark: :blue[Sharpness Sigma] (preset = 0.5)", min_value=0.0, max_value=3.0, value=0.5, step=0.1, key='sharpness')
            #contrast = col4.slider("###### :heavy_check_mark: :blue[Contrast] (preset = 1.0)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key='contrast')
              
        # Live preview of alignment process
        if uploaded_files and len(names) > 0:
            _ , fcol2 , _ = st.columns(3)
            scol1 , scol2 = st.columns(2)

                                   
            ch = scol1.button("Select baseline image to align to")
            fs = scol2.button("Align images to baseline image")
            
            if ch:
                filename = uploaded_files[np.random.randint(len(uploaded_files))].name.split('/')[-1]
                               
                # Display the base image
                fcol2.image(Image.open(filename))
                
                #Set the selected image as the base image
                st.session_state["disp_img"] = filename
                st.write(f"Base Image: {filename}")
            
            # Align images to the selected base image            
            if fs:                
                idx = names.index(st.session_state["disp_img"])                
                print("----------idx------------------------------------")
                print(idx)
                                
                # Remove the selected base image from the list of uploaded files
                base_image = uploaded_files.pop(idx).name.split('/')[-1]
                                               
                print("----------base image------------------------------------")
                print(base_image)
                
                # Extract the filenames from the list of uploadedfile objects
                file_names = [file.name.split('/')[-1] for file in uploaded_files]
                
                print("----------uploaded images------------------------------------")
                print(file_names)
                
                processed_images = align_all_images_in_folder_to_template2(
                             base_image_path  = base_image,
                             input_files = file_names)
            
                # display the base image
                #fcol2.image(Image.open(st.session_state["disp_img"]))
                
                # Display aligned images                
                #for filename, aligned_image in processed_images:
                #    st.image(aligned_image, caption=f"Aligned Image: {filename}")
            
                # ----------------------------------------------------
                
                # Initialize current_index if not already present
                if "current_index" not in st.session_state:
                    st.session_state["current_index"] = 0
                    
                print("----------current index init ------------------------------------")
                print(st.session_state["current_index"])
                                             
                c1, c2 = st.columns(2)
                
                # Display the baseline image in the left column
                with c1:
                    c1.image(st.session_state["disp_img"], caption = f"Baseline Image: : {base_image}", width=350)

                # Display aligned image in the right column
                with c2:
                    
                    # Function to display the aligned image corresponding to the given index
                    def display_aligned_image(index):
                        if 0 <= index < len(processed_images):
                            filename, aligned_image = processed_images[index]                            
                            c2.image(aligned_image, caption = f"Aligned Image: {filename}", width=350)  # Adjust width as needed
                            
                    # Get the current index from session state
                    current_index = st.session_state["current_index"]
                    
                    print("----------current index xxx ------------------------------------")
                    print(st.session_state["current_index"])
                                        
                    print("----------current index after function is apllied ------------------------------------")
                    print(st.session_state["current_index"])

                    # Increment the current index to show the next aligned image
                    current_index += 1
                    print("----------current index after next ------------------------------------")
                    print(current_index)
                    
                    print("Before 'Next' button check")                    
                    # Button to show the next aligned image
                    if c2.button("Next"):
                        print("Inside 'Next' button check")
                        current_index += 1
                        st.session_state["current_index"] = current_index
                        st.empty()  
                        display_aligned_image(current_index)
                                                              
                    print("----------current index after next ------------------------------------")
                    print(current_index)    
                    
                    #display_aligned_image(current_index)

                        # # Ensure the index stays within bounds
                        # current_index %= len(processed_images)

                        # st.session_state["current_index"] = current_index
                        
                        # print("----------current index after resetting 1 ------------------------------------")
                        # print(st.session_state["current_index"])
                        
                        # st.empty()  # Clear previous output
                        
                        # print("----------current index after empty ------------------------------------")
                        # print(st.session_state["current_index"])
                        
                        # display_aligned_image(current_index)
                        
                    # # Button to show the next aligned image
                    # if st.button("Next") and current_index < len(processed_images) - 1:
                    #     current_index += 1
                    #     st.session_state["current_index"] = current_index
                    #     st.empty()  
                    #     display_aligned_image(current_index)

                    # # Button to show the previous aligned image
                    # if st.button("Previous") and current_index > 0:
                    #     current_index -= 1
                    #     st.session_state["current_index"] = current_index
                    #     st.empty()  # Clear previous output
                    #     display_aligned_image(current_index)
                                                
        
                            
            #         # Add a button to scroll between other aligned images
            #         if c2.button("Next"):
            #                 break  # Break after the first button press to show only one aligned image at a time
                            
            #         # Save aligned images to output directory
            #         current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #         output_directory = f"aligned_images_{current_time}"
            #         os.makedirs(output_directory, exist_ok = True)
                                                        
            #         # Save aligned images to output directory
            #         aligned_image_jpeg_path = os.path.join(output_directory, f"{aligned_image_name}_aligned.jpg")
            #         aligned_image_png_path = os.path.join(output_directory, f"{aligned_image_name}_aligned.png")

            #         aligned_image.convert("RGB").save(aligned_image_jpeg_path, format="JPEG")
            #         imageio.imwrite(aligned_image_png_path, np.array(aligned_image), format='PNG')
                
            # st.markdown("---")
                
            # # Download Buttons            
            # st.sidebar.markdown("\n")
            # st.sidebar.markdown(f"#### :floppy_disk: :orange[3. Download Aligned Images] ")
            # st.sidebar.write('In compressed :orange[JPG] or losless :orange[PNG] file format:')
                
            # # Provide download buttons for both formats
            # col1, col2 = st.columns(2)
                
            # with col1:
            #     for filename, aligned_image in processed_images:
            #         aligned_image_jpeg_path = os.path.join(output_directory, f"aligned_{filename}.jpg")
            #         with open(aligned_image_jpeg_path, "rb") as f:
            #             st.sidebar.download_button(
            #                 label=f":orange[JPG] (aligned_{filename}.jpg)",
            #                 data=f,
            #                 file_name=f"aligned_{filename}.jpg",
            #                 key=f"aligned_image_download_jpeg_{filename}"
            #     )

            # with col2:
            #     for filename, aligned_image in processed_images:
            #         aligned_image_png_path = os.path.join(output_directory, f"aligned_{filename}.png")
            #         with open(aligned_image_png_path, "rb") as f:
            #             st.sidebar.download_button(
            #                 label=f":orange[PNG] (aligned_{filename}.png)",
            #                 data=f,
            #                 file_name=f"aligned_{filename}.png",
            #                 key=f"aligned_image_download_png_{filename}"
            #     )
                
# run main function
if __name__ == "__main__":
    main()
    