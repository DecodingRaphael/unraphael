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
from skimage.exposure import match_histograms

sys.path.append('../')

st.set_page_config(layout="wide", page_title = "")

# import the function to align images in the directory
#from unraphael.modules.outline_normalization import (
# align_images, 
#align_all_selected_images_to_template, normalize_brightness, normalize_contrast, 
#normalize_sharpness, normalize_colors)

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
        
def align_images(image, template, maxFeatures = 5000, keepPercent = 0.1):
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
    # convert input image and template to grayscale
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
        
    # derive rotation angle between figures from the homography matrix
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
    print(f'Rotational degree ORB feature: {theta:.2f}') # rotation angle, in degrees
        
    # apply the homography matrix to align the images, including the rotation
    h, w, c = template.shape
    aligned = cv2.warpPerspective(image, H, (w, h), borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))    
    
    return aligned

def eccAlign(im1, im2):
    """
    Aligns two images using the ECC (Enhanced Correlation Coefficient) algorithm.

    Parameters:
    - im1: The template image.
    - im2: The image to be warped to match im1.

    Returns:
    - im2_aligned: The aligned image.

    """
    
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY) # template image
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY) # im2 is image to be warped to match im1
    
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    #warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the ECC number of iterations
    number_of_iterations = 50000
    
    # Specify the ECC threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-5
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

    # Warp im2, based on im1
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography        
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Rotation angle
    row1_col0 = warp_matrix[0,1]
    angle = math.degrees(math.asin(row1_col0))
    print(f'Rotational degree ECC: {angle:.2f}')
    
    return im2_aligned

def align_all_selected_images_to_template(base_image_path, input_files, selected_option, preprocess_options):
   
    """
    Aligns all images in a folder to a template image using the selected alignment method and preprocess options.

    Parameters:
    - base_image_path (str): The file path of the template image.
    - input_files (list): List of file paths of images to be aligned.
    - selected_option (str): The selected alignment method.
    - preprocess_options (dict): Dictionary containing preprocessing options.

    Returns:
    - aligned_images (list): List of tuples containing filename and aligned image.
    """
    
    # load the base image to which we want to align all the other images
    template = cv2.imread(base_image_path)

    aligned_images = []
    
    for filename in input_files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # image to be aligned            
            image = cv2.imread(filename)
            
            # preprocess the image
            preprocessed_image = preprocess_image(template, image, **preprocess_options)
            
            if selected_option == 'ORB feature based alignment':
                aligned = align_images(preprocessed_image, template)
            
            elif selected_option == "Enhanced Correlation Coefficient Maximization":
                aligned = eccAlign(preprocessed_image, image)
            #else:
            #    warp_matrix = translation(image, template)

            # TODO: Add other alignment methods here
            
            # append filename and aligned image to list
            aligned_images.append((filename, aligned))
    return aligned_images

def normalize_colors(template, target):
    """
    Normalize the colors of the target image to match the color distribution of the template image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - normalized_img: Target image with colors normalized to match the template image.
    """        
    matched = match_histograms(target, template, channel_axis = -1)
    
    return matched


def preprocess_image(template, image, brightness=False, contrast=False, sharpness=False, color=False):
    """
    Preprocesses the input image based on the selected enhancement options.

    Parameters:
    - image: The input image in BGR format.
    - brightness: Whether to equalize brightness.
    - contrast: Whether to equalize contrast.
    - sharpness: Whether to equalize sharpness.
    - color: Whether to equalize colors.

    Returns:
    - preprocessed_image: The preprocessed image.
    """

    if brightness:
        # perform brightness equalization
        # implement brightness equalization here
        pass

    if contrast:
        # perform contrast equalization
        # implement contrast equalization here
        pass

    if sharpness:
        # perform sharpness equalization
        # implement sharpness equalization here
        pass

    if color:
        # perform color equalization
        # implement color equalization here
        image = normalize_colors(template, image)

    return image

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
                       'FFT phase correlation',
                       'Fourier Mellin Transform',
                       'User-provided keypoints (from pose estimation)']

            # Display the dropdown menu
            selected_option = col3.selectbox('Select an option:', options)
            
            if selected_option:
                motion_model = col4.selectbox("Select motion model:", ['euclidian', 'homography'])
            
            #TODO: selection of motion model for ecc (e.g., euclidian, homography)       
                      
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

                    for filename, aligned_image in processed_images:
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

                    image_downloads_widget(images = save_aligned_images)

# run main function
if __name__ == "__main__":
    main()