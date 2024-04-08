"""Module for extracting figures and outlines to be used for the similarity analysis.
"""

# IMPORTS ----
import os
from rembg import remove
from PIL import Image
import cv2
import sys
from matplotlib import pyplot as plt

from PIL import ImageDraw
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import imutils

# load a pretrained YOLOv8x segmentation model
model = YOLO("yolov8x-seg.pt") 
       
def remove_background_from_images(input_folder, output_folder):
    """
    Remove background from images in the input folder and save them in the output folder.
    
    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where output images will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):  
            
            # Construct the full path for input and output
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}")

            # Open the input image
            input_image = Image.open(input_path)
            
            # Preprocess the image before removing the background
            #TODO: Add preprocessing steps here

            # Use rembg to remove the background
            output_image = remove(input_image, alpha_matte=True,
                                  only_mask = False, background_color=(0, 0, 0),
                                  alpha_matting_foreground_threshold=200,
                                  alpha_matting_background_threshold=10,
                                  alpha_matting_erode_structure_size=5,
                                  alpha_matting_base_size=500)

            # Convert the image to PNG format
            output_image = output_image.convert('RGBA')
            
            # Save the output image to the output folder
            output_image.save(output_path, format='PNG')
    
def detect_and_save_faces(image_path):
    """
    Detect faces in an image and save the image with detected faces.

    Parameters:
        image_path (str): Path to the input image.
    """
    # Load the cascade classifiers for face and body detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Read the input image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(40, 40)
    )

    print("[INFO] Found {0} Faces!".format(len(faces)))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the new width and height
        w = int(w * 2.50)
        h = int(h * 1.75)

        # Adjust the coordinates to make the bounding box twice as big
        x = max(0, center_x - w // 2)
        y = max(0, center_y - h // 2)

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region
        roi_color = image[y:y + h, x:x + w]

        # Save the extracted face
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)


def segment_and_save_individual_figures_from_image(image_path, output_folder):
    """
    Segment objects in an individual image using a YOLO model and save each segmented object as an individual image with 
    transparent background.

    Parameters:
        image_path (str): Path to the input image.
        output_folder (str): Path to the folder where segmented images will be saved.
    """
    
    # Predict using YOLO model
    res = model.predict(image_path, save=True, save_txt=True)

    # Iterate detection results 
    for idx, r in enumerate(res):
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        # Iterate each object contour
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]

            # Create binary mask
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Create contour mask 
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Isolate object with transparent background (when saved as PNG)
            isolated = np.dstack([img, b_mask])
            
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            
            # Ensure the coordinates are within the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)              
                   
            # Save isolated figure to file
            output_path = os.path.join(output_folder, f'{img_name}_segment{idx}_{label}-{ci}.png')
            _ = cv2.imwrite(output_path, isolated)
            
def run_inference_for_images(image_folder, model):
    """
    Run inference for a folder of images using a YOLOv8 model.

    Args:
        image_folder (str): Path to the folder containing images without background.
        model: YOLOv8 model instance.

    Returns:
        None
    """
    
    # Get a list of image files in the folder
    image_files = Path(image_folder).rglob("*.jpg")
    
    for image_file in image_files:
        # Run inference for each image
        res = model.predict(str(image_file), save=True, save_txt=True)
                
        for idx, r in enumerate(res):
            process_detection_result(r, idx, image_file)

def process_detection_result(result, idx, image_file):
    """
    Process the detection result from YOLOv8 for a single image.

    Args:
        result: Detection result from YOLOv8.
        idx (int): Index of the result.
        image_file: Path to the input image file.

    Returns:
        None
    """
    img = np.copy(result.orig_img)
    img_name = Path(result.path).stem

    for ci, c in enumerate(result):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = create_binary_mask(c)
        isolated = isolate_object(img, b_mask)
        
        # Post-processing steps ----
        isolated = smooth_edges(isolated)
        isolated = morphological_operations(isolated)
        #isolated = smooth_contours(isolated)
        #isolated = conditional_masking(img, isolated)
        #isolated = connected_component_analysis(isolated)
        
        x1, y1, x2, y2 = clip_coordinates(c.boxes.xyxy.cpu().numpy().squeeze(), img.shape)        

        save_isolated_object(isolated, img_name, idx, label, ci)
        
def create_binary_mask(contour_result):
    """
    Create a binary mask from a contour result.

    Args:
        contour_result: Contour result from YOLOv8.

    Returns:
        np.ndarray: Binary mask.
    """
    b_mask = np.zeros(contour_result.orig_img.shape[:2], np.uint8)
    contour = contour_result.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    return b_mask

def isolate_object(img, b_mask):
    """
    Isolate an object with a binary mask.

    Args:
        img: Original image.
        b_mask: Binary mask.

    Returns:
        np.ndarray: Isolated object.
    """
    # OPTION-1: Isolate object with black background ----
    # Create 3-channel mask
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)

    # Isolate object with binary mask
    return cv2.bitwise_and(mask3ch, img)

    # OPTION-2: Isolate object with transparent background (when saved as PNG)
    return np.dstack([img, b_mask])

def clip_coordinates(coordinates, img_shape):
    """
    Clip coordinates to ensure they are within the image boundaries.

    Args:
        coordinates: Coordinates to be clipped.
        img_shape: Shape of the image.

    Returns:
        Tuple: Clipped coordinates (x1, y1, x2, y2).
    """    
    x1, y1, x2, y2 = coordinates.astype(np.int32)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_shape[1], x2), min(img_shape[0], y2)
    return x1, y1, x2, y2

def save_isolated_object(isolated, img_name, idx, label, ci):
    """
    Save the isolated object to a file.

    Args:
        isolated: Isolated object.
        img_name (str): Name of the original image.
        idx (int): Index of the result.
        label (str): Object label.
        ci (int): Index of the contour.

    Returns:
        None
    """
    # Convert isolated object to black background
    black_bg = np.zeros_like(isolated)
    black_bg[:isolated.shape[0], :isolated.shape[1]] = isolated
    
    # Save the isolated object with black background as JPG
    filename = f'../../data/interim/segments/{img_name}_segment{idx}_{label}-{ci}.jpg'
    _ = cv2.imwrite(filename, black_bg)
    print(f'Saved isolated object to: {filename}')
    
# Add the post-processing functions here

def smooth_edges(segmentation_mask, sigma=1):
    """
    Smooth the edges of a segmentation mask using Gaussian blur.

    Args:
        segmentation_mask: Segmentation mask.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed segmentation mask.
    """
    return cv2.GaussianBlur(segmentation_mask, (0, 0), sigma)

def morphological_operations(segmentation_mask, kernel_size=3):
    """
    Apply morphological operations to a segmentation mask.

    Args:
        segmentation_mask: Segmentation mask.
        kernel_size (int): Size of the morphological kernel.

    Returns:
        np.ndarray: Refined segmentation mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(segmentation_mask, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion

def smooth_contours(segmentation_mask):
    """
    Smooth the contours of a segmentation mask.

    Args:
        segmentation_mask: Segmentation mask.

    Returns:
        np.ndarray: Segmentation mask with smoothed contours.
    """
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours on the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    smoothed_mask = np.zeros_like(segmentation_mask)

    # Draw smoothed contours on each color channel
    for i in range(segmentation_mask.shape[2]):
        smoothed_mask[:, :, i] = segmentation_mask[:, :, i] & 0  # Set channel to 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw smoothed contours on each color channel
        for i in range(segmentation_mask.shape[2]):
            cv2.drawContours(smoothed_mask[:, :, i:i+1], [approx], 0, (255), thickness=cv2.FILLED)

    return smoothed_mask


def conditional_masking(original_image, segmentation_mask, threshold=0.5):
    """
    Conditionally mask regions of a segmentation mask based on the original image.

    Args:
        original_image: Original image.
        segmentation_mask: Segmentation mask.
        threshold (float): Threshold for conditional masking.

    Returns:
        np.ndarray: Conditionally masked segmentation mask.
    """
    condition = original_image > threshold
    refined_mask = np.where(condition, segmentation_mask, 0)
    return refined_mask

def connected_component_analysis(segmentation_mask, min_area=100):
    """
    Perform connected component analysis to filter out small components in a segmentation mask.

    Args:
        segmentation_mask: Segmentation mask.
        min_area (int): Minimum area of connected components to retain.

    Returns:
        np.ndarray: Refined segmentation mask.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(segmentation_mask)
    large_components = (stats[:, cv2.CC_STAT_AREA] > min_area)
    refined_mask = np.zeros_like(segmentation_mask)
    refined_mask[labels.isin(np.where(large_components))] = 255
    return refined_mask


def combine_masks_from_image_folder(input_folder_path, output_folder_path):
    """
    Combine segmentation masks from an image into a single binary image and save it in the output folder. All images in the input folder
    are processed and saved at once

    Parameters:
        input_folder_path (str): Path to the folder containing input images.
        output_folder_path (str): Path to the folder where combined mask images will be saved.
    """
        
    for filename in os.listdir(input_folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            
            # Predict using YOLO model
            img_path = os.path.join(input_folder_path, filename)
            res = model.predict(img_path, save=True, save_txt=True)

            # Combine segmentation masks into a single binary image
            img = (res[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
            height, width = img.shape
            masked = np.zeros((height, width), dtype="uint8")

            num_masks = len(res[0].cpu().masks.data)

            for i in range(num_masks):
                masked = cv2.add(masked, (res[0].cpu().masks.data[i].numpy() * 255).astype("uint8"))

            # Convert NumPy array to PIL Image
            mask_img = Image.fromarray(masked, "L")

            # Save the combined mask image
            output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_combined_mask.jpg")
            mask_img.save(output_path)

def extract_colored_outer_contour(input_folder, output_folder):
    """
    Extracts colored outer contours from binary images in the input folder and saves the results in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing the binary images.
    - output_folder (str): Path to the folder where the colored outer contour images will be saved.

    Returns:
    Images with different colored contours in a separate folder
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each binary image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
            
            # Read the binary image
            binary_path = os.path.join(input_folder, filename)
            binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)           
            
            # Use a suitable thresholding method to create a binary image
            _, thresh = cv2.threshold(binary_image, 20, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image; RETR_EXTERNAL retrieves only the extreme outer contours
            contours, _ = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area to keep only larger contours (human-sized)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25]

            # Create an empty black image for contours
            contour_image = np.zeros_like(binary_image)

            # Convert the black image to RGB
            contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)
                        
            # Draw the filtered contours on the RGB image
            for i, contour in enumerate(filtered_contours):
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.drawContours(contour_image_rgb, [contour], -1, color, thickness=2) 
                        
            # Save the result
            output_path = os.path.join(output_folder, f"colored_outer_contour_{filename}")
            cv2.imwrite(output_path, contour_image_rgb)
            
            # Bitwise AND operation to extract the regions with contours from the original image
            #result = cv2.bitwise_and(binary_image, binary_image, mask=contour_image)
            # Save the result
            #output_path = os.path.join(output_folder, f"outer_contour_{filename}")
            #cv2.imwrite(output_path, result)

def get_contour_color(image):
    """
    Get the color of the contour in the given image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: The contour color as a tuple of RGB values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_color = tuple(map(int, cv2.mean(image, mask=thresh)[:3]))
    return contour_color

def overlaying_images(base_image_path, overlay_images, scale_factor=1.50):
    """
    Overlay multiple images on a base image and display the result.

    Args:
        base_image_path (str): The file path of the base image.
        overlay_images (list): A list of file paths of the overlay images.
        scale_factor (float, optional): The scaling factor for the output image. Defaults to 1.50.
    """
    
    # Read the base image
    base_image = cv2.imread(base_image_path)

    # Check if the base image is valid
    if base_image is None:
        print(f"Error: Unable to read {base_image_path}")
        return

    # Initialize the output image with the base image
    output = base_image.copy()

    # Get the title of the base image
    base_title = os.path.basename(base_image_path).split(".")[0]

    # Add the base image title to the legend
    legend_text_base = f"{base_title}"
    cv2.putText(output, legend_text_base, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Initialize y-coordinate for legend text
    text_y = 60
    
    # Iterate over each overlay image and blend it with the output
    for overlay_path in overlay_images:
        overlay = cv2.imread(overlay_path)

        # Check if the overlay image is valid
        if overlay is not None:
            output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

            # Get contour color from the overlay image
            contour_color = get_contour_color(overlay)

            # Add legend text with contour color above the contours
            legend_text = os.path.basename(overlay_path).split(".")[0]  # Extract text before the file extension
            cv2.putText(output, legend_text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, contour_color, 2, cv2.LINE_AA)
            
            # Increment y-coordinate for next legend text
            text_y += 30
        else:
            print(f"Error: Unable to read {overlay_path}")

    # Resize the output image
    output = cv2.resize(output, None, fx=scale_factor, fy=scale_factor)  
            
    cv2.imshow("Overlayed Images", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
