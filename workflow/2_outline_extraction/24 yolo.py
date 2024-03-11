# libraries ----
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

model = YOLO("yolov8x-seg.pt") # load a pretrained YOLOv8x segmentation model

def run_inference_for_images(image_folder, model):
    """
    Run inference for a folder of images using a YOLOv8 model.

    Args:
        image_folder (str): Path to the folder containing images.
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

        print(f'Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}')

        save_isolated_object(isolated, img_name, idx, label, ci)

        print(f'Original Image Dimensions: {img.shape}')
        print(f'Isolated Image Dimensions: {isolated.shape}')

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

# run_inference_for_images ----
image_folder = "../../data/interim/no_bg"

# Run inference for the images in the specified folder
run_inference_for_images(image_folder, model)
