
# This script removes the background of each image in the input folder and saves the output images in 
# the output folder. The background removal is done using the rembg library, with the background set 
# to transparant

# https://www.youtube.com/watch?v=Z3pP1GuQe8g
# https://www.horilla.com/blogs/how-to-remove-the-background-of-image-using-rembg-in-python/

import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Specify the input folder containing images
input_folder = "../../data/raw/Bridgewater"
#input_folder = "../../data/raw/Lamb"

# Create the output folder if it doesn't exist
output_folder = "../../data/interim/no_bg"
#output_folder = "../../data/interim/no_bg_lamb"
os.makedirs(output_folder, exist_ok = True)


def show_images(images, titles):
    """
    Display a grid of images with corresponding titles.
    
    Parameters:
        images (list of numpy.ndarray): List of images to display.
        titles (list of str): List of titles for the images.
    """
    num_images = len(images)
    num_cols = min(num_images, 5)  # Maximum of 5 columns
    num_rows = (num_images + 4) // 5  # Calculate the number of rows for the grid

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 8 * num_rows))

    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])  # Convert single Axes object to a 1x1 array

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].imshow(img, aspect='auto')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    # Hide any remaining empty subplots
    for j in range(num_images, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
            
# performing histogram equalization
def adaptive_hist(img, clipLimit=4.0):
    """
    Apply adaptive histogram equalization to an image.
    
    Parameters:
        img (PIL.Image.Image or numpy.ndarray): Input image.
        clipLimit (float): Limit for contrast limiting.
    
    Returns:
        numpy.ndarray: Image with adaptive histogram equalization applied.
    """
    # Convert PIL Image to NumPy array if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    # Check if the image is grayscale and convert it to 3 channels if necessary
    if len(img.shape) == 2:
        img = cv2.merge([img] * 3)

    window = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    ch1, ch2, ch3 = cv2.split(img_lab)
    img_l = window.apply(ch1)
    img_clahe = cv2.merge((img_l, ch2, ch3))
    return cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

def contrast_enhancement(img, alpha=1.0, beta=0):
    """
    Enhance the contrast of an image using histogram stretching.
    
    Parameters:
        img (numpy.ndarray): Input image.
        alpha (float): Contrast control (default is 1.0).
        beta (float): Brightness control (default is 0).
    
    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    # Convert image to grayscale if it's not already in grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Apply histogram stretching
    enhanced_img = np.clip(alpha * img_gray + beta, 0, 255).astype(np.uint8)

    return enhanced_img

def adjust_brightness(img, alpha=1.0, beta=0):
    """
    Adjust the brightness of an image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        alpha (float): Contrast control (default is 1.0).
        beta (float): Brightness control (default is 0).
    
    Returns:
        numpy.ndarray: Image with adjusted brightness.
    """
    # Apply brightness adjustment
    adjusted_img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    
    return adjusted_img


def noise_reduction(img, kernel_size=(3, 3)):
    """
    Reduce noise in an image using Gaussian blur.
    
    Parameters:
        img (numpy.ndarray): Input image.
        kernel_size (tuple): Size of the Gaussian kernel (default is (3, 3)).
    
    Returns:
        numpy.ndarray: Image with reduced noise.
    """
    return cv2.GaussianBlur(img, kernel_size, 0)

def edge_enhancement(img):
    """
    Enhance edges in an image using Laplacian operator.
    
    Parameters:
        img (numpy.ndarray): Input image.
    
    Returns:
        numpy.ndarray: Image with enhanced edges.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    # Scale the Laplacian result to the range [0, 255]
    scaled_laplacian = cv2.convertScaleAbs(laplacian)
    return scaled_laplacian

def thresholding(img, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """
    Apply thresholding to an image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        threshold_value (float): Threshold value.
        max_value (float): Maximum value to use with the chosen thresholding type.
        threshold_type (int): Type of thresholding to apply (default is cv2.THRESH_BINARY).
    
    Returns:
        numpy.ndarray: Thresholded image.
    """
    _, thresholded_img = cv2.threshold(img, threshold_value, max_value, threshold_type)
    return thresholded_img

def morphological_operations(img, kernel_size=(5, 5), iterations=1, operation=cv2.MORPH_CLOSE):
    """
    Apply morphological operations to an image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        kernel_size (tuple): Size of the structuring element kernel (default is (5, 5)).
        iterations (int): Number of iterations for the morphological operation (default is 1).
        operation (int): Type of morphological operation to apply (default is cv2.MORPH_CLOSE).
    
    Returns:
        numpy.ndarray: Image after applying morphological operations.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(img, operation, kernel, iterations=iterations)



# Iterate through each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png')):  # Check if the file is an image
        # Construct the full path for input and output
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"output_{filename}")

        # Open the input image
        input_image = Image.open(input_path)
        
        # Preprocess the input image with adaptive histogram equalization
        input_image = adaptive_hist(input_image, clipLimit=4.0)
                
        # Apply brightness adjustment
        input_image = adjust_brightness(input_image, alpha=0.9, beta=15)
        
        # Enhance contrast of the image
        input_image = contrast_enhancement(input_image, alpha=1.4, beta=10)
        
        # Reduce noise in the image
        #input_image = noise_reduction(input_image, kernel_size=(3, 3))
        
        # Enhance edges in the image
        #input_image = edge_enhancement(input_image)
        
        # Apply thresholding to the image
        #input_image = thresholding(input_image, threshold_value=127)
        
        # Apply morphological operations to the image
        input_image = morphological_operations(input_image, kernel_size=(4, 4), 
                                               operation=cv2.MORPH_CLOSE, iterations=1)
        
        # plot image
        #show_images([input_image], ["Preprocessed Image"])

        # Use rembg to remove the background
        output_image = remove(input_image,alpha_matte=True, 
                              only_mask = False,
                              background_color=(0, 0, 0),
                              alpha_matting_foreground_threshold=200,
                              alpha_matting_background_threshold=10,
                              alpha_matting_erode_structure_size=5,
                              alpha_matting_base_size=500,
                              post_process_mask = True)

        # Convert the image to RGB mode if it's in RGBA mode
        #if output_image.mode == 'RGBA':
        #    output_image = output_image.convert('RGB')

        # After processing with rembg, convert the NumPy array to a PIL Image
        output_image = Image.fromarray(output_image)

        # Convert the image to PNG format
        output_image = output_image.convert('RGBA')
        # Save the output image to the output folder
        output_image.save(output_path, format='PNG')  # Specify the format as PNG)

print("Background removal completed. Output images are saved in the 'no_bg' folder.")

#######################################################################################################

import cv2
from matplotlib import pyplot as plt

# Open the input image
input_image = Image.open("../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg")
image = cv2.imread("../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg")

plt.imshow(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.equalizeHist(gray_image)

# Use rembg to remove the background
output_image = remove(gray_image)
        
plt.imshow(output_image)
        
        

def remove_background_advanced(input_path, output_path, alpha_matte=False, background_color=(255, 255, 255)):
    with open(input_path, "rb") as input_file, open(output_path, "wb") as output_file:
        input_data = input_file.read()
        
        # Use advanced options
        output_data = remove(input_data, alpha_matte=alpha_matte, background_color=background_color)
        
        output_file.write(output_data)

# Specify input and output paths
#input_image_path = "../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg"
input_image_path = "../../data/raw/Lamb/1 Madrid_Museo del Prado.jpg"
output_image_path_advanced = "output_image_advanced.png"

# Remove background with advanced options
remove_background_advanced(input_image_path, output_image_path_advanced, 
                           alpha_matte=True, background_color=(0, 0, 0))

# Display the results
removed_background_image_advanced = Image.open(output_image_path_advanced)
removed_background_image_advanced.show(title="Image with Removed Background (Advanced)")

