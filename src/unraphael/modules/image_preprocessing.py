"""Module for preprocessing images before extracting figures and outlines.
"""

# IMPORTS ----
import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np 
import matplotlib.pyplot as plt


# performing histogram equalization
def adaptive_hist(img, clipLimit = 4.0):
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
            Available operations:
                cv2.MORPH_OPEN
                cv2.MORPH_CLOSE
                cv2.MORPH_GRADIENT
                cv2.MORPH_TOPHAT
                cv2.MORPH_BLACKHAT
    
    Returns:
        numpy.ndarray: Image after applying morphological operations.
        
    Raises:
        ValueError: If the specified morphological operation is not valid.
    """
    valid_operations = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT
    }
    
    # Validate the specified morphological operation
    if operation not in valid_operations.values():
        raise ValueError("Invalid morphological operation. Choose one of: 'open', 'close', 'gradient', 'tophat', 'blackhat'.")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(img, operation, kernel, iterations=iterations)