"""Module for analyzing and quantifying structural similarity between extracted outline of figures from a template
image with aligned figures from all other images in the dataset.
"""

# IMPORTS ----
import numpy as np
import imutils
import cv2
import os
from skimage.metrics import structural_similarity as compare_ssim

def compare_two_aligned_images(template_image, aligned_image, result_path = "filled_after.jpg"):
    """
    Compare two images and generate an image showing the differences.

    Args:
        template_image (str): The file path of the template image (the image to compare to).
        aligned_image (str): The file path of the aligned image.
        result_path (str, optional): The file path to save the resulting image. 
        Defaults to "filled_after.jpg".

    Returns:
        None
    """
    # Read images
    imageA = template_image
    imageB = aligned_image

    # Convert to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index
    (score, diff) = compare_ssim(grayA, grayB, full = True)
    print("Image Similarity SSIM: {:.4f}%".format(score * 100))
    
    # plot the difference between the two images
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find contours to obtain the regions of the two input images that differ
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    mask = np.zeros(template_image.shape, dtype = 'uint8')
    filled_after = aligned_image.copy()

    # Compute the bounding box of the contour and then draw the bounding box on both
    # input images to represent where the two images differ
    
    # Loop over the contours
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 500:            
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    # Show the output images
    #cv2.imshow("Template", imageA)
    cv2.imshow("Differences", imageB)
    #cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.imshow('mask', mask)
    cv2.imshow('filled after', filled_after)
    cv2.waitKey(0)

    # Save the result image
    cv2.imwrite(result_path, filled_after)

def compare_all_images_to_template(template_path, input_directory):
    """
    Compare aligned images in a directory to a selected template image.

    Parameters:
    - template_path (str): The file path of the template image.
    - input_directory (str): The directory containing the input images to be aligned.
    - output_directory (str): The directory where the aligned images will be saved.

    Returns:
    None
    """
    # load the template image with which we want to compare all other paintings
    template = cv2.imread(template_path)

    # create output directory if it does not exist yet
    os.makedirs(output_directory, exist_ok=True)

    # loop over all images in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # load the input image: this painting we want to compare to the template
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # compare the image by applying the function we defined above
            compare_two_aligned_images(image, template)
