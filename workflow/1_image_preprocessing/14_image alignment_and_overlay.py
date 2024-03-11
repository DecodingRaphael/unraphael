# Image alignment ----
# Applying image alignment to an input image allows us to align it with a template document. 
# Once we have the input image aligned with the template, we can compare the two images. 
# The two images are aligned here using keypoint matching with ORB feature matching and 
# homography. The aligned images are saved in a new directory called "aligned".

# Background material:
# https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
# https://stackoverflow.com/questions/76568361/image-registration-issue-using-opencvpython
# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# https://forum.opencv.org/t/image-difference-after-image-registration-and-alignment/10553/3
# https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

# libraries ----
import math
import numpy as np
import imutils
import cv2
import os
from IPython.display import display, HTML
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

# own developed function to normalize brightness ----
from equal_brightness import equalize_brightness_color

# function to align image ----
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    """
    Aligns an input image with a template image using feature matching and homography transformation.

    Parameters:
        image (numpy.ndarray): The input image to be aligned.
        template (numpy.ndarray): The template image to align the input image with.
        maxFeatures (int, optional): The maximum number of features to detect and extract using ORB. Default is 500.
        keepPercent (float, optional): The percentage of top matches to keep. Default is 0.2.
        debug (bool, optional): Whether to visualize the matched keypoints. Default is False.

    Returns:
        numpy.ndarray: The aligned image.

    """
    # convert both the input image and template to grayscale
    imageGray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local invariant features
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

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

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
    # The homography matrix represents the rotation, translation, and scale
    # to convert (warp) from the plane of our input image to the plane of our
    # template image.
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    
    # By creating a modified homography matrix (H_no_rotation) that excludes the
    # rotation component, i.e., preserving only translation, scale, and shear,
    # we can do image alignment while preserving the differences in rotation.
    H_no_rotation = np.array([[H[0, 0], H[0, 1], H[0, 2]],
                              [H[1, 0], H[1, 1], H[1, 2]],
                              [0, 0, 1]])
    
    ## derive rotation angle between figures from the homography matrix
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
    print(f'Rotational degree: {theta:.2f}') # rotation angle, in degrees
    #print(theta) 
    
    # apply the homography matrix to align the images, including the rotation
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    
    # apply the homography matrix to align the images, without modifying the rotation
    #aligned = cv2.warpPerspective(image, H_no_rotation, (w, h))

    # check sizes
    print(aligned.shape, template.shape)
    
    # side-by-side comparison of the template (left) and the aligned image (right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Aligned image')
    axes[1].axis('off')
    plt.show()

    # return the aligned image
    return aligned

def align_all_images_to_template(template_path, input_directory, output_directory, maxFeatures=500, keepPercent=0.2, debug=False):
    """
    Aligns images in a directory to a template image.

    Parameters:
    - template_path (str): The file path of the template image.
    - input_directory (str): The directory containing the input images to be aligned.
    - output_directory (str): The directory where the aligned images will be saved.
    - maxFeatures (int): The maximum number of features to detect in the images (default: 500).
    - keepPercent (float): The percentage of features to keep during feature matching (default: 0.2).
    - debug (bool): Whether to print debug information (default: False).

    Returns:
    None
    """
    # load the template image/ painting to which we want to align all the other paintings
    template = cv2.imread(template_path)

    # create output directory if it does not exist yet
    os.makedirs(output_directory, exist_ok=True)

    # loop over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # load the input image: this painting we want to align to the template
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # align the images by applying the function we defined above
            aligned = align_images(image, template, maxFeatures, keepPercent, debug)

            # save the aligned image in the output directory
            output_path = os.path.join(output_directory, f"aligned_{filename}")
            cv2.imwrite(output_path, aligned)
            print(f"[INFO] Image {filename} aligned and saved to {output_path}")

# Function to examine image differences ----
def compare_images(image_path1, image_path2, result_path = "filled_after.jpg"):
    """
    Compare two images and generate an image showing the differences.

    Args:
        image_path1 (str): The file path of the first image.
        image_path2 (str): The file path of the second image.
        result_path (str, optional): The file path to save the resulting image. 
        Defaults to "filled_after.jpg".

    Returns:
        None
    """
    # Read images
    imageA = image_path1
    imageB = image_path2

    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two gray images
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    print("Image Similarity SSIM: {:.4f}%".format(score * 100))
        
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

    mask = np.zeros(template.shape, dtype='uint8')
    filled_after = aligned.copy()

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

# load the images to compare ----
print("[INFO] loading images...")

# template to align to
template = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")
#template = cv2.imread("../../data/interim/segments/output_2_Naples_Museo Capodimonte_segment0_person-0.jpg")
#template = cv2.imread("../../data/interim/outlines/outer_contour_aligned_0_combined_mask.jpg")
    
# image to align
image    = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")
#image    = cv2.imread("../../data/interim/segments/output_1_London_Nat_Gallery_segment0_person-0.jpg")  
#image    = cv2.imread("../../data/interim/outlines/outer_contour_aligned_1_combined_mask.jpg")  

# Equalizing colors ----
# The match_histograms function from scikit-image is designed to adjust the color 
# distribution of an image to match that of a reference image
print("[INFO] equalizing colors by matching histograms...")
matched = match_histograms(image, template, channel_axis=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Template')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Original Target')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
axes[2].set_title('Target(adapted colours)')
axes[2].axis('off')
plt.show()

# Equalizing brightness ----
print("[INFO] equalizing brightness between the images...")
eq_image = equalize_brightness_color(template, matched)
#eq_image = equalize_brightness_contrast_sharpness(template, image, alpha_contrast=0.8, alpha_sharpness=0.7)

# Aligning the images ----
print("[INFO] aligning images...")
aligned = align_images(eq_image, template, debug=True)

# Check if dimensions are the same
print(aligned.shape, template.shape)
# Resize both the aligned and template images so we can visualize them on the screen
#aligned  = imutils.resize(aligned, width=800)
#template = imutils.resize(template, width=800)

# Side-by-side stacked visualization of the aligned output
stacked = np.hstack([template,aligned])
cv2.imshow("Image Alignment Stacked", stacked)
cv2.waitKey(0)

# Save resized and aligned image
cv2.imwrite("../../data/interim/aligned/aligned_0_Edinburgh_Nat_Gallery.jpg", template)
cv2.imwrite("../../data/interim/aligned/aligned_1_London_Nat_Gallery.jpg", aligned)


# 2) overlaying the aligned image on the template ----
overlay = template.copy()
output  = aligned.copy()

# transparently blend the two images into a single output image with the pixels
# from each image having equal weight
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

print("Overlay the  aligned and registered image on our template with a 50/50 blend")
# show the overlayed images
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed

# Display using Matplotlib
result = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 3) using trackbars to examine the alignment   
alpha_slider_max = 100
title_window = 'Linear Blend'

## on_trackbar
def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta  = ( 1.0 - alpha )
    dst   = cv2.addWeighted(overlay, alpha, output, beta, 0.0)
    cv2.imshow(title_window, dst)

cv2.namedWindow(title_window)

## create_trackbar
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)

on_trackbar(0)
cv2.waitKey()

# comparing the areas of the figure in black and white ----

# Convert the image to grayscale
templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
align_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to create a binary image (black and white)
_, binary_templ = cv2.threshold(templ_gray, 1, 255, cv2.THRESH_BINARY)
_, binary_align = cv2.threshold(align_gray, 1, 255, cv2.THRESH_BINARY)

# Display the original and black and white images
plt.subplot(1, 2, 1), plt.imshow(binary_templ, cmap='gray'), plt.title('Template (b&w)')
plt.subplot(1, 2, 2), plt.imshow(binary_align, cmap='gray'), plt.title('Aligned image (b&w)')
plt.show()

# Compare images with SSIM ----
# SSIM works best when images are near-perfectly aligned, otherwise, the pixel locations and 
# values would not match up, throwing off the similarity score. We therefore aligned the images first
templ = template.copy()
compare_images(template, aligned)

# Create a list of frames of the animation
frames = []
frames.append(Image.fromarray(templ))
frames.append(Image.open('filled_after.jpg'))

# Save result
frames[0].save('anim.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)


# plotting the original paintings with no background side by side -----
original_template = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")
original_image    = cv2.imread("../../data/interim/no_bg/output_2_Naples_Museo Capodimonte.jpg")

# plotting the original paintings with background side by side -----
original_template = cv2.imread("../../data/raw/Bridgewater/1_London_Nat_Gallery.jpg")
original_image    = cv2.imread("../../data/raw/Bridgewater/2_Naples_Museo Capodimonte.jpg")

# Resize images to a common size
common_size = (min(original_template.shape[1], original_image.shape[1]), min(original_template.shape[0], original_image.shape[0]))
original_template_resized = cv2.resize(original_template, common_size)
original_image_resized = cv2.resize(original_image, common_size)

# double check if dimensions are the same
print(original_template_resized.shape, original_image_resized.shape)

compare_images(original_template_resized, original_image_resized)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_template_resized)
axes[0].set_title('Painting 1')
axes[0].axis('off')
axes[1].imshow(original_image_resized)
axes[1].set_title('Painting 2')
axes[1].axis('off')
plt.show()

# align all images to basis image ----
# set the paths and parameters
template_path = "../../data/raw/0_Edinburgh_Nat_Gallery.jpg"
input_directory = "../../data/raw/Bridgewater"
output_directory = "../../data/interim/aligned"

# align all images to the original painting (edinburgh)
align_all_images_to_template(template_path, input_directory, output_directory, debug=False)
