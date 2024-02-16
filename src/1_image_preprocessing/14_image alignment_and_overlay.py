# Image alignment ----
# Applying image alignment to an input image allows us to align it with a template document. Once we have
# the input image aligned with the template, we can compare the two images. The two images are aligned here 
# using keypoint matching using ORB feature matching and homography.

# The aligned images are saved in a new directory called "aligned".

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

# function to normalize brightness ----
#def equalize_brightness_contrast_sharpness(template, target, alpha_contrast=1.0, alpha_sharpness=1.0):
def equalize_brightness_color(template, target):
    
    """
    Equalizes the brightness of the target image based on the luminance of the template image
    and normalizes contrast and sharpness.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.
    - alpha_contrast: Weight for contrast normalization (default is 1.0).
    - alpha_sharpness: Weight for sharpness normalization (default is 1.0).

    Returns:
    - equalized_img: Adjusted target image with equalized brightness, contrast, and sharpness.
    - ratios: Dictionary containing brightness, contrast, and sharpness ratios.

    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to BGR. The function also normalizes
    contrast and sharpness and returns the adjusted image and ratios.
    """
    
    # Convert the template image to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the template image
    l_template, a_template, b_template = cv2.split(template_lab)

    # Convert the target image to LAB color space
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the target image
    l_target, a_target, b_target = cv2.split(target_lab)

    # Adjust the L channel (brightness) of the target image based on the mean brightness of the template
    l_target = (l_target * (np.mean(l_template) / np.mean(l_target))).clip(0, 255).astype(np.uint8)
    
    # ----------------------
    # Normalize contrast
    #l_target = (l_target - np.mean(l_target)) * (alpha_contrast / np.std(l_target)) + np.mean(l_target)

    # Normalize sharpness
    #blurred_template = cv2.GaussianBlur(l_template, (0, 0), sigmaX=5)
    #sharpened_template = cv2.addWeighted(l_template, 2.5, blurred_template, -1.5, 0)
    #sharpened_template = cv2.resize(sharpened_template, (l_target.shape[1], l_target.shape[0]))  # Resize to match dimensions
    
    # Ensure both l_target and sharpened_template have the same data type
    #l_target = l_target.astype(sharpened_template.dtype)

    # Apply weighted addition
    #l_target = cv2.addWeighted(l_target, 1 - alpha_sharpness, sharpened_template, alpha_sharpness, 0)

    # Clip and convert back to uint8
    #l_target = np.clip(l_target, 0, 255).astype(np.uint8)
    # ----------------------

    # Merge LAB channels back for the adjusted target image
    equalized_img_lab = cv2.merge([l_target, a_target, b_target])

    # Convert the adjusted LAB image back to BGR
    equalized_img = cv2.cvtColor(equalized_img_lab, cv2.COLOR_LAB2BGR)
    
    # evaluate brightness ratios
    
    # Using LAB color space ---- 
    # we convert the images (template and eq_image) to the LAB color space and calculate 
    # the mean brightness from the luminance channel (L) only

    # Calculate the mean of the color images from the luminance channel
    mean_template_lab = np.mean(cv2.split(template_lab)[0])
    mean_eq_image_lab = np.mean(cv2.split(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2LAB))[0])
    # The ratio is computed based on the mean brightness of the L channel for both color images
    ratio_lab = mean_template_lab / mean_eq_image_lab

    # Using RGB color space ----
    # We calculate the mean intensity across all color channels (R, G, B) for both images
    # (template and equalized_image), i.e., the ratio is computed based on the mean intensity
    # across all color channels for both images   
    mean_template_rgb = np.mean(template)
    mean_eq_image_rgb = np.mean(equalized_img)
    # Calculate the ratio of the brightness of the images
    ratio_rgb = mean_template_rgb / mean_eq_image_rgb

    # Using gray images ----
    # Calculate the mean of the grayscale images
    mean_template_gray = np.mean(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    mean_eq_image_gray = np.mean(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY))
    # Calculate the ratio of the brightness of the grayscale images
    ratio_gray = mean_template_gray / mean_eq_image_gray
    
    # ----------------------
    # Calculate brightness, contrast, and sharpness ratios
    #ratio_brightness = np.mean(cv2.split(template_lab)[0]) / np.mean(cv2.split(equalized_img_lab)[0])
    #ratio_contrast = np.std(l_template) / np.std(l_target)
    #ratio_sharpness = np.max(sharpened_template) / np.max(l_target)
    
    # Print ratios
    #print(f'Brightness ratio: {ratio_brightness}')
    #print(f'Contrast ratio: {ratio_contrast}')
    #print(f'Sharpness ratio: {ratio_sharpness}')
    # ----------------------

    # Print brightness ratios ----
    print(f'Brightness ratio (LAB): {ratio_lab}')
    print(f'Brightness ratio (RGB): {ratio_rgb}')
    print(f'Brightness ratio (Grayscale): {ratio_gray}')
    
    # Visualization check ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Painting with equalized brightness')
    axes[1].axis('off')
    plt.show()
    
    return equalized_img

# function to align images ----
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
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

    # compute the homography matrix between the two sets of matched points
    # The homography matrix represents the rotation, translation, and scale
    # to convert (warp) from the plane of our input image to the plane of our
    # template image
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    
    # By creating a modified homography matrix (H_no_rotation) that excludes the
    # rotation component, i.e., preserving only translation, scale, and shear,
    # we can do image alignment while preserving the rotation differences
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
    
    # apply the homography matrix to align the images but without modifying the rotation
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

# Function to look into image differences ----
def compare_images(image_path1, image_path2, result_path = "filled_after.jpg"):
    # Read images
    imageA = image_path1
    imageB = image_path2

    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two gray images
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    print("Image Similarity SSIM: {:.4f}%".format(score * 100))
    #print("SSIM: {}".format(score))
    
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

# --------------------------------------------------------------------------------------------
# load the images to compare ----
print("[INFO] loading images...")
# template to align to
#template = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")
template = cv2.imread("../../data/interim/segments/output_2_Naples_Museo Capodimonte_segment0_person-0.jpg")

#from PIL import Image
#image_path = '../../data/interim/segments_transparant/output_0_Edinburgh_Nat_Gallery_segment0_person-0.png'
#img = Image.open(image_path)
#img.show()
    
# image to align
#image    = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")
image    = cv2.imread("../../data/interim/segments/output_1_London_Nat_Gallery_segment0_person-0.jpg")  

# equaling coloring ----
# The match_histograms function from scikit-image is primarily designed to adjust the color distribution
# of an image to match that of a reference image
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

# equaling level of brightness ----
print("[INFO] equalizing brightness between the images...")
eq_image = equalize_brightness_color(template, matched)
#eq_image = equalize_brightness_contrast_sharpness(template, image, alpha_contrast=0.8, alpha_sharpness=0.7)

# aligning the images ----
print("[INFO] aligning images...")
aligned = align_images(eq_image, template, debug=True) # return the aligned image

# Resize both the aligned and template images so we can visualize them on the screen
aligned  = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# save the resized and aligned images to disk
cv2.imwrite("../../data/interim/aligned/aligned_0_Edinburgh_Nat_Gallery.jpg", template)
cv2.imwrite("../../data/interim/aligned/aligned_1_London_Nat_Gallery.jpg", aligned)

# side-by-side stacked visualization of the output aligned
stacked = np.hstack([template,aligned])
cv2.imshow("Image Alignment Stacked", stacked)
cv2.waitKey(0)

# 2) overlaying the aligned image on the template ----
overlay = template.copy()
output  = aligned.copy()

# transparently blend the two images into a single output image with the pixels
# from each image having equal weight
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

print("This time, we overlay our aligned and registered image on our template with a 50/50 blend")
# show the overlayed images
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed

result = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
# Display the result using Matplotlib
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

#################################################################################################################

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


def align_images_in_directory(template_path, input_directory, output_directory, maxFeatures=500, keepPercent=0.2, debug=False):
    
    # load the template image: to this painting we want to align all other paintings
    template = cv2.imread(template_path)

    # create output directory if not exists
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

# set the paths and parameters
template_path = "../../data/raw/0_Edinburgh_Nat_Gallery.jpg"
input_directory = "../../data/raw/Bridgewater"
output_directory = "../../data/interim/aligned"

# align images in the directory and save them
align_images_in_directory(template_path, input_directory, output_directory, debug=False)
