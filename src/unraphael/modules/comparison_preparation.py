"""Module for preparing correct and accurate comparison between human figures in paintings.
"""

# IMPORTS ----
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
from skimage.exposure import match_histograms



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

def align_all_images_in_folder_to_template(template_path, input_directory, output_directory, maxFeatures=500, keepPercent=0.2, debug=False):
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

            # align the images by applying the alignment function
            aligned = align_images(image, template, maxFeatures, keepPercent, debug)

            # save the aligned image in the output directory
            output_path = os.path.join(output_directory, f"aligned_{filename}")
            cv2.imwrite(output_path, aligned)
            print(f"[INFO] Image {filename} aligned and saved to {output_path}")


def normalize_brightness(template, target):
    """
    Normalizes the brightness of the target image based on the luminance of the template 
    image. This refers to the process of bringing the brightness of the target image
    into alignment with the brightness of the template image. This can help ensure 
    consistency in brightness perception between the two images, which is particularly
    useful in applications such as image comparison, enhancement, or blending.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.
    .

    Returns:
    - equalized_img: Adjusted target image with equalized brightness
    
    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to BGR. 
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
    
    # Merge LAB channels back for the adjusted target image
    equalized_img_lab = cv2.merge([l_target, a_target, b_target])

    # Convert the adjusted LAB image back to BGR
    equalized_img = cv2.cvtColor(equalized_img_lab, cv2.COLOR_LAB2BGR)
    
    ## Using LAB color space
    # we convert the images (template and eq_image) to the LAB color space and calculate 
    # the mean brightness from the luminance channel (L) only

    # Calculate the mean of the color images from the luminance channel
    mean_template_lab = np.mean(cv2.split(template_lab)[0])
    mean_eq_image_lab = np.mean(cv2.split(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2LAB))[0])
    
    # The ratio is computed based on the mean brightness of the L channel for both color images
    ratio_lab = mean_template_lab / mean_eq_image_lab

    ## Using RGB color space
    # We calculate the mean intensity across all color channels (R, G, B) for both images
    # (template and equalized_image), i.e., the ratio is computed based on the mean intensity
    # across all color channels for both images 
    mean_template_rgb = np.mean(template)
    mean_eq_image_rgb = np.mean(equalized_img)
    # Calculate the ratio of the brightness of the images
    ratio_rgb = mean_template_rgb / mean_eq_image_rgb

    ## Using gray images
    # Calculate the mean of the grayscale images
    mean_template_gray = np.mean(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    mean_eq_image_gray = np.mean(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY))
    # Calculate the ratio of the brightness of the grayscale images
    ratio_gray = mean_template_gray / mean_eq_image_gray
    
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

def normalize_contrast(template, target):
    """
    Normalize the contrast of the target image to match the contrast of the template image.

    Parameters:
    - template: Reference image (template) in BGR or grayscale format.
    - target: Target image to be adjusted in BGR or grayscale format.

    Returns:
    - normalized_img: Target image with contrast normalized to match the template image.
    """
    if len(template.shape) == 3 and len(target.shape) == 3:  # Both images are color
        # Convert images to LAB color space
        template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # Calculate contrast metric (standard deviation) for L channel of both images
        std_template = np.std(template_lab[:, :, 0])
        std_target = np.std(target_lab[:, :, 0])

        # Adjust contrast of target image to match template image
        l_target = (target_lab[:, :, 0] * (std_template / std_target)).clip(0, 255).astype(np.uint8)
        normalized_img_lab = cv2.merge([l_target, target_lab[:, :, 1], target_lab[:, :, 2]])

        # Convert the adjusted LAB image back to BGR
        normalized_img = cv2.cvtColor(normalized_img_lab, cv2.COLOR_LAB2BGR)

        # Print contrast values
        print(f'Contrast value (template): {std_template}')
        print(f'Contrast value (target): {std_target}')
        print(f'Contrast ratio: {std_template / std_target}')
        print(f'Adapted value (target): {np.std(normalized_img_lab[:, :, 0])}')
        
    else:  # when both images are grayscale
        # Calculate contrast metric (standard deviation) for grayscale intensity of both images
        std_template = np.std(template)
        std_target = np.std(target)

        # Adjust contrast of target image to match template image
        normalized_img = (target * (std_template / std_target)).clip(0, 255).astype(np.uint8)
        
        # Print contrast values
        print(f'Contrast value (template): {std_template}')
        print(f'Contrast value (target): {std_target}')
        print(f'Contrast ratio: {std_template / std_target}')
        print(f'Adapted value (target): {np.std(normalized_img_lab[:, :, 0])}')

    # Visualization check ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Painting with equalized contrast')
    axes[1].axis('off')
    plt.show()   
            
    return normalized_img

def normalize_sharpness(template, target):
    """
    Normalize the sharpness of the target image to match the sharpness of the template image.

    Parameters:
    - template: Reference image (template) in BGR or grayscale format.
    - target: Target image to be adjusted in BGR or grayscale format.

    Returns:
    - normalized_img: Target image with sharpness normalized to match the template image.
    """
    if len(template.shape) == 3 and len(target.shape) == 3:  # Both images are color
        # Convert images to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
        target_gray = target

    # Calculate image gradients for both images
    grad_x_template = cv2.Sobel(template_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_template = cv2.Sobel(template_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_template = cv2.magnitude(grad_x_template, grad_y_template)
    grad_x_target = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_target = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_target = cv2.magnitude(grad_x_target, grad_y_target)

    # Calculate sharpness metric (mean gradient magnitude) for both images
    mean_grad_template = np.mean(grad_template)
    mean_grad_target = np.mean(grad_target)

    # Adjust sharpness of target image to match template image
    normalized_img = (target * (mean_grad_template / mean_grad_target)).clip(0, 255).astype(np.uint8)

    # Print sharpness values
    print(f'Sharpness value (template): {mean_grad_template}')
    print(f'Sharpness value (target): {mean_grad_target}')
    print(f'Sharpness ratio: {mean_grad_template / mean_grad_target}')

    # Calculate sharpness value for the normalized image
    grad_x_normalized = cv2.Sobel(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
    grad_y_normalized = cv2.Sobel(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
    grad_normalized = cv2.magnitude(grad_x_normalized, grad_y_normalized)
    mean_grad_normalized = np.mean(grad_normalized)
    print(f'Sharpness value (normalized): {mean_grad_normalized}')

    # Visualization check
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(template_gray, cmap='gray')
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY), cmap='gray')
    axes[1].set_title('Image with normalized sharpness')
    axes[1].axis('off')
    plt.show()

    return normalized_img

def normalize_colors(template, target):
    """
    Normalize the colors of the target image to match the color distribution of the template image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - normalized_img: Target image with colors normalized to match the template image.
    """
    matched = match_histograms(target, template, channel_axis=-1)

    # Visualization check ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Original Target')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Target (adapted colors)')
    axes[2].axis('off')
    plt.show()

    return matched

def normalize_image(template, image):
    """
    Normalize the brightness, contrast, sharpness, and color of the target image 
    to match the characteristics of the template image.

    Parameters:
    - template: Reference image (template) in BGR format.
    - image: Target image to be adjusted in BGR format.

    Returns:
    - normalized_img: Target image with normalized characteristics.
    """
    # Step 1: Normalize brightness
    normalized_brightness_img = normalize_brightness(template, image)
    
    # Step 2: Normalize contrast
    normalized_contrast_img = normalize_contrast(template, normalized_brightness_img)
    
    # Step 3: Normalize sharpness
    normalized_sharpness_img = normalize_sharpness(template, normalized_contrast_img)
    
    # Step 4: Normalize colors
    normalized_color_img = normalize_colors(template, normalized_sharpness_img)
    
    return normalized_color_img
