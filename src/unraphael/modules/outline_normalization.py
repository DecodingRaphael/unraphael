"""Module for preparing accurate comparison between human figures in paintings by aligning images
to a selected baseline image.
"""

# IMPORTS ----
import math
import numpy as np
import cv2
from IPython.display import display, HTML
#from __future__ import print_function
#from __future__ import division
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from matplotlib import animation
from IPython.display import display, HTML
from numpy.fft import fft2, ifft2, fftshift
import diplib as dip

def rotationAlign(im1, im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    height, width = im1_gray.shape[0:2]
    
    values = np.ones(360)
    
    for i in range(0,360):
      rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), i, 1)
      rot = cv2.warpAffine(im2_red, rotationMatrix, (width, height))
      values[i] = np.mean(im1_gray - rot)
    
    rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), np.argmin(values), 1)
    rotated = cv2.warpAffine(im2, rotationMatrix, (width, height))
    
    return rotated

#(ORB) feature based alignment
def featureAlign(image, template, maxFeatures = 50000, keepPercent = 0.15):
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
    # convert both the input image and template to grayscale
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
    # The homography matrix represents the rotation, translation, and scale
    # to convert (warp) from the plane of our input image to the plane of our
    # template image.
    (H, mask) = cv2.findHomography(ptsA, ptsB, method = cv2.RANSAC)
    
    # By creating a modified homography matrix (H_no_rotation) that excludes the
    # rotation component, i.e., preserving only translation, scale, and shear,
    # we can do image alignment while preserving the differences in rotation.
    H_no_rotation = np.array([[H[0, 0], H[0, 1], H[0, 2]],
                              [H[1, 0], H[1, 1], H[1, 2]],
                              [0, 0, 1]])
    
    ## derive rotation angle between figures from the homography matrix
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
    print(f'Rotational degree: {theta:.2f}') # rotation angle, in degrees
        
    # apply the homography matrix to align the images, including the rotation
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    
    # apply the homography matrix to align the images, without modifying the rotation
    #aligned = cv2.warpPerspective(image, H_no_rotation, (w, h)) 
    
    return aligned
    
# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(im1,im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY) # template image
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY) # im2 is image to be warped to match im1

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the ECC number of iterations
    number_of_iterations = 5000
    
    # Specify the ECC threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-3
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        # Warp im2 using affine
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine:
        # Warp im2 using affine 
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Print rotation angle
    row1_col0 = warp_matrix[0,1]
    angle = math.degrees(math.asin(row1_col0))
    print(angle)
    
    # Side-by-side stacked visualization of the aligned output
    stacked = np.hstack([im1,im2_aligned])
    cv2.imshow("Image Alignment Stacked", stacked)
    cv2.waitKey(0)

    return im2_aligned

# FFT phase correlation
def translation(im0, im1):
    
    # Convert images to grayscale
    im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]

def fourier_mellin_transform_match(image1_path, image2_path):
    """
    Apply Fourier-Mellin transform to match one image to another.

    Args:
    - image1_path (str): File path to the first input image.
    - image2_path (str): File path to the second input image.

    Returns:
    - out_matrix (numpy.ndarray): Output matrix after applying Fourier-Mellin transform.
    """
    # Load the two images
    img1 = dip.ImageRead(image1_path)
    img2 = dip.ImageRead(image2_path)

    # They're gray-scale images, even if the JPEG file has RGB values
    img1 = img1(1)  # just keep the green channel
    img2 = img2(1)

    # They need to be the same size
    out_size = np.minimum(img1.Sizes(), img2.Sizes())
    img1.Crop(out_size)
    img2.Crop(out_size)

    # Apply Fourier-Mellin to transform one image to match the other
    out = dip.Image()
    matrix = dip.FourierMellinMatch2D(img1, img2, out=out, correlationMethod="don't normalize")
    
    #stacked = np.hstack([img1,out])
    #cv2.imshow("Image Alignment Stacked", stacked)
    #cv2.waitKey(0)
    
    aligned_image = out

    return aligned_image

def visualize_alignment(template, aligned):
    
    # side-by-side comparison of the template (left) and the aligned image (right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template')
    axes[0].axis('off')
        
    axes[1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Aligned image')
    axes[1].axis('off')        
    plt.show()

def align_all_images_in_folder_to_template(base_image_path, input_files):
   
    # load the base image to which we want to align all the other images
    template = cv2.imread(base_image_path)

    # list to store aligned images
    aligned_images = []

    # loop over all images in the input directory
    for filename in input_files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # image to be aligned            
            image = cv2.imread(filename)
            
            # Switch between alignment modes
            if   mode == "feature":                   
                aligned = featureAlign(image, template)
      
            elif mode == "fourier":
                aligned = fourier_mellin_transform_match(image, template)
                
            elif mode == "ecc":
                aligned = eccAlign(image, template)

            elif mode == "rotation":
                aligned = rotationAlign(image, template)
                      
            else:
                warp_matrix = translation(image, template)
            
            # append filename and aligned image to list
            aligned_images.append((filename, aligned))
    return aligned_images


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

# ------------------------------------------------------------------------------
# Animation of alignment
def load_images(image_path1, image_path2):
    painting1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    painting2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return painting1, painting2

def resize_images(img1, img2):
    common_size = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
    painting1_resized = cv2.resize(img1, common_size)
    painting2_resized = cv2.resize(img2, common_size)
    return painting1_resized, painting2_resized

def detect_and_match_features(img1, img2):
    
    orb = cv2.ORB_create()
    
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
    
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descriptors1, descriptors2, None)

    # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * 0.2)
    matches = matches[:keep]
    
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")    
    
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images map to each other
        ptsA[i] = keypoints1[m.queryIdx].pt
        ptsB[i] = keypoints2[m.trainIdx].pt
    
    return ptsA, ptsB

def estimate_homography(points1, points2):
    (H, mask) = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H

def rotational_degree(H):       
    ## derive rotation angle between figures from the homography matrix
    theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
    print(f'Rotational degree: {theta:.2f}') # rotation angle, in degrees
    return theta
    
def animate_images(painting1_resized, painting2_resized, H, num_frames=100):
    fig, ax = plt.subplots()
    blended_image = cv2.addWeighted(painting1_resized, 1, painting2_resized, 0, 0)
    im = ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB), extent=[0, blended_image.shape[1], 0, blended_image.shape[0]])

    def update(frame):
        alpha = frame / num_frames
        #interpolated_H = (1 - alpha) * H + alpha * np.eye(3)
        (h, w) = painting2_resized.shape[:2]
        blended_image = cv2.warpPerspective(painting1_resized, H, (w,h))
        blended_image = cv2.addWeighted(blended_image, 1 - alpha, painting2_resized, alpha, 0)
        im.set_array(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {frame + 1}/{num_frames}')
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=25, repeat=True, repeat_delay=1000)
    display(HTML(ani.to_jshtml()))
    plt.close(fig)  # Close the figure after animation

def main(image_path1, image_path2):
    painting1, painting2 = load_images(image_path1, image_path2)
    painting1_resized, painting2_resized = resize_images(painting1, painting2)
    points1, points2 = detect_and_match_features(painting1_resized, painting2_resized)
    H = estimate_homography(points1, points2)
    animate_images(painting1_resized, painting2_resized, H)