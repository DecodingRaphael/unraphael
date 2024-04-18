import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# function to normalize brightness ----
# def equalize_brightness_contrast_sharpness(template, target, alpha_contrast=1.0, alpha_sharpness=1.0):
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
        

def normalize_brightness(template, target):
    """
    Equalizes the brightness of the target image based on the luminance of the template image

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
        
    else:  # Both images are grayscale
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

# gray
template = cv2.imread("../../data/interim/no_bg/1_London_Nat_Gallery.jpg")
image    = cv2.imread("../../data/interim/no_bg/2_Naples_Museo Capodimonte.jpg")

# color
template = cv2.imread("../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg")
image    = cv2.imread("../../data/interim/no_background/output_1_London_Nat_Gallery.jpg")

normalized_brightness_img = normalize_brightness(template, image)
normalized_contrast_img = normalize_contrast(template, image)
normalized_sharpness_img = normalize_sharpness(template, image)
normalized_color_img = normalize_colors(template, image)

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

normed = normalize_image(template, image)