import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        
#template = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")
#image    = cv2.imread("../../data/interim/no_bg/output_2_Naples_Museo Capodimonte.jpg")

#eq_image = equalize_brightness_color(template, image)