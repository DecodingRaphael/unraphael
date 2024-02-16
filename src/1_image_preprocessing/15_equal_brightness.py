import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_brightness_color(template, target):
    
    """
    Equalizes the brightness of the target image based on the luminance of the template image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - equalized_img: Adjusted target image with equalized brightness.
    - ratios: Dictionary containing brightness ratios for LAB, RGB, and grayscale.

    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to BGR. The function also calculates
    and returns brightness ratios for LAB, RGB, and grayscale color spaces.
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
    
    # Print brightness ratios ----
    print(f'Brightness ratio (LAB): {ratio_lab}')
    print(f'Brightness ratio (RGB): {ratio_rgb}')
    print(f'Brightness ratio (Grayscale): {ratio_gray}')
    
    # Visualization check ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Painting 1 (template)')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Painting with equalized brightness')
    axes[1].axis('off')
    plt.show()
    
    return equalized_img
        
template = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")
image    = cv2.imread("../../data/interim/no_bg/output_2_Naples_Museo Capodimonte.jpg")

eq_image = equalize_brightness_color(template, image)