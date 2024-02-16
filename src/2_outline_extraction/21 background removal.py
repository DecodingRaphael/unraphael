
# This script removes the background of each image in the input folder and saves the output images in 
# the output folder. The background removal is done using the rembg library, with the background set 
# to transparant

# https://www.youtube.com/watch?v=Z3pP1GuQe8g
# https://www.horilla.com/blogs/how-to-remove-the-background-of-image-using-rembg-in-python/

import os
from rembg import remove
from PIL import Image

# Specify the input folder containing images
input_folder = "../../data/raw/Bridgewater"

# Create the output folder if it doesn't exist
output_folder = "../../data/interim/no_bg"
os.makedirs(output_folder, exist_ok = True)

# Iterate through each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png')):  # Check if the file is an image
        # Construct the full path for input and output
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"output_{filename}")

        # Open the input image
        input_image = Image.open(input_path)

        # Use rembg to remove the background
        output_image = remove(input_image,alpha_matte=True, background_color=(0, 0, 0),
                              alpha_matting_foreground_threshold=240,
                              alpha_matting_background_threshold=10,
                              alpha_matting_erode_structure_size=10,
                              alpha_matting_base_size=1000,)

        # Convert the image to RGB mode if it's in RGBA mode
        #if output_image.mode == 'RGBA':
        #    output_image = output_image.convert('RGB')

        # Convert the image to PNG format
        output_image = output_image.convert('RGBA')
        # Save the output image to the output folder
        output_image.save(output_path, format='PNG')  # Specify the format as PNG)

print("Background removal completed. Output images are saved in the 'no_bg' folder.")

#######################################################################################################

import cv2
from matplotlib import pyplot as plt

# Open the input image
        input_image = Image.open("../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg")
        image = cv2.imread("../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg")

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
input_image_path = "../../data/interim/no_background/output_0_Edinburgh_Nat_Gallery.jpg"
input_image_path = "../data/raw/baby.jpg"
output_image_path_advanced = "output_image_advanced.png"

# Remove background with advanced options
remove_background_advanced(input_image_path, output_image_path_advanced, 
                           alpha_matte=True, background_color=(0, 0, 0))

# Display the results
removed_background_image_advanced = Image.open(output_image_path_advanced)
removed_background_image_advanced.show(title="Image with Removed Background (Advanced)")

