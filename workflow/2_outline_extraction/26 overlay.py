# LIBRARIES ----
import numpy as np
import imutils
import cv2
import os
import matplotlib.pyplot as plt

print("[INFO] loading outlines...")

# outline edinburgh
original = cv2.imread("../../data/interim/outlines/0_Edinburgh_Nat_Gallery.jpg")
copy1 = cv2.imread("../../data/interim/outlines/1_London_Nat_Gallery.jpg")

overlay = original.copy()
output  = copy1.copy()

# transparently blend the two images into a single output image with the pixels
# from each image having equal weight
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

print("We overlay the copy on the original from Edinburgh with a 50/50 blend")
# show the overlayed images
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed

# using trackbars to examine the alignment   
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


# overlaying multiple images on a base image ----
# Super-impose a number of these outlines (taken from different paintings) on a 
# black background so we can analyze them 

# Function to get the contour color from an image
def get_contour_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_color = tuple(map(int, cv2.mean(image, mask=thresh)[:3]))
    return contour_color

# Read the base image

base_image_path = "../../data/interim/outlines/0_Edinburgh_Nat_Gallery.jpg"
base_image = cv2.imread(base_image_path)

#base_image = cv2.imread("../../data/interim/outlines/0_Edinburgh_Nat_Gallery.jpg")

# List of overlay images
overlay_images = [#"../../data/interim/outlines/1_London_Nat_Gallery.jpg",
                  #"../../data/interim/outlines/2_Naples_Museo Capodimonte.jpg",
                  #"../../data/interim/outlines/3_Milan_private.jpg",
                  #"../../data/interim/outlines/4_Oxford_Ashmolean.jpg",
                  #"../../data/interim/outlines/5_UK_Nostrell Priory.jpg",
                  "../../data/interim/outlines/6_Oxford_Christ_Church.jpg",
                  #"../../data/interim/outlines/8_London_OrderStJohn.jpg"
                  ]
                  
# Initialize the output image with the base image
output = base_image.copy()

# Get the title of the base image
base_title = os.path.basename(base_image_path).split(".")[0]

# Iterate over each overlay image and blend it with the output
for i, overlay_path in enumerate(overlay_images):
    overlay = cv2.imread(overlay_path)
    output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
    
    # Get contour color from the overlay image
    contour_color = get_contour_color(overlay)

    # Add legend text with contour color
    legend_text = os.path.basename(overlay_path).split(".")[0]  # Extract text before the file extension

    cv2.putText(output, legend_text, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, contour_color, 2, cv2.LINE_AA)


# Display the result
cv2.imshow("Overlayed Images", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


#######################################################

import cv2
import os
import numpy as np

def get_contour_color(image):
    # Assuming the image has colored contours, get the color from the center pixel
    center_pixel = image[image.shape[0] // 2, image.shape[1] // 2]
    return tuple(center_pixel)

def overlay_images(base_image_path, overlay_images):
    # Read the base image
    base_image = cv2.imread(base_image_path)

    # Check if the base image is valid
    if base_image is None:
        print(f"Error: Unable to read {base_image_path}")
        return

    # Initialize the output image with the base image
    output = base_image.copy()

    # Get the title of the base image
    base_title = os.path.basename(base_image_path).split(".")[0]

    # Add the base image title to the legend
    legend_text_base = f"Base Image: {base_title}"
    cv2.putText(output, legend_text_base, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Iterate over each overlay image and blend it with the output
    for i, overlay_path in enumerate(overlay_images):
        overlay = cv2.imread(overlay_path)

        # Check if the overlay image is valid
        if overlay is not None:
            output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

            # Get contour color from the overlay image
            contour_color = get_contour_color(overlay)

            # Extract legend text from the filename
            legend_text = os.path.basename(overlay_path).split(".")[0]  # Extract text before the file extension

            # Add legend text with contour color
            cv2.putText(output, legend_text, (20, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, contour_color, 2, cv2.LINE_AA)
        else:
            print(f"Error: Unable to read {overlay_path}")

    # Display the result
    cv2.imshow("Overlayed Images", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
base_image_path = "../../data/interim/outlines/0_Edinburgh_Nat_Gallery.jpg"
overlay_images_list = [#"../../data/interim/outlines/1_London_Nat_Gallery.jpg",
                  "../../data/interim/outlines/2_Naples_Museo Capodimonte.jpg",
                  "../../data/interim/outlines/3_Milan_private.jpg",
                  "../../data/interim/outlines/4_Oxford_Ashmolean.jpg",
                  "../../data/interim/outlines/5_UK_Nostrell Priory.jpg"]


overlay_images(base_image_path, overlay_images_list)
