"""Module for extracting contours from aligned figures."""

# IMPORTS ----
from __future__ import annotations

import os

import cv2
import numpy as np


def extract_colored_outer_contour(input_folder, output_folder):
    """Extracts colored outer contours from binary images in the input folder
    and saves the results in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing the binary images.
    - output_folder (str): Path to the folder where the colored outer contour images will be saved.

    Returns:
    Images with different colored contours in a separate folder
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each binary image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
            # Read the binary image
            binary_path = os.path.join(input_folder, filename)
            binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

            # Use a suitable thresholding method to create a binary image
            _, thresh = cv2.threshold(binary_image, 20, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image; RETR_EXTERNAL retrieves only the extreme outer contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area to keep only larger contours (human-sized)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25]

            # Create an empty black image for contours
            contour_image = np.zeros_like(binary_image)

            # Convert the black image to RGB
            contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2RGB)

            # Draw the filtered contours on the RGB image
            for i, contour in enumerate(filtered_contours):
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.drawContours(contour_image_rgb, [contour], -1, color, thickness=2)

            # Save the result
            output_path = os.path.join(output_folder, f'colored_outer_contour_{filename}')
            cv2.imwrite(output_path, contour_image_rgb)

            # Bitwise AND operation to extract the regions with contours from the original image
            # result = cv2.bitwise_and(binary_image, binary_image, mask=contour_image)
            # Save the result
            # output_path = os.path.join(output_folder, f"outer_contour_{filename}")
            # cv2.imwrite(output_path, result)


def get_contour_color(image):
    """Get the color of the contour in the given image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: The contour color as a tuple of RGB values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_color = tuple(map(int, cv2.mean(image, mask=thresh)[:3]))
    return contour_color


def overlaying_images(base_image_path, overlay_images, scale_factor=1.50):
    """Overlay multiple images on a base image and display the result.

    Args:
        base_image_path (str): The file path of the base image.
        overlay_images (list): A list of file paths of the overlay images.
        scale_factor (float, optional): The scaling factor for the output image. Defaults to 1.50.
    """

    # Read the base image
    base_image = cv2.imread(base_image_path)

    # Check if the base image is valid
    if base_image is None:
        print(f'Error: Unable to read {base_image_path}')
        return

    # Initialize the output image with the base image
    output = base_image.copy()

    # Get the title of the base image
    base_title = os.path.basename(base_image_path).split('.')[0]

    # Add the base image title to the legend
    legend_text_base = f'{base_title}'
    cv2.putText(
        output,
        legend_text_base,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Initialize y-coordinate for legend text
    text_y = 60

    # Iterate over each overlay image and blend it with the output
    for overlay_path in overlay_images:
        overlay = cv2.imread(overlay_path)

        # Check if the overlay image is valid
        if overlay is not None:
            output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

            # Get contour color from the overlay image
            contour_color = get_contour_color(overlay)

            # Add legend text with contour color above the contours
            legend_text = os.path.basename(overlay_path).split('.')[
                0
            ]  # Extract text before the file extension
            cv2.putText(
                output,
                legend_text,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                contour_color,
                2,
                cv2.LINE_AA,
            )

            # Increment y-coordinate for next legend text
            text_y += 30
        else:
            print(f'Error: Unable to read {overlay_path}')

    # Resize the output image
    output = cv2.resize(output, None, fx=scale_factor, fy=scale_factor)

    cv2.imshow('Overlayed Images', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
