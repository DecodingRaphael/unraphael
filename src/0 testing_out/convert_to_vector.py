import cv2
import numpy as np
from matplotlib import pyplot as plt

def convert_to_vector(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)
    
    # plot the original
    plt.imshow(image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find outlines
    edges = cv2.Canny(gray_image, 40, 100)
    plt.imshow(edges)

    # Find contours (shapes) in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new blank image to draw the vectorized image
    vector_image = np.zeros_like(image)

    # Draw contours on the blank image to form vectorized shapes
    cv2.drawContours(vector_image, contours, -1, (255, 255, 255), 1)

    # Save the vectorized image
    cv2.imwrite(output_image_path, vector_image)
    
    plt.imshow(vector_image)

# Sample input and output file paths
input_image_path = "../data/raw/Bridgewater/6_Oxford_Christ_Church.jpg"

output_image_path = "../data/raw/vector_image.jpg"

# Call the function to convert the image to vectors
convert_to_vector(input_image_path, output_image_path)