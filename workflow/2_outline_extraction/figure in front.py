import cv2
import numpy as np

# Reading the Image
image = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the threshold value (adjust as needed)
threshold_value = 105

# Apply threshold to create binary mask
_, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest contour
largest_contour_mask = np.zeros_like(binary_mask)
cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Apply the mask to the original image to extract the figure in front
result_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)

# Display the result
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


##########

import cv2
import numpy as np

# Callback function for the trackbar
def on_trackbar(value):
    global threshold_value
    threshold_value = value
    update_mask()

# Function to update the mask based on the current threshold_value
def update_mask():
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    largest_contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image to extract the figure in front
    result_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)

    # Display the result
    cv2.imshow("Result", result_image)

# Reading the Image
image = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the threshold_value
threshold_value = 128

# Create a window to display the result
cv2.namedWindow("Result")

# Create a trackbar
cv2.createTrackbar("Threshold", "Result", threshold_value, 255, on_trackbar)

# Initial update of the mask
update_mask()

# Wait for a key event
cv2.waitKey(0)
cv2.destroyAllWindows()





###############################

import cv2
import numpy as np

# Callback function for the trackbar
def on_trackbar(value):
    global threshold_value
    threshold_value = value
    update_mask()

# Function to update the mask based on the current threshold_value
def update_mask():
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply color thresholding on the 'a' channel
    _, color_mask = cv2.threshold(lab_image[:, :, 1], threshold_value, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Combine color mask and edges
    combined_mask = cv2.bitwise_or(color_mask, edges)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    largest_contour_mask = np.zeros_like(combined_mask)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image to extract the figure in front
    result_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)

    # Display the result
    cv2.imshow("Result", result_image)

# Reading the Image
image = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the threshold_value
threshold_value = 128

# Create a window to display the result
cv2.namedWindow("Result")

# Create a trackbar
cv2.createTrackbar("Threshold", "Result", threshold_value, 255, on_trackbar)

# Initial update of the mask
update_mask()

# Wait for a key event
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################################

import cv2
import numpy as np

def separate_figures(image, k_value):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # Convert to float type for k-means
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def on_trackbar(value):
    global img_gray1, contours2

    # Apply blurring
    blur_value = cv2.getTrackbarPos('Blur', 'Segmented Image with Contours')
    if blur_value % 2 == 0:  # Ensure that the blur kernel size is odd
        blur_value += 1

    # Get threshold value from trackbar
    threshold_value = cv2.getTrackbarPos('Threshold', 'Segmented Image with Contours')

    # Get k-means parameters from trackbar
    k_value = cv2.getTrackbarPos('K', 'Segmented Image with Contours')

    # Apply thresholding
    ret, thresh1 = cv2.threshold(img_gray1, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Separate figures based on color using k-means
    color_segmented_image = separate_figures(image1, k_value)

    # Draw contours on the segmented image
    for contour in contours2:
        cv2.drawContours(color_segmented_image, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Segmented Image with Contours', color_segmented_image)

# Load image
image1 = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")
img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create window
cv2.namedWindow('Segmented Image with Contours')

# Create trackbars for threshold, blurring, and k-means parameters
cv2.createTrackbar('Threshold', 'Segmented Image with Contours', 150, 255, on_trackbar)
cv2.createTrackbar('Blur', 'Segmented Image with Contours', 5, 20, on_trackbar)
cv2.createTrackbar('K', 'Segmented Image with Contours', 2, 10, on_trackbar)

# Initialize trackbar callback
on_trackbar(150)

# Wait for a key event
key = cv2.waitKey(0)

# Save image if 's' key is pressed
if key == ord('s'):
    cv2.imwrite('contour_and_segmented_image.jpg', color_segmented_image)

# Cleanup
cv2.destroyAllWindows()
