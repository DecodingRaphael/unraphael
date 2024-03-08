
import cv2
import numpy as np

# Callback function for trackbar changes
def nothing(x):
    pass

print("Contour detection parameters")
print("Use sliders to set lower and upper threshols.")

# Load the image
image = cv2.imread("../../data/raw/0_Edinburgh_Nat_Gallery.jpg")
assert image is not None, "file could not be read, check with os.path.exists()"

window_title   = '0_Edinburgh_Nat_Gallery.jpg'

# Create a window to display the original image
cv2.namedWindow(window_title) 

# Create trackbars for parameters
cv2.createTrackbar('Blur', window_title, 1, 10, nothing)
cv2.createTrackbar('Threshold', window_title, 0, 255, nothing)
cv2.createTrackbar('Min Area', window_title, 0, 5000, nothing)

# Trackbars for image enhancement
cv2.createTrackbar('Histogram Equalization', window_title, 0, 100, nothing)
cv2.createTrackbar('Dilation Kernel Size', window_title, 1, 10, nothing)
cv2.createTrackbar('Erosion Kernel Size', window_title, 1, 10, nothing)

# Trackbar for contour retrieval mode
cv2.createTrackbar('Contour Retrieval Mode', window_title, 0, 1, nothing)


while True:
      
    k = cv2.waitKey(1) & 0xFF
    
    # Break the loop if 'Esc' key is pressed
    if k == 27:  # escape key
        break       
    
    # Get trackbar positions
    blur_value       = cv2.getTrackbarPos('Blur', window_title)
    threshold_value  = cv2.getTrackbarPos('Threshold', window_title)
    min_contour_area = cv2.getTrackbarPos('Min Area', window_title)

    # Get trackbar positions for image enhancement
    apply_histogram_equalization = cv2.getTrackbarPos('Histogram Equalization', window_title) / 100.0
        
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization with adjustable strength
    if apply_histogram_equalization > 0:
        clahe = cv2.createCLAHE(clipLimit=apply_histogram_equalization, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply GaussianBlur based on trackbar value
    blurred = cv2.GaussianBlur(gray, (2 * blur_value + 1, 2 * blur_value + 1), 0)

    # Use a suitable thresholding method to create a binary image
    _, thresh    = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image; RETR_EXTERNAL retrieves only the extreme outer contours
    contours, hierarchy = cv2.findContours(thresh, 
                                           cv2.RETR_EXTERNAL if cv2.getTrackbarPos('Contour Retrieval Mode', window_title) == 0 else cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # set to cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy.
    # The choice of retrieval mode can influence the contours you obtain.
    
    # Filter contours based on area to keep only larger contours (human-sized)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create an empty black image for contours
    contour_image = np.zeros_like(gray)

    # Draw the filtered contours on the black image
    cv2.drawContours(contour_image, filtered_contours, -1, (255), thickness = 2)  # Use thickness=2 for thin lines
    #cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

    # Bitwise AND operation to extract the regions with contours from the original image
    result = cv2.bitwise_and(image, image, mask=contour_image)   
    
    # Apply dilation and erosion
    dilation_kernel_size = cv2.getTrackbarPos('Dilation Kernel Size', window_title)
    erosion_kernel_size = cv2.getTrackbarPos('Erosion Kernel Size', window_title)

    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)

    result = cv2.dilate(result, dilation_kernel, iterations=1)
    result = cv2.erode(result, erosion_kernel, iterations=1)
    
    # to display image side by side
    numpy_horizontal_concat = np.concatenate((image, result), axis=1)
    cv2.imshow(window_title, numpy_horizontal_concat)
       
    if k == ord('s'):  # press 's' to save the image
        result_filename = 'contour_result.jpg'
        cv2.imwrite(result_filename, result)
        print(f"Result saved as {result_filename}")
    
cv2.destroyAllWindows()

print("Selected values:")
print("lowerThreshold: " , lowerThreshold)
print("Upper threshold: ", upperThreshold)
print("aperture Size: "  , apertureSize)
