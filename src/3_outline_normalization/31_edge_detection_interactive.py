# https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    pass

print("Canny edge detection parameters")
print("Use sliders to set lower and upper threshold, and apertureSize.")

image_filename = '../../data/raw/Bridgewater/1_London_Nat_Gallery.jpg'
window_title   = '1_London_Nat_Gallery.jpg'

img = cv2.imread(image_filename, 0)  # read image as grayscale
plt.imshow(img, cmap = 'gray')

canny = cv2.Canny(img, 85, 255)

cv2.namedWindow(window_title)  # make a window with name 'image'
cv2.createTrackbar('L', window_title, 0, 5000, callback)  # lower threshold trackbar for window 'image
cv2.createTrackbar('U', window_title, 0, 5000, callback)  # upper threshold trackbar for window 'image
cv2.createTrackbar('a', window_title, 0, 2, callback)  # aperture size 3, 5, 7

while (True):
    # to display image side by side
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1)
    cv2.imshow(window_title, numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break

    lowerThreshold = cv2.getTrackbarPos('L', window_title)
    upperThreshold = cv2.getTrackbarPos('U', window_title)
    apertureSize   = cv2.getTrackbarPos('a', window_title) * 2 + 3

    canny = cv2.Canny(img, lowerThreshold, upperThreshold, apertureSize = apertureSize)
    
    if k == ord('s'):  # press 's' to save the image
        result_filename = 'canny_result.jpg'
        cv2.imwrite(result_filename, canny)
        print(f"Result saved as {result_filename}")
    
cv2.destroyAllWindows()

print("Selected values:")
print("lowerThreshold: ", lowerThreshold)
print("Upper threshold: ", upperThreshold)
print("aperture Size: ", apertureSize)
