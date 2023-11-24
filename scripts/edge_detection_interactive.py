import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def callback(x):
    pass


print("Canny edge detection parameters")
print("Use sliders to set lower and upper threshold, and apertureSize.")

img = cv2.imread('./data/raw/Bridgewater/0_Edinburgh_Nat_Gallery.jpg', 0) #read image as grayscale

canny = cv2.Canny(img, 85, 255)

cv2.namedWindow('image')  # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 5000, callback)  # lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 5000, callback)  # upper threshold trackbar for window 'image
cv2.createTrackbar('a', 'image', 0, 2, callback)  # aperture size 3, 5, 7

while (True):
    # to display image side by side
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1)
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break

    lowerThreshold = cv2.getTrackbarPos('L', 'image')
    upperThreshold = cv2.getTrackbarPos('U', 'image')
    apertureSize = cv2.getTrackbarPos('a', 'image') * 2 + 3

    canny = cv2.Canny(img, lowerThreshold, upperThreshold, apertureSize=apertureSize)

cv2.destroyAllWindows()

print("Selected values:")
print("lowerThreshold: ", lowerThreshold)
print("Upper threshold: ", upperThreshold)
print("aperture Size: ", apertureSize)
