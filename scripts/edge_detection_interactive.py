import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def callback(x):
    print('Slider value is: ', x)


img = cv2.imread('./data/raw/Bridgewater/0_Edinburgh_Nat_Gallery.jpg', 0) #read image as grayscale

canny = cv2.Canny(img, 85, 255)

cv2.namedWindow('image')  # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 1000, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 1000, callback) #upper threshold trackbar for window 'image

while (True):
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break

    lowerThreshold = cv2.getTrackbarPos('L', 'image')
    upperThreshold = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, lowerThreshold, upperThreshold)

cv2.destroyAllWindows()
