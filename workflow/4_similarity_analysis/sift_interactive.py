import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

image_filename1 = '../data/raw/Bridgewater/0_Edinburgh_Nat_Gallery.jpg'
image_filename2 = '../data/raw/Bridgewater/1_London_Nat_Gallery.jpg'

window_title   = '1_London_Nat_Gallery.jpg'

img1 = cv.imread(image_filename1, 0)  # read image as grayscale
img2 = cv.imread(image_filename2, 0)  # read image as grayscale

plt.imshow(img1, cmap = 'gray')
plt.imshow(img2, cmap = 'gray')

# Initiate SIFT detector
sift = cv.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(img1, None)
kp_2, desc_2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann         = cv.FlannBasedMatcher(index_params, search_params)


def nothing(x):
   pass

# create trackbars for color change
cv.namedWindow(window_title)
cv.createTrackbar('R',window_title, 0, 5000, nothing)
cv.createTrackbar('G',window_title, 0, 5000, nothing)
cv.createTrackbar('B',window_title, 0, 5000, nothing)

while(1):
   # to display image side by side
    #numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
    #cv.imshow(window_title, numpy_horizontal_concat)
    result = cv.drawMatches(img1, kp_1, img1, kp_2, good_points[0:100], None)
    k = cv.waitKey(1) & 0xFF
    if k == 27:  # escape key
        break
    
   # get current positions of four trackbars
    r = cv.getTrackbarPos('R',window_title)
    g = cv.getTrackbarPos('G',window_title)
    b = cv.getTrackbarPos('B',window_title)

    # Flann
    matches = flann.knnMatch(desc_1, desc_2, k = 2)
    
    good_points = []
    
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_points.append(m)
            
    
       
        

   
cv.destroyAllWindows()

print("Selected values:")
print("lowerThreshold: ", r)
print("Upper threshold: ", g)
print("aperture Size: ", b)