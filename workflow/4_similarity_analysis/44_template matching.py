import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

# read images as grayscale
image1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0) # loads the image in grayscale
image2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)

# resize image so that they are equal
(H, W) = image1.shape

image2 = cv2.resize(image2, (W, H))

# double check dimensions
print(image1.shape, image2.shape)

# Apply Canny edge detection to find outlines
edges1 = cv2.Canny(image1, 80, 500) # we use this as the template

# we need to have the contour or form of - for example - the baby
# and use that form as the template to look for in the other images

# contour of forms is possible (compare wit canny edge detection))

 

# plot images
plt.imshow(edges1)

img2 = image2.copy()
w, h = edges1.shape[::-1]

# binarized template ----
#ret,template = cv2.threshold(edges1, 20, 255, cv2.THRESH_BINARY)

template = edges1
# plot template
cv2.imshow('Binarized Image', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    
    img = img2.copy()
   
    method = eval(meth)
    
    # Apply template matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()