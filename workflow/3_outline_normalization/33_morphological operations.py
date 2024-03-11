import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
# Load the images
img = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg',1)

plt.imshow(img) 
# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

# morphology edgeout = dilated_mask - mask
# morphology dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# get absolute difference between dilate and thresh
diff = cv2.absdiff(dilate, thresh)

# invert
edges = 255 - diff

# write result to disk
cv2.imwrite("cartoon_thresh.jpg", thresh)
cv2.imwrite("cartoon_dilate.jpg", dilate)
cv2.imwrite("cartoon_diff.jpg", diff)
cv2.imwrite("cartoon_edges.jpg", edges)

# display it
cv2.imshow("thresh", thresh)
cv2.imshow("dilate", dilate)
cv2.imshow("diff", diff)
cv2.imshow("edges", edges)
cv2.waitKey(0)


