import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("../data/raw/0_Edinburgh_Nat_Gallery.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../data/raw/Bridgewater/7_UK_Warrington Museum.jpg",cv2.IMREAD_GRAYSCALE)

plt.imshow(img1, cmap='gray')
plt.imshow(img2, cmap='gray')

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching: It takes the descriptor of one feature in first set and is matched 
# with all other features in second set using some distance calculation. And the closest one is returned.

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 50 matches
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract matched keypoints
points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate the homography matrix
homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

# Apply the homography to image1
registered_image = cv2.warpPerspective(img1, homography_matrix, (img2.shape[1], img2.shape[0]))

# Calculate the absolute difference between the registered image and image2
difference_image = cv2.absdiff(img2, registered_image)

# Display the registered image and the difference image
cv2.imshow('Registered Image', registered_image)
cv2.imshow('Difference Image', difference_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
