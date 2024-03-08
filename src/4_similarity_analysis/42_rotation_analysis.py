# GOAL: rotation between two images

# https://nl.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html;jsessionid=d208fbee0a343c0b7f3726bae40b
# https://stats.stackexchange.com/questions/590278/how-to-calculate-the-transalation-and-or-rotation-of-two-images-using-fourier-tr
# https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
# https://answers.opencv.org/question/203890/how-to-find-rotation-angle-from-homography-matrix/


# libraries ----
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
#from PIL import Image
#import os

# 0. Load two paintings ----
painting1 = cv2.imread("../../data/raw/Bridgewater/0_Edinburgh_Nat_Gallery.jpg", cv2.IMREAD_GRAYSCALE)
painting2 = cv2.imread('../../data/raw/Bridgewater/1_London_Nat_Gallery.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Plot the two images side by side ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(painting1, cmap='gray')
axes[0].set_title('Painting 1')
axes[0].axis('off')

axes[1].imshow(painting2, cmap='gray')
axes[1].set_title('Painting 2')
axes[1].axis('off')

# Show the plot
plt.show()

# 2 Feature Detection ----
# Use a feature detection algorithm, such as SIFT, SURF, or ORB, to identify keypoints
# and descriptors in both paintings. These algorithms can detect distinctive points in 
# the images.

# Initialize ORB detector
orb = cv2.ORB_create()

 # Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(painting1, None)
keypoints2, descriptors2 = orb.detectAndCompute(painting2, None)

# 3 Matching Keypoints ----
# Match the keypoints between the two paintings. This establishes correspondences 
# between points in the images.


# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Visualize matched keypoints
# Draw the matches
img_matches = cv2.drawMatches(painting1, keypoints1, painting2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.show()

# Extract matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 4 Homography Estimation ----
# Once feature points are matched, one can compute the transformation between two images:
# - change in scale
# - rotation
# - translation
# - shearing or deformation

# We use the matched keypoints to estimate a homography transformation matrix. 
# The homography matrix represents the geometric transformation (including rotation) 
# between the two images.

# compute the homography matrix between the two sets of matched points with RANSAC
# Find homography
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

## derive rotation angle from homography
theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
        
# direct extraction
angle1 = - math.atan2(H[1,0], H[0,0]) # -2.38

# SVD decomposition methods
u, _, vh = np.linalg.svd(H[0:2, 0:2])
R = u @ vh
angle2 = - math.atan2(R[1,0], R[0,0]) # -2.49

print(angle1, angle2) # -2.38 -2.49

angle_degrees_1 = math.degrees(angle1)
angle_degrees_2 = math.degrees(angle2)

print("Rotation Angle 1 (degrees):", angle_degrees_1)
print("Rotation Angle 2 (degrees):", angle_degrees_2)

# 5 Decompose Homography ----
# Decompose the homography matrix to extract rotation information. The rotation matrix 
# can be obtained from the decomposition. Once you have the homography matrix H, you 
# can decompose it into its components matrixes: the rotation matrix, transformation 
# matrix (unused), and normals (unused)

#rotations: A list of rotation matrices representing the possible rotations. 
# In the context of homography, it typically represents the camera rotations.
# translations: A list of translation vectors corresponding to the rotations. It 
# represents the translation component of the transformation.
# normals: A list of normal vectors to the planes. It provides additional information 
# about the homography

#retval = 4: The homography matrix is a full rank matrix, indicating a perspective transformation.
# since retval is 4, it suggests that the provided homography matrix is valid for a perspective transformation. 
# The decomposition was successful.
retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, np.eye(3))

# 6 Calculate Rotation Angle ----
# Extract the rotation angle from the rotation matrix. This angle represents the relative 
# rotation between the central figures in the two paintings (assuming one rotation)
rotation_matrix = rotations[0]

# Extract rotation angle from rotation matrix
rotation_angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
print(rotation_angle_rad) # 0.78

# Convert rotation angle to degrees
rotation_angle_deg = np.degrees(rotation_angle_rad) #44 degrees
print("Rotation Angle (degrees):", rotation_angle_deg)

# Extract rotation angle from homography matrix
# convert the angle from radians to degrees
rotation_angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))





