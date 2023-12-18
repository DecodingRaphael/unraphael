# Determining the rotation of central figures in paintings relative to each other involves several 
# steps

# https://nl.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html;jsessionid=d208fbee0a343c0b7f3726bae40b

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

def get_image_size_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            resolution = img.info.get("dpi", (0, 0))
            return width, height, resolution
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def pixels_to_cm(pixels, dpi):
    if dpi != 0:
        inches = pixels / dpi
        cm = inches * 2.54
        return cm
    else:
        return None

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            size_resolution = get_image_size_resolution(image_path)
            if size_resolution:
                width, height, resolution = size_resolution
                width_cm = pixels_to_cm(width, resolution[0])
                height_cm = pixels_to_cm(height, resolution[1])
                if width_cm is not None and height_cm is not None:
                    print(f"Image: {filename}, Size: {width} x {height} pixels, Resolution: {resolution} dpi, Size in cm: {width_cm:.2f} x {height_cm:.2f} cm")
                else:
                    print(f"Image: {filename}, Size: {width} x {height} pixels, Resolution: {resolution} dpi, Size in cm: Not available (dpi is zero)")


# Provide the path to your image folder
image_folder_path = "../data/raw/Bridgewater"

# Process images in the folder
process_images_in_folder(image_folder_path)


# 0. Load two paintings
painting1 = cv2.imread("../data/raw/0_Edinburgh_Nat_Gallery.jpg", cv2.IMREAD_GRAYSCALE)
painting2 = cv2.imread('../data/raw/Bridgewater/1_London_Nat_Gallery.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Plot the two images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(painting1, cmap='gray')
axes[0].set_title('Painting 1')
axes[0].axis('off')

axes[1].imshow(painting2, cmap='gray')
axes[1].set_title('Painting 2')
axes[1].axis('off')

# Show the plot
plt.show()

# 2 Feature Detection:
# Use a feature detection algorithm, such as SIFT, SURF, or ORB, to identify keypoints and descriptors
# in both paintings. These algorithms can detect distinctive points in the images.
# Use ORB for feature detection and matching
orb = cv2.ORB_create()

# These descriptors provide scale, direction and geometry of the interest points
keypoints1, descriptors1 = orb.detectAndCompute(painting1, None)
keypoints2, descriptors2 = orb.detectAndCompute(painting2, None)

# 3 Matching Keypoints:
# Match the keypoints between the two paintings. This establishes correspondences between points in 
# the images.
# Use a matcher (e.g., BFMatcher) to find matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to obtain good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.85 * n.distance:
        good_matches.append(m)


# Visualize matched keypoints
img_matches = cv2.drawMatches(painting1, keypoints1, painting2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.show()

# Extract matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 4 Homography Estimation:
# Use the matched keypoints to estimate a homography transformation matrix. The homography matrix
# represents the geometric transformation (including rotation) between the two images.

# compute the homography matrix between the two sets of matched points with RANSAC
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)

# use the homography matrix to align the images
(h, w) = painting1.shape[:2]

# Apply perspective warp to align images
aligned = cv2.warpPerspective(painting2, H, (w, h))
#aligned = cv2.warpPerspective(painting2, H, (painting1.shape[1], painting1.shape[0]))

# Display the transformed image
plt.imshow(aligned)

# Visualize the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(painting1, cv2.COLOR_BGR2RGB)), plt.title('Painting 1')
plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(painting2, cv2.COLOR_BGR2RGB)), plt.title('Painting 2')
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)), plt.title('Aligned Painting')
plt.show()


# 5. Decompose Homography:
# Decompose the homography matrix to extract rotation information. The rotation matrix can be
# obtained from the decomposition.
# Once you have the homography matrix H, you can decompose it into its components matrixes: 
# translation, rotation and scale
#Extracting the rotation matrix, transformation matrix (unused), and normals (unused)
retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, np.eye(3))

# 6. Calculate Rotation Angle:
# Extract the rotation angle from the rotation matrix. This angle represents the relative 
# rotation between the central figures in the two paintings.
# Extract rotation matrix from rotations (assuming one rotation)
rotation_matrix = rotations[0]

# Extract rotation angle from rotation matrix
rotation_angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

print(rotation_angle_rad)

# Convert rotation angle to degrees
rotation_angle_deg = np.degrees(rotation_angle_rad)

print("Rotation Angle (degrees):", rotation_angle_deg)