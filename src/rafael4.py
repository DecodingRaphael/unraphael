
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

# Specify the path to the subfolder containing the images
subfolder_path = "../data/raw/Bridgewater/"

# Get a list of all files in the subfolder
files = os.listdir(subfolder_path)

# Filter the list to include only image files (you can add more extensions if needed)
image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", "psd"]
image_files = [file for file in files if any(file.endswith(ext) for ext in image_extensions)]

# Create a dictionary to store images with their filenames as keys
images = {}

# Iterate through the image files and load each image
for image_file in image_files:
    image_path = os.path.join(subfolder_path, image_file)
    
    try:
        with Image.open(image_path) as img:
            # Store the image in the dictionary with its filename as the key
            images[image_file] = img
    except Exception as e:
        print(f"Error opening {image_path}: {e}")

# Create a new list to store the grayscale images ----
grayscale_images = []

for filename in os.listdir(subfolder_path):
    if any(filename.endswith(ext) for ext in image_extensions):
        image_path = os.path.join(subfolder_path, filename)
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            grayscale_images.append(img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
  
            
# Apply edge detection using the Canny edge detector

# Initialize a list to store the edge-detected images
edge_images = []

# Define Canny edge detection parameters
low_threshold = 50
high_threshold = 150

# Iterate through the grayscale images and apply Canny edge detection -----
for grayscale_img in grayscale_images:
    try:
        # Apply Canny edge detection
        edges = cv2.Canny(grayscale_img, low_threshold, high_threshold)
        
        # Append the edge-detected image to the list
        edge_images.append(edges)
    except Exception as e:
        print(f"Error applying edge detection: {e}")
        


# Initialize dictionaries to store keypoints and descriptors for each detector
keypoints_dict = {"SIFT": [], "ORB": [], "SURF": [], "BRISK": []}
descriptors_dict = {"SIFT": [], "ORB": [], "SURF": [], "BRISK": []}

# Initialize detectors
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
#surf = cv2.xfeatures2d.SURF_create()
brisk = cv2.BRISK_create()

# Create a dictionary for detectors
detectors = {"SIFT": sift, "ORB": orb, "BRISK": brisk}

# Iterate through the grayscale images and compute keypoints and descriptors for each detector
for detector_name, detector in detectors.items():
    for grayscale_img in grayscale_images:
        try:
            # Detect keypoints and compute descriptors
            keypoints, descriptors = detector.detectAndCompute(grayscale_img, None)
            
            # Append keypoints and descriptors to their respective lists in the dictionaries
            keypoints_dict[detector_name].append(keypoints)
            descriptors_dict[detector_name].append(descriptors)
        except Exception as e:
            print(f"Error with {detector_name}: {e}")

# 'keypoints_dict' contains keypoints for each detector
# 'descriptors_dict' contains descriptors

# match descriptors using:

# Match the descriptors using a brute-force matcher
bf = cv2.BFMatcher()

matches = []

for detector_name, detector in detectors.items():
    for i in range(len(grayscale_images)):
        for j in range(i + 1, len(grayscale_images)):
            match = bf.knnMatch(descriptors[i], descriptors[j], k=2)
            #matches.append(match)

 # Apply the ratio test to filter out bad matches
good_matches = []
for match in matches:
    if match[0].distance < 0.75 * match[1].distance:
        good_matches.append(match)
 

# Calculate the similarity matrix
similarity_matrix = np.zeros((len(grayscale_images), len(grayscale_images)))

for i in range(len(grayscale_images)):
    for j in range(len(grayscale_images)):
        num_good_matches = len([m for m in good_matches if m[0].queryIdx == i and m[0].trainIdx == j])
        num_keypoints_j = len(keypoints[j])
        similarity_matrix[i, j] = num_good_matches / num_keypoints_j

# Display the similarity matrix
print(similarity_matrix)

# brute-force matching, 
# FLANN (Fast Approximate Nearest Neighbor Search), and 
# K-Nearest Neighbors (KNN) for the images, 
# We'll iterate through the detectors and matching methods, 
# and we'll continue matching as long as the matches are accurate based
# on a certain threshold

# Initialize a dictionary to store matched keypoints and descriptors for each detector
matched_dict = {}

# Threshold for matching accuracy (adjust as needed)
matching_threshold = 0.7

# Iterate through detectors
for detector_name, detector in detectors.items():
    matched_dict[detector_name] = {}
    
    for i, descriptors in enumerate(descriptors_dict[detector_name]):
        try:
            # Specifically handle the case when descriptors are None for the ORB detector
            if descriptors is None and detector_name == "ORB":
                matched_dict[detector_name][f"Image{i}"] = None
                continue
            
            # Create a list of descriptors for other images
            other_descriptors = [desc.astype(descriptors.dtype) for j, desc in enumerate(descriptors_dict[detector_name]) if j != i]
            
            # Convert query descriptors to the same data type as train descriptors
            descriptors = descriptors.astype(other_descriptors[0].dtype)
            
            # Initialize the brute-force matcher
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            
            # Match descriptors with the descriptors of other images using the brute-force matcher
            matches = matcher.match(descriptors, other_descriptors)
            
            # Apply the Lowe's ratio test to keep only good matches
            good_matches = []
            for match in matches:
                if match.distance < matching_threshold:
                    good_matches.append(match)
            
            # Store the good matches in the dictionary
            matched_dict[detector_name][f"Image{i}"] = good_matches
        except Exception as e:
            print(f"Error matching with {detector_name} for Image{i}: {e}")

# Now 'matched_dict' contains the matched keypoints and descriptors for each detector,
# using only the brute-force matcher.
# For ORB, if descriptors are None, the entry in the dictionary will be None.
# You can access and work




# Initialize a dictionary to store matched keypoints and descriptors for each detector and matching method
matched_dict = {}

# Threshold for matching accuracy (adjust as needed)
matching_threshold = 0.7

# Iterate through detectors
for detector_name, detector in detectors.items():
    matched_dict[detector_name] = {}
    
    for matching_method in ["BruteForce", "FLANN", "KNN"]:
        # Initialize the matcher based on the matching method
        if matching_method == "BruteForce":
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif matching_method == "FLANN":
            flann_index_params = dict(algorithm=0, trees=5)
            flann_search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        elif matching_method == "KNN":
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Iterate through descriptors for this detector
        for i, descriptors in enumerate(descriptors_dict[detector_name]):
            try:
                # Create a list of descriptors for other images
                other_descriptors = [desc for j, desc in enumerate(descriptors_dict[detector_name]) if j != i]
                
                # Match descriptors with the descriptors of other images
                matches = matcher.knnMatch(descriptors, other_descriptors, k=2)
                
                # Apply the Lowe's ratio test to keep only good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < matching_threshold * n.distance:
                        good_matches.append(m)
                
                # Store the good matches in the dictionary
                matched_dict[detector_name][f"Image{i}"] = good_matches
            except Exception as e:
                print(f"Error matching with {detector_name} using {matching_method} for Image{i}: {e}")

# Now 'matched_dict' contains the matched keypoints and descriptors for each detector and matching method.
# You can access and work with these matches as needed.


# Match the descriptors using a different descriptor matching algorithm, such as FLANN or KNN
# FLANN is a faster alternative to the brute-force matcher that is still accurate for many types of images
bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
matches = []
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        match = bf.match(image_descriptors[i], image_descriptors[j])
        matches.append(match)

# Apply the ratio test to filter out bad matches
good_matches = []
for match in matches:
    if match.distance < 0.75 * match[1].distance:
        good_matches.append(match)

# Calculate the similarity matrix
similarity_matrix = np.zeros((len(images), len(images)))
for i in range(len(images)):
    for j in range(len(images)):
        num_good_matches = len([m for m in good_matches if m.queryIdx == i and m.trainIdx == j])
        num_keypoints_j = len(image_keypoints[j])
        similarity_matrix[i, j] = num_good_matches / num_keypoints_j

# Perform image recognition registration using a different image recognition registration algorithm, such as ORB-SLAM or LSD-SLAM
# ORB-SLAM is a more robust alternative to the homography matrix estimation algorithm

# Calculate the homography matrices
homography_matrices = []
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        if len(good_matches) >= 4:
            src_points = np.float32([image_keypoints[i][m.queryIdx].pt for m in good_matches if m.queryIdx == i])
            dst_points = np.float32([image_keypoints[j][m.trainIdx].pt for m in good_matches if m.queryIdx == i])
            # Use a different image recognition registration algorithm, such as ORB-SLAM or LSD-SLAM
            # ORB-SLAM is a more robust alternative to the homography matrix estimation algorithm
            # homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            homography_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
            homography_matrices.append(homography_matrix)

