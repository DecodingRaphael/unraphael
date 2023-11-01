
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
%matplotlib inline
plt.rcParams["axes.grid"] = False


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
        

original         = grayscale_images[0]
image_to_compare = grayscale_images[1:]


# Sift and Flann
sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)



# use the function detectAndCompute to get the keypoints
keypoints_1, descriptors_1 = sift.detectAndCompute(original,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image_to_compare,None)

# print the total number of keypoints found in each image
print("Keypoints 1st image: " + str(len(keypoints_1))) 
print("Keypoints 2nd image: " + str(len(keypoints_2))) 

# match the features from image 1 with features from image 2
## with brute-force 
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

## with flann
indexParams = dict(algorithm = 0, trees = 5)
searchParams = dict()
flann = cv2.FlannBasedMatcher(indexParams,searchParams)

# function match() from the BFmatcher (brute force match) module
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

# flann match
matches = flann.knnMatch(descriptors_1,descriptors_2, k= 2)

# how many keypoints are matched?
len(matches)

good_points = []
# m is the first image, n is the second image
for m, n in matches:
     #the lower the distance, the better the match
     if m.distance < 0.6*n.distance:
        good_points.append(m)
        
number_keypoints = 0
if len(keypoints_1) <= len(keypoints_2):
    number_keypoints = len(keypoints_1)
else:
    number_keypoints = len(keypoints_2)

print("GOOD matches: ", len(good_points))

perc_similarity = len(good_points)/number_keypoints * 100
print("Similarity: " + str( int(perc_similarity)) + "%")

# draw lines between the features that match both images
result = cv2.drawMatches(original, keypoints_1, image_to_compare, keypoints_2, good_points, None)
plt.imshow(result), plt.show()

# part 3: Detect how similar two images are with Opencv and Python
# https://www.youtube.com/watch?v=ND5vGDNvN0s

# part 4: # Check if a set of images match the original one
# https://www.youtube.com/watch?v=ADuHe4JNLXs

# https://pysource.com/2018/07/27/check-if-a-set-of-images-match-the-original-one-with-opencv-and-python/



