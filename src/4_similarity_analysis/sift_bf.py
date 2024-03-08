# Brute-Force Matching with SIFT Descriptors and Ratio Test

# The keypoints and descriptors computed by SIFT are used for matching, 
# making the template matching scale-invariant

# libraries
import numpy as np
import glob
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
%matplotlib inline
plt.rcParams["axes.grid"] = False

# original painting, read in in gray scale
original = cv2.imread("../data/raw/0_Edinburgh_Nat_Gallery.jpg", cv2.IMREAD_GRAYSCALE)

# plot the original
plt.imshow(original, cmap='gray')
    
# Initiate SIFT detector
sift = cv2.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)

# look at keypoints in the original
img_1 = cv2.drawKeypoints(original, kp_1, original)
plt.title('SIFT Algorithm for original painting Edinburgh_Nat_Gallery')
plt.imshow(img_1)

# Brute force matcher with default params
bf = cv2.BFMatcher()

# We can set crossCheck=True for a more accurate but slower matching
bf.set_crossCheck(True)

# Load all the copies of the original painting
all_images_to_compare = []

titles = []

# reading images in gray format
for f in glob.iglob("../data/raw/Bridgewater/*"):
    image = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    titles.append(f)
    all_images_to_compare.append(image)

# plot the copies of the original painting    
for image in all_images_to_compare:
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()

for image_to_compare, title in zip(all_images_to_compare, titles):
    
    # find the keypoints and descriptors with SIFT
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    
    # Brute Force
    matches = bf.knnMatch(desc_1, desc_2, k = 2)
        
    # print the total number of keypoints found in each image
    print("Keypoints 1st image: " + str(len(kp_1))) 
    print("Keypoints 2nd image: " + str(len(kp_2))) 
        
    # Find all the good matches as per Lowe's ratio test.
    # For each pair of descriptors in the matches, we check if the distance to the
    # best match (m) is significantly smaller than the distance to the second-best
    # match (n). If it satisfies the condition, we consider it a good match and add
    # it to the good_matches list.
    
    good_points = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_points.append([m])            
       
    number_keypoints = 0
    
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    print("Title: " + title)
    print("Good matches: ", len(good_points))
    
    percentage_similarity = 100 - (len(good_points) / number_keypoints * 100)
    print("Similarity: " + str(int(percentage_similarity)) + "%" + "\n")
    
    # draw lines between the features that match both images
    # cv.drawMatchesKnn expects list of lists as matches.
    result = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, good_points[0:100], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Show the matching result
    plt.figure(figsize=(16, 16))
    plt.imshow(result)
    plt.show()
        


