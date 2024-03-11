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

# original painting
original = cv2.imread("../data/raw/0_Edinburgh_Nat_Gallery.jpg", cv2.IMREAD_GRAYSCALE)

# plot the original
plt.imshow(original, cmap='gray')
    
# Sift and Flann
# TODO: Compare SIFT with SURF and ORB: which is better for paintings? 

# Initiate SIFT detector
sift = cv2.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)

# look at keypoints in the original
img_1 = cv2.drawKeypoints(original, kp_1, original)
plt.imshow(img_1)

# flann
FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann         = cv2.FlannBasedMatcher(index_params, search_params)

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
    
    # Flann
    matches = flann.knnMatch(desc_1, desc_2, k = 2)
        
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
        if m.distance < 0.9 * n.distance:
            good_points.append(m)
            
    # TODO: Use the PROSAC algorithm to further filter matches for accuracy
    
    # Define how similar they are
    number_keypoints = 0
    
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
        
    print("Title: " + title)
    print("Good matches: ", len(good_points))
    
    percentage_similarity = len(good_points) / number_keypoints * 100
    
    print("Similarity: " + str(int(percentage_similarity)) + "%" + "\n")
    
    # draw lines between the features that match both images
    # cv.drawMatchesKnn expects list of lists as matches.
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points[0:100], None)
       
    # Show the matching result
    plt.imshow(result)
    plt.show()
    
    cv2.imwrite("feature_matching_"+str(image_to_compare)+".jpg", result)
    
# slightly different approach to drawMatchesKnn using matchesMask to draw only good matches ----
for image_to_compare, title in zip(all_images_to_compare, titles):
    
    # find the keypoints and descriptors with SIFT
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    
    # Flann
    matches = flann.knnMatch(desc_1, desc_2, k = 2)
        
    # print the total number of keypoints found in each image
    print("Keypoints 1st image: " + str(len(kp_1))) 
    print("Keypoints 2nd image: " + str(len(kp_2)))        
   
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
  # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1,0]
            
    number_keypoints = 0
    
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor   = (255,0,0),
                   matchesMask        = matchesMask,
                   flags              = cv2.DrawMatchesFlags_DEFAULT)
                 # flags              = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # draw lines between the features that match both images
    result = cv2.drawMatchesKnn(original,kp_1,image_to_compare,kp_2, matches, None,**draw_params)
    # cv.drawMatchesKnn expects list of lists as matches                        
    
    # Show the matching result
    plt.imshow(result)
    plt.show()
    

