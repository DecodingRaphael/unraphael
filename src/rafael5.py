
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
        
# list with colored images ----
colored_images = []

for filename in os.listdir(subfolder_path):
    if any(filename.endswith(ext) for ext in image_extensions):
        image_path = os.path.join(subfolder_path, filename)
        try:
            img = cv2.imread(image_path)
            
            # Convert the color format from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            colored_images.append(img_rgb)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Display the images
for image in colored_images:
    plt.imshow(image)
    plt.grid(None)
    plt.show()    
    
# list with grayscale images: grayscale is a necessary step in SIFT ----
grayscale_images = []

for filename in os.listdir(subfolder_path):
    if any(filename.endswith(ext) for ext in image_extensions):
        image_path = os.path.join(subfolder_path, filename)
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            grayscale_images.append(img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
  
for image in grayscale_images:
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.grid(None) 
    plt.show()
    
# visualizing SIFT
sift = cv2.SIFT_create()

original         = grayscale_images[0]
image_to_compare = grayscale_images[1]
all_images       = grayscale_images[1:]

# look at keypoints in the image
keypoints_1, descriptors_1 = sift.detectAndCompute(original, None)
img_1 = cv2.drawKeypoints(original,keypoints_1,original)
plt.imshow(img_1)

# Compare two images
figure, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(original, cmap='gray')
ax[1].imshow(image_to_compare, cmap='gray')

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
print("How good is the match: ", len(good_points)/number_keypoints * 100,"%")

# draw lines between the features that match both images
result = cv2.drawMatches(original, keypoints_1, image_to_compare, keypoints_2, good_points, None)
plt.imshow(result), plt.show()

# part 3: Detect how similar two images are with Opencv and Python
# https://www.youtube.com/watch?v=ND5vGDNvN0s

# part 4: # Check if a set of images match the original one
# https://www.youtube.com/watch?v=ADuHe4JNLXs





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
        
# Initialize dictionaries to store keypoints and descriptors for each detector ----
keypoints_dict = {"SIFT": []}   # 'keypoints_dict' contains keypoints for each detector
descriptors_dict = {"SIFT": []} # 'descriptors_dict' contains descriptors

# Initialize detectors
sift = cv2.SIFT_create()

# Create a dictionary for detectors
detectors = {"SIFT": sift}

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

# Match the descriptors using a brute-force matcher ----
bf = cv2.BFMatcher()
matches = []

for detector_name, detector in detectors.items():
    for i in range(len(grayscale_images)):
        for j in range(i + 1, len(grayscale_images)):
            match = bf.knnMatch(descriptors[i], descriptors[j], k=2)
            matches.append(match)

# Apply the ratio test to filter out bad matches
good_matches = []

# This code iterates through the matches and checks if the distance of the best match (m) 
# is less than 75% of the distance of the second-best match (n). If this condition is met, 
# the match is considered a good match and added to the good_matches list.
for match in matches:
    for m, n in match:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m) # Now 'good_matches' contains the filtered matches

# You can print the number of good matches to see how many were retained
print("Number of good matches:", len(good_matches))

# Calculate the similarity matrix -----------------------------------------------------------------------

# we create a similarity matrix where each entry represents the similarity between two images
# based on the number of good matches. The diagonal entries are set to 1 because each image is
# always similar to itself

# Create a similarity matrix
similarity_matrix = np.zeros((len(grayscale_images), len(grayscale_images)))

for i in range(len(grayscale_images)):
    for j in range(len(grayscale_images)):
        num_good_matches = len([m for m in good_matches if m.queryIdx == i and m.trainIdx == j])
        num_keypoints_j = len(keypoints_dict["SIFT"][j])
        similarity_matrix[i, j] = num_good_matches / num_keypoints_j
       
         
# Iterate through the images and calculate the similarity based on the number of good matches
for i in range(len(grayscale_images)):
    for j in range(len(grayscale_images)):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Images are always similar to themselves
        else:
            # Check if 'i' and 'j' are valid indices in keypoints_dict["SIFT"]
            if i < len(keypoints_dict["SIFT"]) and j < len(keypoints_dict["SIFT"]):
                # Count the number of good matches between image i and image j
                num_good_matches = 0
                keypoints_i = keypoints_dict["SIFT"][i]
                keypoints_j = keypoints_dict["SIFT"][j]
                
                for match in good_matches:
                    if match.queryIdx < len(keypoints_i) and match.trainIdx < len(keypoints_j):
                        num_good_matches += 1
                        
                # Use the number of good matches as the similarity value
            #    similarity_matrix[i, j] = num_good_matches
            #else:
            #    similarity_matrix[i, j] = 0  # No keypoints for one or both images        

                # Calculate the similarity based on the number of good matches
                similarity = num_good_matches
                #similarity = num_good_matches / max(len(keypoints_i), len(keypoints_j))
                similarity_matrix[i, j] = similarity
            else:
                similarity_matrix[i, j] = 0.0  # No keypoints for one or both images

# Visualize the matrix using a heatmap

sns.heatmap(similarity_matrix, annot=True)
plt.show()


###################################################################################

def display_sift_features(image, image_name):
    """
    Extracts SIFT keypoints from the luminance (Y) component of an image
    and displays the image with the keypoints highlighted.

    Args:
    - image (cv: img): Image from which to extract SIFT keypoints.
    - image_name (str): name of the image
    """
    working_image = image.copy()
    yuv = cv2.cvtColor(working_image, cv2.COLOR_BGR2YUV)

    # Extract the Y (luminance) component from the YUV image
    y_channel = yuv[:,:,0]

    # Detect SIFT keypoints from the Y component
    kp = sift_extractor(y_channel)[0]

    # Loop through each detected keypoint to draw a cross
    for keypoint in kp:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        line_length = 5
        cv2.line(working_image, (x - line_length, y), (x + line_length, y), (0, 0, 255), 1)
        cv2.line(working_image, (x, y - line_length), (x, y + line_length), (0, 0, 255), 1)

    # Draw the keypoints with circles and orientation lines on the working image
    image_w_keypoints = cv2.drawKeypoints(image, kp, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Stack the original image and the image with keypoints side-by-side for comparison
    final_image = np.hstack([image, image_w_keypoints])
    display_image(final_image)
    print(f"# of keypoints in {image_name} is {len(kp)}")
