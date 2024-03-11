# Comparing the proportions of the main figures in the images
# Workplan

# 1. Image Preprocessing:
# Read the two images
# Convert the images to grayscale to simplify further processing
# Apply any necessary preprocessing steps (e.g., resizing, normalization)

# 2. Object Detection:
# Use an object detection algorithm to identify the main figures in each image.
# You can use a pre-trained model like YOLO or SSD for this purpose.

# 3. Feature Extraction:
# Extract features from the detected main figures. You can use the contours of the figures for this.
# Normalize the features to account for variations in image size.

# 4. Proportion Calculation:
# Calculate the proportions of the main figures based on the extracted features.

# 5. Similarity Measurement:
# Use a similarity metric (e.g., Euclidean distance, cosine similarity) to quantify the similarity between the proportions of the main figures.


# Step 1: Image Preprocessing ----
# libraries 
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import signal, spatial
from scipy.spatial import distance

print(cv2.__version__) #Version 4.8.0

# search for location of xml files
print(cv2.data.haarcascades)

# Read images in gray
image1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg',1)
image2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg',1)

# Display the image
cv2.imshow('Image', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#plt.imshow(gray_image1,cmap='gray')
#plt.imshow(gray_image2,cmap='gray')


# Apply resizing if necessary

# ......

# Step 2: Object Detection

# Object detection (i.e., faces and/or body) in the painting with detect_objects_haar
def detect_objects_haar(image, scaleFactor=1.08, minNeighbors=5, scale_factor=2.0):
    
    # Load the pre-trained Haar Cascade classifier for faces
    face_cascade = cv2.CascadeClassifier(
        '../data/haarcascades/cascades/haarcascade_frontalface_default.xml')
    
    # Load the pre-trained Haar Cascade classifier for the full or upper body
    # body_cascade = cv2.CascadeClassifier(
    #   '../data/haarcascades/cascades/haarcascade_fullbody.xml')
    
    # Convert the image to grayscale (Haar Cascades work with grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)

    for (x, y, w, h) in faces:
        
         # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate the new width and height
        w = int(w * scale_factor)
        h = int(h * scale_factor)

        # Adjust the coordinates to make the bounding box twice as big        
        x = max(0, center_x - w // 2)
        y = max(0, center_y - h // 2)
        
        # Draw the adjusted bounding box
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)        
        
    # Display the image with the double-sized bounding box
    cv2.imshow('Maria Detected!', image)
    cv2.imwrite('./maria_detected.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return bounding boxes of detected faces
    return faces

# Call the function for both images
objects_image1 = detect_objects_haar(image1) # 'objects_image1' is the list of bounding boxes
objects_image2 = detect_objects_haar(image2) # 'objects_image2' is the list of bounding boxes

# Initialize SIFT detector
sift = cv2.SIFT_create()
    
# Step 3: Feature Extraction
def extract_features(image, objects, scale_factor =2, canny_threshold1=50, canny_threshold2=150):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 1.4)
    
     # Initialize an image to accumulate Canny edges
    # edges_accumulated = np.zeros_like(blurred)
    
    # Initialize a list to store features
    features = []

    for (x, y, w, h) in objects:
        # Calculate the new width and height
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Adjust the coordinates to make the bounding box twice as big
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)

        # Extract the region of interest (ROI) from the grayscale image
        roi = blurred[new_y:new_y + new_h, new_x:new_x + new_w]
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(roi)

        # Perform Canny edge detection within the adjusted bounding box
        #edges = cv2.Canny(roi, canny_threshold1, canny_threshold2)
        
         # Adaptive thresholding
         
         # Parameters for adaptive thresholding
        block_size = 10  # Size of the neighborhood for adaptive thresholding
        c_value = 4      # Constant subtracted from the mean for adaptive thresholding
        edges = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)
        
        # Accumulate the edges onto the overall image
        #edges_accumulated[new_y:new_y + new_h, new_x:new_x + new_w] |= edges
          
        # Dilation and erosion for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
            
        # Calculate aspect ratio and solidity using contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            aspect_ratio = float(new_w) / new_h
            solidity = cv2.contourArea(contour) / (new_w * new_h)
        
        else:
            aspect_ratio = 0.0
            solidity = 0.0
        
        # Detect SIFT features and compute descriptors on the Canny edges
        keypoints, descriptors = sift.detectAndCompute(edges, None)
        
        # Filter keypoints to keep only those on the edges
        keypoints_on_edges = [kp for kp in keypoints if edges[int(kp.pt[1]), int(kp.pt[0])] == 255]

        # Draw circles at the filtered keypoints' locations on the Canny edges
        image_with_keypoints = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        #cv2.drawKeypoints(image_with_keypoints, keypoints, image_with_keypoints, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(image_with_keypoints, keypoints_on_edges, 
                          image_with_keypoints, color=(0, 255, 0), 
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Place the edges onto the original image preserving original pixels
        image_with_edges = image.copy()
        image_with_edges[new_y:new_y + new_h, new_x:new_x + new_w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        image_with_edges[new_y:new_y + new_h, new_x:new_x + new_w] &= image_with_keypoints

        # Plot the original image with the adjusted bounding box, Canny edges, and SIFT keypoints
        plt.imshow(cv2.cvtColor(image_with_edges, cv2.COLOR_BGR2RGB))
        plt.title(f'SIFT Keypoints on Edges for Object: {x, y, w, h}')
        plt.show()
        
        # Add features to the list
        features.append({
            'bbox': (new_x, new_y, new_x + new_w, new_y + new_h),
            'keypoints': keypoints_on_edges,
            'descriptors': descriptors,
            'aspect_ratio': float(new_w) / new_h,
            'solidity': 0.0  # Placeholder for solidity as contours are not used
        })

    return features

features_image1 = extract_features(image1, objects_image1,canny_threshold1=10, canny_threshold2=600)
features_image2 = extract_features(image2, objects_image2,canny_threshold1=50, canny_threshold2=100)

# Step 4: Proportion Calculation ----

# Calculate the proportions of the main figures based on the extracted features.

# We can normalize the distances using the inherent property of the bounding box itself.
# We use the diagonal length of the bounding box as a reference for normalization. 
# This way, the distances are normalized with respect to the size of the bounding box, 
# making the average_distance measure independent of the face size
def calculate_proportion(features):
    
    # Initialize a list to store proportion measures
    proportion_measures = []

    for feature in features:
        
        # Extract bounding box coordinates
        bbox = feature['bbox']
        
        # Calculate the diagonal length of the bounding box
        # diagonal_length is calculated as the Euclidean distance between the top-left 
        # and bottom-right corners of the bounding box. This length is then used to 
        # normalize the distances
        diagonal_length = np.linalg.norm(np.array(feature['bbox'][:2]) - np.array(feature['bbox'][2:]))
        
        # Extract all keypoints
        all_keypoints = feature['keypoints']
        total_keypoints = len(all_keypoints)  # Calculate the total number of keypoints for each bounding box
               
        # Calculate pairwise distances between all keypoints
        distances = []
        
        for i in range(len(all_keypoints)):
            for j in range(i + 1, len(all_keypoints)):
                distance = np.linalg.norm(np.array(all_keypoints[i].pt) - np.array(all_keypoints[j].pt))
                distances.append(distance)

        # Combine distances into one measure (e.g., average) and normalize by the diagonal length of the box
        if distances:
            average_distance = np.mean(distances) / diagonal_length
        else:
            average_distance = 0.0

        # Add the proportion measure to the list along with the total number of keypoints
        proportion_measures.append({
            'average_distance': average_distance,
            'aspect_ratio': feature['aspect_ratio'],
            'solidity': feature['solidity'],
            'total_keypoints': total_keypoints
        })

        print(f"Total number of keypoints for bounding box: {total_keypoints}")

    return proportion_measures

# each row in the outcome corresponds to one of the bounding boxes 
proportion_measures_image1 = calculate_proportion(features_image1)
proportion_measures_image2 = calculate_proportion(features_image2)


# Step 5: Similarity Measurement ----
# TODO!
similarity = distance.cosine(proportions_image1, proportions_image2)

## cosine similarity, akin to similarity score ranging from 0 to 1
result = spatial.distance.cosine(edges1_norm.flatten(), edges2_norm.flatten())

# Step 6: Display Result
print(f'Similarity between images: {1 - similarity}')

