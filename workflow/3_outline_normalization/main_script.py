# Comparing the proportions of the main figures in the images
# Workplan

# 1. Image Preprocessing:
# Read the two images.
# Convert the images to grayscale to simplify further processing.
# Apply any necessary preprocessing steps (e.g., resizing, normalization).

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

print(cv2.__version__)

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

# face
face_cascade = cv2.CascadeClassifier(
    '../data/haarcascades/cascades/haarcascade_frontalface_default.xml')

# body
#body_cascade = cv2.CascadeClassifier(
#    '../data/haarcascades/cascades/haarcascade_fullbody.xml')
   
img = image2

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.08, 5)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
  
cv2.imshow('Maria Detected!', img)
cv2.imwrite('./maria_detected.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


def detect_objects_haar(image, scaleFactor=1.08, minNeighbors=5, scale_factor=2.0):
    # Load the pre-trained Haar Cascade classifier for faces
    face_cascade = cv2.CascadeClassifier(
        '../data/haarcascades/cascades/haarcascade_frontalface_default.xml')

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return bounding boxes of detected faces
    return faces

# Call the function for both images
objects_image1 = detect_objects_haar(image1)
objects_image2 = detect_objects_haar(image2)


# Step 3: Feature Extraction
def extract_features(image, objects, scale_factor =2, canny_threshold1=50, canny_threshold2=150):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 1.4)
    
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

        # Perform Canny edge detection within the adjusted bounding box
        edges = cv2.Canny(roi, canny_threshold1, canny_threshold2)

        # Visualize the Canny edge detection on the original image
        image_with_edges = image.copy()

        # Place the edges onto the original image preserving original pixels
        image_with_edges[new_y:new_y + new_h, new_x:new_x + new_w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        image_with_edges[new_y:new_y + new_h, new_x:new_x + new_w] &= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Plot the original image with the adjusted bounding box and Canny edges
        plt.imshow(cv2.cvtColor(image_with_edges, cv2.COLOR_BGR2RGB))
        plt.title(f'Canny Edges in Adjusted Box for Object: {x, y, w, h}')
        plt.show()

        # Add features to the list
        features.append({
            'aspect_ratio': float(new_w) / new_h,
            'solidity': 0.0  # Placeholder for solidity as contours are not used
        })

    return features

features_image1 = extract_features(image1, objects_image1,canny_threshold1=10, canny_threshold2=600)
features_image2 = extract_features(image2, objects_image2,canny_threshold1=50, canny_threshold2=100)



# Step 4: Proportion Calculation ----
def calculate_proportions(features):
    
    
    # ... code for proportion calculation
    
    
    return proportions

proportions_image1 = calculate_proportions(features_image1)
proportions_image2 = calculate_proportions(features_image2)

# Step 5: Similarity Measurement ----
from scipy.spatial import distance

similarity = distance.cosine(proportions_image1, proportions_image2)

# Step 6: Display Result
print(f'Similarity between images: {1 - similarity}')

