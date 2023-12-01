import cv2
import numpy as np
from scipy.stats import pearsonr

# Function to extract edge features from an image
def extract_edge_features(image):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Extract features (e.g., number of edge pixels)
    features = np.sum(edges)

    return features

image1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0)
image2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)


# Extract edge features from each image
features_image1 = extract_edge_features(image1)
features_image2 = extract_edge_features(image2)

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(features_image1, features_image2)

print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
