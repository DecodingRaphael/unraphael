import cv2
import numpy as np
from scipy.stats import pearsonr
from scipy import signal, spatial
from matplotlib import pyplot as plt

# Function to extract edge features from an image
def extract_edge_features(image):
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Extract features (e.g., number of edge pixels)
    features = np.sum(edges)

    return features

# analysis ----
# read images as grayscale
image1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0) # loads the image in grayscale
image2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)

# resize image so that they are equal
(H, W) = image1.shape

image2 = cv2.resize(image2, (W, H))

# double check dimensions
print(image1.shape, image2.shape)

# Apply Canny edge detection to find outlines
edges1 = cv2.Canny(image1, 60, 200)
edges2 = cv2.Canny(image2, 40, 100)

# plot images
plt.imshow(edges1)
plt.imshow(edges2)

# apply thresholding to convert grayscale to binary image
ret,thresh1 = cv2.threshold(edges1,70,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(edges2,70,255,cv2.THRESH_BINARY)

# Display the original gray image and binarized image (optional)
cv2.imshow('Original Image', image2)
cv2.imshow('Binarized Image', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display Original, Grayscale and Binary Images
plt.subplot(131),plt.imshow(image1,cmap = 'gray'),plt.title('Original Image'), plt.axis('off')
plt.subplot(132),plt.imshow(edges1,cmap = 'gray'),plt.title('Canny Image'),plt.axis('off')
plt.subplot(133),plt.imshow(thresh1,cmap = 'gray'),plt.title('Binary Image'),plt.axis('off')
plt.show()

#convert to float
edges1_norm  = cv2.normalize(thresh1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
edges2_norm  = cv2.normalize(thresh2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

## correlation coefficient
cor = np.corrcoef(edges1_norm.reshape(-1), edges2_norm.reshape(-1))[0][1]
# or
cor = np.corrcoef(edges1_norm.flatten(), edges2_norm.flatten())

## cosine similarity
# akin to similarity score ranging from 0 to 1
result = spatial.distance.cosine(edges1_norm.flatten(), edges2_norm.flatten())

res = cv2.matchTemplate(image1, image2, cv2.TM_SQDIFF_NORMED)  

# Extract edge features from each image
features_image1 = extract_edge_features(image1)
features_image2 = extract_edge_features(image2)

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(features_image1, features_image2)

print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
