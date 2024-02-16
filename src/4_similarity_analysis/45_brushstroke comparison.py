# with the following code, I like to compare a set of images with each other. 
# Specifically, I focus on the brushstrokes in the images (which are paintings). 
# These brushstrokes carry the essence of an artist’s technique, often manifest as edges 
# in an image. Therefore, capturing these subtle cues through edge detection can provide
# insights into the artist’s unique style and potentially aid in measuring the degree 
# of similarity between the paintings

# Before drawing conclusions, it's crucial to validate the approach. We still need a subset
# of our paintings for which we  know the ground truth similarity. This will help to assess
# how well this  method aligns with our expectations.

# libraries ----
import os
import cv2
import numpy as np
import pandas as pd
import glob
from PIL import Image
from plotnine import ggplot, aes, geom_tile, geom_text, theme_minimal,labs,theme
import plotnine as pn
from scipy.cluster import hierarchy
import plotly.express as px

# Once the edge features are computed, each time the standard deviation is calculated for 
# every individual edge feature obtained from an image. The standard deviation serves as
# an effective metric to quantify the variability and intensity of edge features in the image.

# We capture various edge features using different operators, providing a comprehensive 
# representation of brushstrokes.

# Function to calculate edge features using Canny edge detector
def calculate_canny_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  
    return np.std(edges)

# Function to calculate edge features using Sobel operator
def calculate_sobel_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return np.std(sobelx), np.std(sobely)

# Function to calculate edge features using Laplacian operator
def calculate_laplacian_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.std(laplacian)

# Function to calculate edge features using Scharr operator
def calculate_scharr_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.std(scharrx), np.std(scharry)

# Function to calculate all edge features from Canny, Sobel, Laplacian, and Scharr employed to capture 
# all the different edge features of an image. 
def calculate_features(img):    
    canny_edges                    = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y   = calculate_sobel_edges(img)
    laplacian_edges                = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)    
    
    # The function returns an array of standard deviations for various edge features 
    # extracted from the image
    return np.array([canny_edges, 
                     sobel_edges_x, sobel_edges_y, 
                     laplacian_edges, 
                     scharr_edges_x, scharr_edges_y])       

# Load all the paintings
all_images = []
titles     = []

# reading in all the images from the folder and storing them in a list  
for f in glob.iglob("../../data/raw/Bridgewater/*"):
    image = cv2.imread(f)
    titles.append(os.path.basename(f))  # Use only the file name as the title
    all_images.append(image)

# Create an empty matrix to store the mean differences in standard deviation for each pair of images
matches_matrix = np.zeros((len(all_images), len(all_images)))

for i, (image1, title1) in enumerate(zip(all_images, titles)):
    
    # extract sd's for the current image
    features_1 = calculate_features(image1)
    
    # Normalize features in order to get weights. Divide each individual standard deviation in
    # features_1 by the total sum of standard deviations
    # weights = features_1 / np.sum(features_1)   
            
    for j, (image2, title2) in enumerate(zip(all_images, titles)):
        if i != j:  # Avoid comparing an image with itself
            
            # find the sd for the other image
            features_2 = calculate_features(image2)  
            
            # Calculate the total feature values and the count of images
            #features_2_weighted = np.zeros_like(features_1)
                        
            # Add to total and increment count (multiply by weights here)
            #features_2_weighted += features_2 * weights                       
                        
            # We now compare the weighted extracted features from one image with the weighted extracted features 
            # derived from another image using the absolute difference between them.            
            difference = np.abs(features_1 - features_2)
            
            # The mean of these differences (mean_diff) is then calculated and used as a metric to 
            # quantify the brushstroke similarity between the two images. The lower the mean of the
            # differences, the more similar the  brushstrokes between the images are. Thus, it 
            # provides a metric representing the average difference in standard deviation across 
            # various edge features in the two images.           
            mean_diff = np.mean(difference)      
                       
            # Store the mean difference in sd in the matrix
            matches_matrix[i, j] = mean_diff
                             
# Calculate the normalized mean difference across the entire dataset so that the normalization
# is consistent across all pairs of images

# Set diagonal values to a large number to ignore them in min/max calculation
np.fill_diagonal(matches_matrix, np.inf)

global_min_diff = np.min(matches_matrix)
global_max_diff = np.max(matches_matrix[np.isfinite(matches_matrix)])  # Exclude infinite values

# normalize the matrix (all values are scaled between 0 and 1)
matches_matrix_norm = (matches_matrix - global_min_diff)/(global_max_diff - global_min_diff)
np.fill_diagonal(matches_matrix_norm, 0) # set diagonal values to 0 again
heatmap_data = pd.DataFrame(matches_matrix_norm, columns=titles, index=titles)

# Create a DataFrame for the heatmap
heatmap_data = pd.DataFrame(matches_matrix, columns=titles, index=titles)
heatmap_data = heatmap_data.round(decimals = 2) 
heatmap_data = heatmap_data.rename({'0_Edinburgh_Nat_Gallery.jpg': '0_Edinburgh', 
                                    '1_London_Nat_Gallery.jpg'   : '1_London_Nat',
                                    '2_Naples_Museo Capodimonte.jpg': '2_Naples',
                                    '3_Milan_private.jpg'        : '3_Milan',
                                    '4_Oxford_Ashmolean.jpg'     : '4_Oxford',
                                    '5_UK_Nostrell Priory.jpg'   : '5_Nostrell',
                                    '6_Oxford_Christ_Church.jpg' : '6_Oxford_Christ',	
                                    '7_UK_Warrington Museum.jpg' : '7_Warrington',
                                    '8_London_OrderStJohn.jpg'   : '8_London_Order',	
                                    '9_Zurich_KunyCollection.jpg': '9_Zurich',
                                    '10_Nolay_MaisonRetraite.jpg': '10_Nolay',
                                    }, axis=1)

heatmap_data.index = heatmap_data.columns

# Compute hierarchical clustering
linkage_matrix = hierarchy.linkage(heatmap_data.values, method='ward')

# Create a dendrogram plot ----
dendrogram_row = hierarchy.dendrogram(linkage_matrix, orientation='right', labels=heatmap_data.index)
dendrogram_row = hierarchy.dendrogram(linkage_matrix, orientation='left', 
                                      labels=heatmap_data.index, p=50)

# transform the DataFrame heatmap_data into a "long" format suitable for plotting with plotnine.
heatmap_data_long = heatmap_data.reset_index().melt(id_vars='index')

# Plot the larger heatmap with matching feature points using plotnine
(ggplot(heatmap_data_long, aes(x='variable', y='index', fill='value'))
 + geom_tile()
 + geom_text(aes(label='value'), color='white', size=30, ha='center', va='center')  # Adjust the size parameter
 + theme_minimal()
 + labs(title='SIFT Feature Matches Heatmap', x='Image 2', y='Image 1', fill='Number of Matches')
 + theme(figure_size=(25, 20))  # Adjust the figure size as needed
)


df = heatmap_data
fig = px.imshow(df,text_auto=True,aspect="auto")
fig.show()