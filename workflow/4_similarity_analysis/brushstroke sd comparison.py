# libraries
import os
import cv2
import numpy as np
import pandas as pd
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_tile, geom_text, theme_minimal,labs,theme
import plotnine as pn
from scipy.cluster import hierarchy

# Once the edge features are computed, each time the standard deviation is calculated for 
# every individual edge feature obtained from an image. The standard deviation serves as
# an effective metric to quantify the variability and intensity of edge features in the image.
    
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
# all the different edge features of an image. The function returns a vector of edge features for the input
# image. 
def calculate_features(img):    
    canny_edges                    = calculate_canny_edges(img)
    sobel_edges_x, sobel_edges_y   = calculate_sobel_edges(img)
    laplacian_edges                = calculate_laplacian_edges(img)
    scharr_edges_x, scharr_edges_y = calculate_scharr_edges(img)    
       
    return np.array([canny_edges, 
                     sobel_edges_x, sobel_edges_y, 
                     laplacian_edges, 
                     scharr_edges_x, scharr_edges_y])       

# Load all the paintings
all_images = []
titles     = []

# reading images
for f in glob.iglob("../../data/raw/Bridgewater/*"):
    image = cv2.imread(f)
    titles.append(os.path.basename(f))  # Use only the file name as the title
    all_images.append(image)

# Create a matrix to store the mean differences in standard deviation for each pair of images
matches_matrix = np.zeros((len(all_images), len(all_images)))

for i, (image1, title1) in enumerate(zip(all_images, titles)):
    
    # extract sd for the current image
    features_1 = calculate_features(image1)
    
    # Normalize test features in order to get weights
    weights = features_1 / np.sum(features_1)
    
    # Calculate the total feature values and the count of images
    features_2_weighted = np.zeros_like(features_1)
    image_count = 0
        
    for j, (image2, title2) in enumerate(zip(all_images, titles)):
        if i != j:  # Avoid comparing an image with itself
            
            # find the sd for the other image
            features_2 = calculate_features(image2)  
            
            # Add to total and increment count (multiply by weights here)
            features_2_weighted += features_2 * weights
            image_count += 1    
            
            # Calculate the weighted average feature values
            features_2_weighted_2 = features_2_weighted / image_count if image_count else np.zeros_like(features_1)                        
                        
            # We now compare the weighted extracted features from one image with the weighted extracted features 
            # derived from another image using the absolute difference between them.            
            difference = np.abs(features_1 - features_2_weighted_2)
            
            # The sum of these differences is then calculated and used as a metric to quantify the similarity 
            # between the image and the other image. The lower the sum of differences, the more similar the
            # brushstrokes between the images are.
            sum_diff = np.sum(difference) 
            
            # The sum of differences is then normalized to a range of [0, 1]
            max_diff = np.max(difference)
            min_diff = np.min(difference)
            mean_diff = np.mean(difference)
            edge_threshold = (mean_diff - min_diff) / mean_diff
            
            print("########  mean_diff: ", mean_diff)   
            
            # Store the difference in sd in the matrix
            matches_matrix[i, j] = mean_diff            

# Create a DataFrame for the heatmap
heatmap_data = pd.DataFrame(matches_matrix, columns=titles, index=titles)

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
 + geom_text(aes(label='value'), color='white', size=26, ha='center', va='center')  # Adjust the size parameter
 + theme_minimal()
 + labs(title='SIFT Feature Matches Heatmap', x='Image 2', y='Image 1', fill='Number of Matches')
 + theme(figure_size=(25, 20))  # Adjust the figure size as needed
)


import plotly.express as px

df = heatmap_data
fig = px.imshow(df,text_auto=True,aspect="auto")
fig.show()