# Brute-Force Matching with SIFT Descriptors and Ratio Test

# libraries
import numpy as np
import pandas as pd
import glob
import cv2 as cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from plotnine import ggplot, aes, geom_tile, geom_text, theme_minimal, labs, theme, element_text

import plotnine as pn
from scipy.cluster import hierarchy
%matplotlib inline
plt.rcParams["axes.grid"] = False
  
# Initiate SIFT detector
sift = cv2.SIFT_create()

# Brute force matcher with default params
bf = cv2.BFMatcher()

# Load all the paintings in the folder
all_images = []

# Create a list to store the titles of the images
titles = []

# reading images in gray format
for f in glob.iglob("../../data/raw/Bridgewater/*"):
    image = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    titles.append(os.path.basename(f))  # Use only the file name as the title
    all_images.append(image)

# Create a matrix to store the number of good matches between each pair of images
matches_matrix = np.zeros((len(all_images), len(all_images)))

for i, (image1, title1) in enumerate(zip(all_images, titles)):
    
    # find the keypoints and descriptors with SIFT for the current image
    kp_1, desc_1 = sift.detectAndCompute(image1, None)
    
    for j, (image2, title2) in enumerate(zip(all_images, titles)):
        # image2 represents the image associated with the second image at index j.
        # title2 represents the  name associated with the second image at index j.
        if i != j:  # Avoid comparing an image with itself
            
            # find the keypoints and descriptors with SIFT for the other image
            kp_2, desc_2 = sift.detectAndCompute(image2, None)            

            # Brute Force
            matches = bf.knnMatch(desc_1, desc_2, k = 2)
                                  
            # Find all the good matches with Lowe's ratio test
            good_points = []
            
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_points.append([m])            
            
            # Store the number of good matches in the matrix
            matches_matrix[i, j] = len(good_points)
            
             # Ensure symmetry in the matrix
            matches_matrix[j, i] = len(good_points)

# plot with plotnine ----
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
 + geom_text(aes(label='value'), color='white', size=36, ha='center', va='center')  # Adjust the size parameter
 + theme_minimal()
 + labs(title='SIFT Feature Matches Heatmap', x='Image 2', y='Image 1', fill='Number of Matches')
 + theme(figure_size=(50, 30), axis_text_x=element_text(size=35,angle=45), axis_text_y=element_text(size=35))  # Adjust the figure size and label size as needed
)           

