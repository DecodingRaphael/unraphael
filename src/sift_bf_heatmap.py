# Brute-Force Matching with SIFT Descriptors and Ratio Test

# libraries
import numpy as np
import pandas as pd
import glob
import cv2 as cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from plotnine import ggplot, aes, geom_tile, geom_text, theme_minimal,labs,theme
import plotnine as pn
from scipy.cluster import hierarchy
%matplotlib inline
plt.rcParams["axes.grid"] = False
  
# Initiate SIFT detector
sift = cv2.SIFT_create()

# Brute force matcher with default params
bf = cv2.BFMatcher()

# Load all the copies of the original painting
all_images = []

titles = []

# reading images in gray format
for f in glob.iglob("../data/raw/Bridgewater/*"):
    image = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    titles.append(os.path.basename(f))  # Use only the file name as the title
    all_images.append(image)

# Create a matrix to store the number of good matches between each pair of images
matches_matrix = np.zeros((len(all_images), len(all_images)))

for i, (image1, title1) in enumerate(zip(all_images, titles)):
    
    # find the keypoints and descriptors with SIFT for the current image
    kp_1, desc_1 = sift.detectAndCompute(image1, None)
    
    for j, (image2, title2) in enumerate(zip(all_images, titles)):
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
 + geom_text(aes(label='value'), color='white', size=20, ha='center', va='center')  # Adjust the size parameter
 + theme_minimal()
 + labs(title='SIFT Feature Matches Heatmap', x='Image 2', y='Image 1', fill='Number of Matches')
 + theme(figure_size=(25, 20))  # Adjust the figure size as needed
)           

# plot with seaborn ----

# Create a clustered heatmap with Seaborn, including annotations and dendrogram
sns.set(style="whitegrid")

# Compute hierarchical clustering
linkage_matrix = hierarchy.linkage(heatmap_data.values,metric="euclidean", method='ward')

# Create a cluster map with Seaborn, including annotations and dendrogram
sns.clustermap(heatmap_data, cmap = "vlag", linewidths=.5, figsize=(15, 15), row_linkage=linkage_matrix, col_linkage=linkage_matrix, annot=True)
plt.show()

heatmap_sns = sns.clustermap(heatmap_data, metric="cosine", standard_scale=1, method="ward", cmap="viridis")

df_ro = heatmap_sns.data2d(heatmap_data) 

# plot with plotly ----
# see https://plotly.com/python/heatmaps/

import plotly.express as px

df = heatmap_data
fig = px.imshow(df,text_auto=True,aspect="auto")
fig.show()

import dash_bio

#https://dash.plotly.com/dash-bio/clustergram
