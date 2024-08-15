# Tutorials

## Clustering Images with the Streamlit Application

This tutorial will guide you through the process of clustering a collection of images using the Streamlit application. The goal is to group images based on their structural similarities, focusing on patterns, textures, and shapes rather than the content of the images. This process can be particularly useful in fields like art history, where identifying pieces created using the same techniques or within the same workshop is of interest.

## Step 1: Displaying Images

### Overview

The first step in the process is to upload and visualize your images. The application provides a widget that displays the uploaded images in a grid format. This allows you to inspect all the images at once and get an initial sense of their visual characteristics.

### Instructions

1. **Upload Your Images**: Use the image uploader in the application to select the images you want to cluster.
2. **Adjust the Grid**: Use the provided slider to set the number of columns in the image grid.
3. **Inspect the Images**: Take a moment to visually inspect the images in the grid to understand their general appearance.

---

## Step 2: Equalizing Images

### Background

Equalization is a crucial preprocessing step that standardizes the visual properties of the images, such as brightness, contrast, and sharpness. This step helps to ensure that variations in these properties do not interfere with the clustering process.

### Instructions

1. **Equalize Brightness**: Use the application to adjust the brightness of all images to a common reference level. This ensures that all images appear uniformly lit.
2. **Equalize Contrast**: Next, equalize the contrast to make sure the differences between light and dark areas are consistent across all images.
3. **Equalize Sharpness**: Finally, standardize the sharpness to ensure that details across the images are uniformly clear.

The application will display metrics that show how the images have been adjusted, helping you to understand the impact of these equalization steps.

---

## Step 3: Aligning Images to a Mean Image

### Background

Alignment is essential for ensuring that all images are consistently positioned and oriented, which is necessary for accurate clustering. This step corrects minor variations in position or rotation that could otherwise affect the clustering results.

### Instructions

1. **Choose a Motion Model**: Select the **Affine** transformation model. This model allows for rotation, scaling, translation, and skewing, providing the flexibility needed for aligning images that might have different perspectives or slight distortions.
2. **Align to the Mean Image**: Set the feature method to align all images to the mean image. Aligning to the mean image ensures consistency across the entire dataset, making it easier to compare images during clustering.

Once aligned, the images will be standardized in terms of position, scale, and orientation, preparing them for the clustering process.

---

## Step 4: Clustering Images

### Background

Clustering involves grouping the images based on their structural similarities. This method is particularly useful when you want to identify images that share the same patterns, textures, or shapes, which might indicate that they were created using the same techniques or in the same workshop.

### Instructions

1. **Choose a Clustering Method**: Use the **Agglomerative Clustering** method. This hierarchical approach builds clusters based on the features of the images and is effective for identifying groups with similar structural properties.

2. **Set the Linkage Type**: Choose the **Ward** linkage type. This method minimizes the variance within each cluster, resulting in more compact and distinct clusters.

3. **Evaluate the Clusters**: Use the **Silhouette Score** to evaluate the quality of the clusters. This metric measures how similar each image is to its own cluster compared to other clusters, providing insight into how well the clustering algorithm has performed.

### Viewing Cluster Results

After the clustering process is complete, the application will display the results in several formats:

1. **Clustered Images**: The images within each cluster will be displayed, allowing you to visually inspect how well the algorithm grouped similar images together.
2. **Scatter Plot**: The scatter plot visualizes the clusters in a 2D space, giving you a sense of how the images are distributed across the different clusters.
3. **Dendrogram**: A hierarchical tree diagram that shows the relationships between clusters, helping you to understand the structure of the clustering process.

---

## Conclusion

By following these steps, you can effectively group your images based on their structural components. This clustering process can reveal valuable insights, such as identifying images that might have been created using the same techniques or within the same workshop, which might not be immediately obvious through visual inspection alone.
