# Clustering Images

This page of the application provides a comprehensive tool for clustering images based on their structural components, focusing on identifying similarities in image characteristics rather than semantic content.

By following the steps described below, you can effectively group your images based on their structural components. This clustering can provide valuable insights, such as identifying images created within the same workshop by having used a template for mechanical reproduction which might not be evident through visual inspection alone.

# Key Features

### Image Preprocessing

- Equalization of image brightness, contrast, and sharpness
- Alignment of images using various transformation models

### Clustering Methods

- We make use of functionality provided by the [clusteval package](https://erdogant.github.io/clusteval/pages/html/index.html) to derive the optimal number of clusters using silhouette, dbindex, and derivatives in combination with clustering methods, such as agglomerative, kmeans, dbscan and hdbscan. 
- For more information on the methods or interpreting the results, we highly recommend looking into the [clusteval documentation](https://erdogant.github.io/clusteval/pages/html/index.html).
- Multiple clustering algorithms

- Clustering based on a similarity matrix composed of various similarity measures, or on the images' features directly
- Various similarity measures can be selected to compute the similarity matrix
- Cluster based on either outer contours or the extracted main figures

# Workflow

### 1 Displaying Images

The first step is to display the uploaded images, presented in a grid format where you can control the number of columns displayed using the provided slider. This allows you to get an initial look at the images before starting any preprocessing or clustering.

### 2 Equalizing Images

Equalization is a preprocessing step that adjusts the brightness, contrast, and sharpness of images. This step ensures that the images have consistent visual properties, making them easier to compare during clustering. For example, two images with similar content but different lighting conditions can be made more comparable by equalizing their brightness.

How to Use the Widget
- **Equalize Brightness**: Adjusts the brightness of all images to match a common reference level.
- **Equalize Contrast**: Ensures that the contrast between light and dark areas is consistent across all images.
- **Equalize Sharpness**: Standardizes the sharpness, making details across images more uniform.

Metrics for these adjustments are displayed, providing feedback on how the images have been modified.

### 3 Aligning Images to a Mean Image

Alignment is crucial when comparing images, as it ensures that all images are in the same position and orientation. This step helps to account for minor variations in position or rotation that could interfere with accurate clustering.

How to Use the Widget
- **Motion Model**: Choose a transformation model (i.e., translation, rigid body, scaled rotation, affine, bilinear) to align the images.
- **Feature Method**: Select how the images should be aligned, either to the first image, the mean image, or in a sequential manner. Note that, in most cases involving the clustering of images, **aligning to the mean image** is highly recommended(!) to ensure consistency across all images.

### Further Explanation of Motion Models

- **Translation**: Translation involves shifting the image in the x and y directions without changing its orientation or size. This method is useful when images are slightly off-center but otherwise aligned. It ensures that all images are positioned correctly relative to each other.

- **Rigid Body**: Rigid body alignment maintains the shape and size of the image while allowing for rotations and translations. This model is suitable for images where the objects need to be aligned but can be rotated or shifted. It preserves the relative proportions of the image features during alignment.

- **Scaled Rotation**: Scaled rotation allows for both rotation and scaling of the images in addition to translation. This method is useful when images may differ in size and orientation. It ensures that images are adjusted for both their position and scale to improve alignment.

- **Affine**: Affine transformation provides more flexibility by allowing rotation, scaling, translation, and skewing of the images. It preserves parallelism in the image, making it ideal for aligning images with different perspectives or distortions. This method is suitable for more complex adjustments where simple transformations are insufficient.

- **Bilinear**: Bilinear transformation involves a linear mapping of the image's coordinates, which can handle translation, scaling, and rotation. This method interpolates pixel values to maintain smoothness and is used when precise adjustments are needed without significant distortion. It is especially useful for correcting minor alignment issues in images.

These aligned images are now prepared for clustering, having been standardized in terms of position, scale, and orientation.

### 4 Clustering Images

Two primary clustering approaches are available: 

- *Outer Contours Clustering* 
- *Complete Figures Clustering*. 

Both of these clustering processes group images based on structural similarities. Unlike semantic clustering, which might group images based on their color and content (e.g., animals, landscapes), structural clustering focuses on patterns, textures, and shapes.


### A. Outer Contours Clustering

Extracts and analyzes image outlines. The cluster process allows one - or a combination of - the following feature components used as input:

- **Fourier Descriptors:** Describes the shape of the contour using Fourier coefficients, capturing the contour's overall structure.
- **Hu Moments:** A set of seven moments that describe the shape of the contour, providing information about its orientation, size, and shape.
- **HOG Features:** Histogram of Oriented Gradients (HOG) captures the distribution of gradient orientations in the contour, useful for detecting edges and shapes.
- **Aspect Ratio:** The ratio of the width to the height of the bounding box around the contour, providing information about the contour's shape.
- **Contour Length:** The total length of the contour, which can indicate the complexity or level of detail in the shape.
- **Centroid Distance:** The distance between the centroid of the contour and the image center, useful for understanding the contour's position within the image.
- **Hausdorff Distance:** The Hausdorff distance measures the similarity between two contours by computing the maximum distance from a point on one contour to the nearest point on the other contour. In this implementation, the Hausdorff distance is computed for each pair of contours, resulting in a distance matrix where each entry corresponds to the Hausdorff distance between two contours. For each contour, the average Hausdorff distance to all other contours is calculated.
- **Procrustes Distance**: The Procrustes distance quantifies the similarity between two contours by first aligning them through translation, scaling, and rotation, and then computing the Euclidean distance between corresponding points. Here, the Procrustes distance is calculated for each pair of contours, generating a matrix similar to the Hausdorff distance matrix. For each contour, the average Procrustes distance to all other contours is computed.

### B. Complete Figures Clustering
Uses entire image for clustering, preferably with the background removed. Multiple algorithms are supported:

#### Spectral Clustering
- **Description**: Spectral Clustering uses eigenvalues of a similarity matrix to reduce dimensionality and group images based on structural features. It works well for clusters with complex boundaries.
- **Application**: Ideal for grouping images with subtle structural similarities that are not easily captured by other clustering methods.

#### Affinity Propagation
- **Description**: Affinity Propagation does not require the number of clusters to be specified beforehand. It works by passing messages between data points to identify exemplars, which represent clusters.
- **Application**: Useful when you want the algorithm to determine the optimal number of clusters based on the data itself.

#### Agglomerative Clustering
- **Description**: Agglomerative Clustering is a hierarchical method that builds clusters by successively merging pairs of clusters. It is particularly effective for small datasets with clear hierarchical relationships.
- **Application**: Suitable for applications where the relationships between images can be naturally represented in a hierarchical structure.

#### KMeans
- **Description**: KMeans partitions the images into clusters by minimizing the variance within each cluster. It requires specifying the number of clusters beforehand.
- **Application**: Best used when you have a good estimate of the number of clusters and need a quick, straightforward clustering approach.

#### DBSCAN
- **Description**: DBSCAN identifies clusters based on the density of points. It can find clusters of arbitrary shape and is robust to noise.
- **Application**: Effective for datasets where clusters vary in shape and size, especially when dealing with noise or outliers.

### Similarity Indices
When clustering complete figures, various similarity indices can be used as input for matrix-based clustering:

- **SIFT (Scale-Invariant Feature Transform)**: Detects and describes local features in images, making it robust to changes in scale, rotation, and illumination.
- **SSIM (Structural Similarity Index)**: Measures the similarity between two images by comparing luminance, contrast, and structure, making it sensitive to perceived changes in image quality. Widely used and foundational metric for structural similarity.
- **CW-SSIM (Complex Wavelet Structural Similarity)**: A variation of SSIM that operates in the wavelet domain, making it more robust to image shifts and rotations. More robust than SSIM, but computationally more expensive.
- **IW-SSIM (Information Content Weighted Structural Similarity Index)**: This metric extends SSIM by weighting the similarity based on local information content, making it particularly useful in high-variance regions of an image.
- **FSIM (Feature Similarity Index Measure)**: Uses phase congruency and gradient magnitude to measure similarity, focusing on perceptually important features. Gaining traction in the field for its perceptual relevance and accuracy.
- **MSE (Mean Squared Error)**: Traditional metric measuring the average squared difference between corresponding pixels of two images, often used as a baseline for comparing image similarity.
- **Brushstrokes**: Focuses on the texture and style of brushstrokes, capturing fine-grained artistic techniques used in painting. For this, a combination of several edge detection algorithms is used.

### Cluster Evaluation

- **Silhouette Method**: Evaluates the quality of clusters by measuring how similar an image is to its own cluster compared to other clusters.
- **DBIndex (Davies-Bouldin Index)**: Measures the average similarity ratio of each cluster with the cluster that is most similar to it, where a lower index indicates better clustering.
- **Derivative Method**: Estimates the optimal number of clusters by analyzing the rate of change in the clustering criterion, helping to identify a natural "elbow" point.

### Linkage Types

- **Ward**: Minimizes the variance within clusters, producing compact and spherical clusters.
- **Single**: Uses the smallest distance between clusters, resulting in elongated and chain-like clusters.
- **Complete**: Considers the maximum distance between clusters, leading to more compact clusters.
- **Average**: Calculates the average distance between all points of two clusters, balancing between single and complete linkage.
- **Weighted**: Similar to average linkage but gives more weight to larger clusters.
- **Centroid**: Uses the distance between the centroids of clusters, effective for well-separated clusters.
- **Median**: Similar to centroid linkage but considers the median of all pairwise distances, less affected by outliers.


### 5 Viewing Cluster Results
After clustering, the application provides multiple visualizations:

- **Silhouette plot**: depicts how well our data points fit into the clusters they've been assigned to, informing on the quality of fit cohesion.
- **Scatter Plot**: Visualizes the clusters in a 2D space, providing insight into how the images are grouped into different clusters.
- **Dendrogram**: A hierarchical tree diagram that shows the relationships between clusters.
- **Performance metrics**: Displays metrics such as the silhouette_score, Davies Bouldin Score, and Calinski Harabasz Score, helping you evaluate the quality of the clustering.
- **Clustered Images**: The images in each cluster are shown, allowing you to visually inspect how well the clustering has grouped similar images.

### Performance metrics
In practice, the following metrics are often used alongside each other to evaluate clustering quality from multiple perspectives.

**Silhoutte Score** measures how similar an object is to its own cluster compared to other clusters. The score ranges from -1 to 1, where: +1 indicates that the sample is far away from the neighboring clusters and very close to the cluster it is assigned to. 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters. -1 indicates that the sample might have been assigned to the wrong cluster, as it is closer to a neighboring cluster than to the cluster it is assigned to.

**Davies-Bouldin Index (DBI)** is a metric used to evaluate the quality of clustering algorithms. It measures the average similarity ratio of each cluster with its most similar cluster, taking into account both intra-cluster similarity (compactness) and inter-cluster difference (separation).

- Lower DBI Values indicate better clustering quality**, as it implies that clusters are more compact and better separated.
- Higher DBI Values suggest poor clustering, as it means clusters are either not very compact or not well separated.

Advantages: The DBI is easy to compute and useful for comparing different clustering configurations.

Limitations: It can be sensitive to the shape and size of clusters and may not work well with clusters of varying density or elongated shapes


**Calinski-Harabasz Index** is calculated based on the ratio of the dispersion between clusters to the dispersion within clusters. The idea is that well-separated, compact clusters should have a high between-cluster variance and a low within-cluster variance. By maximizing the between-cluster dispersion and minimizing the within-cluster dispersion, the Calinski-Harabasz Index helps in selecting the optimal number of clusters and evaluating the performance of different clustering algorithms.

- Higher Calinski-Harabasz Index values indicate better clustering quality. A higher value suggests that clusters are well separated (high between-cluster dispersion) and compact (low within-cluster dispersion).
- Lower Calinski-Harabasz Index values indicate poorer clustering, where clusters may be overlapping or not distinct.

Advantages: The Calinski-Harabasz Index is straightforward to calculate and works well for convex, spherical clusters of similar sizes.

Limitations: It may not perform as well with clusters that have irregular shapes or vary widely in density It may not be suitable for datasets with outliers or noise.
