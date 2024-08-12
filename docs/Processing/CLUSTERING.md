# Clustering Images

This Streamlit page helps you cluster a collection of images. The goal is to group images that might have been created using the same techniques or even within the same artist's workshop, focusing on their structural components rather than their semantic content. Some methods (i.e., SpectralClustering and AffinityPropagation) require a similarity matrix, while others (i.e., Agglomerative, KMeans, DBSCAN, HDBSCAN) use the images' features directly.

---

## Step 1: Displaying Images

The first step is to display the images you've uploaded. The `show_images_widget` function presents the images in a grid format, where you can control the number of columns displayed using the provided slider. This allows you to get an initial look at the images before starting any preprocessing or clustering.

---

## Step 2: Equalizing Images

### Background on Equalization
Equalization is a preprocessing step that adjusts the brightness, contrast, and sharpness of images. This step ensures that the images have consistent visual properties, making them easier to compare during clustering. For example, two images with similar content but different lighting conditions can be made more comparable by equalizing their brightness.

### How to Use the Widget
- **Equalize Brightness**: Adjusts the brightness of all images to match a common reference level.
- **Equalize Contrast**: Ensures that the contrast between light and dark areas is consistent across all images.
- **Equalize Sharpness**: Standardizes the sharpness, making details across images more uniform.

Metrics for these adjustments are displayed, providing feedback on how the images have been modified.

---

## Step 3: Aligning Images to a Mean Image

### Background on Alignment
Alignment is crucial when comparing images, as it ensures that all images are in the same position and orientation. This step helps to account for minor variations in position or rotation that could interfere with accurate clustering.

### How to Use the Widget
- **Motion Model**: Choose a transformation model (i.e., translation, rigid body, scaled rotation, affine, bilinear) to align the images.
- **Feature Method**: Select how the images should be aligned, either to the first image, the mean image, or in a sequential manner. Note that, in most cases involving the clustering of images, aligning to the mean image is recommended to ensure consistency across all images.

### Explanation of Motion Models

- **Translation**: Translation involves shifting the image in the x and y directions without changing its orientation or size. This method is useful when images are slightly off-center but otherwise aligned. It ensures that all images are positioned correctly relative to each other.

- **Rigid Body**: Rigid body alignment maintains the shape and size of the image while allowing for rotations and translations. This model is suitable for images where the objects need to be aligned but can be rotated or shifted. It preserves the relative proportions of the image features during alignment.

- **Scaled Rotation**: Scaled rotation allows for both rotation and scaling of the images in addition to translation. This method is useful when images may differ in size and orientation. It ensures that images are adjusted for both their position and scale to improve alignment.

- **Affine**: Affine transformation provides more flexibility by allowing rotation, scaling, translation, and skewing of the images. It preserves parallelism in the image, making it ideal for aligning images with different perspectives or distortions. This method is suitable for more complex adjustments where simple transformations are insufficient.

- **Bilinear**: Bilinear transformation involves a linear mapping of the image’s coordinates, which can handle translation, scaling, and rotation. This method interpolates pixel values to maintain smoothness and is used when precise adjustments are needed without significant distortion. It is especially useful for correcting minor alignment issues in images.

These aligned images are now prepared for clustering, having been standardized in terms of position, scale, and orientation.

---

## Step 4: Clustering Images

### Background on Clustering
The clustering processes grouping images here are based on structural similarities. Unlike semantic clustering, which might group images based on their content (e.g., animals, landscapes), structural clustering focuses on patterns, textures, and shapes. This focus on structural similairity method is particularly useful in art history or forensic analysis, where you might want to identify pieces created in the same workshop or using the same template.

### Clustering Methods

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

#### HDBSCAN
- **Description**: HDBSCAN is an extension of DBSCAN that builds a hierarchy of clusters and extracts the most stable ones. It does not require the number of clusters to be specified.
- **Application**: Ideal for large, complex datasets where you need to discover clusters of varying densities without having to predetermine the number of clusters.

### Similarity Indices

- **SIFT (Scale-Invariant Feature Transform)**: Detects and describes local features in images, making it robust to changes in scale, rotation, and illumination.
- **SSIM (Structural Similarity Index)**: Measures the similarity between two images by comparing luminance, contrast, and structure, making it sensitive to perceived changes in image quality.
- **CW-SSIM (Complex Wavelet Structural Similarity)**: A variation of SSIM that operates in the wavelet domain, making it more robust to image shifts and rotations.
- **MSE (Mean Squared Error)**: Measures the average squared difference between corresponding pixels of two images, often used as a baseline for comparing image similarity.
- **Brushstrokes**: Focuses on the texture and style of brushstrokes, capturing fine-grained artistic techniques used in painting.

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


### Viewing Cluster Results
After clustering, the results are displayed:

- **Clustered Images**: The images in each cluster are shown, allowing you to visually inspect how well the clustering has grouped similar images.
- **Scatter Plot**: Visualizes the clusters in a 2D space, providing insight into how the images are grouped.
- **Dendrogram**: A hierarchical tree diagram that shows the relationships between clusters.

---

## Conclusion
By following these steps, you can effectively group your images based on their structural components. This clustering can provide valuable insights, such as identifying images created within the same workshop or using the same template, which might not be evident through visual inspection alone.

## References and links for more information
- We make use of functionality provided by the [clustimage package](https://erdogant.github.io/clustimage/pages/html/index.html), which is a tool for unsupervised images *clustering*.

- We also use the [clusteval package](https://erdogant.github.io/clusteval/pages/html/index.html), which is a tool for for unsupervised cluster *evaluation*. Specifically, we make use of its tooling to derive the optimal number of clusters using silhouette, dbindex, and derivatives in combination with clustering methods, such as agglomerative, kmeans, dbscan and hdbscan.
