# Guide to Clustering Images Based on Structural Similarity

This Streamlit page helps you cluster a collection of images based on their structural similarity. The goal is to group images that might have been created using the same techniques or even within the same artist's workshop, focusing on their structural components rather than their semantic content.

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
- **Feature Method**: Select how the images should be aligned, either to the first image, the mean image, or in a sequential manner. Note that, in most cases, aligning to the mean image is recommended!

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
Clustering groups images based on structural similarities. Unlike semantic clustering, which might group images based on their content (e.g., animals, landscapes), structural clustering focuses on patterns, textures, and shapes. This method is particularly useful in art history or forensic analysis, where you might want to identify pieces created in the same workshop or using the same template.

### Clustering Methods
- **Spectral Clustering & Affinity Propagation**: These methods cluster images based on similarity matrices, which quantify the similarity between each pair of images.
  - **Similarity Measures**: You can choose between different similarity measures (e.g., SIFT, SSIM) that best capture the structural features you're interested in.
  - **Option to Specify Clusters**: In Spectral Clustering, you can choose whether to specify the number of clusters beforehand.

- **Agglomerative, KMeans, DBSCAN, HDBSCAN**: These methods cluster images based on their features rather than a predefined similarity matrix.
  - **Cluster Evaluation**: Choose how the quality of the clusters should be evaluated (e.g., silhouette, DB index).
  - **Linkage Type**: Select the method used to calculate the distance between clusters (e.g., ward, complete).

### Viewing Cluster Results
After clustering, the results are displayed:

- **Clustered Images**: The images in each cluster are shown, allowing you to visually inspect how well the clustering has grouped similar images.
- **Scatter Plot**: Visualizes the clusters in a 2D space, providing insight into how the images are grouped.
- **Dendrogram**: A hierarchical tree diagram that shows the relationships between clusters.

---

## Conclusion
By following these steps, you can effectively group your images based on their structural components. This clustering can provide valuable insights, such as identifying images created within the same workshop or using the same template, which might not be evident through visual inspection alone.

## References and links for more information
