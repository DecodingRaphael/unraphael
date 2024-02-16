# https://copyprogramming.com/howto/checking-images-for-similarity-with-opencv

#### Detecting Image Similarities Using OpenCV Compact Vector Representations.

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import pandas as pd
import plotly.express as px
import numpy as np

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Next we compute the embeddings
# To encode an image, you can use the following code:
# from PIL import Image
# encoded_image = model.encode(Image.open(filepath))

image_names = list(glob.glob('../../data/raw/Bridgewater/*.jpg'))
print("Images:", len(image_names))
encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

# Now we run the clustering algorithm. This function compares images against all other images and
# returns a list with the pairs that have the highest cosine similarity score
processed_images = util.paraphrase_mining_embeddings(encoded_image)
NUM_SIMILAR_IMAGES = 10000

# Get dimensions
num_rows = len(processed_images)
num_columns = len(processed_images[0])

type(processed_images)

# Save embeddings to file
np.save("all_vecs.npy", encoded_image)
# Save image names to file
np.save("all_names.npy", image_names)
np.save("processed_images.npy", processed_images)

# =================
# DUPLICATES
# =================

print('Finding duplicate images...')
# Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
# A duplicate image will have a score of 1.00. It may be 0.9999 due to lossy image compression (.jpg)
duplicates = [image for image in processed_images if image[0] >= 0.999]
# Output the top X duplicate images
for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_names[image_id1])
    print(image_names[image_id2])

# =================
# NEAR DUPLICATES
# =================

print('Finding near duplicate images...')
# Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
# A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.

# Varied images yield diverse outcomes, with higher scores indicating greater similarity and lower scores 
# indicating less similarity

# The threshold is a similarity threshold used to determine whether two images are considered "similar" (> 0/99) 
# or "near-duplicates". Thus, any pair of images with a similarity score below this threshold is considered a 
# "near-duplicate"
threshold = 0.99

# select all pairs that have a similarity score less than 0.99, meaning it includes pairs that are not identical
near_duplicates = [image for image in processed_images if image[0] < threshold]
for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_names[image_id1])
    print(image_names[image_id2])

# Extract similarity scores and image indices from processed_images
similarity_scores = [image[0] for image in processed_images]
image_indices = [(image[1], image[2]) for image in processed_images]

# Create a matrix with similarity scores
matrix_size = len(image_names)
similarity_matrix = [[0] * matrix_size for _ in range(matrix_size)]

# Get dimensions
num_rows = len(similarity_matrix)
num_columns = len(similarity_matrix[0]) if similarity_matrix else 0

print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# Create a matrix with similarity scores, applying the threshold
for score, (image_id1, image_id2) in zip(similarity_scores, image_indices):
    #if score >= threshold:
    similarity_matrix[image_id1][image_id2] = score * 100
    similarity_matrix[image_id2][image_id1] = score * 100

# Round each number to two decimal places
similarity_matrix = [[round(num, 2) for num in row] for row in similarity_matrix]

# Extract characters between the underscore (_) and the first dot (.)
def extract_label(name):
    start_index = name.find('_') + 1
    end_index = name.find('.', start_index)
    return name[start_index:end_index]

# Apply the extraction function to each image name
extracted_labels = [extract_label(os.path.basename(name)) for name in image_names]

# Convert to a Pandas DataFrame
df = pd.DataFrame(similarity_matrix, columns=extracted_labels, index=extracted_labels)

# Create a heatmap using Plotly Express with numerical axes
fig = px.imshow(df, text_auto=True, aspect="auto", x=extracted_labels, y=extracted_labels)
fig.show()

