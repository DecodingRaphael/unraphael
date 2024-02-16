import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import pandas as pd
import plotly.express as px

# Load the OpenAI CLIP Model
model = SentenceTransformer('clip-ViT-B-32')

# Next we compute the embeddings
image_names = list(glob.glob('../../data/raw/Bridgewater/*.jpg'))
encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

# Now we run the clustering algorithm. This function compares images against all other images and
# returns a list with the pairs that have the highest cosine similarity score
processed_images = util.paraphrase_mining_embeddings(encoded_image)
NUM_SIMILAR_IMAGES = 100

# Streamlit app
st.title("Image Similarity Analysis with OpenAI CLIP Model")

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.99, 0.01)

# Display duplicate images
st.header("Duplicate Images")
duplicates = [image for image in processed_images if image[0] >= 0.999]
for score, image_id1, image_id2 in duplicates[:NUM_SIMILAR_IMAGES]:
    st.subheader(f"Score: {score * 100:.3f}%")
    st.image(image_names[image_id1], caption="Image 1", use_column_width=True)
    st.image(image_names[image_id2], caption="Image 2", use_column_width=True)

# Display near-duplicate images based on the selected threshold
st.header("Near Duplicate Images")
near_duplicates = [image for image in processed_images if image[0] < threshold]
for score, image_id1, image_id2 in near_duplicates[:NUM_SIMILAR_IMAGES]:
    st.subheader(f"Score: {score * 100:.3f}%")
    st.image(image_names[image_id1], caption="Image 1", use_column_width=True)
    st.image(image_names[image_id2], caption="Image 2", use_column_width=True)
