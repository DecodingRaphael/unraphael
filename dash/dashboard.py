from __future__ import annotations

from pathlib import Path

import streamlit as st
from scipy.cluster.hierarchy import linkage
from seaborn import clustermap
from sidebar_logo import add_sidebar_logo
from skimage.feature import SIFT
from widgets import show_images

from unraphael.feature import (
    detect_and_extract,
    get_heatmaps,
    heatmap_to_condensed_distance_matrix,
)
from unraphael.io import load_images_from_drc

add_sidebar_logo()

st.title('Input images')

with st.sidebar:
    image_drc = st.text_input(label='Image directory', value='../data/raw/Bridgewater')
    image_drc = Path(image_drc)

    width = st.number_input('Width', value=540, step=10)

if not image_drc.exists():
    st.error(f'Cannot find {image_drc}.')

images = load_images_from_drc(image_drc, width=width)

show_images(images, n_cols=4)

st.title('Feature extraction')

bar1 = st.progress(0, text='Extracting features')

extractor = SIFT()
features = detect_and_extract(images=images, extractor=extractor, progress=bar1.progress)

st.title('Image similarity')

bar2 = st.progress(0, text='Calculating similarity features')

heatmap, heatmap_inliers = get_heatmaps(features, progress=bar2.progress)

st.title('Heatmap')

# single average complete median weighted centroid ward
method = 'average'
names = tuple(features.keys())

d = heatmap_to_condensed_distance_matrix(heatmap)
z = linkage(d, method=method)

fig1 = clustermap(
    heatmap,
    xticklabels=names,
    yticklabels=names,
    annot=True,
    fmt='d',
    row_linkage=z,
    col_linkage=z,
)

st.pyplot(fig1)

st.title('Heatmap inliers')

d = heatmap_to_condensed_distance_matrix(heatmap_inliers)
z = linkage(d, method=method)

fig2 = clustermap(
    heatmap_inliers,
    xticklabels=names,
    yticklabels=names,
    annot=True,
    fmt='d',
    row_linkage=z,
    col_linkage=z,
)

st.pyplot(fig2)
