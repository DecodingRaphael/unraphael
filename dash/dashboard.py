from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from seaborn import clustermap
from sidebar_logo import add_sidebar_logo
from widgets import show_images

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

st.title('Image similarity')

# SIFT

st.title('Heatmap')

n = len(image_fns)
heatmap = np.random.random((n, n))

fig = clustermap(heatmap, xticklabels=labels, yticklabels=labels)

st.pyplot(fig)
