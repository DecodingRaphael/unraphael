from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from seaborn import clustermap
from sidebar_logo import add_sidebar_logo

add_sidebar_logo()

st.title('Input images')

with st.sidebar:
    image_dir = st.text_input(label='Image directory', value='../data/raw/Bridgewater')
    image_dir = Path(image_dir)

if not image_dir.exists():
    st.error(f'Cannot find {image_dir}.')

image_fns = list(image_dir.glob('*'))

cols = st.columns(4)

labels = []
for i, fn in enumerate(image_fns):
    col = cols[i % 4]
    label = fn.stem
    labels.append(label)
    col.image(str(fn), caption=label)

st.title('Image similarity')

# SIFT

st.title('Heatmap')


n = len(image_fns)
heatmap = np.random.random((n, n))

fig = clustermap(heatmap, xticklabels=labels, yticklabels=labels)

st.pyplot(fig)
