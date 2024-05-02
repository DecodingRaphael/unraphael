from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import imageio.v3 as imageio
import streamlit as st
from config import (
    _load_config,
    _update_session_state,
    dump_session_state,
    to_session_state,
)
from scipy.cluster.hierarchy import linkage
from seaborn import clustermap

from unraphael.feature import (
    heatmap_to_condensed_distance_matrix,
)
from unraphael.io import load_images_from_drc, load_images
from importlib.resources import files

if TYPE_CHECKING:
    import numpy as np

data_directory = files('unraphael.data')

_load_images = st.cache_data(load_images)
_load_images_from_drc = st.cache_data(load_images_from_drc)


def show_images_widget(images: dict[str, np.ndarray], *, n_cols: int = 4):
    """Widget to show images with given number of columns."""
    col1, col2 = st.columns(2)
    n_cols = col1.number_input('Number of columns', value=8, min_value=1, step=1)
    selected = col2.selectbox('Selected', options=images.keys())

    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]
        col.image(im, use_column_width=True, caption=name)

    return selected


def load_image() -> tuple[str, np.ndarray]:
    """Widget to load a single image with default."""
    load_example = st.sidebar.checkbox('Load example', value=False)
    uploaded_file = st.sidebar.file_uploader('Upload Image ', type=['JPG', 'JPEG'])

    if load_example:
        image_drc = Path('../data/raw/Bridgewater')
        image_file = image_drc / '0_Edinburgh_Nat_Gallery.jpg'
    else:
        if not uploaded_file:
            st.info('Upload image to continue')
            st.stop()

        image_file = uploaded_file

    name, _ = image_file.name.rsplit('.')
    image = imageio.imread(image_file)

    return name, image


def load_config_widget():
    """Widget to load config file."""
    uploaded_file = st.sidebar.file_uploader('Load config ', type=['yml', 'yaml'])
    if not uploaded_file:
        uploaded_file = data_directory / 'config.yaml'

    config = _load_config(uploaded_file)
    if 'config' not in st.session_state:
        _update_session_state(config)

    st.download_button(
        label='Download config',
        data=dump_session_state(),
        file_name='my_config.yaml',
        mime='text/yaml',
        disabled=False,
    )


def load_images_widget():
    """Widget to load images."""

    load_example = st.sidebar.checkbox('Load example', value=False)
    uploaded_files = st.file_uploader('Upload Images', accept_multiple_files=True)

    width = st.number_input(
        'Width',
        step=10,
        key='width',
        on_change=to_session_state,
        kwargs={'key': 'width'},
    )

    if load_example:
        image_drc = Path('../data/raw/Bridgewater')
        images = _load_images_from_drc(image_drc, width=width)
    else:
        if not uploaded_files:
            st.info('Upload images to continue')
            st.stop()

        images = _load_images(uploaded_files, width=width)

    return images


def show_heatmaps_widget(heatmaps: dict[str, np.ndarray], labels: list[str]):
    """Widget to show heatmaps."""
    st.title('Heatmaps')

    col, _ = st.columns(2)

    options = 'single', 'average', 'complete', 'median', 'weighted', 'centroid', 'ward'
    method = col.selectbox('Linking method', options=options, index=1)

    cols = st.columns(len(heatmaps))

    for col, (name, heatmap) in zip(cols, heatmaps.items()):
        col.subheader(name.capitalize())

        d = heatmap_to_condensed_distance_matrix(heatmap)
        z = linkage(d, method=method)

        fig = clustermap(
            heatmap,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='d',
            row_linkage=z,
            col_linkage=z,
        )

        col.pyplot(fig)
