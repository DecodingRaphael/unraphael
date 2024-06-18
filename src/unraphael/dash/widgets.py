from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as imageio
import streamlit as st
from config import (
    dump_session_state,
    load_config,
)
from scipy.cluster.hierarchy import linkage
from seaborn import clustermap
from skimage import img_as_ubyte

from unraphael.feature import (
    heatmap_to_condensed_distance_matrix,
)
from unraphael.io import load_images, load_images_from_drc, resize_to_width
from unraphael.locations import image_directory
from unraphael.types import ImageType

if TYPE_CHECKING:
    import numpy as np


_load_images = st.cache_data(load_images)
_load_images_from_drc = st.cache_data(load_images_from_drc)


def show_images_widget(
    images: list[ImageType],
    *,
    n_cols: int = 4,
    key: str = 'show_images',
    message: str = 'Select image',
) -> None | ImageType:
    """Widget to show images with given number of columns."""
    col1, col2 = st.columns(2)
    n_cols = col1.number_input(
        'Number of columns for display', value=8, min_value=1, step=1, key=f'{key}_cols'
    )
    options = [None] + [image.name for image in images]
    selected = col2.selectbox(message, options=options, key=f'{key}_sel')
    selected_image = None

    cols = st.columns(n_cols)

    for i, image in enumerate(images):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]
        if image.name == selected:
            selected_image = image

        col.image(image.data, use_column_width=True, caption=image.name)

    return selected_image


def load_image_widget() -> ImageType:
    """Widget to load a single image with default."""
    load_example = st.sidebar.checkbox('Load example', value=False, key='load_example')
    uploaded_file = st.sidebar.file_uploader('Upload Image ', type=['JPG', 'JPEG'])

    if load_example:
        image_file = image_directory / '0_edinburgh_nat_gallery.jpg'
    else:
        if not uploaded_file:
            st.info('Upload image to continue')
            st.stop()

        image_file = uploaded_file

    name, _ = image_file.name.rsplit('.')
    image = imageio.imread(image_file)

    return ImageType(data=image, name=name)


def load_config_widget():
    """Widget to load config file."""
    uploaded_file = st.sidebar.file_uploader('Load config ', type=['yml', 'yaml'])
    load_config(uploaded_file)

    st.download_button(
        label='Download config',
        data=dump_session_state(),
        file_name='my_config.yaml',
        mime='text/yaml',
        disabled=False,
    )


def load_images_widget(as_ubyte: bool = False, **loader_kwargs) -> list[ImageType]:
    """Widget to load images."""

    load_example = st.sidebar.checkbox('Load example', value=False, key='load_example')
    uploaded_files = st.file_uploader('Upload Images', accept_multiple_files=True)

    if load_example:
        images = _load_images_from_drc(image_directory, **loader_kwargs)
    else:
        if not uploaded_files:
            st.info('Upload images to continue')
            st.stop()

        images = _load_images(uploaded_files, **loader_kwargs)

    if not images:
        raise ValueError('No images were loaded')

    images = [ImageType(name=name, data=data) for name, data in images.items()]

    images = equalize_width_widget(images)

    if as_ubyte:
        images = [image.apply(img_as_ubyte) for image in images]

    return images


def equalize_width_widget(images: list[ImageType]) -> list[ImageType]:
    """This widget equalizes the width of the images."""
    enabled = st.checkbox('Equalize width', value=True)

    width = st.number_input(
        'Width',
        value=240,
        step=10,
        key='width',
        disabled=(not enabled),
    )

    if enabled:
        return [image.apply(resize_to_width, width=width) for image in images]

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


def image_downloads_widget(*, images: list[ImageType], basename: str | None = None):
    """This widget takes a dict of images and shows them with download
    buttons."""
    st.title('Download Images')

    prefix = f'{basename}_' if basename else ''

    cols = st.columns(len(images))

    for col, image in zip(cols, images):
        col.image(image.data, caption=image.name.upper(), use_column_width=True)

        filename = f'{prefix}{image.name}.png'

        col.download_button(
            label=f'Download ({filename})',
            data=imageio.imwrite('<bytes>', image.data, extension='.png'),
            file_name=filename,
            key=filename,
        )
