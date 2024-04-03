from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from config import _load_config, _update_session_state, dump_session_state, to_session_state

from unraphael.io import load_images_from_drc

_load_images_from_drc = st.cache_data(load_images_from_drc)


if TYPE_CHECKING:
    import numpy as np


def show_images(images: dict[str, np.ndarray], *, n_cols: int = 4):
    """Widget to show images with given number of columns."""
    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        col = cols[i % n_cols]
        col.image(im, use_column_width=True, caption=name)


def load_config():
    """Widget to load config file."""
    config_fn = st.text_input(label='Config file', value='config.yaml')
    config = _load_config(config_fn)
    if 'config' not in st.session_state:
        _update_session_state(config)

    st.download_button(
        label='Download config',
        data=dump_session_state(),
        file_name='my_config.yaml',
        mime='text/yaml',
        disabled=False,
    )


def load_images():
    """Widget to load images."""
    image_drc = st.text_input(label='Image directory', value='../data/raw/Bridgewater')
    image_drc = Path(image_drc)

    if not image_drc.exists():
        st.error(f'Cannot find {image_drc}.')

    width = st.number_input(
        'Width', step=10, key='width', on_change=to_session_state, kwargs={'key': 'width'}
    )

    return _load_images_from_drc(image_drc, width=width)
