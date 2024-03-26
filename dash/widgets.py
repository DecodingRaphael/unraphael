from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import numpy as np


def show_images(images: dict[str, np.ndarray], *, n_cols: int = 4):
    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        col = cols[i % n_cols]
        col.image(im, use_column_width=True, caption=name)
