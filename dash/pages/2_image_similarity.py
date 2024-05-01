from __future__ import annotations

import streamlit as st
from config import to_session_state
from image_sim import image_similarity_feat_ransac
from styling import set_custom_css
from widgets import load_config, load_images, show_heatmaps, show_images


def main():
    set_custom_css()

    with st.sidebar:
        load_config()
        images = load_images()

    st.title('Input images')

    _ = show_images(images)

    st.title('Image similarity')

    col, _ = st.columns(2)

    method_options = ['sift', 'orb', 'outline', 'scale']
    method = col.selectbox(
        'Select similarity metric',
        options=method_options,
        key='method',
        on_change=to_session_state,
        kwargs={'key': 'method'},
    )

    if method in ('sift', 'orb'):
        heatmaps, features = image_similarity_feat_ransac(images, method=method)
    else:
        raise NotImplementedError(method)
        st.stop()

    show_heatmaps(heatmaps=heatmaps, labels=tuple(features.keys()))


main()
