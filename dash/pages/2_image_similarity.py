from __future__ import annotations

import streamlit as st
from config import to_session_state
from image_sim import image_similarity_feat_ransac_widget
from styling import set_custom_css
from widgets import (
    load_config_widget,
    load_images_widget,
    show_heatmaps_widget,
    show_images_widget,
)


def main():
    set_custom_css()

    with st.sidebar:
        load_config_widget()
        st.write('---')
        images = load_images_widget()

    st.title('Input images')

    _ = show_images_widget(images)

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
        heatmaps, features = image_similarity_feat_ransac_widget(images, method=method)
    else:
        raise NotImplementedError(method)
        st.stop()

    show_heatmaps_widget(heatmaps=heatmaps, labels=tuple(features.keys()))


if __name__ == '__main__':
    main()
