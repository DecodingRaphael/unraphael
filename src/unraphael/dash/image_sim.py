from __future__ import annotations

import streamlit as st
from config import dump_config, to_session_state

from unraphael.feature import (
    detect_and_extract,
    get_heatmaps,
)

_detect_and_extract = st.cache_data(detect_and_extract)
_get_heatmaps = st.cache_data(get_heatmaps)


def image_similarity_feat_ransac_widget(images, *, method: str):
    """Image similarity for ORB/SIFT."""
    col1, col2 = st.columns(2)

    col1.subheader('Feature extraction')
    col2.subheader('Image matching')

    if method == 'sift':
        label1 = '[SIFT config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT)'
        st.session_state.col1 = dump_config(st.session_state.config['sift'])
    elif method == 'orb':
        label1 = '[ORB config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.ORB)'
        st.session_state.col1 = dump_config(st.session_state.config['orb'])
    else:
        raise ValueError(method)

    col1.text_area(
        label=label1,
        key='col1',
        height=200,
        on_change=to_session_state,
        kwargs={'key': 'col1', 'section': method},
    )

    label2 = '[RANSAC config parameters](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac)'
    st.session_state.col2 = dump_config(st.session_state.config['ransac'])

    col2.text_area(
        label=label2,
        key='col2',
        height=200,
        on_change=to_session_state,
        kwargs={'key': 'col2', 'section': 'ransac'},
    )

    if not st.checkbox('Continue...', key='continue_ransac'):
        st.stop()

    features = _detect_and_extract(
        images=images, method=method, **st.session_state.config[method]
    )

    heatmaps = _get_heatmaps(features, **st.session_state.config['ransac'])

    return heatmaps, features
