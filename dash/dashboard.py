from __future__ import annotations

import streamlit as st
from config import dump_config, to_session_state
from sidebar_logo import add_sidebar_logo
from widgets import load_config, load_images, show_heatmaps, show_images

from unraphael.feature import (
    detect_and_extract,
    get_heatmaps,
)

_detect_and_extract = st.cache_data(detect_and_extract)
_get_heatmaps = st.cache_data(get_heatmaps)


def main():
    st.set_page_config(
        page_title='Unraphael dashboard', page_icon=':framed-picture:', layout='wide'
    )
    st.write(
        '<style>textarea[class^="st-"] { font-family: monospace; font-size: 14px; }</style>',
        unsafe_allow_html=True,
    )
    add_sidebar_logo()

    with st.sidebar:
        load_config()
        images = load_images()

    st.title('Input images')

    _ = show_images(images)

    st.title('Image similarity')

    col1, col2 = st.columns(2)

    method_options = ['sift', 'orb', 'outline', 'scale']
    method = col1.selectbox(
        'Select similarity metric',
        options=method_options,
        key='method',
        on_change=to_session_state,
        kwargs={'key': 'method'},
    )

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
        label1 = 'nrst'

    if st.session_state.method in ('sift', 'orb'):
        label2 = '[RANSAC config parameters](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac)'
        st.session_state.col2 = dump_config(st.session_state.config['ransac'])
    else:
        label2 = 'heai'

    col1.text_area(
        label=label1,
        key='col1',
        height=200,
        on_change=to_session_state,
        kwargs={'key': 'col1', 'section': method},
    )
    col2.text_area(
        label=label2,
        key='col2',
        height=200,
        on_change=to_session_state,
        kwargs={'key': 'col2', 'section': 'ransac'},
    )

    if st.session_state.method in ('outline', 'scale'):
        raise NotImplementedError(st.session_state.method)
        st.stop()

    if not st.checkbox('Continue...'):
        st.stop()

    features = _detect_and_extract(
        images=images, method=method, **st.session_state.config[method]
    )

    heatmaps = _get_heatmaps(features, **st.session_state.config['ransac'])

    show_heatmaps(heatmaps=heatmaps, labels=tuple(features.keys()))


main()
