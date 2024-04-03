from __future__ import annotations

import streamlit as st
from config import dump_config, to_session_state
from scipy.cluster.hierarchy import linkage
from seaborn import clustermap
from sidebar_logo import add_sidebar_logo
from skimage.feature import ORB, SIFT
from streamlit_image_select import image_select
from widgets import load_config, load_images, show_images

from unraphael.feature import (
    detect_and_extract,
    get_heatmaps,
    heatmap_to_condensed_distance_matrix,
)


def main():
    st.set_page_config(
        page_title='Unraphael dashboard', page_icon=':framed-picture:', layout='wide'
    )
    st.write(
        '<style>textarea[class^="st-"] { font-family: monospace; font-size: 14px; }</style>',
        unsafe_allow_html=True,
    )
    add_sidebar_logo()

    st.title('Input images')

    with st.sidebar:
        load_config()
        images = load_images()

    show_images(images, n_cols=8)

    import numpy as np
    from PIL import Image
    from skimage.color import gray2rgb

    ims = [
        Image.fromarray(gray2rgb(im * 255).astype(np.uint8), 'RGB') for im in images.values()
    ]

    img = image_select(
        label='Select an image',
        images=ims,
        captions=list(images.keys()),
    )

    st.stop()

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

    if not st.button('Start!', type='primary'):
        st.stop()

    st.subheader('Feature extraction')

    bar1 = st.progress(0, text='Extracting features')

    if method == 'sift':
        extractor = SIFT(**st.session_state.config[method])
    elif method == 'orb':
        extractor = ORB(**st.session_state.config[method])

    features = detect_and_extract(images=images, extractor=extractor, progress=bar1.progress)

    st.subheader('Image matching')

    bar2 = st.progress(0, text='Calculating similarity features')

    heatmap, heatmap_inliers = get_heatmaps(
        features, progress=bar2.progress, **st.session_state.config['ransac']
    )

    col1, col2 = st.columns(2)
    col1.title('Heatmap')

    # single average complete median weighted centroid ward
    method = 'average'
    names = tuple(features.keys())

    d = heatmap_to_condensed_distance_matrix(heatmap)
    z = linkage(d, method=method)

    fig1 = clustermap(
        heatmap,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt='d',
        row_linkage=z,
        col_linkage=z,
    )

    col1.pyplot(fig1)

    col2.title('Heatmap inliers')

    d = heatmap_to_condensed_distance_matrix(heatmap_inliers)
    z = linkage(d, method=method)

    fig2 = clustermap(
        heatmap_inliers,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt='d',
        row_linkage=z,
        col_linkage=z,
    )

    col2.pyplot(fig2)


main()
