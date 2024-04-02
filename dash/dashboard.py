from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml
from scipy.cluster.hierarchy import linkage
from seaborn import clustermap
from sidebar_logo import add_sidebar_logo
from skimage.feature import SIFT
from widgets import show_images

from unraphael.feature import (
    detect_and_extract,
    get_heatmaps,
    heatmap_to_condensed_distance_matrix,
)
from unraphael.io import load_images_from_drc


@st.cache_data
def _load_images_from_drc(*args, **kwargs):
    return load_images_from_drc(*args, **kwargs)


def _dump_config(cfg: dict) -> str:
    def cfg_str_representer(dumper, in_str):
        if '\n' in in_str:  # use '|' style for multiline strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', in_str, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', in_str)

    yaml.representer.SafeRepresenter.add_representer(str, cfg_str_representer)
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


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
        image_drc = st.text_input(label='Image directory', value='../data/raw/Bridgewater')
        image_drc = Path(image_drc)

        config_fn = st.text_input(label='Config file', value='config.yaml')
        config_fn = Path(config_fn)

        with open(config_fn) as f:
            st.session_state.unraph_config = yaml.safe_load(f)

        st.download_button(
            label='Download config',
            data=_dump_config(st.session_state.unraph_config),
            file_name='my_config.yaml',
            mime='text/yaml',
        )

        width = st.number_input('Width', value=50, step=10)

        # method = st.number_input('Width', value=50, step=10)

    if not image_drc.exists():
        st.error(f'Cannot find {image_drc}.')

    st.session_state.width = st.session_state.unraph_config['width']
    st.session_state.method = st.session_state.unraph_config['method']
    st.session_state.sift_config = _dump_config(st.session_state.unraph_config['sift_config'])
    st.session_state.orb_config = _dump_config(st.session_state.unraph_config['orb_config'])
    st.session_state.ransac_config = _dump_config(
        st.session_state.unraph_config['ransac_config']
    )

    images = _load_images_from_drc(image_drc, width=width)

    show_images(images, n_cols=8)

    st.title('Image similarity')

    col1, col2 = st.columns(2)

    method_options = [None, 'sift', 'orb', 'outline', 'scale']
    method = col1.selectbox(
        'Select similarity metric',
        options=method_options,
        index=method_options.index(st.session_state.method),
    )

    col1, col2 = st.columns(2)

    if method == 'sift':
        col1.text_area(
            label='[SIFT config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT)',
            key='sift_config',
            height=200,
        )
    elif method == 'orb':
        col1.text_area(
            label='[ORB config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.ORB)',
            key='sift_config',
            height=200,
        )

    if method in ('sift', 'orb'):
        col2.text_area(
            label='[RANSAC config parameters](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac)',
            key='ransac_config',
            height=200,
        )

    if not method:
        st.stop()

    if method in ('outline', 'scale'):
        raise NotImplementedError(method)
        st.stop()

    if not st.button('Start!', type='primary'):
        st.stop()

    st.subheader('Feature extraction')

    bar1 = st.progress(0, text='Extracting features')

    extractor = SIFT()
    features = detect_and_extract(images=images, extractor=extractor, progress=bar1.progress)

    st.subheader('Image matching')

    bar2 = st.progress(0, text='Calculating similarity features')

    heatmap, heatmap_inliers = get_heatmaps(features, progress=bar2.progress)

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
