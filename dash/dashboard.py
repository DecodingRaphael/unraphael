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

    width = st.number_input('Width', value=50, step=10)

    # method = st.number_input('Width', value=50, step=10)

if not image_drc.exists():
    st.error(f'Cannot find {image_drc}.')

images = load_images_from_drc(image_drc, width=width)

show_images(images, n_cols=4)


def _dump_config(cfg: dict) -> str:
    def cfg_str_representer(dumper, in_str):
        if '\n' in in_str:  # use '|' style for multiline strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', in_str, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', in_str)

    yaml.representer.SafeRepresenter.add_representer(str, cfg_str_representer)
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


sift_config = {
    'upsampling': 2,
    'n_octaves': 8,
    'n_scales': 3,
    'sigma_min': 1.6,
    'sigma_in': 0.5,
    'c_dog': 0.013333333333333334,
    'c_edge': 10,
    'n_bins': 36,
    'lambda_ori': 1.5,
    'c_max': 0.8,
    'lambda_descr': 6,
    'n_hist': 4,
    'n_ori': 8,
}

st.session_state.sift_config = _dump_config(sift_config)

orb_config = {
    'downscale': 1.2,
    'n_scales': 8,
    'n_keypoints': 500,
    'fast_n': 9,
    'fast_threshold': 0.08,
    'harris_k': 0.04,
}

st.session_state.orb_config = _dump_config(orb_config)

ransac_config = {
    'max_trials': 100,
    'stop_sample_num': float('inf'),
    'stop_residuals_sum': 0,
    'stop_probability': 1,
    'rng': None,
}

st.session_state.ransac_config = _dump_config(ransac_config)

col1, col2 = st.columns(2)

method = col1.radio(
    'Select similarity metric',
    ['None', 'SIFT', 'ORB', 'Outline', 'Scale'],
    captions=[
        'Do nothing',
        'https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html',
        'https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_orb.html',
        'Not implemented',
        'Not implemented',
    ],
)

if method == 'SIFT':
    col2.text_area(
        label='[SIFT config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT)',
        key='sift_config',
        height=200,
    )
elif method == 'ORB':
    col2.text_area(
        label='[ORB config parameters](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.ORB)',
        key='sift_config',
        height=200,
    )

if method in ('SIFT', 'ORB'):
    col2.text_area(
        label='[RANSAC config parameters](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac)',
        key='ransac_config',
        height=200,
    )

if method == 'None':
    st.stop()

if method in ('Outline', 'Scale'):
    raise NotImplementedError(method)
    st.stop()

st.title('Feature extraction')

bar1 = st.progress(0, text='Extracting features')

extractor = SIFT()
features = detect_and_extract(images=images, extractor=extractor, progress=bar1.progress)

st.title('Image similarity')

bar2 = st.progress(0, text='Calculating similarity features')

heatmap, heatmap_inliers = get_heatmaps(features, progress=bar2.progress)

st.title('Heatmap')

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

st.pyplot(fig1)

st.title('Heatmap inliers')

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

st.pyplot(fig2)
