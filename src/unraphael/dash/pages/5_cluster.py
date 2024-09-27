from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st
from styling import set_custom_css
from widgets import load_images_widget

from unraphael.dash.image_clustering import (
    align_images_to_mean,
    build_similarity_matrix,
    cluster_images,
    clustimage_clustering,
    compute_metrics,
    equalize_images,
    plot_clusters,
    plot_dendrogram,
)


def show_images_widget(
    images: dict[str, np.ndarray],
    *,
    n_cols: int = 4,
    key: str = 'show_images',
    message: str = 'Select image',
) -> None | str:
    """Widget to show images with given number of columns."""
    col1, col2 = st.columns(2)
    n_cols = col1.number_input(
        'Number of columns for display', value=4, min_value=1, step=1, key=f'{key}_cols'
    )

    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]

        col.image(im, use_column_width=True, caption=name, clamp=True)

    return None  # primarily for display


def display_equalization_options() -> dict[str, bool]:
    """Display the equalization options UI and return the selected options."""
    st.subheader('Equalization parameters')
    brightness = st.checkbox('Equalize brightness', value=False)
    contrast = st.checkbox('Equalize contrast', value=False)
    sharpness = st.checkbox('Equalize sharpness', value=False)

    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
    }


def display_metrics(metrics: dict[str, float], title: str) -> None:
    """Display the metrics with a specified title."""
    st.subheader(title)
    col3, col4 = st.columns(2)

    # Display mean and SD for each metric separately
    col3.metric('Mean Normalized Brightness', metrics['mean_normalized_brightness'])
    col4.metric('SD Normalized Brightness', metrics['sd_normalized_brightness'])

    col3.metric('Mean Normalized Contrast', metrics['mean_normalized_contrast'])
    col4.metric('SD Normalized Contrast', metrics['sd_normalized_contrast'])

    col3.metric('Mean Normalized Sharpness', metrics['mean_normalized_sharpness'])
    col4.metric('SD Normalized Sharpness', metrics['sd_normalized_sharpness'])


def equalize_images_widget(*, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Widget for equalizing images in terms of brightness, contrast, and
    sharpness."""

    # Compute metrics before equalization
    all_images = list(images.values())
    before_metrics = compute_metrics(all_images)

    # Display equalization options first
    preprocess_options = display_equalization_options()

    # Create columns for metrics
    col1, col2 = st.columns(2)

    with col1:
        # Display metrics before equalization
        display_metrics(before_metrics, 'Metrics Before Equalization')

    with col2:
        # Initialize a variable for equalized images
        equalized_images = all_images.copy()  # Keep original images initially

        # Perform equalization if any checkbox is ticked
        if any(preprocess_options.values()):
            equalized_images = equalize_images(all_images, **preprocess_options)
            after_metrics = compute_metrics(equalized_images)
            display_metrics(after_metrics, 'Metrics After Equalization')

    # Map the equalized images back to their original names
    return {name: equalized_images[i] for i, name in enumerate(images.keys())}


def align_to_mean_image_widget(
    *,
    images: dict[str, np.ndarray],
) -> Optional[dict[str, np.ndarray]]:
    """This widget aligns all images, preferably to their mean value but other
    aligning options are also available.

    The aligned images are input for the later clustering of the images.
    """

    st.subheader('Alignment parameters')
    st.write(
        (
            'This section allows aligning the images (typically on their mean image) '
            'to ensure the best possible clustering outcomes, mitigating variations in '
            'perspective, orientation, and scale that may exist among the images in '
            'the data. For more information on the available techniques, expand the'
            'help box below.'
        )
    )

    with st.expander('Help on transformation models', expanded=False):
        st.write(
            (
                '**Translation:** Moves images in the x and/or y direction. '
                'This is suitable for cases where images have been shifted but are '
                'otherwise aligned.\n\n'
                '**Rigid Body:** (translation + rotation). It is used when images might '
                'have been taken from slightly different angles.\n\n'
                '**Scaled Rotation:** (translation + rotation + scaling). It '
                'combines rotation with scaling, useful when images might be taken '
                'from different distances.\n\n'
                '**Affine:** (translation + rotation + scaling + shearing), suited '
                'for images that may have been distorted or taken from varying '
                'perspectives.\n\n'
                '**Bilinear:** non-linear transformation; does not preserve straight lines.'
            )
        )

    motion_model = st.selectbox(
        'Transformation model:',
        [None, 'translation', 'rigid body', 'scaled rotation', 'affine', 'bilinear'],
        index=0,  # default to none
        help=(
            'The transformation model defines the geometric transformation'
            'one wants to apply.'
        ),
    )

    if motion_model is None:
        st.warning('Please select a transformation model to proceed.')

    feature_method = None

    if motion_model is not None:
        feature_method = st.selectbox(
            'Register and transform a stack of images:',
            [
                None,
                'to first image',
                'to mean image',
                'each image to the previous (already registered) one',
            ],
            index=0,  # default to none
            help=('For more help, see https://pystackreg.readthedocs.io/en/latest/readme.html'),
        )

    if motion_model is not None and feature_method is None:
        st.warning('Please select how you want to align your stack of images to proceed.')

    # Proceed only if both motion_model and feature_method are selected
    if motion_model is not None and feature_method is not None:
        aligned_images = align_images_to_mean(
            images=images,
            motion_model=motion_model,
            feature_method=feature_method,
        )
        return aligned_images

    return None


def cluster_image_widget(images: dict[str, np.ndarray]):
    """Widget to cluster images, using clustering methods suitable for small
    datasets.

    The clustering is based on the similarity of the images, which is
    computed using the selected similarity measure.
    """

    st.subheader('Clustering')

    image_list = list(images.values())
    image_names = list(images.keys())

    cluster_method = st.selectbox(
        'clustering algorithms:',
        [
            'SpectralClustering',
            'AffinityPropagation',
            'agglomerative',
            'kmeans',
            'dbscan',
            'hdbscan',
        ],
        help='The cluster method defines the way in which the images are grouped.',
    )

    # clustering based on similarity measures
    if cluster_method in ['SpectralClustering', 'AffinityPropagation']:
        n_clusters = None

        if cluster_method == 'SpectralClustering':
            specify_clusters = st.checkbox(
                'Do you want to specify the number of clusters beforehand?', value=False
            )
            if specify_clusters:
                n_clusters = st.number_input(
                    'Number of clusters:', min_value=2, step=1, value=4
                )

        measure = st.selectbox(
            'Select the similarity measure to cluster on:',
            ['SIFT', 'SSIM', 'CW-SSIM', 'IW-SSIM', 'FSIM', 'MSE', 'Brushstrokes'],
            help='Select a similarity measure used as the basis for clustering the images:',
        )

        matrix = build_similarity_matrix(np.array(image_list), algorithm=measure)
        st.subheader(f'Similarity matrix based on pairwise {measure} indices')
        st.write(np.round(matrix, decimals=2))

        c, n_clusters = cluster_images(
            image_list,
            algorithm=measure,
            n_clusters=n_clusters,
            method=cluster_method,
            print_metrics=True,
            labels_true=None,
        )

        if c is None:
            st.error('Clustering failed. Please check the parameters and try again.')
            return

        st.subheader('Cluster results')

        num_clusters = len(set(c))

        for n in range(num_clusters):
            cluster_label = n + 1
            st.write(f'\n --- Images from cluster #{cluster_label} ---')

            cluster_indices = np.argwhere(c == n).flatten()
            cluster_images_dict = {image_names[i]: image_list[i] for i in cluster_indices}
            show_images_widget(
                cluster_images_dict,
                key=f'cluster_{cluster_label}_images',
                message=f'Images from Cluster #{cluster_label}',
            )

        col1, col2 = st.columns(2)

        pca_clusters = plot_clusters(
            images, c, n_clusters, title=f'{cluster_method} Clustering results'
        )
        col1.subheader('Scatterplot')
        col1.pyplot(pca_clusters)

        dendrogram = plot_dendrogram(images, title=f'{cluster_method} Dendrogram')
        col2.subheader('Dendrogram')
        col2.pyplot(dendrogram)

    # clustering based on the image features
    if cluster_method in ['agglomerative', 'kmeans', 'dbscan', 'hdbscan']:
        cluster_evaluation = st.selectbox(
            'Select the cluster evaluation method:',
            ['silhouette', 'dbindex', 'derivatives'],
            help='Select the methos used as the basis for clustering the images:',
        )

        cluster_linkage = st.selectbox(
            'Select the cluster evaluation method:',
            ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'],
            help='Select the methos used as the basis for clustering the images:',
        )

        figures = clustimage_clustering(
            image_list,
            method=cluster_method,
            evaluation=cluster_evaluation,
            linkage_type=cluster_linkage,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.header('Scatter Plot')
            if isinstance(figures['scatter_plot'], tuple):
                # Select the figure part of the tuple
                st.pyplot(figures['scatter_plot'][0])
            else:
                st.pyplot(figures['scatter_plot'])

        with col2:
            st.header('Dendrogram Plot')
            if isinstance(figures['dendrogram_plot'], dict):
                # Select the figure part from 'ax'
                st.pyplot(figures['dendrogram_plot']['ax'])
            else:
                st.pyplot(figures['dendrogram_plot'])


def main():
    set_custom_css()

    st.title('Clustering of images')
    st.write(
        'This page groups a set of images based on their structural similarity. '
        'Optimal clustering performance is obtained via equalization'
        'of image parameters and by aligning images to a common reference point,'
        'thereby improving the quality of the data being analyzed.'
    )

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.error('No images loaded. Please upload images.')
        st.stop()

    st.subheader('The images')
    show_images_widget(images, key='original_images', message='Your selected images')

    images = {name: image for name, image in images.items()}

    st.markdown('---')

    images = equalize_images_widget(images=images)

    # aligned images are now in similar size and in grayscale
    aligned_images = align_to_mean_image_widget(images=images)

    if aligned_images:
        # Convert to uint8 if necessary
        aligned_images = {
            name: (image * 255).astype(np.uint8)
            if image.dtype == np.float64
            else image.astype(np.uint8)
            for name, image in aligned_images.items()
        }

        st.markdown('---')
        st.subheader('The aligned images')

        show_images_widget(aligned_images, message='The aligned images')

        st.markdown('---')
        cluster_image_widget(aligned_images)


if __name__ == '__main__':
    main()
