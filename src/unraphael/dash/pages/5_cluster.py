from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from image_clustering import (
    align_images_to_mean,
    build_similarity_matrix,
    compute_metrics,
    equalize_images,
    extract_and_scale_features,
    extract_outer_contours_from_aligned_images,
    feature_based_clustering,
    matrix_based_clustering,
    plot_dendrogram,
    plot_pca_mds_scatter,
    plot_scatter,
    preprocess_images,
    visualize_clusters,
    visualize_outer_contours,
)
from styling import set_custom_css
from widgets import load_images_widget, show_images_widget

from unraphael.types import ImageType

_compute_metrics = st.cache_data(compute_metrics)
_equalize_images = st.cache_data(equalize_images)
_align_images_to_mean = st.cache_data(align_images_to_mean)
_build_similarity_matrix = st.cache_data(build_similarity_matrix)
_matrix_based_clustering = st.cache_data(matrix_based_clustering)
_feature_based_clustering = st.cache_data(feature_based_clustering)
_extract_outer_contours = st.cache_data(extract_outer_contours_from_aligned_images)
_extract_and_scale_features = st.cache_data(extract_and_scale_features)


def display_components_options() -> dict[str, bool]:
    """Display the components used for examining contour similarity in a
    Streamlit app.

    Returns:
        dict[str, bool]: A dictionary where keys are the names of the contour
        similarity components and values are booleans indicating whether each
        component is selected by the user.
    """

    col1, col2 = st.columns(2)
    with col1:
        fourier_descriptors = st.checkbox('Fourier Descriptors')
        hu_moments = st.checkbox('Hu Moments')
        hog_features = st.checkbox('HOG Features')
        aspect_ratio = st.checkbox('Aspect Ratio')

    with col2:
        contour_length = st.checkbox('Contour Length')
        centroid_distance = st.checkbox('Centroid Distance')
        hausdorff_distance = st.checkbox('Hausdorff Distance')
        procrustes = st.checkbox('Procrustes Distance')

    return {
        'fourier_descriptors': fourier_descriptors,
        'hu_moments': hu_moments,
        'hog_features': hog_features,
        'aspect_ratio': aspect_ratio,
        'contour_length': contour_length,
        'centroid_distance': centroid_distance,
        'hausdorff_distance': hausdorff_distance,
        'procrustes': procrustes,
    }


def display_equalization_options() -> dict[str, bool]:
    """This function creates a user interface with checkboxes for equalizing
    brightness, contrast, and sharpness. It returns a dictionary with the
    selected options.

    Returns
    -------
    dict[str, bool]
        A dictionary with keys 'brightness', 'contrast', and 'sharpness',
        each mapped to a boolean indicating whether the corresponding
        equalization option was selected.
    """
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
    """Display the metrics with a specified title.

    Parameters
    ----------
    metrics : dict[str, float]
        A dictionary containing the metrics to be displayed. The keys
        are the names of the metrics,
        and the values are the corresponding metric values.
    title : str
        The title to be displayed above the metrics.
    """
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
    sharpness.

    Parameters, before and after equalization.
    ----------
    images : dict[str, np.ndarray]
        A dictionary where keys are image names and values are image arrays.
    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are image names and values are the equalized image arrays.
    """

    # before equalization
    all_images = list(images.values())
    before_metrics = _compute_metrics(all_images)
    preprocess_options = display_equalization_options()
    col1, col2 = st.columns(2)

    with col1:
        # Display metrics before equalization
        display_metrics(before_metrics, 'Metrics before equalization')

    with col2:
        # Initialize a variable for equalized images
        equalized_images = all_images.copy()

        # Perform equalization if any checkbox is ticked
        if any(preprocess_options.values()):
            equalized_images = _equalize_images(all_images, **preprocess_options)
            after_metrics = _compute_metrics(equalized_images)
            display_metrics(after_metrics, 'Metrics after equalization')

    # Map the equalized images back to their original names
    return {name: equalized_images[i] for i, name in enumerate(images.keys())}


def align_to_mean_image_widget(*, images: dict[str, np.ndarray]) -> Optional[dict[str, np.ndarray]]:
    """This widget aligns a set of images to a reference image using various
    transformation models, typically to their mean value, but other aligning
    options are also available. The aligned images are used for later
    clustering.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        A dictionary where keys are image identifiers and values are the image arrays.
    Returns
    -------
    Optional[dict[str, np.ndarray]]
        A dictionary of aligned images if alignment is successful, otherwise None.
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
        help=('The transformation model defines the geometric transformationone wants to apply.'),
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

    if motion_model is not None and feature_method is not None:
        aligned_images = _align_images_to_mean(
            images=images,
            motion_model=motion_model,
            feature_method=feature_method,
        )
        return aligned_images

    return None


def cluster_image_widget(images: Dict[str, np.ndarray]) -> None:
    """Widget to cluster images or outer contours using suitable clustering
    methods."""

    # input is the aligned images
    image_list = list(images.values())
    image_names = list(images.keys())

    # Step 1: Selection of cluster approach
    cluster_approach = select_cluster_approach()

    if cluster_approach == 'Outer contours':
        cluster_on_outer_contours(images, image_names, image_list)
    elif cluster_approach == 'Complete figures':
        cluster_on_complete_figures(images, image_names, image_list)


def select_cluster_approach() -> str:
    """Select the cluster approach (outer contours or complete figures)."""
    st.subheader('Cluster on outer contours or complete figures')
    return st.selectbox(
        'Choose clustering basis:',
        ['Select an option', 'Outer contours', 'Complete figures'],
        help='Defines the basis for grouping images.',
        key='cluster_approach',
    )


def cluster_on_outer_contours(images: Dict[str, np.ndarray], image_names: List[str], image_list: List[np.ndarray]) -> None:
    """Handle clustering based on outer contours."""
    st.write('Extracting outer contours from the aligned images...')
    contours_dict = _extract_outer_contours(images)

    st.subheader('Outer contours of the aligned images')
    visualize_outer_contours(images, contours_dict)

    selected_features = select_contour_features()

    if selected_features:
        st.write('Extracting and scaling features...')
        image_shape = next(iter(images.values())).shape
        features, _ = _extract_and_scale_features(contours_dict, selected_features, image_shape)

        st.write('Scatter plot to understand feature distribution.')
        plot_scatter(features)

        cluster_on_features(features, contours_dict, image_names, image_list)


def select_contour_features() -> List[str]:
    """Select features for clustering based on outer contours."""
    st.subheader('Select components for contour similarity')
    components = display_components_options()
    feature_map = {
        'fourier_descriptors': 'fd',
        'hu_moments': 'hu',
        'hog_features': 'hog',
        'aspect_ratio': 'aspect_ratio',
        'contour_length': 'contour_length',
        'centroid_distance': 'centroid_distance',
        'hausdorff_distance': 'hd',
        'procrustes': 'procrustes',
    }
    return [feature_map[key] for key, value in components.items() if value]


def cluster_on_features(
    features: np.ndarray,
    contours_dict: Dict,
    image_names: List[str],
    image_list: List[np.ndarray],
) -> None:
    """Handle clustering based on selected features."""
    st.subheader('Select cluster method and evaluation')

    cluster_method = st.selectbox(
        'Choose clustering algorithm:',
        ['agglomerative', 'dbscan', 'kmeans'],
        help='Defines the grouping method.',
        key='cluster_method',
    )

    cluster_evaluation = select_cluster_evaluation(cluster_method)
    cluster_linkage = select_cluster_linkage()

    n_clusters, labels = _feature_based_clustering(
        features,
        cluster_method,
        cluster_evaluation,
        cluster_linkage,
        name_dict=contours_dict,
        is_similarity_matrix=False,
    )

    visualize_clusters(labels, image_names, image_list, contours_dict)


def select_cluster_evaluation(cluster_method: str) -> str:
    """Select the cluster evaluation method based on the clustering
    algorithm."""
    if cluster_method == 'dbscan':
        st.selectbox('Cluster evaluation method:', ['silhouette'], index=0, disabled=True)
        return 'silhouette'

    else:
        return st.selectbox('Cluster evaluation method:', ['silhouette', 'dbindex', 'derivative'])


def select_cluster_linkage() -> str:
    """Select the linkage method for clustering."""
    return st.selectbox(
        'Select linkage method:',
        ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'],
        help='Linkage method for clustering.',
    )


def cluster_on_complete_figures(images: Dict[str, np.ndarray], image_names: List[str], image_list: List[np.ndarray]) -> None:
    """Handle clustering based on complete figures."""
    st.subheader('The aligned images')

    show_images_widget(
        [ImageType(name=name, data=data) for name, data in images.items()],
        message='The aligned images',
    )

    cluster_method = select_cluster_method_for_complete_figures()

    if cluster_method in ['SpectralClustering', 'AffinityPropagation', 'DBSCAN']:
        n_clusters = None
        if cluster_method == 'SpectralClustering':
            specify_clusters = st.checkbox('Specify number of clusters?', value=False)
            if specify_clusters:
                n_clusters = st.number_input('Number of clusters:', min_value=2, step=1, value=4)

        measure = select_similarity_measure()

        st.markdown('---')
        st.title('Results')

        st.subheader(f'Similarity matrix based on pairwise {measure} indices')
        matrix = _build_similarity_matrix(np.array(image_list), algorithm=measure)
        st.write(np.round(matrix, decimals=2))
        labels, metrics, n_clusters = _matrix_based_clustering(matrix, algorithm=measure, n_clusters=n_clusters, method=cluster_method)

        if labels is None:
            st.error('Clustering failed. Check parameters and try again.')
            return

        st.subheader('Dendrogram')
        dendrogram = plot_dendrogram(matrix, labels, title=f'{cluster_method}-{measure}')
        st.pyplot(dendrogram)

        st.subheader('Scatterplot')
        pca_clusters = plot_pca_mds_scatter(
            data=matrix,
            labels=labels,
            contours_dict=images,
            is_similarity_matrix=True,
            title=f'{cluster_method}-{measure} MDS Dimensions',
        )
        st.pyplot(pca_clusters)

        st.subheader(f'Performance metrics for {cluster_method}')
        num_clusters = len(set(labels))
        st.metric('Number of clusters found:', num_clusters)

        for metric_name, metric_value in metrics.items():
            st.metric(metric_name, f'{metric_value:.2f}')

        visualize_clusters(labels, image_names, image_list, image_names)

    elif cluster_method in ['agglomerative', 'dbscan', 'kmeans']:
        cluster_evaluation = st.selectbox('Cluster evaluation method:', ['silhouette', 'dbindex', 'derivative'])
        cluster_linkage = st.selectbox(
            'Linkage method:',
            ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'],
        )

        features = preprocess_images(image_list)

        n_clusters, labels = _feature_based_clustering(
            features,
            cluster_method,
            cluster_evaluation,
            cluster_linkage,
            name_dict=images,
            is_similarity_matrix=False,
        )

        visualize_clusters(labels, image_names, image_list, images)


def select_cluster_method_for_complete_figures() -> str:
    """Select the cluster method for clustering complete figures."""
    return st.selectbox(
        'Choose clustering algorithm:',
        [
            'Select an option',
            'SpectralClustering',
            'AffinityPropagation',
            'DBSCAN',
            'agglomerative',
            'dbscan',
            'kmeans',
        ],
        help='Defines the grouping method.',
        key='cluster_method',
    )


def select_similarity_measure() -> str:
    """Select the similarity measure for clustering complete figures."""
    return st.selectbox(
        'Select similarity measure:',
        ['SIFT', 'SSIM', 'IW-SSIM', 'FSIM', 'MSE', 'Brushstrokes'],
        help='Basis for clustering images.',
        key='similarity_measure',
    )


def main():
    set_custom_css()

    st.title('Clustering of images')
    st.write(
        'This page groups a set of images based on the structural similarity between '
        'the extracted figures or their outlines. Optimal clustering performance '
        'is obtained via equalization of image parameters and by aligning images to '
        'a common reference point, thereby improving the quality of the data being '
        'analyzed. Several cluster methods and metrics are available to cluster your '
        'images.'
    )

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.error('No images loaded. Please upload images.')
        st.stop()

    st.subheader('The images')
    show_images_widget(images, key='original_images', message='Your selected images')

    images = {image.name: image.data for image in images}

    st.markdown('---')

    images = equalize_images_widget(images=images)

    st.markdown('---')

    # Silent check for alignment (grayscale and size consistency)
    unaligned_images = []

    # Check if all images are in grayscale (not RGB)
    for name, image in images.items():
        if len(image.shape) == 3 and image.shape[2] == 3:
            unaligned_images.append(name)  # This image is not grayscale

    # Check if all images have the same size
    image_shape = None
    for name, image in images.items():
        if image_shape is None:
            image_shape = image.shape
        elif image.shape != image_shape:
            unaligned_images.append(name)

    # Automatically align if there are unaligned images
    if unaligned_images:
        st.warning("It appears your images are not aligned yet. Let's do that in the following step...")
        aligned_images = align_to_mean_image_widget(images=images)
    else:
        st.success("All images appear to be already aligned. Let's proceed to the following step...")
        aligned_images = images

    if aligned_images:
        # Convert to uint8 if necessary
        aligned_images = {name: (image * 255).astype(np.uint8) if image.dtype == np.float64 else image.astype(np.uint8) for name, image in aligned_images.items()}

        st.markdown('---')
        cluster_image_widget(aligned_images)


if __name__ == '__main__':
    main()
