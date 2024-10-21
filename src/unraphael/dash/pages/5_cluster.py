from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st
from styling import set_custom_css
from widgets import load_images_widget

from unraphael.dash.image_clustering import (
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
    show_images_widget,
    visualize_clusters,
    visualize_outer_contours,
)


@st.cache_data
def cached_compute_metrics(all_images):
    return compute_metrics(all_images)


@st.cache_data
def cached_equalize_images(all_images, **preprocess_options):
    return equalize_images(all_images, **preprocess_options)


@st.cache_data
def cached_align_images_to_mean(images, motion_model, feature_method):
    return align_images_to_mean(
        images=images, motion_model=motion_model, feature_method=feature_method
    )


@st.cache_data
def cached_build_similarity_matrix(image_array, algorithm):
    return build_similarity_matrix(image_array, algorithm=algorithm)


@st.cache_data
def cached_matrix_based_clustering(image_list, algorithm, n_clusters, method):
    return matrix_based_clustering(
        image_list, algorithm=algorithm, n_clusters=n_clusters, method=method
    )


@st.cache_data
def cached_feature_based_clustering(
    features,
    cluster_method,
    cluster_evaluation,
    cluster_linkage,
    name_dict,
    is_similarity_matrix=False,
):
    """Caches the feature- or matrix based clustering process and passes the
    relevant arguments."""
    return feature_based_clustering(
        features,
        cluster_method,
        cluster_evaluation,
        cluster_linkage,
        name_dict,
        is_similarity_matrix,
    )


@st.cache_data
def cached_extract_outer_contours(images: dict[str, np.ndarray]) -> dict:
    """Cache the extraction of outer contours from aligned images."""
    return extract_outer_contours_from_aligned_images(images)


@st.cache_data
def cached_extract_and_scale_features(
    contours_dict: dict, selected_features: list, image_shape: tuple
) -> tuple:
    """Cache the feature extraction and scaling."""
    return extract_and_scale_features(contours_dict, selected_features, image_shape)


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
    before_metrics = cached_compute_metrics(all_images)
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
            equalized_images = cached_equalize_images(all_images, **preprocess_options)
            after_metrics = cached_compute_metrics(equalized_images)
            display_metrics(after_metrics, 'Metrics after equalization')

    # Map the equalized images back to their original names
    return {name: equalized_images[i] for i, name in enumerate(images.keys())}


def align_to_mean_image_widget(
    *, images: dict[str, np.ndarray]
) -> Optional[dict[str, np.ndarray]]:
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

    if motion_model is not None and feature_method is not None:
        aligned_images = cached_align_images_to_mean(
            images=images,
            motion_model=motion_model,
            feature_method=feature_method,
        )
        return aligned_images

    return None


def cluster_image_widget(images: dict[str, np.ndarray]) -> None:
    """Widget to cluster images or outer contours, using clustering methods
    suitable for small to medium datasets.

    The clustering is based on image features or a matrix of similarity
    indices, depending on what cluster method is chosen.
    """

    image_list = list(images.values())
    image_names = list(images.keys())

    st.subheader('Cluster on outer contours or complete figures')
    st.write(
        (
            'The choice between them depends on the specifics of your dataset '
            'and the aspects of similarity that matter most for your specific use case.'
        )
    )

    cluster_approach = st.selectbox(
        'Please choose what you want to cluster on:',
        [
            'Select an option',
            'Outer contours',
            'Complete figures',
        ],
        help='The cluster method defines the way in which the images are grouped.',
        key='cluster_approach',
    )

    if cluster_approach in ['Outer contours']:
        st.write('Extracting outer contours from the aligned images...')
        st.markdown('---')
        contours_dict = cached_extract_outer_contours(images)

        st.subheader('Outer contours of the aligned images')
        visualize_outer_contours(images, contours_dict)

        st.markdown('---')
        st.subheader('Select components used for examining contour similarity')
        st.write(
            (
                'Select components to include for the clustering process. '
                'By extracting various (combinations of) shape descriptors '
                'like the ones below, you create a multidimensional '
                'representation of the contours. This captures different '
                'aspects of the shape and allows you to differentiate '
                'between contours effectively.'
            )
        )

        components = display_components_options()
        selected_features = []

        if components['fourier_descriptors']:
            selected_features.append('fd')
        if components['hu_moments']:
            selected_features.append('hu')
        if components['hog_features']:
            selected_features.append('hog')
        if components['aspect_ratio']:
            selected_features.append('aspect_ratio')
        if components['contour_length']:
            selected_features.append('contour_length')
        if components['centroid_distance']:
            selected_features.append('centroid_distance')
        if components['hausdorff_distance']:
            selected_features.append('hd')
        if components['procrustes']:
            selected_features.append('procrustes')

        # Extract and scale features from contours
        image_shape = next(iter(images.values())).shape

        if selected_features:
            st.write('Extracting and scaling features...')
            features, _ = cached_extract_and_scale_features(
                contours_dict, selected_features, image_shape
            )

            st.write(
                'Before the actual clustering below, you can use the '
                'following scatter plot to understand the distribution '
                'of the image data based on the selected features above. '
                'This can help to inform your clustering decisions.'
            )

            plot_scatter(features)

            st.markdown('---')
            st.subheader('Select cluster method and evaluation')

            cluster_method = st.selectbox(
                'Please choose the clustering algorithm:',
                [
                    'agglomerative',
                    'dbscan',
                    'kmeans',
                ],
                help='The cluster method defines the way in which the images are grouped.',
                key='cluster_method',
            )

            # Evaluation method selection
            if cluster_method == 'dbscan':
                # Freeze the selection for dbscan
                cluster_evaluation = 'silhouette'
                st.selectbox(
                    'Select the cluster evaluation method:',
                    ['silhouette', 'dbindex', 'derivative'],
                    index=0,  # Always select the first option
                    disabled=True,  # Make it disabled so it can't be changed
                )
            else:
                # Allow selection for other methods
                cluster_evaluation = st.selectbox(
                    'Select the cluster evaluation method:',
                    ['silhouette', 'dbindex', 'derivative'],
                    help='Select the method used as the basis for clustering the images:',
                )

            cluster_linkage = st.selectbox(
                'Select the cluster linkage method:',
                ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'],
                help='Select the linkage method used for clustering the images:',
            )

            st.markdown('---')
            st.title('Results')
            n_clusters, labels = cached_feature_based_clustering(
                features,
                cluster_method,
                cluster_evaluation,
                cluster_linkage,
                name_dict=contours_dict,
                is_similarity_matrix=False,
            )

            visualize_clusters(labels, image_names, image_list, contours_dict)

    # clustering images based on complete figures
    if cluster_approach in ['Complete figures']:
        st.markdown('---')
        st.subheader('The aligned images')
        show_images_widget(images, message='The aligned images')

        st.markdown('---')
        st.subheader('Select cluster method and evaluation')

        cluster_method = st.selectbox(
            'Please choose the clustering algorithm:',
            [
                'Select an option',
                'SpectralClustering',
                'AffinityPropagation',
                'DBSCAN',
                'agglomerative',
                'dbscan',
                'kmeans',
            ],
            help='The cluster method defines the way in which the images are grouped.',
            key='cluster_method',
        )

        if cluster_method in ['SpectralClustering', 'AffinityPropagation', 'DBSCAN']:
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
                key='similarity_measure',
            )

            st.markdown('---')
            st.title('Results')

            st.subheader(f'Similarity matrix based on pairwise {measure} indices')
            matrix = cached_build_similarity_matrix(np.array(image_list), algorithm=measure)
            st.write(np.round(matrix, decimals=2))

            labels, metrics, n_clusters = cached_matrix_based_clustering(
                image_list,
                algorithm=measure,
                n_clusters=n_clusters,
                method=cluster_method,
            )

            if labels is None:
                st.error('Clustering failed. Please check the parameters and try again.')
                return

            dendrogram = plot_dendrogram(
                images, labels, title=f'{cluster_method}-{measure} Dendrogram'
            )
            st.subheader('Dendrogram')
            st.pyplot(dendrogram)

            pca_clusters = plot_pca_mds_scatter(
                data=matrix,
                labels=labels,
                contours_dict=images,
                is_similarity_matrix=True,  # run MDS
                title=f'{cluster_method}-{measure} MDS Dimensions',
            )

            st.subheader('Scatterplot')
            st.pyplot(pca_clusters)

            st.subheader(f'Performance metrics for {cluster_method}')
            num_clusters = len(set(labels))
            st.metric('Number of clusters found:', num_clusters)

            for metric_name, metric_value in metrics.items():
                metric_str = f'{metric_value:.2f}'
                st.metric(metric_name, metric_str)

            visualize_clusters(labels, image_names, image_list, image_names)

        if cluster_method in ['agglomerative', 'dbscan', 'kmeans']:
            cluster_evaluation = st.selectbox(
                'Select the cluster evaluation method:',
                ['silhouette', 'dbindex', 'derivative'],
                help='Select the method used as the basis for clustering the images:',
            )

            cluster_linkage = st.selectbox(
                'Select the cluster linkage method:',
                ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median'],
                help='Select the linkage method used for clustering the images:',
            )

            st.markdown('---')
            st.title('Results')
            # extracts features from the raw image data
            features = preprocess_images(image_list)

            n_clusters, labels = cached_feature_based_clustering(
                features,
                cluster_method,
                cluster_evaluation,
                cluster_linkage,
                name_dict=images,
                is_similarity_matrix=False,
            )

            visualize_clusters(labels, image_names, image_list, images)


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

    images = {name: image for name, image in images.items()}

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
        st.warning(
            "It appears your images are not aligned yet. Let's do that in the following step..."
        )
        aligned_images = align_to_mean_image_widget(images=images)
    else:
        st.success(
            "All images appear to be already aligned. Let's proceed to the following step..."
        )
        aligned_images = images

    if aligned_images:
        # Convert to uint8 if necessary
        aligned_images = {
            name: (image * 255).astype(np.uint8)
            if image.dtype == np.float64
            else image.astype(np.uint8)
            for name, image in aligned_images.items()
        }

        st.markdown('---')
        cluster_image_widget(aligned_images)


if __name__ == '__main__':
    main()
