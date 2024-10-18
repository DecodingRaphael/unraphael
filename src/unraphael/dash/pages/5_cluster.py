from __future__ import annotations

from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from clusteval import clusteval
from scatterd import scatterd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from styling import set_custom_css
from widgets import load_images_widget

from unraphael.dash.image_clustering import (
    align_images_to_mean,
    build_similarity_matrix,
    cluster_images,
    clustimage_clustering,
    compute_metrics,
    equalize_images,
    extract_and_scale_features,
    extract_outer_contours_from_aligned_images,
    plot_clusters,
    plot_clusters2,
    plot_dendrogram,
    show_images_widget,
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
def cached_cluster_images(image_list, algorithm, n_clusters, method):
    return cluster_images(image_list, algorithm=algorithm, n_clusters=n_clusters, method=method)


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
    suitable for small datasets.

    The clustering is based on the structural similarity measure of the
    images.
    """

    image_list = list(images.values())
    image_names = list(images.keys())

    st.subheader('Select object for clustering')
    st.write(
        (
            'Select whether you want to cluster on the outer contours or on '
            'the complete figures. The choice between them depends on the '
            'specifics of your dataset and the aspects of similarity that '
            'matter most for your specific use case.'
        )
    )

    cluster_approach = st.selectbox(
        'Please choose what you want to cluster on:',
        [
            'Outer contours',
            'Complete figures',
        ],
        help='The cluster method defines the way in which the images are grouped.',
        key='cluster_approach',
    )

    # clustering images based on outer contours of figures
    if cluster_approach in ['Outer contours']:
        st.write('Extracting outer contours from the aligned images...')
        contours_dict = cached_extract_outer_contours(images)

        # Extract the shape of the aligned images
        image_shape = next(iter(images.values())).shape

        st.subheader('Outer contours of the aligned images')
        visualize_outer_contours(images, contours_dict)

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
        if selected_features:
            st.write('Extracting and scaling features...')
            features, _ = cached_extract_and_scale_features(
                contours_dict, selected_features, image_shape
            )

            # Before clustering, visualize the data using a scatter plot
            # or similar to understand the distribution.
            def plot_scatter(features):
                scatterd(features[:, 0], features[:, 1])

                fig = plt.gcf()
                st.pyplot(fig)

            plot_scatter(features)

            st.subheader('Clustering Outer Contours')

            cluster_method = st.selectbox(
                'Please choose the clustering algorithm:',
                [
                    'kmeans',
                    'agglomerative',
                    'dbscan',
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

            n_clusters = None
            ce = clusteval(
                cluster=cluster_method, evaluate=cluster_evaluation, linkage=cluster_linkage
            )

            results = ce.fit(features)

            # Check if results is not None and contains 'labx'
            if results is not None and 'labx' in results:
                cluster_labels = results['labx']
            else:
                st.error('No clusterlabels found. Check your clustering method and parameters.')
                cluster_labels = None  # Set to None to avoid further errors
            # cluster_labels = results['labx']

            # extract optimal number of clusters
            if cluster_method in ['kmeans', 'agglomerative']:
                scores = results['score']['score']
                cluster_numbers = results['score']['clusters']

                # Find the index of the maximum silhouette score
                optimal_index = np.argmax(scores)
                n_clusters = cluster_numbers[optimal_index]

            elif cluster_method == 'dbscan':
                unique_labels = set(cluster_labels) if cluster_labels is not None else set()
                n_clusters = len(unique_labels) - (
                    1 if -1 in unique_labels else 0
                )  # Exclude noise

            dendrogram_plot = ce.dendrogram(X=results, linkage=cluster_linkage)['ax'].figure
            st.subheader('Dendrogram plot')
            st.pyplot(dendrogram_plot)

            silhouette_plot, ax, _ = ce.plot_silhouette(
                X=features,
                dot_size=200,
                jitter=0.01,
                cmap='Set2',
                savefig={'fname': None, 'format': 'png', 'dpi': 100},
            )

            st.subheader('Silhouette plot')
            st.pyplot(fig=silhouette_plot)

            # Plot the clusters
            pca_clusters = plot_clusters2(
                features, cluster_labels, contours_dict, title='PCA dimensions'
            )
            st.subheader('Scatterplot')
            st.pyplot(pca_clusters)

            # show metrics
            st.metric('Number of clusters found:', n_clusters)

            if cluster_method in ['kmeans', 'agglomerative']:
                if n_clusters is not None and n_clusters > 1:
                    silhouette_avg = silhouette_score(features, cluster_labels)
                    davies_bouldin = davies_bouldin_score(features, cluster_labels)

                    st.metric(label='Silhouette Score', value=f'{silhouette_avg:.2f}')
                    st.metric(label='Davies Bouldin Score', value=f'{davies_bouldin:.2f}')

            elif cluster_method == 'dbscan':
                # Exclude noise points (-1)
                if len(set(cluster_labels)) > 1:
                    valid_labels = cluster_labels[cluster_labels != -1]
                    if len(set(valid_labels)) > 1:  # Ensure there's more than 1 cluster
                        silhouette_avg = silhouette_score(
                            features[cluster_labels != -1], valid_labels
                        )
                        davies_bouldin = davies_bouldin_score(
                            features[cluster_labels != -1], valid_labels
                        )

                        st.metric(label='Silhouette Score', value=f'{silhouette_avg:.2f}')
                        st.metric(label='Davies Bouldin Score', value=f'{davies_bouldin:.2f}')
                    else:
                        st.metric(label='Silhouette Score could not be computed')
                        st.metric(label='Davies Bouldin Score could not be computed')

            else:
                st.metric(label='Silhouette Score could not be computed')
                st.metric(label='Davies Bouldin Score could not be computed')

    # clustering images based on complete figures
    if cluster_approach in ['Complete figures']:
        cluster_method = st.selectbox(
            'Please choose the clustering algorithm:',
            [
                'SpectralClustering',
                'AffinityPropagation',
                'DBSCAN',
                'agglomerative',
                'kmeans',
                'dbscan',
                'hdbscan',
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

            st.title('Results')

            matrix = cached_build_similarity_matrix(np.array(image_list), algorithm=measure)
            st.subheader(f'Similarity matrix based on pairwise {measure} indices')
            st.write(np.round(matrix, decimals=2))

            labels, metrics, n_clusters = cached_cluster_images(
                image_list,
                algorithm=measure,
                n_clusters=n_clusters,
                method=cluster_method,
            )

            if labels is None:
                st.error('Clustering failed. Please check the parameters and try again.')
                return

            num_clusters = len(set(labels))

            st.subheader(f'Performance metrics for {cluster_method}')

            st.metric('Number of clusters found:', num_clusters)

            for metric_name, metric_value in metrics.items():
                metric_str = f'{metric_value:.2f}'
                st.metric(metric_name, metric_str)

            st.subheader('Visualizing the clusters')
            for n in range(num_clusters):
                cluster_label = n + 1
                st.write(f'\n --- Images from cluster #{cluster_label} ---')

                cluster_indices = np.argwhere(labels == n).flatten()
                cluster_images_dict = {image_names[i]: image_list[i] for i in cluster_indices}
                show_images_widget(
                    cluster_images_dict,
                    key=f'cluster_{cluster_label}_images',
                    message=f'Images from Cluster #{cluster_label}',
                )

            col1, col2 = st.columns(2)

            images_dict = dict(zip(image_names, image_list))

            pca_clusters = plot_clusters(
                images_dict,
                labels,
                n_clusters,
                title=f'{cluster_method}-{measure} PCA dimensions',
            )
            col1.subheader('Scatterplot')
            col1.pyplot(pca_clusters)

            dendrogram = plot_dendrogram(
                images_dict, labels, title=f'{cluster_method}-{measure} Dendrogram'
            )
            col2.subheader('Dendrogram')
            col2.pyplot(dendrogram)

        if cluster_method in ['agglomerative', 'kmeans', 'dbscan', 'hdbscan']:
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

            results = clustimage_clustering(
                image_list,
                method=cluster_method,
                evaluation=cluster_evaluation,
                linkage_type=cluster_linkage,
            )

            labels = results['labels']
            metrics = results['metrics']
            figures = results['figures']

            num_clusters = len(set(labels))

            st.title('Results')
            st.subheader(f'Performance metrics for {cluster_method} clustering')
            st.metric('Number of Clusters:', num_clusters)

            for metric_name, metric_value in metrics.items():
                metric_str = f'{metric_value:.2f}'
                st.metric(metric_name, metric_str)

            st.subheader('Visualizing the clusters')
            for n in range(num_clusters):
                cluster_label = n + 1
                st.write(f'\n --- Images from cluster #{cluster_label} ---')

                cluster_indices = np.argwhere(labels == n).flatten()
                cluster_images_dict = {image_names[i]: image_list[i] for i in cluster_indices}
                show_images_widget(
                    cluster_images_dict,
                    key=f'cluster_{cluster_label}_images',
                    message=f'Images from cluster #{cluster_label}',
                )

            col1, col2 = st.columns(2)

            images_dict = dict(zip(image_names, image_list))

            st.subheader('Cluster evaluation')
            st.pyplot(figures['scatter_plot'][0])
            st.pyplot(figures['dendrogram_plot'])


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

    # Prompt user to select "Yes" or "No" for aligning images
    alignment_option = st.radio('Are your images already aligned?', ('Yes', 'No'))

    if alignment_option == 'No':
        # aligned images will be in similar size and in grayscale
        aligned_images = align_to_mean_image_widget(images=images)
    elif alignment_option == 'Yes':
        # Check if all images are in grayscale
        for name, image in images.items():
            if (
                len(image.shape) == 3 and image.shape[2] == 3
            ):  # Image has 3 channels (i.e., it's in RGB format)
                images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Check if all images have the same dimension
        image_shape = None
        for name, image in images.items():
            if image_shape is None:
                image_shape = image.shape
            else:
                if image.shape != image_shape:
                    st.error(
                        f'Image {name} has a different size ({image.shape}) '
                        f'compared to others ({image_shape}).'
                    )
                    st.stop()

        # Assume images are already aligned if they pass the grayscale and size checks
        aligned_images = images
    else:
        st.error("Please select either 'Yes' or 'No' to proceed.")
        st.stop()

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
