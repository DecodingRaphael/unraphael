from __future__ import annotations

import numpy as np
import streamlit as st
from styling import set_custom_css
from widgets import load_images_widget

# from skimage import color
from unraphael.dash.image_clustering import (
    align_images_to_mean,
    cluster_images,
    clustimage_clustering,
    compute_metrics,
    equalize_images,
    matrix_of_similarities,
    plot_clusters,
    plot_dendrogram,
)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import io


def show_images_widget(
    images: dict[str, np.ndarray],
    *,
    n_cols: int = 4,
    key: str = 'show_images',
    message: str = 'Select image',
) -> None | str:
    """Widget to show images with given number of columns."""
    col1, col2 = st.columns(2)

    # Ensure each number_input has a unique key
    n_cols_key = f'{key}_cols'
    n_cols = col1.number_input(
        'Number of columns for display', value=4, min_value=1, step=1, key=n_cols_key
    )

    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[i % n_cols]

        col.image(im, use_column_width=True, caption=name, clamp=True)

    return None  # Indicate that this function is primarily for display


def equalize_images_widget(*, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """This widget helps with equalizing images in terms of brightness,
    contrast and sharpness.

    The mean brightness, contrast and sharpness of the images are
    computed
    """

    col1, col2 = st.columns(2)

    with col1:
        col1.subheader('Equalization parameters')

        brightness = st.checkbox('Equalize brightness', value=False)
        contrast = st.checkbox('Equalize contrast', value=False)
        sharpness = st.checkbox('Equalize sharpness', value=False)

        preprocess_options = {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
        }

        all_images = [img for name, img in images.items()]

        equalized_images = equalize_images(all_images, **preprocess_options)

    with col2:
        metrics = compute_metrics(equalized_images)

        col2.subheader('Metrics after equalization')

        col3, col4 = st.columns((2))
        col3.metric('Mean Normalized Brightness', metrics['mean_normalized_brightness'])
        col4.metric('SD Normalized Brightness', metrics['sd_normalized_brightness'])

        col3.metric('Mean Normalized Contrast', metrics['mean_normalized_contrast'])
        col4.metric('SD Normalized Contrast', metrics['sd_normalized_contrast'])

        col3.metric('Mean Normalized Sharpness', metrics['mean_normalized_sharpness'])
        col4.metric('SD Normalized Sharpness', metrics['sd_normalized_sharpness'])

    # Map the equalized images back to their original names
    return {name: equalized_images[i] for i, name in enumerate(images.keys())}


def align_to_mean_image_widget(*, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """This widget aligns all images, preferably to their mean value but other
    aligning options are also available.

    The aligned images are input for thr later clustering of the images
    """

    st.subheader('Alignment parameters')

    # Displaying the alignment help text
    with st.expander('Help on Motion Models', expanded=False):
        st.write(
            (
                'This section allows to align the images (typically on their mean image)'
                'to ensure the best possible clustering outcomes, mitigating variations in'
                'perspective, orientation, and scale that may exist among the images in'
                'the data. The following methods are available:\n\n'
                '**Translation:** Moves images in the x and/or y direction.'
                'This is suitable for cases where images have been shifted but are '
                'otherwise aligned.\n\n'
                '**Rigid Body:** Allows rotation and translation without deformation.'
                'It is used when images might have been taken from slightly different'
                'angles.\n\n'
                '**Scaled Rotation:** Combines rotation with scaling, useful when images'
                'might be taken from different distances.\n\n'
                '**Affine:** A more complex transformation that allows for translation,'
                'rotation, scaling, and shearing. It is suitable for images that may have'
                'been distorted or taken from varying perspectives.\n\n'
                '**Bilinear:** Another transformation method, often used for general '
                'image alignment.'
            )
        )

    motion_model = st.selectbox(
        'Motion model:',
        [None, 'translation', 'rigid body', 'scaled rotation', 'affine', 'bilinear'],
        help=('The motion model defines the specific transformation one wants to apply. '),
    )

    feature_method = st.selectbox(
        'Register and transform a stack of images:',
        [
            'to first image',
            'to mean image',
            'each image to the previous (already registered) one',
        ],
        index=1,  # default 'to mean image'
        help=('For more help, see https://pystackreg.readthedocs.io/en/latest/readme.html'),
    )

    aligned_images = align_images_to_mean(
        images=images,
        motion_model=motion_model,
        feature_method=feature_method,
    )

    return aligned_images


def cluster_image_widget(images: dict[str, np.ndarray]):
    """Widget to cluster images, focusing on clustering methods suitable for
    small datasets.

    The clustering is based on the similarity of the images, which is
    computed using the selected similarity measure
    """

    st.subheader('Clustering')

    image_list = list(images.values())
    image_names = list(images.keys())

    # Ensure images are grayscale before stacking
    # image_list = [image if image.ndim == 2 else color.rgb2gray(image) for image in image_list]

    cluster_method = st.selectbox(
        'clustering algorithms:',
        [
            'SpectralClustering',
            'AffinityPropagation',  # similarity based
            'agglomerative',
            'kmeans',
            'dbscan',
            'hdbscan',
        ],  # clustimage
        help='The cluster method defines the way in which the images are grouped.',
    )

    # clustering based on similarity measures
    if cluster_method in ['SpectralClustering', 'AffinityPropagation']:
        if cluster_method == 'SpectralClustering':
            specify_clusters = st.checkbox(
                'Do you want to specify the number of clusters beforehand?', value=False
            )
            if specify_clusters:
                n_clusters = st.number_input(
                    'Number of clusters:', min_value=2, step=1, value=4
                )
            else:
                n_clusters = None
        else:
            n_clusters = None

        measure = st.selectbox(
            'Select the similarity measure to cluster on:',
            ['SIFT', 'SSIM', 'CW-SSIM', 'MSE', 'Brushstrokes'],
            help='Select a similarity measure used as the basis for clustering the images:',
        )

        matrix = matrix_of_similarities(np.array(image_list), algorithm=measure)
        st.subheader(f'Similarity matrix based on pairwise {measure} indices')
        st.write(np.round(matrix, decimals=3))

        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.heatmap(matrix, annot=True, fmt=".2f",annot_kws={"size": 8})
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # plt.close(fig)

        # st.image(buf, use_column_width = True)

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
        'This page groups a set of images based on their structural similarity.'
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

    # st.markdown('---')
    # with st.expander('Help for aligning images', expanded=False):
    #    alignment_help_widget()

    # creates aligned images which are now in similar size and gray scale
    images = align_to_mean_image_widget(images=images)

    # Convert to uint8 if necessary
    images = {
        name: (image * 255).astype(np.uint8)
        if image.dtype == np.float64
        else image.astype(np.uint8)
        for name, image in images.items()
    }

    st.markdown('---')
    st.subheader('The aligned images')
    st.write(
        'If selected, these aligned images are equalized in brightness,'
        'contrast, and sharpness'
    )
    show_images_widget(images, message='The aligned images')

    st.markdown('---')
    cluster_image_widget(images)


if __name__ == '__main__':
    main()
