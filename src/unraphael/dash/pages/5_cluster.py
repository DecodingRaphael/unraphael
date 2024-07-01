from __future__ import annotations
#from typing import Any
import numpy as np
import streamlit as st
from unraphael.dash.image_clustering import align_images_to_mean, equalize_images, cluster_images
from styling import set_custom_css
from widgets import load_images_widget
from matplotlib import pyplot as plt
from skimage import color

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

        col.image(im, use_column_width=True, caption=name,clamp=True)
    

#TODO: Implement equalize_images_widget by equalizing accross all images 
def equalize_images_widget(*, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """This widget helps with equalizing images."""
    st.subheader('Equalization parameters')

    brightness = st.checkbox('Equalize brightness', value=False)
    contrast = st.checkbox('Equalize contrast', value=False)
    sharpness = st.checkbox('Equalize sharpness', value=False)

    preprocess_options = {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,        
    }
    
    # Collect all images into a list
    all_images = [img for name, img in images.items()]
    
    # Equalize the entire set of images
    equalized_images = equalize_images(all_images, **preprocess_options)

    #return {name: equalize_images(image, **preprocess_options) for name, image in images.items()}
    #return {name: equalize_images(images, **preprocess_options) for name, images in images.items()}
    
    # Map the equalized images back to their original names
    return {name: equalized_images[i] for i, name in enumerate(images.keys())}
    

def align_to_mean_image_widget(*, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """This widget helps with aligning all images to their mean value."""
    st.subheader('Alignment parameters')
    
    motion_model = st.selectbox(
            'Motion model:',
            ['translation',
             'rigid body',
             'scaled rotation',
             'affine',
             'bilinear'],
            help=(
                'The motion model defines the transformation one wants'
                'to account for. '          
            ),
        )
    
    feature_method = st.selectbox(
            'Register and transform a stack of images:',
            ['to first image',
             'to mean image',
             'each image to the previous (already registered) one'],
            help=(
                'For more help, see https://pystackreg.readthedocs.io/en/latest/readme.html'          
            ),
        )    
        
    aligned_images = align_images_to_mean(
        images    = images,
        motion_model = motion_model,
        feature_method  = feature_method,
        )

    return aligned_images


def alignment_help_widget():
    st.write(
        (
            'The following methods are used for image registration and alignment. '
            'Depending on your specific alignment requirements and computational '
            'constraints, you may choose one method over the other. Example usage '
            'scenarios and comparative analysis can help you choose the most suitable '
            'alignment technique for your specific requirements.'
        )
    )
    

def cluster_image_widget(images: dict[str, np.ndarray]):    
    """Widget to cluster images."""
    
    st.subheader('Clustering')
                
    cluster_method = st.selectbox(
        'Unsupervised clustering algorithms:',
        ['SpectralClustering', 'AffinityPropagation', 'KMeans', 'DBSCAN'],
        help='The cluster method defines the way in which the images are grouped.'
    )    
    
    if cluster_method == 'SpectralClustering':
        specify_clusters = st.checkbox('Do you want to specify the number of clusters?', value=False)
        if specify_clusters:
            n_clusters = st.number_input('Number of clusters:', min_value=2, step=1, value=4)
        else:
            n_clusters = None
    else:
        n_clusters = None
        
    measure = st.selectbox(
        'Select the similarity measure to cluster on:',
        ['SIFT', 'SSIM', 'CW-SSIM', 'MSE','Brushstrokes'],
        help='Select a similarity measure used for clustering the images:')
            
    col1, col2 = st.columns((2))
        
    image_list = list(images.values())
    image_names = list(images.keys()) 
        
    # Ensure images are 3D arrays (grayscale) before stacking
    image_list = [image if image.ndim == 2 else color.rgb2gray(image) for image in image_list]
    
    c = cluster_images(np.array(image_list), 
                           algorithm     = measure, 
                           n_clusters    = n_clusters, 
                           method        = cluster_method, 
                           print_metrics = True, 
                           labels_true   = None)

    if c is None:
        st.error("Clustering failed. Please check the parameters and try again.")
        return
    
    num_clusters = len(set(c))
    for n in range(num_clusters):
            cluster_label = n + 1
            st.write(f"\n --- Images from cluster #{cluster_label} ---")
                        
            cluster_indices = np.argwhere(c == n).flatten()
            cluster_images_dict = {image_names[i]: image_list[i] for i in cluster_indices}
            show_images_widget(cluster_images_dict, key=f'cluster_{cluster_label}_images', message=f'Images from Cluster #{cluster_label}')

            
def main():
    set_custom_css()

    st.title('Clustering of images')
    st.write('Group a set of images based on their structural similarity.')

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.error("No images loaded. Please upload images.")
        st.stop()

    st.subheader('The images')
    show = show_images_widget(images, key='original_images', message='Your selected images')

    #if not show:
    #    st.stop()

    images = {name: image for name, image in images.items() }

    col1, col2 = st.columns(2)

    with col1:
        images = equalize_images_widget(images = images)    
    
    with st.expander('Help for parameters for aligning images to their mean', expanded=False):
        alignment_help_widget()
        
    with col2:
        aligned_images = align_to_mean_image_widget(images = images) # creates aligned images, similar size and gray scale        

    st.subheader('The aligned images (with equalized brightness, contrast, and sharpness)')
    show_images_widget(aligned_images, message='Your aligned images')
    
    #TODO: Necessary to work with transparant images so that uniform background color is not included in the clustering process? 
    cluster_image_widget(aligned_images)
    
    # add heatmap of similarity matrix
    
if __name__ == '__main__':
    main()
