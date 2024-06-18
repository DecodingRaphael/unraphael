from __future__ import annotations
from typing import Any
import numpy as np
import streamlit as st
from unraphael.dash.image_clustering import align_images_to_mean, equalize_images,cluster_images
from styling import set_custom_css
from widgets import load_images_widget, show_images_widget
from matplotlib import pyplot as plt

#TODO: Implement equalize_images_widget by equalizing accross all images 
def equalize_images_widget(*, images: dict[str, np.ndarray]):
    """This widget helps with equalizing images."""
    st.subheader('Equalization parameters')

    brightness = st.checkbox('Equalize brightness', value=False)
    contrast = st.checkbox('Equalize contrast', value=False)
    sharpness = st.checkbox('Equalize sharpness', value=False)
    color = st.checkbox('Equalize colors', value=False)
    reinhard = st.checkbox('Reinhard color transfer', value=False)

    preprocess_options = {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'color': color,
        'reinhard': reinhard,
    }

    return {
        name: equalize_images(image, **preprocess_options)
        for name, image in images.items()
    }

def align_to_mean_image_widget(*, images: dict[str, np.ndarray]):
    """This widget helps with aligning all images to their mean value."""
    st.subheader('Alignment parameters')

    options = [
        None,
        'pyStackReg',        
    ]

    selected_option = st.selectbox(
        'Alignment procedure:',
        options,
        help=(
            '**pyStackReg**: Aligns images'            
            ),
    )

    if not selected_option:
        st.stop()
    
    elif selected_option == 'pyStackReg':
        motion_model = st.selectbox(
            'Motion model:',
            ['translation','rigid body (translation + rotation)',
             'scaled rotation (translation + rotation + scaling)',
             'affine (translation + rotation + scaling + shearing)',
             'bilinear (non-linear transformation; does not preserve straight lines)'],
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
        
    aligned_images_stack = align_images_to_mean(
        image    = images,       
        selected_option = selected_option,     
        motion_model = motion_model,
        feature_method  = feature_method,
        )

    return aligned_images_stack


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
    

def cluster_image_widget(
    images: dict[str, dict[str, Any]],
):
    """Widget to cluster images."""
    st.subheader('Clustering')    

    options = [
        None,
        'pyStackReg',        
    ]

    selected_option = st.selectbox(
        'Alignment procedure:',
        options,
        help=(
            '**pyStackReg**: Aligns images based on detected features using '
            'algorithms like SIFT, SURF, or ORB.'
        ),
    )

    if not selected_option:
        st.stop()
    
    elif selected_option == 'pyStackReg':
        cluster_method = st.selectbox(
            'Unsupervised clustering algorithms:',
            ['SpectralClustering',
             'AffinityPropagation',
             'KMeans',
             'DBSCAN'],
            help=(
                'The cluster method defines the way in which the images are grouped.'                          
            ),
        )
        measure = st.selectbox(
            ['SIFT: Scale-invariant Feature Transform',
             'SSIM: Structural Similarity Index',
             'CW-SSIM: Complex Wavelet Structural Similarity Index',
             'MSE: Mean Squared Error'],
            help=(
                'For more help, see .....'          
            ),
        )

    col1, col2 = st.columns((0.3, 0.7))

    image_name = col1.selectbox('Pick image', options=tuple(images.keys()))
    data = images[image_name]

    image = data['image']
    angle_str = f"{data['angle']:.2f}°"

    with col1:
         # Perform clustering on aligned images
        c = cluster_images(images, 
                          algorithm      = 'SSIM', # measure
                          n_clusters     = None,
                          method         = 'affinity', # cluster_method
                          print_metrics  = True, 
                          labels_true    = None)

    with col2:
        # Post-clustering visualization
        num_clusters = len(set(c))
        #images = os.listdir(input_dir)  
        images = names #TODO
       
        for n in range(num_clusters):
            cluster_label = n + 1  #  cluster label starting from 1
            print(f"\n --- Images from cluster #{cluster_label} ---")

            for i in np.argwhere(c == n).flatten():
                print(f"Image {images[i]}")
                img = images[i]
                plt.figure()
                plt.imshow(img, cmap='gray')
                plt.title(f'Cluster #{cluster_label}, Image: {images[i]}')
                plt.axis('off')
                plt.show()


def main():
    set_custom_css()

    st.title('Clustering of images')
    st.write('For a set of images, group images based on their structural similarity.')

    with st.sidebar:
        images = load_images_widget(as_gray=False, as_ubyte=True)

    if not images:
        st.stop()

    st.subheader('The images')
    show = show_images_widget(images, message='Your sleected images')

    if not show:
        st.stop()

    images = {name: image for name, image in images.items() }

    col1, col2 = st.columns(2)
    
    #TODO: Implement equalize_images_widget by equalizing accross all images 
    # with col1:
    #     images = equalize_images_widget(base_image=base_image, images=images)

    with col2:
        aligned_images_stack = align_to_mean_image_widget(images = images)

    with st.expander('Help for parameters for aligning images to their mean', expanded=False):
        alignment_help_widget()
    
    input_dir = 'no_background'
    
    images = cluster_image_widget(aligned_images_stack)  
       
    # Perform clustering on aligned images
    c = cluster_images(aligned_images_stack, 
                              algorithm      = 'SSIM', 
                              n_clusters     = None,
                              method         = 'affinity', 
                              print_metrics  = True, 
                              labels_true    = None)
    
    
if __name__ == '__main__':
    main()
