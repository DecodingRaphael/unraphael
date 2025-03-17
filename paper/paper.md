---
title: 'UnRaphael: A Python pipeline tool for analysing the similarity between paintings and their reproductions'
tags:
  - python
  - computer vision
  - computational art analysis
  - paintings
  - image-processing
authors:
  - name: Thijs Vroegh^[Corresponding author]
    corresponding: true
    orcid: 0000-0002-6311-0798
    equal-contrib: true
    affiliation: "1"
  - name: Stef Smeets
    orcid: 0000-0002-5413-9038
    equal-contrib: true
    affiliation: "1"
  - name: Lisandra S. Costiner
    orcid: 0000-0002-6580-9164
    equal-contrib: true
    affiliation: "2"  
affiliations:
 - name: Netherlands eScience Center, The Netherlands
   index: 1  
 - name: Utrecht University, The Netherlands
   index: 2
date: 16 January 2025
bibliography: paper.bib

---

# Summary

Taking its name from the Renaissance artist Raphael and the popular practice of reproducing paintings, `Unraphael`[@vroegh2024] is a Python workflow tool designed for art historical research. It aims to assist in the comparison of paintings, in particular of originals and their copies. It is designed to streamline the preprocessing, background removal, image alignment, outline extraction, and clustering of photographs of these paintings, grouping the images into similarity-based clusters. The tool specifically targets the challenge of comparing near-identical images by offering a comprehensive pipeline that includes state-of-the-art similarity measures, clustering techniques, together with several options for close visual inspection. Unraphael facilitates easy quantitative comparison of extracted figure-outlines across images of painting reproductions. Integrated with a user-friendly Streamlit interface, the package provides an efficient, flexible and accessible solution for art historians, museum professionals, digital humanities scholars and the general public to experiment, analyse and interpret visual data.

# Statement of need

The comparison of designs or figures across paintings is of key interests to art historians, museum professionals and specialists, and is traditionally carried out by manually copying the originals using tracing paper [@holmes2004] [humfrey2022]. Among its many functions, it assists in uncovering the relationship between paintings, in understanding the practices used by an artist in the production and reproduction of a particular composition, in tracing the dissemination of designs within and beyond an artistic workshop, and in exploring creative and collaborative working methods [@bambach1999]. The process of producing manual tracings comes with many challenges, including access to original artworks which may be on museum display or in situ, as well as concerns about working with and handling fragile original artworks. Recently a range of computational approaches have been added to the art historian’s resources for studying paintings [@stork2023], although these have not focused specifically on the comparison of figures or compositions.

`Unraphael` addresses this challenge by providing an integrated pipeline that automates the essential steps of comparing elements within paintings. This relies on photographs of paintings alone and provides an intuitive and easy-to-use interface, accessible without specialised knowledge. This includes the steps of image pre-processing, segmentation, alignment, and clustering analysis based on structural similarities. In contrast to recent advancements in deep - and machine learning [@smith2024] [@ugail2023] as well as GPT-based agents [@tang2024], which have influenced the study of art and its attribution, `Unraphael` avoids these black box methods to ensure that the applied techniques can be interpreted and better understood by researchers from the art-historical domain. It integrates third-party algorithms from established Python libraries in the computer-vision field (e.g., OpenCV and scikit-image), allowing researchers to focus on interpreting analytical results rather than managing complex image processing tasks. It is open-source and available on `Github` [@unraphael] under the Apache 2.0 license.

# Overview

`Unraphael` integrates several image-processing functionalities into a cohesive pipeline, which can be used in sequence or independently from each other. It is supported by a [streamlit application](https://unraphael.streamlit.app) which offers an interactive web-based environment for conducting and visualising the analyses. Its user-friendly interface guides researchers through the entire process from data ingestion to result interpretation, enhancing the overall analytical experience. Several tutorials are available on the accompanying [documentation page](https://unraphael.readthedocs.io/en/latest/), explaining how  `Unraphael` can be applied using a small dataset of reproductions of paintings. The first **preprocessing page** allows users to upload one or multiple images (in *jpg* or *png*) and enables automatic standardisation through resizing, colour normalisation and enhancement to ensure consistency across images (see \autoref{fig:dash_preprocess}).

![Visualisation of the preprocessing page\label{fig:dash_preprocess}](dash_preprocess.png)

In particular, the pre-processing consists of the following features:

- Bilateral Filter Strength: This filter smooths the image while preserving edges, useful for reducing noise while keeping important details intact.
- Color Saturation: Adjusts the vibrancy of colours in the image, enhancing saturation which makes features stand out more clearly.
- CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast in the image, making details in darker or lighter regions more visible.
- Sharpness Sigma: Controls the sharpness of the image by applying a sharpening filter, which can help highlight fine details.
- Gamma: Adjusts the brightness and contrast of the image.
- Gain: Similar to gamma, but specifically enhances the overall brightness of the image.
- Sharpening Radius: Defines the extent of the sharpening effect applied to the image.

In addition, several options allow users to control and optimise the background removal process. The extracted figures in the foreground can be saved and used for subsequent analysis.

The **segmentation page** allows the isolation of key objects within each image such as individual figures, for further in-depth analysis. Additionally, analysing the pose structure enables a first visual comparison between the images. Next, the **alignment page** enables the alignment of multiple images to a selected base image. This feature is essential for comparing images against a consistent frame of reference. If desired, users can also adjust features such as brightness, contrast, sharpness, and colour to match the base image, ensuring uniformity before alignment and allowing for optimal (visual) comparison. Images can be aligned using:

- Feature-based alignment: Aligns images using detected features based on SIFT, SURF or ORB.
- Enhanced correlation coefficient maximization: Maximizes correlation between images.
- Fourier Mellin transform: Aligns images based on frequency content.
- FFT phase correlation: Aligns images using phase correlation in the Fourier domain.
- Rotational alignment: Aligns images by correcting rotational differences.

In addition, several motion models are available (see \autoref{fig:motion_models}). These models are used to describe the transformation between the base image and the other images.

![Available motion models.\label{fig:motion_models}](motion_models.png)

To facilitate detailed visual analysis, a number of ways of visualising the information are provided. An animation can be produced that slowly fades one of the images into the second one to aid comparison. A slider can be used to compare the base and aligned image., A third feature, produces an overlay of the outlines of the regions of interest (eg., figures in the painting). This is accompanied by the computations of similarity metrics such as *Frechet*-, *Procrustes*- and *Hausdorff* distances. These different features empower the user to more closely analyse and compare the results from different perspectives

The **cluster page** of `Unraphael` is specifically designed to cluster images based on what in computer science is termed structural similarities, while in in art history formal similarities, while being resilient to noise, outliers, or variations in the data [@yang2004]. First, an option is presented to equalise brightness, sharpness, and contrast to reduces variations due to lighting varying photographic conditions. For instance, copies of paintings may have been photographed under different lighting conditions, or might have been printed or created with different sharpness and contrast levels. Equalising these factors ensures that the clustering focuses on the actual content of the images and not on extraneous factors introduced during photographing, such as lighting or image quality.

Next, aligning images to their mean ensures that the key elements of each image, typically the elements to be compared, are in the same spatial location across images. This is critical for structural similarity metrics, which assume that the images being compared are geometrically aligned. Clustering takes place on either the foregrounded figures in the image (i.e., with background removed) or the outer outlines of these figures. In the former case, clustering is based on a) the extracted features from the figures or b) a similarity matrix containing pairwise structural similarity indices. These indices include the *Structural Similarity Index* [@wang2004], *Complex Wavelet Structural Similarity Index* [@sampat2009] and a metric for *brushstroke signatures* based on a combination of edge detection techniques [@ugail2023]. *Spectral clustering*, *density-based spatial clustering of applications with noise* (DBSCAN) and *affinity propagation* are among the available clustering methods attuned to these similarity matrices. To assess the effectiveness of the clustering process, performance metrics such as the *Silhouette-* *Davies-Bouldin-* and *Calinski Harabasz score* are displayed. These metrics provide valuable insights into the cohesion and separation of clusters, ensuring that the groupings reflect meaningful similarities. The process is complemented by several visualizations. The first is a dendrogram which visualises relationships between the paintings. The second and third are a silhouette- with a scatterplot that map how images group together into clusters using a 2D projection based on Principal Component Analysis (see \autoref{fig:silhoutte_plot}).

![(left) Example of silhoutte plot and (right) 2D PCA plot of image clustering.\label{fig:silhoutte_plot}](silhoutte_plot.png)

# Conclusion

`Unraphael` is a powerful tool that offers a user-friendly way of examining the visual relationships between original paintings and their copies or derivatives. This new computational approach not only aids in investigating copying methods—such as the use of stencils or cartoons—but also contributes to broader discussions in art history about the transmission of compositions and techniques across time and geography. Unraphael represents a significant advancement in computational art analysis that can be integrated into the toolbox of art historians, museum professionals, and art enthusiasts to further the understanding of paintings, their methods of production and histories.

# Acknowledgements

We acknowledge contributions from Christiaan Meijer and Elena Ranguelova for scientific input and helpful discussions guiding the development of `Unraphael`. The development of this project was generously supported by the Netherlands eScience Center, under grant number NLESC.SSIH.2022a.SSIH016.

# References
