# Image Preprocessing and Background Removal

This section walks you through the image preprocessing and background removal steps available in the app. These steps help to optimize your image for further analysis by enhancing important features and removing unnecessary background elements.

## Image Preprocessing

The preprocessing stage is crucial for preparing the image, ensuring that the background removal and subsequent analysis are more accurate and effective. The app provides several adjustable filters to refine your image:

- **Bilateral Filter Strength**: This filter smooths the image while preserving edges. It's useful for reducing noise while keeping important details intact.
- **Color Saturation**: Adjusts the vibrancy of colors in the image. Enhancing saturation can make features stand out more clearly.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast in the image, making details in darker or lighter regions more visible. Adjust the clip limit and tile grid size to control how aggressively the contrast is enhanced.
- **Sharpness Sigma**: Controls the sharpness of the image by applying a sharpening filter. Increasing sharpness can help highlight fine details.
- **Gamma**: Adjusts the brightness and contrast of the image. A higher gamma makes the image brighter, while a lower gamma darkens it.
- **Gain**: Similar to gamma, but specifically enhances the overall brightness of the image.
- **Sharpening Radius**: Defines the extent of the sharpening effect applied to the image. A larger radius sharpens broader areas, while a smaller radius focuses on fine details.

The app displays a side-by-side comparison of the original and processed images, allowing you to fine-tune the parameters until you're satisfied with the result.

##  Background Removal

After preprocessing, the next step is to remove the background from the image. This step isolates the subject of interest, making it easier to analyze without interference from background elements:

- **Alpha Matting**: A technique that refines the edges of the mask, ensuring a smoother and more natural transition between the subject and the background.
- **Keep Mask Only**: If you only need the mask (the outline of the subject), this option will remove the image and leave just the mask.
- **Postprocess Mask**: Further refines the mask after the initial processing, which can help in obtaining cleaner results.

You can also choose the background color that will replace the removed background:

- **Background Color**: Options include Transparent, White, or Black. This setting determines what color will fill the area where the background is removed.

Additional parameters help fine-tune the background removal process:

- **Background Threshold**: Controls how aggressively the background is identified and removed. A lower threshold may retain more background, while a higher threshold removes more.
- **Foreground Threshold**: Similar to the background threshold, but for the foreground (the subject). Adjusting this helps in clearly defining the subject's boundaries.
- **Erode Size**: Determines how much the mask should shrink after background removal, which helps in fine-tuning the edges of the subject.

The app shows the resulting mask and the image with the background removed, allowing you to compare and adjust settings as needed.

## Final Steps and Download

After preprocessing and background removal, you can download the processed images for further analysis. The app allows you to save:

- The original image
- The image with the background removed
- An extracted version, where only the subject remains, isolated from both the background and any irrelevant details

These steps ensure that the image is optimized and ready for further analysis, such as aligning the images or clustering images based on their features.
