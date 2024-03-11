# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import torch
from typing import Tuple

# https://stackoverflow.com/questions/56918877/color-match-in-images
# https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html

# https://github.com/pytorch/vision/issues/598
def color_transfer(
    input: torch.Tensor,
    source: torch.Tensor,
    mode: str = "pca",
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Transfer the colors from one image tensor to another, so that the target image's
    histogram matches the source image's histogram. Applications for image histogram
    matching includes neural style transfer and astronomy.

    The source image is not required to have the same height and width as the target
    image. Batch and channel dimensions are required to be the same for both inputs.

    Gatys, et al., "Controlling Perceptual Factors in Neural Style Transfer", arXiv, 2017.
    https://arxiv.org/abs/1611.07865

    Args:

        input (torch.Tensor): The NCHW or CHW image to transfer colors from source
            image to from the source image.
        source (torch.Tensor): The NCHW or CHW image to transfer colors from to the
            input image.
        mode (str): The color transfer mode to use. One of 'pca', 'cholesky', or 'sym'.
            Default: "pca"
        eps (float): The desired epsilon value to use.
            Default: 1e-5

    Returns:
        matched_image (torch.tensor): The NCHW input image with the colors of source
            image. Outputs should ideally be clamped to the desired value range to
            avoid artifacts.
    """

    assert input.dim() == 3 or input.dim() == 4
    assert source.dim() == 3 or source.dim() == 4
    input = input.unsqueeze(0) if input.dim() == 3 else input
    source = source.unsqueeze(0) if source.dim() == 3 else source
    assert input.shape[:2] == source.shape[:2]

    # Handle older versions of PyTorch
    torch_cholesky = (
        torch.linalg.cholesky if torch.__version__ >= "1.9.0" else torch.cholesky
    )

    def torch_symeig_eigh(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        torch.symeig() was deprecated in favor of torch.linalg.eigh()
        """
        if torch.__version__ >= "1.9.0":
            L, V = torch.linalg.eigh(x, UPLO="U")
        else:
            L, V = torch.symeig(x, eigenvectors=True, upper=True)
        return L, V

    def get_mean_vec_and_cov(
        x_input: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert input images into a vector, subtract the mean, and calculate the
        covariance matrix of colors.
        """
        x_mean = x_input.mean(3).mean(2)[:, :, None, None]

        # Subtract the color mean and convert to a vector
        B, C = x_input.shape[:2]
        x_vec = (x_input - x_mean).reshape(B, C, -1)

        # Calculate covariance matrix
        x_cov = torch.bmm(x_vec, x_vec.permute(0, 2, 1)) / x_vec.shape[2]

        # This line is only important if you get artifacts in the output image
        x_cov = x_cov + (eps * torch.eye(C, device=x_input.device)[None, :])
        return x_mean, x_vec, x_cov

    def pca(x: torch.Tensor) -> torch.Tensor:
        """Perform principal component analysis"""
        eigenvalues, eigenvectors = torch_symeig_eigh(x)
        e = torch.sqrt(torch.diag_embed(eigenvalues.reshape(eigenvalues.size(0), -1)))
        # Remove any NaN values if they occur
        if torch.isnan(e).any():
            e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
        return torch.bmm(torch.bmm(eigenvectors, e), eigenvectors.permute(0, 2, 1))

    # Collect & calculate required values
    _, input_vec, input_cov = get_mean_vec_and_cov(input, eps)
    source_mean, _, source_cov = get_mean_vec_and_cov(source, eps)

    # Calculate new cov matrix for input
    if mode == "pca":
        new_cov = torch.bmm(pca(source_cov), torch.inverse(pca(input_cov)))
    elif mode == "cholesky":
        new_cov = torch.bmm(
            torch_cholesky(source_cov), torch.inverse(torch_cholesky(input_cov))
        )
    elif mode == "sym":
        p = pca(input_cov)
        pca_out = pca(torch.bmm(torch.bmm(p, source_cov), p))
        new_cov = torch.bmm(torch.bmm(torch.inverse(p), pca_out), torch.inverse(p))
    else:
        raise ValueError(
            "mode has to be one of 'pca', 'cholesky', or 'sym'."
            + " Received '{}'.".format(mode)
        )

    # Multiply input vector by new cov matrix
    new_vec = torch.bmm(new_cov, input_vec)

    # Reshape output vector back to input's shape &
    # add the source mean to our output vector
    return new_vec.reshape(input.shape) + source_mean

reference = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")
image    = cv2.imread("../../data/interim/no_bg/output_8_London_OrderStJohn.jpg")

# Assuming 'image' and 'reference' are your NumPy arrays
reference_tensor = torch.from_numpy(reference.transpose(2, 0, 1)).float() / 255.0
image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

# Example for standard PyTorch images with value ranges of [0-1]
matched_image_tensor = color_transfer(image_tensor, reference_tensor).clamp(0, 1)

# convert the result back to a NumPy array
matched_image_numpy = (matched_image_tensor[0].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Template')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Original Target')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(matched_image_numpy, cv2.COLOR_BGR2RGB))
axes[2].set_title('Target(adapted colours)')
axes[2].axis('off')
plt.show()



# The match_histograms function from scikit-image is primarily designed to adjust the color distribution
# of an image to match that of a reference image
matched = match_histograms(image, reference, channel_axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Template')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Original Target')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
axes[2].set_title('Target(adapted colours)')
axes[2].axis('off')
plt.show()

# This function performs the actual transfer of color from the source image (the first argument) to
# the target image (the second argument). 
# Because it relies on global color statistics,  large regions with similar pixel intensities
# values can dramatically influence the mean (and thus the overall color transfer). We therefore remove
# black pixels from the source and target images before computing the color statistics.
# The function returns the color transferred image.
# The function is based on the paper "Color Transfer between Images" by Erik Reinhard, Michael Ashikhmin,
# Bruce Gooch, and Peter Shirley, 2001.
 
def color_transfer_excluding_black(source, target):
    # Convert the images from the RGB to L*ab* color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Mask black pixels in the source and target images
    black_mask_source = (source[:, :, 0] == 0) & (source[:, :, 1] == 0) & (source[:, :, 2] == 0)
    black_mask_target = (target[:, :, 0] == 0) & (target[:, :, 1] == 0) & (target[:, :, 2] == 0)

    # Compute color statistics for the source and target images excluding black pixels
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source_lab, black_mask_source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target_lab, black_mask_target)

    # Subtract the means from the target image excluding black pixels
    target_lab[~black_mask_target, 0] = (target_lab[~black_mask_target, 0] - lMeanTar) * (lStdSrc / lStdTar) + lMeanSrc
    target_lab[~black_mask_target, 1] = (target_lab[~black_mask_target, 1] - aMeanTar) * (aStdSrc / aStdTar) + aMeanSrc
    target_lab[~black_mask_target, 2] = (target_lab[~black_mask_target, 2] - bMeanTar) * (bStdSrc / bStdTar) + bMeanSrc

    # Convert LAB back to BGR
    transfer = cv2.cvtColor(target_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    return transfer

def image_stats(image, mask=None):
    if mask is not None:
        image = image[mask]

    # Check if the image is grayscale or color
    if len(image.shape) == 2:  # Grayscale image
        mean_channel = np.mean(image)
        std_channel = np.std(image)
        return (mean_channel, std_channel, mean_channel, std_channel, mean_channel, std_channel)
    elif len(image.shape) == 3:  # Color image
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        return (lMean, lStd, aMean, aStd, bMean, bStd)
    else:
        raise ValueError("Unsupported image format")

def equalize_brightness_color(template, target):
    
    """
    Equalizes the brightness of the target image based on the luminance of the template image.

    Parameters:
    - template: Reference image (template) in BGR color format.
    - target: Target image to be adjusted in BGR color format.

    Returns:
    - equalized_img: Adjusted target image with equalized brightness.
    - ratios: Dictionary containing brightness ratios for LAB, RGB, and grayscale.

    The function converts both the template and target images to the LAB color space,
    adjusts the L channel of the target image based on the mean brightness of the template,
    and then converts the adjusted LAB image back to BGR. The function also calculates
    and returns brightness ratios for LAB, RGB, and grayscale color spaces.
    """
    
    # Convert the template image to LAB color space
    template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the template image
    l_template, a_template, b_template = cv2.split(template_lab)

    # Convert the target image to LAB color space
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Split LAB channels of the target image
    l_target, a_target, b_target = cv2.split(target_lab)

    # Adjust the L channel (brightness) of the target image based on the mean brightness of the template
    l_target = (l_target * (np.mean(l_template) / np.mean(l_target))).clip(0, 255).astype(np.uint8)

    # Merge LAB channels back for the adjusted target image
    equalized_img_lab = cv2.merge([l_target, a_target, b_target])

    # Convert the adjusted LAB image back to BGR
    equalized_img = cv2.cvtColor(equalized_img_lab, cv2.COLOR_LAB2BGR)
    
    # evaluate brightness ratios
    
    # Using LAB color space ---- 
    # we convert the images (template and eq_image) to the LAB color space and calculate 
    # the mean brightness from the luminance channel (L) only

    # Calculate the mean of the color images from the luminance channel
    mean_template_lab = np.mean(cv2.split(template_lab)[0])
    mean_eq_image_lab = np.mean(cv2.split(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2LAB))[0])
    # The ratio is computed based on the mean brightness of the L channel for both color images
    ratio_lab = mean_template_lab / mean_eq_image_lab

    # Using RGB color space ----
    # We calculate the mean intensity across all color channels (R, G, B) for both images
    # (template and equalized_image), i.e., the ratio is computed based on the mean intensity
    # across all color channels for both images   
    mean_template_rgb = np.mean(template)
    mean_eq_image_rgb = np.mean(equalized_img)
    # Calculate the ratio of the brightness of the images
    ratio_rgb = mean_template_rgb / mean_eq_image_rgb

    # Using gray images ----
    # Calculate the mean of the grayscale images
    mean_template_gray = np.mean(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    mean_eq_image_gray = np.mean(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY))
    # Calculate the ratio of the brightness of the grayscale images
    ratio_gray = mean_template_gray / mean_eq_image_gray
    
    # Print brightness ratios ----
    print(f'Brightness ratio (LAB): {ratio_lab}')
    print(f'Brightness ratio (RGB): {ratio_rgb}')
    print(f'Brightness ratio (Grayscale): {ratio_gray}')
    
    # Visualization check ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Template')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Painting with equalized brightness')
    axes[1].axis('off')
    plt.show()
    
    return equalized_img

def compare_color_histograms(image1, image2):
    # Convert images to LAB color space
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    # Calculate histograms for each channel (L, A, B) separately
    hist1_l = cv2.calcHist([lab1], [0], None, [256], [0, 256])
    hist1_a = cv2.calcHist([lab1], [1], None, [256], [0, 256])
    hist1_b = cv2.calcHist([lab1], [2], None, [256], [0, 256])

    hist2_l = cv2.calcHist([lab2], [0], None, [256], [0, 256])
    hist2_a = cv2.calcHist([lab2], [1], None, [256], [0, 256])
    hist2_b = cv2.calcHist([lab2], [2], None, [256], [0, 256])

    # Normalize histograms to values between 0 and 1
    hist1_l /= hist1_l.sum()
    hist1_a /= hist1_a.sum()
    hist1_b /= hist1_b.sum()

    hist2_l /= hist2_l.sum()
    hist2_a /= hist2_a.sum()
    hist2_b /= hist2_b.sum()

    # Calculate the Bhattacharyya coefficient for each channel
    bhattacharyya_l = np.sum(np.sqrt(hist1_l * hist2_l))
    bhattacharyya_a = np.sum(np.sqrt(hist1_a * hist2_a))
    bhattacharyya_b = np.sum(np.sqrt(hist1_b * hist2_b))

    # Average Bhattacharyya coefficient over all channels
    average_bhattacharyya = (bhattacharyya_l + bhattacharyya_a + bhattacharyya_b) / 3

    plt.plot(hist1_l, color='blue', label='Template L')
    plt.plot(hist2_l, color='cyan', label='Image L')
    plt.plot(hist1_a, color='green', label='Template A')
    plt.plot(hist2_a, color='lime', label='Image A')
    plt.plot(hist1_b, color='red', label='Template B')
    plt.plot(hist2_b, color='orange', label='Image B')
    plt.legend()
    plt.show()

    return average_bhattacharyya

# -----------------------------------------------------------------

# source
template = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")
# target
image    = cv2.imread("../../data/interim/no_bg/output_1_London_Nat_Gallery.jpg")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Template')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Original Target')
axes[1].axis('off')
plt.show() 

similarity_score = compare_color_histograms(template, image)
print(f"Color Similarity Score: {similarity_score}")


import numpy as np
from skimage.io import imread, imsave
from skimage import exposure
from skimage.exposure import match_histograms

# Load left and right images
L = template
R = image

# Match using the right side as reference
matched = match_histograms(R,L,channel_axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(L, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Template')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(R, cv2.COLOR_BGR2RGB))
axes[1].set_title('Original Target')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
axes[2].set_title('Target(adapted colours)')
axes[2].axis('off')

plt.show() 

# Place side-by-side and save
result = np.hstack((matched,R))
imsave('result.png',result)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
for i, img in enumerate((R, L, matched)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = exposure.cumulative_distribution(img[..., c])
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)

axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')

plt.tight_layout()
plt.show()


similarity_score = compare_color_histograms(template, matched)
print(f"Color Similarity Score: {similarity_score}")

# Super Fast Color Transfer between Images ----
print("[INFO] Super Fast Color Transfer between Images ...")
#eq_image = color_transfer(template, image)
eq_image = color_transfer(template, image)

similarity_score = compare_color_histograms(template, eq_image)
print(f"Color Similarity Score: {similarity_score}")


# side-by-side comparison of the template (left) and the aligned image (right) 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0].set_title('Template (from which the colours came from)')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Target (adapted colours)')
axes[1].axis('off')
plt.show() 

# equaling level of brightness ----
print("[INFO] equal brightness between the images...")
eq_image_bright = equalize_brightness_color(template, matched)


#The coefficient ranges from 0 to 1, where 0 indicates no similarity, and 1 indicates identical color distributions.
similarity_score = compare_color_histograms(template, eq_image)
print(f"Color Similarity Score: {similarity_score}")



def color_transfer(source, target):
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
 
 # compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	#l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar
	# scale by the standard deviations
	#l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b
	# add in the source mean
	#l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
 
	#transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB).astype("float32") / 255.
    
	return transfer

def image_stats(image):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)
 


eq_image = color_transfer(template, image)


# side-by-side comparison of the template (left) and the aligned image (right) 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0].set_title('Template (from which the colours came from)')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Target (adapted colours)')
axes[1].axis('off')
plt.show() 

similarity_score = compare_color_histograms(template, eq_image)
print(f"Color Similarity Score: {similarity_score}")
