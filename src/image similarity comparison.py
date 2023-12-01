# Comparing images using ORB similarity, structural similarity index, and other measures

# libraries ----
from skimage.metrics import structural_similarity
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageChops
import numpy as np
from skimage.transform import resize

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

# functions ----

# orb_similarity works well with images of different dimensions
def orb_sim(img1, img2):

  # initiate ORB detector
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


# structural_similarity needs images to have same dimensions
def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full = True)
  return sim

# Analysis ----

## Histogram-Based Approach ----

## reading data (color format)
img00 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 1)
img01 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 1)

img00.shape # 800 x 540 pixels
img01.shape # 1200 x 905 pixels

# plot the original
plt.imshow(img00) 
plt.imshow(img01)

hist_img00 = cv2.calcHist([img00], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
hist_img00[255, 255, 255] = 0 #ignore all white pixels

cv2.normalize(hist_img00, hist_img00, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

hist_img01 = cv2.calcHist([img01], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
hist_img01[255, 255, 255] = 0  #ignore all white pixels

cv2.normalize(hist_img01, hist_img01, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Find the metric value
metric_val = cv2.compareHist(hist_img00, hist_img01, cv2.HISTCMP_CORREL)
print(f"Similarity Score: ", round(metric_val, 2))
# Similarity Score: 0.03

## orb_similarity approach ----

## reading data (gray format)
img00 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0)
img01 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)

# plot the original
plt.imshow(img00,cmap="gray") 
plt.imshow(img01,cmap="gray")

orb_similarity = orb_sim(img00, img01)  

# 1.0 means identical. Lower value = not similar
print("Similarity using ORB is: ", orb_similarity) # 0.47

## Structural Similarity Index (SSIM) approach ----
#Resize an image for SSIM ----
(H, W) = img00.shape

img01 = cv2.resize(img01, (W, H))
#img01 = cv2.resize(img01, (W, H),interpolation = cv2.INTER_AREA)

# double check dimensions
print(img00.shape, img01.shape)

ssim = structural_sim(img00, img01) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", round(ssim,2)) # 0.34 based on gray images

# now check the similarity after we first conduct edge detection ----

# One way to get rid of the noise on the image, is by applying Gaussian blur to smooth it
# gaussian filter
def apply_gaussian_filter(image, kernel_size, sigma):
    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply the filter using cv2.filter2D
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image  
  
  # Set the kernel size and sigma
kernel_size = 5
sigma = 1.4

# Apply the Gaussian filter
img00_g = apply_gaussian_filter(img00, kernel_size, sigma)
img01_g = apply_gaussian_filter(img01, kernel_size, sigma)

cv2.imshow('Original Image', img00)
cv2.imshow('Filtered Image', img00_g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Canny edge detection on gaussian-filtered images
img00_canny = cv2.Canny(img00_g, 75, 255)
img01_canny = cv2.Canny(img01_g, 30, 100)

plt.imshow(img00_canny, cmap="gray")
plt.imshow(img01_canny, cmap="gray")

print("Image 1:", img00_canny.shape)
print("Image 2:", img01_canny.shape)

# similarity after edge detection
ssim_canny = structural_sim(img00_canny, img01_canny) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", round(ssim_canny,2)) # 0.86


# Plotting differences between scaled-images with PIL ----

# Load the images
img1 = Image.open('../data/raw/0_Edinburgh_Nat_Gallery.jpg')
img2 = Image.open('../data/raw/Bridgewater/8_London_OrderStJohn.jpg')

# Convert PIL Image objects to NumPy arrays
img1_array = np.array(img1)
img2_array = np.array(img2)

# Resize img2 to match the size of img1
img2_resized = resize(img2_array, (img1_array.shape[0], img1_array.shape[1]), anti_aliasing=True, preserve_range=True)

# Convert the resized array back to PIL Image
img2_resized_pil = Image.fromarray(img2_resized.astype(np.uint8))

width, height = img2_resized_pil.size
print(f"Image Dimensions: {width} pixels (width) x {height} pixels (height)")

width, height = img1.size
print(f"Image Dimensions: {width} pixels (width) x {height} pixels (height)")

# Now, img2_resized_pil has the same size as img1
diff = ImageChops.difference(img1, img2_resized_pil)

ImageChops.overlay(img1, img2_resized_pil)

if diff.getbbox():
    diff.show() #Shows the difference between the two images
else:
    print("Images are not the same")
    
# Measuring similarity in two images using several metrics ----
print("MSE: ", mse(img00,img01))
print("RMSE: ", rmse(img00, img01))
print("PSNR: ", psnr(img00, img01))
print("SSIM: ", ssim) 
#print("SSIM: ", ssim(img00, img01))
print("UQI: ", uqi(img00, img01))
print("MSSSIM: ", msssim(img00, img01))
print("ERGAS: ", ergas(img00, img01))
print("SCC: ", scc(img00, img01))
print("RASE: ", rase(img00, img01))
print("SAM: ", sam(img00, img01))
print("VIF: ", vifp(img00, img01))

# Measuring similarity in two canny-edge images using several metrics ----
print("MSE: ", mse(img00_canny,img01_canny))
print("RMSE: ", rmse(img00_canny, img01_canny))
print("PSNR: ", psnr(img00_canny, img01_canny))
print("SSIM: ", ssim_canny) 
print("UQI: ", uqi(img00_canny, img01_canny))
print("MSSSIM: ", msssim(img00, img01))
print("ERGAS: ", ergas(img00, img01))
print("SCC: ", scc(img00, img01))
print("RASE: ", rase(img00, img01))
print("SAM: ", sam(img00, img01))
print("VIF: ", vifp(img00, img01))

import image_similarity_measures
from image_similarity_measures.quality_metrics import psnr,uiq,sam,sre,issm,fsim,ssim,rmse

out_rmse = rmse(img00, img01)
out_psnr = psnr(img00, img01)
# ad... more metrics

# CALCULATE ALL VALUES FOR ALL PAIRS
# STANDARDIZE METRICS
# BRING BAKE TO ONE METRIC
# PLOT

# TOWARDS CORRELATION MATRIX REPRESENTING SIMILARITY BETWEEN IMAGES

# USE DAG ANALYSIS WITH TIERS (E.G. BASED ON YEAR OF PRODUCTION) TO MAKE A 
# FAMILY TREE REPRESENTING THE UNDRLYING GENEALOGICAL AFFILIATIONS

# PROBLEM: LITTLE DATA

