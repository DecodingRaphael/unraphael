# Comparing images using ORB similarity, structural similarity index, and other measures

# libraries ----
from skimage.metrics import structural_similarity
import cv2
import imutils
from matplotlib import pyplot as plt
from PIL import Image, ImageChops
import imagehash
import numpy as np
from skimage.transform import resize
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import image_similarity_measures
from image_similarity_measures.quality_metrics import psnr,uiq,sam,sre,issm,fsim,ssim,rmse

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

## A) Histogram-Based comparison ----

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

## B) ORB similarity approach ----

## reading data (gray format)
img00 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0)
img01 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)

# plot the original
plt.imshow(img00,cmap="gray") 
plt.imshow(img01,cmap="gray")

orb_similarity = orb_sim(img00, img01)  

# 1.0 means identical. Lower value = not similar
print("Similarity using ORB is: ", orb_similarity) # 0.47

## C) Structural Similarity Index (SSIM) approach ----

# resize an image for SSIM comparison -
(H, W) = img00.shape

img01 = cv2.resize(img01, (W, H))
#img01 = cv2.resize(img01, (W, H),interpolation = cv2.INTER_AREA)

# double check dimensions
print(img00.shape, img01.shape)

# compute the Structural Similarity Index (SSIM) between the two images
ssim = structural_sim(img00, img01) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", round(ssim,2)) # 0.34 based on gray images

# Ensure that the difference image is returned
(score, diff) = structural_similarity(img00, img01, full=True)
diff = (diff * 255).astype("uint8")

# Threshold the diff image, and find contours which will showcase the regions in the images
# that are different
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours and create bounding boxes on our two images
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img00, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img01, (x, y), (x + w, y + h), (255, 0, 0 ), 2)
    
plt.figure(figsize=(15,10))
plt.title("Original")
plt.imshow(img00)
    
plt.figure(figsize=(15,10))
plt.title("Copy")
plt.imshow(img01)

plt.figure(figsize=(15,10))
plt.suptitle('More darker the shade area is, the higher image difference there is!', fontsize=14, fontweight='bold')
plt.title("Difference")
plt.imshow(diff)

plt.figure(figsize=(15,10))
plt.suptitle('The yellowish part is showing the difference!', fontsize=14, fontweight='bold')
plt.title("Thresh")
plt.imshow(thresh)

## D) Structural Similarity Index based after canny edge detection ----

# One way to get rid of the noise on the image, is by applying Gaussian blur to smooth it
# A Gaussian blur is an image filter that uses a kind of function called a Gaussian to 
# transform each pixel in the image. It has the result of smoothing out image noise and reducing detail.

# The purpose of the GaussianBlur filter is to smooth out the image 
# and reduce noise. The amount of blur is controlled by the standard deviation of
# the Gaussian function, which determines the width of the bell curve. A larger 
# standard deviation results in a wider and smoother blur, while a smaller standard
# deviation produces a narrower and more detailed blur. 
# 
# The GaussianBlur filter is often used as a preprocessing step in image processing tasks
# such as edge detection, object recognition, and image segmentation.

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

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

## E) Finding Image Difference using ImageChops module of Python Imaging Library (PIL)

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

# the absolute value of the pixel-by-pixel difference between the two images
diff = ImageChops.difference(img1, img2_resized_pil)

# visualize the difference map
ImageChops.overlay(img1, img2_resized_pil)

if diff.getbbox():
    diff.show() #Shows the difference between the two images
    image = diff
else:
    print("Images are not the same")
    
image.save("Difference.jpg")  # Saving the output file as Image.
plt.figure(figsize=(15,10))
plt.suptitle('More Lighter the shade area is, the Higher Image difference is there!', fontsize=14, fontweight='bold')
plt.title("Heatmap Difference between 2 Images!")
plt.imshow(image)
   
## F) Measuring similarity of two images using several other metrics ----
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

out_rmse = rmse(img00, img01)
out_psnr = psnr(img00, img01)
# ad... more metrics

## G) Measuring similarity of two images with imagehash ----

# Load the images
img1 = Image.open('../data/raw/0_Edinburgh_Nat_Gallery.jpg')
img2 = Image.open('../data/raw/Bridgewater/8_London_OrderStJohn.jpg')

hash1 = imagehash.average_hash(img1)
hash2 = imagehash.average_hash(img2)
diff = hash1 - hash2
print(diff)
# 8


# CALCULATE ALL VALUES FOR ALL PAIRS
# STANDARDIZE METRICS
# BRING BAKE TO ONE METRIC
# PLOT

# TOWARDS CORRELATION MATRIX REPRESENTING SIMILARITY BETWEEN IMAGES

# USE DAG ANALYSIS WITH TIERS (E.G. BASED ON YEAR OF PRODUCTION) TO MAKE A 
# FAMILY TREE REPRESENTING THE UNDRLYING GENEALOGICAL AFFILIATIONS

# PROBLEM: LITTLE DATA

