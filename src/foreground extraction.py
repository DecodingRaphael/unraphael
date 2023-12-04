# libraries ----
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

def load(path, size=128):
    img= cv2.resize(cv2.imread(path),(size,size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show():

    f, ax = plt.subplots(3, 4, figsize=(40,20))
    for filename in tqdm(os.listdir("../data/raw/Bridgewater/")):
        
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])

            path = os.path.join("../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)        

            ax[i//4][i%4].imshow(load(path, 300), aspect='auto')
            ax[i//4][i%4].set_title(img_id)
            ax[i//4][i%4].set_xticks([]); ax[i//4][i%4].set_yticks([])
    plt.show()
    
show()
    
# Edge detection with required Morphological Transformations
def show_edges(n_colors = 4):
    
    f, ax = plt.subplots(3, 4, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../data/raw/Bridgewater/")):
        
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])

            path = os.path.join("../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)
            #img = k_means(img, n_colors=n_colors)      
        
            img_gray = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2GRAY)
           
            # MedianBlur is a filter commonly used in image processing to 
            # reduce noise in an image. The filter works by replacing each 
            # pixel value with the median value of its neighboring pixels 
            # within a specified window size.
            img_gray = cv2.medianBlur(img_gray,5)
            edges    = cv2.Canny(img_gray,100,200)
            
            ax[i//4][i%4].imshow(edges, aspect='auto')
            ax[i//4][i%4].set_title(img_id)
            ax[i//4][i%4].set_xticks([]); ax[i//4][i%4].set_yticks([])
    plt.show()
    
show_edges(n_colors =3)
    
# Object detection (Drawing bounding boxes around target) ----
def find_box(edges):
    #contour masking
    co, hi = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con = max(co,key=cv2.contourArea)
    conv_hull = cv2.convexHull(con)
    
    top    = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
    return top, bottom, left, right


def draw_bound_box():
    
    f, ax = plt.subplots(3, 4, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../data/raw/Bridgewater/")):
        
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])
            
            print(img_number)

            path = os.path.join("../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)
        
            org=img.copy()
            #img= k_means(img , n_colors= 10)
            
            img_gray = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
            img_gray = cv2.medianBlur(img_gray,7)
            edges    = cv2.Canny(img_gray,100,200)
            
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
            edges    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            top,bottom,left,right = find_box(edges)
            org=cv2.rectangle(org, (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)
            
            ax[i//4][i%4].imshow(org, aspect='auto')
            ax[i//4][i%4].set_title(img_id)
            ax[i//4][i%4].set_xticks([]); ax[i//4][i%4].set_yticks([])
    plt.show()
        
draw_bound_box()

# Foreground extraction
def forgrd_ext(img, rec):
    mask    = np.zeros(img.shape[:2], np.uint8)
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rec, bgmodel, fgmodel, 3, cv2.GC_INIT_WITH_RECT)
    mask2= np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img= img*mask2[:,:,np.newaxis]
    img[np.where((img == [0,0,0]).all(axis = 2))] = [255.0, 255.0, 255.0]
    return img

def ext_frgd():
    
    f, ax = plt.subplots(3, 4, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../data/raw/Bridgewater/")):
        
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])

            path = os.path.join("../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)        
        
            org=img.copy()
            #img= k_means(img , n_colors= 10)
            
            img_gray= cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
            img_gray= cv2.medianBlur(img_gray,7)
            edges = cv2.Canny(img_gray,100,200)
            
            kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            top,bottom,left,right = find_box(edges)
            rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
            forground_img= forgrd_ext(org, rec)
            
            ax[i//4][i%4].imshow(forground_img, aspect='auto')
            ax[i//4][i%4].set_title(img_id)
            ax[i//4][i%4].set_xticks([]); ax[i//4][i%4].set_yticks([])
    plt.show()
    
ext_frgd()
   
# Create extracted images 

# make a directory to store extractions in
#!mkdir image

def image_extract():
    
    f, ax = plt.subplots(3, 4, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../data/raw/Bridgewater/")):
    
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])

            path = os.path.join("../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            im1 = load(path, 300)            
            #im2= img= k_means(im1 , n_colors= 6)
            im2 = im1
            im3 = cv2.cvtColor(np.uint8(im2*255), cv2.COLOR_RGB2GRAY)
            im3 = cv2.medianBlur(im3,5)
            im3 = cv2.Canny(im3,100,200)
            
            kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            im31 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)
            top,bottom,left,right = find_box(im31)
            
            if top!=(0,0) and top!=bottom:
                im4=cv2.rectangle(im1.copy(), (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)
                rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
                im5= forgrd_ext(im1, rec)
                cv2.imwrite('./image/'+ path.split('/')[-1],cv2.cvtColor(im5,cv2.COLOR_BGR2RGB))
                return im5
            else:
                return im1

image_extract()
    

IMG_DIR='../data/raw/Bridgewater'

files0=os.listdir(IMG_DIR)
files=[]
for item in files0:
    if item[-4:]=='.jpg':
        files+=[item]
files        


path = os.path.join("../data/raw/Bridgewater/", files[1])
im1= load(path, 300)

im3= cv2.cvtColor(np.uint8(im1*255), cv2.COLOR_RGB2GRAY)
im3= cv2.medianBlur(im3,5)
im3 = cv2.Canny(im3,100,200)
kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
im31 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)

top,bottom,left,right = find_box(im31)
im4=cv2.rectangle(im1.copy(), (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)

rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
im5= forgrd_ext(im1, rec)

cv2.imwrite('im2.png',cv2.cvtColor(im5,cv2.COLOR_BGR2RGB))

imgs=[im1,im3,im31,im4,im5]
f, ax = plt.subplots(2, 3, figsize=(18,10))
for i in tqdm(range(6)):
    r=i//3
    c=i%3
    ax[r][c].imshow(imgs[i], aspect='auto')
    ax[r][c].set_xticks([]) 
    ax[r][c].set_yticks([])
plt.show()


# GrabCut ----
# GrabCut is an interactive image segmentation algorithm that combines the user's input
# and image features to perform segmentation. It requires the user to provide initial seed
# points to mark foreground and background regions. OpenCV provides the grabCut function, 
# which takes an input image and the user's initial markings to iteratively refine the 
# segmentation. It can be used to separate foreground objects from the background in an image.

path = os.path.join("../data/raw/Bridgewater/", files[1])
im1= load(path, 300)

# Convert to grayscale
gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(im1, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Contours', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a mask
mask = np.zeros(im1.shape[:2], np.uint8)

# Specify the region of interest (foreground and background)
#rect = (x, y, width, height)
rect = (50,50,450,290)

# Apply GrabCut algorithm
bgd_model = np.zeros((1,65), np.float64)
fgd_model = np.zeros((1,65), np.float64)

cv2.grabCut(im1, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Assign foreground and probable foreground pixels a value of 1, others to 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the original image
segmented_image = im1 * mask2[:, :, np.newaxis]

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Watershed Transform ----
# Load image
image = im1

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to remove noise and fill gaps
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that sure background is not 0 but 1
markers = markers + 1

# Mark the unknown region as 0
markers[unknown == 255] = 0

# Apply watershed algorithm
cv2.watershed(image, markers)

# Generate segmented image
segmented_image = np.zeros(image.shape, dtype=np.uint8)
segmented_image[markers > 1] = [0, 255, 0]

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()