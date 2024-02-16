# https://www.kaggle.com/code/akhileshdkapse/foreground-extraction-opencv
# https://www.kaggle.com/code/stpeteishii/face-image-foreground-extraction
# https://www.kaggle.com/code/stpeteishii/whale-dolphin-image-foreground-extraction
# https://www.kaggle.com/code/stpeteishii/large-fish-foreground-extraction-opencv
# https://kaggle.com/code/stpeteishii/bird-image-foreground-extraction
# https://www.kaggle.com/code/markdaniellampa/foreground-extraction-in-fish

# https://www.kaggle.com/code/mlvprasad/opencv-in-depth-course-2023-for-indian-kaggler
# see also https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

# libraries ----
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import random


# a list of all the images in the directory ----
IMG_DIR = '../../data/raw/Bridgewater'

files0=os.listdir(IMG_DIR)
files=[]
for item in files0:
    if item[-4:]=='.jpg':
        files+=[item]
files        

n=len(files0)
N=list(range(n))
random.seed(2022)
random.shuffle(N)
paths=np.array(files0)[N[0:300]]

def load(path, size=128):
    img= cv2.resize(cv2.imread(path),(size,size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show():
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
            
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            
            # Extracting the number from the filename
            img_number = int(filename.split("_")[0])
            
            path = os.path.join("../../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)        

            ax[i//5][i%5].imshow(load(path, 300), aspect='auto')
            ax[i//5][i%5].set_title(img_id)
            ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
show()
    
# Adaptive histogram equalization technique
def adaptive_hist(img, clipLimit= 4.0):
    
    window= cv2.createCLAHE(clipLimit= clipLimit, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    ch1, ch2, ch3 = cv2.split(img_lab)
    img_l = window.apply(ch1)
    img_clahe = cv2.merge((img_l, ch2, ch3))
    return cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)


def show_adhist(clipLimit=4.0):        
    f, ax = plt.subplots(3, 5, figsize=(40,20))
        
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):             
        img_number = int(filename.split("_")[0])        
        path = os.path.join("../../data/raw/Bridgewater/", filename)
                
        img_id = str(img_number)
        i = int(img_number)                
                                 
        #path= os.path.join(IMG_DIR, files[i])
        img = load(path, 300)
        img = adaptive_hist(img, clipLimit)
        
        ax[i//5][i%5].imshow(img, aspect='auto')
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()

show_adhist(2.0)    

# Color Quantization using K-Means
# The purpose of Color Quantization using K-Means in the provided code is to reduce 
# the number of distinct colors in an image to a specified value (n_colors). Overall, 
# the color quantization step using K-Means helps simplify the images by reducing the 
# number of colors, making subsequent image processing steps more efficient and 
# potentially improving the effectiveness of object detection and foreground extraction

# - The k_means function applies the K-Means clustering algorithm to perform color 
#   quantization
# - The image is reshaped into a 2D array of pixels, and K-Means is applied to cluster 
#   these pixels into n_colors clusters
# - The codebook (cluster centers) and labels are obtained, and a new image is created 
#   by assigning each pixel the color of its corresponding cluster center

def k_means(img, n_colors= 4):
    w, h, d = original_shape = tuple(img.shape)
    img= img/255.0
    image_array = np.reshape(img, (w * h, d))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    
    """Recreate the (compressed) image from the code book & labels"""
    codebook= kmeans.cluster_centers_
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# The show_kmean function displays images after color quantization using K-Means.
def show_kmean(n_colors=4):
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
    
        img_number = int(filename.split("_")[0])

        path = os.path.join("../../data/raw/Bridgewater/", filename)
        img_id = str(img_number)
        i = int(img_number)           
            
        img = load(path, 300)         
        
        img= k_means(img , n_colors= n_colors)
        
        ax[i//5][i%5].imshow(img, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
show_kmean(n_colors= 4)

# Edge detection with required morphological transformations
def show_edges(n_colors = 4):    
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
            
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])
            #img_number = int(filename.split("_")[1])

            path = os.path.join("../../data/raw/Bridgewater/", filename)
                        
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)
            img = k_means(img, n_colors=n_colors)      
        
            img_gray = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2GRAY)
           
            # MedianBlur is a filter commonly used in image processing to 
            # reduce noise in an image. The filter works by replacing each 
            # pixel value with the median value of its neighboring pixels 
            # within a specified window size.
            img_gray = cv2.medianBlur(img_gray,5)
            edges    = cv2.Canny(img_gray,100,200)
            
            ax[i//5][i%5].imshow(edges, aspect='auto')
            ax[i//5][i%5].set_title(img_id)
            ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
show_edges(n_colors = 4)
    
# The foreground extraction code in the provided example uses the GrabCut algorithm for segmentation. 
# GrabCut is a powerful method, but its success heavily depends on accurate initialization. Since the 
# code uses a bounding box obtained from edge detection, the quality of the segmentation heavily relies
# on the accuracy of the bounding box

# This code now checks for both face and body detections and combines them into a single bounding box 
# if both are present
def find_box_with_face_and_body_detection(edges):
    
    # Load the pre-trained face classifier
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('../../data/haarcascades/cascades/haarcascade_frontalface_default.xml')

    # Load the pre-trained full body classifier
    #body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    body_cascade = cv2.CascadeClassifier('../../data/haarcascades/cascades/haarcascade_fullbody.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(edges, scaleFactor=1.08, minNeighbors=5)

    # Detect full bodies in the image
    bodies = body_cascade.detectMultiScale(edges, scaleFactor=1.08, minNeighbors=5)

    print(faces)
    # Combine the results of face and body detection
    if len(faces) > 0 and len(bodies) > 0:
        
        # If both face and body are detected, create a bounding box encompassing both
        print("Face and body detected")
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = bodies[0]
        top = min(y1, y2)
        bottom = max(y1 + h1, y2 + h2)
        left = min(x1, x2)
        right = max(x1 + w1, x2 + w2)
    # If only faces are detected, create a bounding box encompassing those 
    elif len(faces) > 0:
        print("Only face detected")
        x, y, w, h = faces[0]
        top, bottom, left, right = y, y + h, x, x + w
    
    # If only bodies are detected, create a bounding box encompassing those
    elif len(bodies) > 0:
        print("Only body detected")
        x, y, w, h = bodies[0]
        top, bottom, left, right = y, y + h, x, x + w
    
    # If neither faces nor bodies are detected, use the contour-based approach               
    else:  
        print("neither faces nor bodies are detected, using the contour-based approach now")
        co, hi = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if co!=():
            con = max(co, key=cv2.contourArea)
            conv_hull = cv2.convexHull(con)
            
            top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
            bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
            left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
            right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
            
            return top, bottom, left, right
        
        else:
            return (0,0),(0,0),(0,0),(0,0)

# Object detection (Drawing bounding boxes around target) ----
def find_box(edges):    
    #contour masking 
    co, hi = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    #valid_contours = [cnt for cnt in co if cv2.contourArea(cnt) > 0.05 and cv2.contourArea(cnt) < 0.05]
    #con = max(valid_contours, key=cv2.contourArea)
    if co!=():
        
        con = max(co, key = cv2.contourArea)
        conv_hull = cv2.convexHull(con)
        
        top    = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
        left   = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
        right  = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
        return top, bottom, left, right
    else:
        return (0,0),(0,0),(0,0),(0,0)


def draw_bound_box():    
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
        
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])
            
            print(img_number)

            path = os.path.join("../../data/raw/Bridgewater/", filename)
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)
            org = img.copy()
            #img = k_means(img , n_colors= 4)
            
            gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
            img_gray = cv2.medianBlur(img_gray,7)
            edges    = cv2.Canny(img_gray,100,200)
            
            # Perform morphological operations to remove noise and fill gaps
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
            edges    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Get bounding box using face detection or contour-based approach
            
            # original
            top,bottom,left,right = find_box(edges)
            
            # adapted
            #top,bottom,left,right = find_box_with_face_and_body_detection(gray)
                        
            # Draw bounding box on the image
            org = cv2.rectangle(org, (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)
            
            ax[i//5][i%5].imshow(org, aspect='auto')
            ax[i//5][i%5].set_title(img_id)
            ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
        
draw_bound_box()

# Foreground extraction

def forgrd_ext(img, rec):    
    # Create a mask
    mask    = np.zeros(img.shape[:2], np.uint8)
    
    # Apply GrabCut algorithm
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)   
   
    # GrabCut is an interactive image segmentation algorithm that combines the user's input
    # and image features to perform segmentation. It requires the user to provide initial seed
    # points to mark foreground and background regions. OpenCV provides the grabCut function, 
    # which takes an input image and the user's initial markings to iteratively refine the 
    # segmentation. It can be used to separate foreground objects from the background in an image.
    # https://www.kaggle.com/code/mlvprasad/opencv-in-depth-course-2023-for-indian-kaggler
    cv2.grabCut(img, mask, rec, bgmodel, fgmodel, 3, cv2.GC_INIT_WITH_RECT)
    
    # Assign foreground and probable foreground pixels a value of 1, others to 0
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    img   = img*mask2[:,:,np.newaxis]
    
    img[np.where((img == [0,0,0]).all(axis = 2))] = [255.0, 255.0, 255.0]
    return img


def ext_frgd():    
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
            
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            # Extracting the number from the filename
            
            img_number = int(filename.split("_")[0])
            
            path = os.path.join("../../data/raw/Bridgewater/", filename)
                        
            img_id = str(img_number)
            i = int(img_number)
            
            # Your image loading and processing code here
            img = load(path, 300)        
            org = img.copy()
            img = k_means(img , n_colors= 4)
            
            img_gray = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
            img_gray = cv2.medianBlur(img_gray,7)
            edges    = cv2.Canny(img_gray,100,200)
            
            # Perform morphological operations to remove noise and fill gaps
            kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            top,bottom,left,right = find_box(edges)
            rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
            forground_img= forgrd_ext(org, rec)
            
            ax[i//5][i%5].imshow(forground_img, aspect='auto')
            ax[i//5][i%5].set_title(img_id)
            ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
ext_frgd()
   
# make a directory to store extractions in
#!mkdir image

# Create extracted images 
def image_extract():
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    
    for filename in tqdm(os.listdir("../../data/raw/Bridgewater/")):
    
        if filename.endswith(".jpg"):  # Assuming your images are in jpg format
            
            # Extracting the number from the filename            
            img_number = int(filename.split("_")[0])

            path = os.path.join("../../data/raw/Bridgewater/", filename)
            
            img_id = str(img_number)
            i = int(img_number)
            
            # Step 1: Load the original image
            im1 = load(path, 300)            
            
            # Step 2: Apply K-Means color quantization
            im2 = k_means(im1 , n_colors= 4)
                                    
            # Step 3: Convert to grayscale
            im3 = cv2.cvtColor(np.uint8(im2*255), cv2.COLOR_RGB2GRAY) 
            
            # Step 3: Apply median blur to reduce noise
            im3 = cv2.medianBlur(im3,5)
            
            # Step 3: Apply Canny edge detection
            im3 = cv2.Canny(im3,100,200)
            
            kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            
            # Step 4: Apply morphological closing
            im31 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)
            
            # Step 5: Find bounding box coordinates
            top,bottom,left,right = find_box(im31)
            
            # Step 5: Draw bounding box
            if top!=(0,0) and top!=bottom:
                im4=cv2.rectangle(im1.copy(), (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)
                
                # Step 6: Define bounding box region
                rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
                
                # Step 7: Perform foreground extraction using GrabCut
                im5= forgrd_ext(im1, rec)
                cv2.imwrite('./image/'+ path.split('/')[-1],cv2.cvtColor(im5,cv2.COLOR_BGR2RGB))
                return im5
            else:
                return 

image_extract() 

  