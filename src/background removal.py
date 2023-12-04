# Background Removal

import numpy as np
import pandas as pd
import cv2
import os
import random
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

paths=[]
t=0

for dirname, _, filenames in os.walk('../data/raw/Bridgewater'):
    if t<30:
        for filename in filenames:
            paths+=[(os.path.join(dirname, filename))]
            t+=1
            
def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

train_images = pd.Series(random.sample(paths,11)).progress_apply(load_image)

def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask

plt.imshow(init_grabcut_mask(3*136, 3*205))

def add_contours(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) != 0:
        cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0) ,2)
        
def remove_background(image):
    h, w = image.shape[:2]
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = cv2.bitwise_and(image, image, mask = mask_binary)
    add_contours(result, mask_binary) # optional, adds visualizations
    return result

rows, cols = (len(train_images), 2)

axes_pad = 0.2
fig_h = 4.0 * rows + axes_pad * (rows-1) 
fig_w = 4.0 * cols + axes_pad * (cols-1) 
fig = plt.figure(figsize=(fig_w, fig_h))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.2)   
        
for i, ax in enumerate(grid):
    im = cv2.resize(train_images.iloc[i // 2], (3*205, 3*136))
    if i % 2 == 1:
        im = remove_background(im)
    ax.imshow(im)    
    
for img in train_images:
    img=cv2.resize(img, dsize=(3*205, 3*136))
    #img=cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
    H=img.shape[0]
    W=img.shape[1]
    Ps=[[0,0],[10,0],[0,10], 
        [H-1,0],[H-11,0],[H-1,10], 
        [0,W-1],[10,W-1],[0,W-11], 
        [H-1,W-1],[H-11,W-1],[H-1,W-11]]
    
    for P in Ps:
        p0=P[0]
        p1=P[1]
        a0=img[p0,p1,0]*0.7
        a1=img[p0,p1,0]*1.3
        b0=img[p0,p1,1]*0.7
        b1=img[p0,p1,1]*1.3
        c0=img[p0,p1,2]*0.7
        c1=img[p0,p1,2]*1.3

        for h in range(H):
            for w in range(W):
                if a0<img[h,w,0]<a1 and b0<img[h,w,1]<b1 and c0<img[h,w,2]<c1:
                    img[h,w,:]=np.array([255,255,255])

    plt.imshow(img)
    for yx in Ps:
        y=yx[0]
        x=yx[1]
        plt.plot(x,y,'ro',markersize=1)
    plt.axis('off')
    plt.show()