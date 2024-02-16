# https://stackoverflow.com/questions/58803312/how-to-count-number-of-white-and-black-pixels-in-color-picture-in-python-how-to
# https://www.tutorialspoint.com/opencv-python-how-to-convert-a-colored-image-to-a-binary-image

import cv2 
import numpy as np 
from matplotlib import pyplot as plt
  
# reading the image data from desired directory 
img = cv2.imread("../../data/b_w.jpg") 

# Convert the image to black and white (grayscale)
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(bw_img, cmap='gray')
plt.axis('off')
plt.show()

total = bw_img.shape[0] * bw_img.shape[1] # 505890
  
# counting the number of pixels 
number_of_white_pix = np.sum(bw_img == 255) # 79486
number_of_black_pix = np.sum(bw_img == 0) # 415289
  
print('Number of white pixels:', number_of_white_pix) 
print('Number of black pixels:', number_of_black_pix)

number_of_white_pix / total # 16%
number_of_black_pix / total # 82%

from PIL import Image
from IPython.display import display

im = Image.open('../../data/b_w.jpg')

# Convert the image to grayscale
bw_image = im.convert('L')

# Set a threshold (adjust as needed)
threshold = 128

# Convert to binary black and white
bw_image = bw_image.point(lambda x: 0 if x < threshold else 255)

# Display the image
display(bw_image)

