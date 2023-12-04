import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt

class FLANNExplorer:
    def __init__(self, root, img1_path, img2_path):
        self.root = root
        self.root.title("FLANN Parameter Explorer")

        self.img1 = cv2.imread(img1_path, 0)
        self.img2 = cv2.imread(img2_path, 0)

        self.init_flann_params()

        self.create_widgets()

    def init_flann_params(self):
        self.index_params = dict(algorithm=cv2.FlannBasedMatcher_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)

    def create_widgets(self):
        ttk.Label(self.root, text="FLANN Parameters Explorer", font=("Helvetica", 16)).pack(pady=10)

        # FLANN Parameters Sliders
        ttk.Label(self.root, text="Trees:").pack()
        self.trees_var = tk.IntVar(value=self.index_params['trees'])
        trees_slider = ttk.Scale(self.root, from_=1, to=50, variable=self.trees_var, orient=tk.HORIZONTAL)
        trees_slider.pack()

        ttk.Label(self.root, text="Checks:").pack()
        self.checks_var = tk.IntVar(value=self.search_params['checks'])
        checks_slider = ttk.Scale(self.root, from_=1, to=100, variable=self.checks_var, orient=tk.HORIZONTAL)
        checks_slider.pack()

        # Button to Apply Parameters
        ttk.Button(self.root, text="Apply Parameters", command=self.apply_parameters).pack(pady=10)

        # Display Images
        self.display_images()

    def apply_parameters(self):
        # Update FLANN parameters
        self.index_params['trees'] = self.trees_var.get()
        self.search_params['checks'] = self.checks_var.get()

        # Perform FLANN matching
        flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Draw matches and display the result
        img_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)

    def display_images(self):
        # Display the images
        cv2.imshow('Image 1', self.img1)
        cv2.imshow('Image 2', self.img2)

# Load the images and compute SIFT features
img1 = cv2.imread('../data/raw/0_Edinburgh_Nat_Gallery.jpg', 0)
img2 = cv2.imread('../data/raw/Bridgewater/8_London_OrderStJohn.jpg', 0)

# plot the original
plt.imshow(img1) 
plt.imshow(img2)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create the main window
root = tk.Tk()
app = FLANNExplorer(root, 'image1.jpg', 'image2.jpg')

# Run the GUI
root.mainloop()
