import cv2
import numpy as np
from IPython.display import display, HTML
from matplotlib import pyplot as plt, animation

# 0. Load two paintings
painting1 = cv2.imread("../data/raw/0_Edinburgh_Nat_Gallery.jpg", cv2.IMREAD_GRAYSCALE)
painting2 = cv2.imread('../data/raw/Bridgewater/3_Milan_private.jpg', cv2.IMREAD_GRAYSCALE)

# Resize images to a common size
common_size = (min(painting1.shape[1], painting2.shape[1]), min(painting1.shape[0], painting2.shape[0]))
painting1_resized = cv2.resize(painting1, common_size)
painting2_resized = cv2.resize(painting2, common_size)

# Use ORB for feature detection and matching
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(painting1_resized, None)
keypoints2, descriptors2 = orb.detectAndCompute(painting2_resized, None)

# Use a matcher (e.g., BFMatcher) to find matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to obtain good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate homography
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Define the number of frames for the animation
num_frames = 100

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize the image with the first frame
blended_image = cv2.addWeighted(painting1_resized, 1, painting2_resized, 0, 0)
im = ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB), extent=[0, blended_image.shape[1], 0, blended_image.shape[0]])

# Update function for animation
def update(frame):
    alpha = frame / num_frames
    
# Interpolate the homography matrix
    interpolated_H = (1 - alpha) * H + alpha * np.eye(3)
    
    # Apply perspective warp using interpolated homography matrix
    blended_image = cv2.warpPerspective(painting1_resized, interpolated_H, (painting1.shape[1], painting1.shape[0]))
    blended_image = cv2.addWeighted(blended_image, 1 - alpha, painting2_resized, alpha, 0)
    
    im.set_array(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Frame {frame + 1}/{num_frames}')

# Create animation
ani = animation.FuncAnimation(fig, update, 
                              frames=num_frames, 
                              interval=25,
                              repeat=True, 
                              repeat_delay=1000)

# Display the animation
display(HTML(ani.to_jshtml()))