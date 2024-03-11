# Goal: 
# Extracts translation, rotation, scale, and shear components from a 3x3 homography matrix.
# The homography matrix represents the rotation, translation, and scale to convert (warp) from
# the plane of our input image to the plane of our template image. 

# Translation:
# The translation component represents the displacement or shift of the image along the x and
# y axes. It is a 2D vector that describes how much the image has been moved horizontally
# (along the x-axis) and vertically (along the y-axis) after the transformation. 
# The translation vector is often denoted as (tx, ty), where tx is the horizontal translation 
# and ty is the vertical translation. This vector indicates the amount by which each point in 
# the image has been shifted to align with its corresponding point in the transformed image.
# For example, if (tx, ty) = (5, 3), it means that each point in the image has been shifted 
# 5 units to the right (positive x-direction) and 3 units upward (positive y-direction).

# Scale:
# The scale component represents the scaling factor applied to the image. It indicates how 
# much the image has been resized or stretched in both the horizontal and vertical directions.

# Scaling factor:
# is often denoted as s, and it is a single value. The scale factor is typically calculated
# by the Euclidean norm of the scaling part. If s is greater than 1, it indicates that the 
# image has been enlarged, and if s is less than 1, it indicates that the image has been 
# reduced in size. The absolute value of s reflects the uniform scaling applied to the entire
# image.

# Shear:
# he shear component represents a form of deformation or skew applied to the image. 
# Shear causes the image to slant or distort along one or more axes. Unlike translation, 
# which moves the entire image, shear affects the relative positions of points within the image.
# A non-zero shear value indicates that the image has been sheared, causing a slanting effect. 
# Positive shear typically slants the image to the right, while negative shear slants it to the left.

# Theta:
# Theta typically represents the rotation angle. Specifically, for a 2D transformation matrix 
# like a homography matrix (3x3) The rotation angle θ can be derived from the elements of 
# the matrix as atan2(b,a) (in radians). This angle represents the rotation applied to the image
# during the transformation. The atan2 function is used to handle the signs of a and b 
# appropriately and return an angle in the correct quadrant. f you choose to express the angle 
# in degrees, you can use the conversion factor 180/π to convert the angle from radians to degrees.

# References:
# https://stats.stackexchange.com/questions/590278/how-to-calculate-the-transalation-and-or-rotation-of-two-images-using-fourier-tr
# https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
# https://answers.opencv.org/question/203890/how-to-find-rotation-angle-from-homography-matrix/
# https://stackoverflow.com/questions/15420693/how-to-get-rotation-translation-shear-from-a-3x3-homography-matrix-in-c-sharp

# libraries ----
import cv2
import numpy as np
import math
import imutils

#  Function to extract rotation, translation, scale, shear from a 3x3 homography matrix
def getComponents(image, template, maxFeatures = 500, keepPercent = 0.2, debug = False):
    
    """
    Extracts translation, rotation, scale, and shear components from a 3x3 Homography matrix.

    Parameters:
        image (numpy.ndarray): Input image.
        template (numpy.ndarray): Template image.
        maxFeatures (int): Maximum number of features for ORB.
        keepPercent (float): Percentage of top matches to keep.
        debug (bool): Optional flag for visualization.

    Returns:
        tuple: A tuple containing translation, rotation, scale, scale factor, and shear components.
    """
        
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the top matches
    # we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images  map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points    
    (H, mask) = cv2.findHomography(ptsA, ptsB, method = cv2.RANSAC)
           
    a = H[0,0]
    b = H[0,1]
    c = H[0,2]
    d = H[1,0]
    e = H[1,1]
    f = H[1,2]

    p = math.sqrt(a*a + b*b)
    r = (a*e - b*d)/(p)
    q = (a*d+b*e)/(a*e - b*d)

    translation_x, translation_y = (c,f)
    
    scale = (p,r)
    scale_factor = math.sqrt(a ** 2 + d ** 2)
    
    shear = q
    theta = math.atan2(b,a) * 180 / math.pi
    
    # print the components 
    print("Translation x:", translation_x) # moved horizontally (along the x-axis), where negative values indicate a shift to the left, and positive values indicate a shift to the right.
    print("Translation y:", translation_y) # moved vertically (along the y-axis) direction, where negative values indicate a shift up, and positive values indicate a shift down.
    
    print("Scale:", scale) # 1 for no scale, 0.5 for half the size, 2 for double the size, etc.
    print("Scale factor:", scale_factor) # 1 for no scale, 0.5 for half the size, 2 for double the size, etc.
    
    print("Shear:", shear) # a scalar. 0 for no shear, 1 for shear
    print("Rotation (in degrees):", theta) # a scalar. 0 for no rotation, 90 for a 90-degree rotation, etc.
    
    return (translation_x, translation_y, scale, scale_factor, shear, theta)

# 0. Load two paintings ----
template = cv2.imread("../../data/interim/no_bg/output_2_Naples_Museo Capodimonte.jpg")
image    = cv2.imread("../../data/interim/no_bg/output_3_Milan_private.jpg")

# apply function
components = getComponents(image, template,debug=True)
print(components)

