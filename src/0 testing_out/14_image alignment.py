# Image alignement and registration

# Here, we align all images in a directory to the template image using ORB feature matching and
# homography.
# The aligned images are saved in a new directory called "aligned".
# The template image is the painting "Edinburgh National Gallery".
# The input images are all copies

# see also https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

# libraries ----
import numpy as np
import imutils
import cv2
import os

def align_images(image, template, maxFeatures = 500, keepPercent = 0.2,	debug = False):
    
	# convert both the input image and template to grayscale
	imageGray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
    # use ORB to detect keypoints and extract (binary) local invariant features
	orb            = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	
    # match the features
	method  = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)
 
    # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	
    # keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	
    # check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)
  
    # allocate memory for the keypoints (x, y)-coordinates from the top matches -- we'll use these
    # coordinates to compute our homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	
    # loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images  map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
  
    # compute the homography matrix between the two sets of matched points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	
    # use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	
    # return the aligned image
	return aligned

# load the input image and template from disk
print("[INFO] loading images...")

def align_images_in_directory(template_path, input_directory, output_directory, maxFeatures=500, keepPercent=0.2, debug=False):
    
    # load the template image: to this painting we want to align all other paintings
    template = cv2.imread(template_path)

    # create output directory if not exists
    os.makedirs(output_directory, exist_ok=True)

    # loop over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            
            # load the input image: this painting we want to align to the template
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # align the images by applying the function we defined above
            aligned = align_images(image, template, maxFeatures, keepPercent, debug)

            # save the aligned image in the output directory
            output_path = os.path.join(output_directory, f"aligned_{filename}")
            cv2.imwrite(output_path, aligned)
            print(f"[INFO] Image {filename} aligned and saved to {output_path}")

# set the paths and parameters
template_path = "../../data/raw/0_Edinburgh_Nat_Gallery.jpg"
input_directory = "../../data/raw/Bridgewater"
output_directory = "../../data/interim/aligned"

# align images in the directory and save them
align_images_in_directory(template_path, input_directory, output_directory, debug=False)
