from roboflow import Roboflow
import math
import numpy as np
import imutils
import cv2
import os

rf = Roboflow(api_key="JhIaNszXgcj3f7qgfyGn")
project = rf.workspace().project("people-in-art")
model = project.version(3).model

template = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")

# infer on a local image
print(model.predict("../../data/raw/0_Edinburgh_Nat_Gallery.jpg").json())

# infer on an image hosted elsewhere
print(model.predict("URL_OF_YOUR_IMAGE").json())

# save an image annotated with your predictions
model.predict("../../data/raw/Lamb/0 Raphael_Holy_Family_with_the_Lamb Prado.jpg").save("prediction.jpg")