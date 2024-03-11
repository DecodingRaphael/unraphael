# https://stackoverflow.com/questions/75357936/how-to-install-detectron2

#import torch
#print("Torch version:",torch.__version__)
#print("Is CUDA enabled?",torch.cuda.is_available())
#print(torch.cuda.is_available())

# Import necessary libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load an image
im = cv2.imread("../../data/interim/no_bg/output_0_Edinburgh_Nat_Gallery.jpg")

# Create a Detectron2 config
cfg = get_cfg()

# Use a built-in model from the model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set the device to CPU
cfg.MODEL.DEVICE = "cpu"

# Set up the predictor
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # get weights
predictor = DefaultPredictor(cfg)

# Make a prediction
outputs = predictor(im)

# Visualize the prediction
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Display the image using Matplotlib
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()


