from pathlib import Path
import sys

# Get the absolute path of the current file (only works in .py files) - path to this file ./settings.py
file_path = Path(__file__).resolve()

# Get the parent directory of the current file (main file: /yolov8-streamlit)
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there : allows for things like helper.process_license_plate()
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the main folder (basically IMAGES_DIR = ../yolov8-streamlit/'images')
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / '../../data/images'
DEFAULT_IMAGE = IMAGES_DIR / '1_london_nat_gallery.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8x-seg.pt'
POSE_MODEL = MODEL_DIR / 'YOLOv8x-pose.pt'
