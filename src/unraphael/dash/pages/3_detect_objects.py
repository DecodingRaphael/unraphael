from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import platformdirs
import streamlit as st
from styling import set_custom_css
from ultralytics import YOLO
from widgets import image_downloads_widget, load_image_widget

from unraphael.pose import BodyDrawer

CACHEDIR = Path(platformdirs.user_cache_dir('unraphael'))

MODELS = {
    'Detection': CACHEDIR / 'yolov8n.pt',
    'Segmentation': CACHEDIR / 'yolov8x-seg.pt',
    'Pose': CACHEDIR / 'yolov8x-pose.pt',
}


@st.cache_resource
def load_model(model_type: str):
    model_path = MODELS[model_type]
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(f'Unable to load model. Check the specified path: {model_path}')
        st.error(ex)
    return model


def _detection_task(*, results):
    objects = {}

    for result in results:
        for ci, c in enumerate(result):
            image = result.orig_img

            label = c.names[c.boxes.cls.tolist().pop()]
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(int)
            # Crop the object from the image
            new_image = image[y1:y2, x1:x2]

            objects[f'{label}_{ci}'] = new_image

    return objects


def _segmentation_task(*, results):
    objects = {}

    for result in results:
        for ci, c in enumerate(result):
            image = result.orig_img

            label = c.names[c.boxes.cls.tolist().pop()]
            b_mask = np.zeros(image.shape[:2], np.uint8)
            contour = c.masks.xy.pop().astype(int).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            new_image = np.dstack([image, b_mask])

            objects[f'{label}_{ci}'] = new_image

    return objects


def _pose_task(*, results):
    objects = {}

    for idx, result in enumerate(results):
        for ci, keypoints in enumerate(result.keypoints.data):
            img = result.orig_img

            frame_height, frame_width = img.shape[:2]

            new_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            keypoints = keypoints.cpu().numpy()
            label = f'pose_{idx}'

            BodyDrawer(keypoints).draw(image=new_image)

            objects[f'{label}_{ci}'] = new_image

    return objects


def yolo_task_widget(image: np.ndarray, /) -> dict[str, np.ndarray]:
    """This widget takes an image and provides the user with some choices of which tasks
    to perform: Detection, Segmentation, Pose analysis."""
    col1, col2, col3 = st.columns([0.30, 0.35, 0.35])

    task_name = col1.radio('Select Task', [None, 'Detection', 'Segmentation', 'Pose'])
    add_box = col1.checkbox('Add bounding box', value=True)
    confidence = float(col1.slider('Select Model Confidence', 10, 100, 25)) / 100

    col2.image(image, caption='Uploaded Image', use_column_width=True)

    if not task_name:
        st.info('Select a task to continue')
        st.stop()

    model = load_model(task_name)

    results = model.predict(
        image,
        conf=confidence,
        show_boxes=True,
        save=True,
        save_txt=True,
    )

    if len(results[0].boxes) == 0:
        st.error('No objects detected in the image.')

    result_image = results[0].plot(boxes=add_box)

    col3.image(result_image, caption='Detected Image', use_column_width=True)

    if task_name == 'Detection':
        images = _detection_task(results=results)
    elif task_name == 'Segmentation':
        images = _segmentation_task(results=results)
    elif task_name == 'Pose':
        images = _pose_task(results=results)

    return images


def main():
    set_custom_css()

    st.title('Segmentation of figures in a painting')

    name, image = load_image_widget()

    images = yolo_task_widget(image)

    image_downloads_widget(images=images)


if __name__ == '__main__':
    main()
