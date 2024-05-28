from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import numpy as np
import cv2
from widgets import image_downloads_widget, load_image_widget
from styling import set_custom_css
import platformdirs
from dataclasses import dataclass


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


def detection_task(*, results):
    objects = {}

    for result in results:
        img = result.orig_img

        for ci, c in enumerate(result):
            label = c.names[c.boxes.cls.tolist().pop()]
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(int)
            # Crop the object from the image
            isolated = img[y1:y2, x1:x2]

            objects[f'{label}_{ci}'] = isolated

    return objects


def segmentation_task(*, results):
    objects = {}

    for result in results:
        img = result.orig_img

        for ci, c in enumerate(result):
            label = c.names[c.boxes.cls.tolist().pop()]
            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = c.masks.xy.pop().astype(int).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            isolated = np.dstack([img, b_mask])

            objects[f'{label}_{ci}'] = isolated

    return objects


@dataclass
class BodyDrawer:
    keypoints: np.ndarray
    HEAD: tuple[int, int, int] = (0, 255, 0)  # green
    BODY: tuple[int, int, int] = (0, 0, 255)  # blue
    LEGS: tuple[int, int, int] = (255, 165, 0)  # orange
    REST: tuple[int, int, int] = (128, 0, 128)  # purple

    def draw(self, *, image):
        self.draw_points_head(image=image)
        self.draw_points_legs(image=image)
        self.draw_points_body(image=image)
        self.draw_skeleton_legs(image=image)
        self.draw_skeleton_body(image=image)
        self.draw_skeleton_head(image=image)
        self.draw_additional(image=image)

    def _draw_points(self, *, image, indices, color):
        for j, kp in enumerate(self.keypoints):
            if j in indices:
                if kp[2] > 0.5:
                    x, y = (
                        int(kp[0]),
                        int(kp[1]),
                    )
                    cv2.circle(image, (x, y), 5, color, -1)

    def draw_points_head(self, *, image):
        # indices for eyes, ears, and nose
        head_indices = [0, 1, 2, 3, 4]
        self._draw_points(image=image, indices=head_indices, color=self.HEAD)

    def draw_points_body(self, *, image):
        # indices for shoulders, elbows, and wrists
        body_indices = [5, 6, 7, 8, 9, 10]
        self._draw_points(image=image, indices=body_indices, color=self.BODY)

    def draw_points_legs(self, *, image):
        # indices for hips, knees, and ankles
        leg_indices = [11, 13, 15, 12, 14, 16]
        self._draw_points(image=image, indices=leg_indices, color=self.LEGS)

    def _draw_skeleton(self, *, image, indices, color):
        for j1, j2 in indices:
            kp1 = self.keypoints[j1]
            kp2 = self.keypoints[j2]

            # check confidence of both keypoints
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                p1 = (int(kp1[0]), int(kp1[1]))
                p2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(image, p1, p2, color, 2)

    def draw_skeleton_legs(self, *, image):
        # left hip to left knee, left knee to left ankle
        # right hip to right knee, right knee to right ankle
        indices = [(11, 13), (13, 15), (12, 14), (14, 16)]
        self._draw_skeleton(image=image, indices=indices, color=self.LEGS)

    def draw_skeleton_body(self, *, image):
        # shoulder to shoulder, shoulder to elbow, elbow to wrist
        indices = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
        self._draw_skeleton(image=image, indices=indices, color=self.BODY)

    def draw_skeleton_head(self, *, image):
        # nose to eyes, eyes to ears
        indices = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 2), (2, 4)]
        self._draw_skeleton(image=image, indices=indices, color=self.HEAD)

    def draw_additional(self, *, image):
        # left ear to left shoulder, right ear to right shoulder
        self._draw_skeleton(image=image, indices=[(3, 5)], color=self.HEAD)
        # right ear to right shoulder, left ear to left shoulder
        self._draw_skeleton(image=image, indices=[(4, 6)], color=self.HEAD)
        # left shoulder to left hip
        self._draw_skeleton(image=image, indices=[(5, 11)], color=self.REST)
        # right shoulder to right hip
        self._draw_skeleton(image=image, indices=[(6, 12)], color=self.REST)
        # left shoulder to right shoulder
        self._draw_skeleton(image=image, indices=[(5, 6)], color=self.REST)
        # left hip to right hip
        self._draw_skeleton(image=image, indices=[(11, 12)], color=self.REST)


def pose_task(*, results):
    objects = {}

    for idx, result in enumerate(results):
        img = result.orig_img
        frame_height, frame_width = img.shape[:2]

        for ci, keypoints in enumerate(result.keypoints.data):
            black_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            keypoints = keypoints.cpu().numpy()

            BodyDrawer(keypoints).draw(image=black_image)

            # Save the black image with keypoints to dictionary
            objects[f'pose_{idx}_{ci}'] = black_image

    return objects


def main():
    set_custom_css()

    st.title('Segmentation of figures in a painting')

    model_type = st.sidebar.radio('Select Task', ['Detection', 'Segmentation', 'Pose'])
    add_box = st.sidebar.checkbox('Add bounding box', value=True)
    confidence = float(st.sidebar.slider('Select Model Confidence', 10, 100, 25)) / 100

    model = load_model(model_type)

    name, source_image = load_image_widget()

    col1, col2 = st.columns(2)

    with col1:
        st.image(source_image, caption='Uploaded Image', use_column_width=True)

    if not st.sidebar.button('Detect Objects', key='detect_button'):
        st.stop()

    results = model.predict(
        source_image,
        conf=confidence,
        show_boxes=True,
        save=True,
        save_txt=True,
    )

    if len(results[0].boxes) == 0:
        st.error('No objects detected in the image.')

    result_image = results[0].plot(boxes=add_box)
    with col2:
        st.image(result_image, caption='Detected Image', use_column_width=True)

    if model_type == 'Detection':
        images = detection_task(results=results)
    elif model_type == 'Segmentation':
        images = segmentation_task(results=results)
    elif model_type == 'Pose':
        images = pose_task(results=results)

    image_downloads_widget(images=images)


if __name__ == '__main__':
    main()
