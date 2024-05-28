from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import numpy as np
import cv2
from widgets import image_downloads_widget, load_image_widget
from styling import set_custom_css
import platformdirs

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


def pose_task(*, results):
    objects = {}

    for idx, result in enumerate(results):
        img = result.orig_img
        frame_height, frame_width = img.shape[:2]

        for ci, keypoints in enumerate(result.keypoints.data):
            black_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            keypoints = keypoints.cpu().numpy()

            # Draw keypoints for legs (orange)
            for j in range(len(keypoints)):
                if j in [11, 13, 15, 12, 14, 16]:
                    # indices for hips, knees, and ankles
                    if keypoints[j][2] > 0.5:
                        x, y = (
                            int(keypoints[j][0]),
                            int(keypoints[j][1]),
                        )
                        cv2.circle(black_image, (x, y), 5, (255, 165, 0), -1)

            # Draw keypoints for body (blue)
            for j in range(len(keypoints)):
                if j in [5, 6, 7, 8, 9, 10]:
                    # indices for shoulders, elbows, and wrists
                    if keypoints[j][2] > 0.5:
                        x, y = (
                            int(keypoints[j][0]),
                            int(keypoints[j][1]),
                        )
                        cv2.circle(black_image, (x, y), 5, (0, 0, 255), -1)

            # Draw keypoints for head (green)
            for j in range(len(keypoints)):
                if j in [0, 1, 2, 3, 4]:
                    # indices for eyes, ears, and nose
                    if keypoints[j][2] > 0.5:
                        x, y = (
                            int(keypoints[j][0]),
                            int(keypoints[j][1]),
                        )
                        cv2.circle(black_image, (x, y), 5, (0, 255, 0), -1)

            # Define the skeleton (pairs of keypoints to connect) according to COCO dataset order
            # Left Hip to Left Knee, Left Knee to Left Ankle
            # Right Hip to Right Knee, Right Knee to Right Ankle
            skeleton_legs = [(11, 13), (13, 15), (12, 14), (14, 16)]
            # Shoulder to Shoulder, Shoulder to Elbow, Elbow to Wrist
            skeleton_body = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
            # Nose to Eyes, Eyes to Ears
            skeleton_head = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 2), (2, 4)]

            # Define additional skeleton connections
            additional_connections = [
                (3, 5),
                (4, 6),
                (5, 6),  # left shoulder to right shoulder
                (11, 12),  # left hip to right hip
                (5, 11),  # Left Shoulder to Left Hip
                (6, 12),  # Right Shoulder to Right Hip
            ]

            # Draw skeleton for legs (orange)
            for j1, j2 in skeleton_legs:
                # check confidence of both keypoints
                if keypoints[j1][2] > 0.5 and keypoints[j2][2] > 0.5:
                    p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                    p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
                    # orange
                    cv2.line(black_image, p1, p2, (255, 165, 0), 2)

            # Draw skeleton for body
            for j1, j2 in skeleton_body:
                # check confidence of both keypoints
                if keypoints[j1][2] > 0.5 and keypoints[j2][2] > 0.5:
                    p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                    p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
                    # blue
                    cv2.line(black_image, p1, p2, (0, 0, 255), 2)

            # Draw skeleton for head (green)
            for j1, j2 in skeleton_head:
                # check confidence of both keypoints
                if keypoints[j1][2] > 0.5 and keypoints[j2][2] > 0.5:
                    p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                    p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
                    # green
                    cv2.line(black_image, p1, p2, (0, 255, 0), 2)

            # Draw additional skeleton connections
            for j1, j2 in additional_connections:
                # check confidence of both keypoints
                if keypoints[j1][2] > 0.5 and keypoints[j2][2] > 0.5:
                    p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                    p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))

                    if j1 in [3, 5] and j2 in [3, 5]:
                        # left ear to left shoulder, right ear to right shoulder
                        # green
                        cv2.line(black_image, p1, p2, (0, 255, 0), 2)
                    elif j1 in [4, 6] and j2 in [4, 6]:
                        # right ear to right shoulder, left ear to left shoulder
                        # green
                        cv2.line(black_image, p1, p2, (0, 255, 0), 2)
                    elif j1 in [0, 5, 11] and j2 in [0, 5, 11]:
                        # shoulders and hips
                        cv2.line(black_image, p1, p2, (128, 0, 128), 2)
                    elif j1 in [0, 6, 12] and j2 in [0, 6, 12]:
                        # shoulders and hips
                        cv2.line(black_image, p1, p2, (128, 0, 128), 2)
                    elif j1 in [5, 6] and j2 in [5, 6]:
                        # left and right shoulders
                        cv2.line(black_image, p1, p2, (0, 0, 255), 2)
                    elif j1 in [11, 12] and j2 in [11, 12]:
                        # left and right hips
                        cv2.line(black_image, p1, p2, (128, 0, 128), 2)

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
