from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import draw


@dataclass
class BodyDrawer:
    keypoints: np.ndarray
    HEAD: tuple[int, int, int] = (0, 255, 0)  # lime
    BODY: tuple[int, int, int] = (255, 140, 0)  # orange
    LEGS: tuple[int, int, int] = (0, 255, 255)  # cyan
    REST: tuple[int, int, int] = (255, 0, 255)  # magenta

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
                    rr, cc = draw.disk((kp[1], kp[0]), radius=5, shape=image.shape)
                    image[rr, cc] = color

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
                rr, cc = draw.line(int(kp1[1]), int(kp1[0]), int(kp2[1]), int(kp2[0]))
                image[rr, cc] = color

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
