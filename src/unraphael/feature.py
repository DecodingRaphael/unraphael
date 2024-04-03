from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import ORB, SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


@dataclass
class FeatureContainer:
    name: str
    image: np.array
    descriptors: np.array | None = None
    keypoints: np.array | None = None
    scales: np.array | None = None

    def plot_keypoints(self):
        fig, ax = plt.subplots()

        ax.imshow(self.image, cmap=plt.cm.gray)
        ax.scatter(
            self.keypoints[:, 1],
            self.keypoints[:, 0],
            2**self.scales,
            facecolors='none',
            edgecolors='r',
        )
        ax.set_title(self.name)
        return fig


def detect_and_extract(
    images: dict[str, np.ndarray], *, method: str, **kwargs
) -> dict[str, FeatureContainer]:
    """`extractor` must have `detect_and_extract` method."""
    features = {}

    if method == 'sift':
        extractor = SIFT(**kwargs)
    elif method == 'orb':
        extractor = ORB(**kwargs)
    else:
        raise ValueError(method)

    for i, (name, im) in enumerate(images.items()):
        extractor.detect_and_extract(im)

        features[name] = FeatureContainer(
            name=name,
            image=im,
            keypoints=extractor.keypoints,
            descriptors=extractor.descriptors,
            scales=extractor.scales,
        )

    return features


def get_heatmaps(
    features: dict[str, FeatureContainer],
    progress: Any = None,
    min_samples=8,
    residual_threshold=1,
    max_trials=5000,
    **kwargs,
) -> dict[str, np.array]:
    n = len(features)

    heatmap = np.zeros((n, n), dtype=int)
    heatmap_inliers = np.zeros((n, n), dtype=int)

    features_tup = tuple(features.values())

    for (i1, i2), _ in np.ndenumerate(heatmap):
        if i1 >= i2:
            continue

        ft1 = features_tup[i1]
        ft2 = features_tup[i2]

        matches = match_descriptors(ft1.descriptors, ft2.descriptors, cross_check=True)

        heatmap[i1, i2] = heatmap[i2, i1] = len(matches)

        try:
            model, inliers = ransac(
                (ft1.keypoints[matches[:, 0]], ft2.keypoints[matches[:, 1]]),
                FundamentalMatrixTransform,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
                **kwargs,
            )

        except ValueError:
            inliers = np.zeros(len(matches), dtype=bool)

        heatmap_inliers[i1, i2] = heatmap_inliers[i2, i1] = inliers.sum()

    return {'all': heatmap, 'inliers': heatmap_inliers}


def heatmap_to_condensed_distance_matrix(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap / heatmap.max()
    dmat = np.sqrt(1 - heatmap**2)

    # array must be a condensed distance matrix
    tri = np.triu_indices_from(dmat, k=1)
    d = dmat[tri]
    return d
