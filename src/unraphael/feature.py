from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_descriptors
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
    images: dict[str, np.ndarray], *, extractor, progress: Any = None
) -> dict[str, FeatureContainer]:
    """`extractor` must have `detect_and_extract` method."""
    features = {}

    n_tot = len(images)

    for i, (name, im) in enumerate(images.items()):
        extractor.detect_and_extract(im)

        features[name] = FeatureContainer(
            name=name,
            image=im,
            keypoints=extractor.keypoints,
            descriptors=extractor.descriptors,
            scales=extractor.scales,
        )

        if progress:
            progress(i / n_tot, name)

    return features


def get_heatmaps(
    features: dict[str, FeatureContainer], seed: int = 1337, progress: Any = None
) -> tuple[np.array, np.array]:
    n = len(features)

    heatmap = np.zeros((n, n), dtype=int)
    heatmap_inliers = np.zeros((n, n), dtype=int)

    features_tup = tuple(features.values())

    for (i1, i2), _ in np.ndenumerate(heatmap):
        if i1 == i2:
            continue

        ft1 = features_tup[i1]
        ft2 = features_tup[i2]

        matches = match_descriptors(ft1.descriptors, ft2.descriptors, cross_check=True)

        heatmap[i1, i2] = len(matches)

        rng = np.random.default_rng(seed)

        try:
            model, inliers = ransac(
                (ft1.keypoints[matches[:, 0]], ft2.keypoints[matches[:, 1]]),
                FundamentalMatrixTransform,
                min_samples=8,
                residual_threshold=1,
                max_trials=5000,
                rng=rng,
            )

        except ValueError:
            inliers = np.zeros(len(matches), dtype=bool)

        heatmap_inliers[i1, i2] = inliers.sum()

        print(f'{ft1.name}->{ft2.name} matches: {len(matches)}, inliers: {inliers.sum()}')

        if progress:
            progress((i1 * n + i2) / (n * n), f'Matching {ft1.name} -> {ft2.name}')

    return heatmap, heatmap_inliers


def heatmap_to_condensed_distance_matrix(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap / heatmap.max()
    dmat = np.sqrt(1 - heatmap**2)

    # array must be a condensed distance matrix
    tri = np.triu_indices_from(dmat, k=1)
    d = dmat[tri]
    return d
