from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as imageio
from skimage.color import rgb2gray
from skimage.transform import resize

if TYPE_CHECKING:
    import numpy as np


def load_images(
    image_files: Sequence[Any], *, width: int, as_gray: bool = True
) -> dict[str, np.ndarray]:
    """Load images through `imageio.imread` and do some preprocessing."""
    images = {}

    for image_file in image_files:
        im = imageio.imread(image_file)
        name, _ = image_file.name.rsplit('.')

        if as_gray and im.ndim > 2:
            im = rgb2gray(im)

        x, y = im.shape
        k = y / width
        new_shape = int(x / k), int(y / k)
        im = resize(im, new_shape)

        images[name] = im

    return images


def load_images_from_drc(drc: Path, **kwargs) -> dict[str, np.array]:
    """Load all images in directory.

    kwargs are passed to `load_images`.
    """
    fns = list(drc.glob('*'))

    return load_images(fns, **kwargs)
