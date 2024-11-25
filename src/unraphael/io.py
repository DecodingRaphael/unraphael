from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as imageio
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize

if TYPE_CHECKING:
    import numpy as np


def resize_to_width(image: np.ndarray, *, width: int) -> np.ndarray:
    """Resize image to given width."""
    new_shape: tuple[int, ...]

    if len(image.shape) == 2:
        x, y = image.shape
        k = y / width
        new_shape = int(x / k), int(y / k)
    else:
        x, y, z = image.shape
        k = y / width
        new_shape = int(x / k), int(y / k), z

    return resize(image, new_shape)


def load_images(
    image_files: Sequence[Any],
    *,
    width: None | int = None,
    as_gray: bool = True,
) -> dict[str, np.ndarray]:
    """Load images through `imageio.imread` and do some preprocessing."""
    images = {}

    for image_file in image_files:
        im = imageio.imread(image_file)
        name, _ = image_file.name.rsplit('.')

        if as_gray:
            if im.ndim > 2:
                im = rgb2gray(im)
        else:
            if im.ndim <= 2:
                im = gray2rgb(im)

        if width:
            im = resize_to_width(im, width=width)

        images[name] = im

    return images


def load_images_from_drc(drc: Path, **kwargs) -> dict[str, np.array]:
    """Load all images in directory.

    kwargs are passed to `load_images`.
    """
    fns = list(drc.glob('*'))

    return load_images(fns, **kwargs)
