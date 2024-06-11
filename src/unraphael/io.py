from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as imageio
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize

if TYPE_CHECKING:
    import numpy as np


def load_images(
    image_files: Sequence[Any],
    *,
    width: int,
    as_gray: bool = True,
    as_ubyte: bool = False,
) -> dict[str, np.ndarray]:
    """Load images through `imageio.imread` and do some preprocessing."""
    images = {}

    for image_file in image_files:
        im = imageio.imread(image_file)
        name, _ = image_file.name.rsplit('.')

        if as_gray:
            if im.ndim > 2:
                im = rgb2gray(im)

            x, y = im.shape
            k = y / width
            new_shape2 = int(x / k), int(y / k)
            im = resize(im, new_shape2)

        else:
            if im.ndim <= 2:
                im = gray2rgb(im)

            x, y, z = im.shape
            k = y / width
            new_shape3 = int(x / k), int(y / k), z
            im = resize(im, new_shape3)

        if as_ubyte:
            im = img_as_ubyte(im)

        images[name] = im

    return images


def load_images_from_drc(drc: Path, **kwargs) -> dict[str, np.array]:
    """Load all images in directory.

    kwargs are passed to `load_images`.
    """
    fns = list(drc.glob('*'))

    return load_images(fns, **kwargs)
