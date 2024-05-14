from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skimage.io import imread
from skimage.transform import resize

if TYPE_CHECKING:
    import numpy as np


def load_images_from_drc(
    drc: Path, *, width: int, as_gray: bool = True
) -> dict[str, np.array]:
    fns = list(drc.glob('*'))

    images = {}

    for fn in fns:
        im = imread(fn, as_gray=as_gray)
        name = fn.stem

        x, y = im.shape
        k = y / width
        new_shape = int(x / k), int(y / k)
        im = resize(im, new_shape)

        images[name] = im

    return images
