from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np


@dataclass
class ImageType:
    data: np.ndarray
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes):
        return replace(self, **changes)

    def apply(self, func: Callable, **kwargs):
        return self.replace(data=func(self.data, **kwargs))
