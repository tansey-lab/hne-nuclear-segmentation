"""Segmenter protocol."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from shapely.geometry import Polygon


class Segmenter(Protocol):
    """Polygon-producing segmenter.

    Coordinates returned are in the *input image's pixel space*, i.e. the
    space of the array passed to predict_batch. Callers project to level-0.
    """

    name: str
    target_mpp: float

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Polygon]]:
        ...
