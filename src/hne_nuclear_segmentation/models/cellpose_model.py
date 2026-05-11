"""Cellpose-SAM wrapper."""
from __future__ import annotations

import numpy as np
from rasterio import features
from shapely.geometry import Polygon, shape


class CellposeSegmenter:
    name = "cellpose"

    def __init__(
        self,
        target_mpp: float = 0.5,
        gpu: bool = True,
        batch_size: int = 8,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
    ):
        import torch
        from cellpose import models

        # Avoid torch's libomp racing other OMP runtimes (numpy/scipy/vips) on
        # macOS, which segfaults inside __kmp_create_worker on first parallel
        # region. Must be set before any torch op constructs the thread pool.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # interop thread count can only be set once per process; ignore.
            pass

        self.target_mpp = target_mpp
        self.batch_size = batch_size
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.model = models.CellposeModel(gpu=gpu)

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Polygon]]:
        masks_list, _flows, _styles = self.model.eval(
            images,
            batch_size=self.batch_size,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
            normalize=True,
        )
        return [_masks_to_polygons(m) for m in masks_list]


def _masks_to_polygons(mask: np.ndarray) -> list[Polygon]:
    if mask is None or mask.max() == 0:
        return []
    mask = mask.astype(np.int32)
    polys: list[Polygon] = []
    for geom, val in features.shapes(mask, mask=mask > 0):
        if val == 0:
            continue
        g = shape(geom)
        if g.geom_type != "Polygon":
            continue
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.geom_type != "Polygon" or g.area <= 0:
            continue
        polys.append(g)
    return polys
