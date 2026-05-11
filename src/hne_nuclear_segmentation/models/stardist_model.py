"""Stardist 2D_versatile_he wrapper."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


class StardistSegmenter:
    name = "stardist"

    def __init__(self, target_mpp: float = 0.5, model_name: str = "2D_versatile_he"):
        from csbdeep.utils import normalize  # noqa: F401  (used in predict)
        from stardist.models import StarDist2D

        self.target_mpp = target_mpp
        self.model = StarDist2D.from_pretrained(model_name)

    def predict_batch(self, images: list[np.ndarray]) -> list[list[Polygon]]:
        from csbdeep.utils import normalize

        results: list[list[Polygon]] = []
        for img in images:
            if img.dtype != np.float32:
                img_f = img.astype(np.float32)
            else:
                img_f = img
            img_n = normalize(img_f, 1.0, 99.8, axis=(0, 1))
            _labels, details = self.model.predict_instances(img_n)
            coords = details.get("coord")
            polys: list[Polygon] = []
            if coords is not None and len(coords) > 0:
                # coords shape: (n_objects, 2, n_rays); axis 0 is y, axis 1 is x
                for obj in coords:
                    ys, xs = obj[0], obj[1]
                    poly = Polygon(zip(xs.tolist(), ys.tolist()))
                    if poly.is_valid and poly.area > 0:
                        polys.append(poly)
                    else:
                        fixed = poly.buffer(0)
                        if not fixed.is_empty and fixed.geom_type == "Polygon":
                            polys.append(fixed)
            results.append(polys)
        return results
