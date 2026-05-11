"""Tissue detection on a low-resolution thumbnail."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.affinity import scale as shapely_scale
from shapely.geometry import shape
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects
from tqdm import tqdm

from .slide import Slide

log = logging.getLogger("hne.tissue")


def detect_tissue(
    slide: Slide,
    thumbnail_mpp: float = 8.0,
    max_thumbnail_px: int = 1024,
    min_tissue_area_um2: float = 5000.0,
    close_radius_px: int = 5,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of tissue polygons in level-0 pixel coords."""
    longest = max(slide.width, slide.height)
    effective_mpp = max(thumbnail_mpp, slide.mpp * longest / max_thumbnail_px)
    log.info("rendering thumbnail @ %.2f µm/px (slide mpp=%.4f)", effective_mpp, slide.mpp)
    thumb, scale = slide.thumbnail(effective_mpp)
    log.info("thumbnail shape=%s", thumb.shape)
    if thumb.ndim == 2:
        thumb = np.stack([thumb] * 3, axis=-1)

    log.info("computing HSV + Otsu threshold")
    hsv = rgb2hsv(thumb)
    sat = hsv[..., 1]

    try:
        thr = threshold_otsu(sat)
    except ValueError:
        thr = 0.05
    mask = sat > max(thr, 0.05)
    log.info("threshold=%.3f, foreground=%.2f%%", max(thr, 0.05), 100.0 * mask.mean())

    if close_radius_px > 0:
        log.info("morphological closing (r=%d)", close_radius_px)
        mask = binary_closing(mask, footprint=disk(close_radius_px))

    # min area in thumbnail pixels
    px_per_um_thumb = 1.0 / effective_mpp
    min_px_thumb = int(min_tissue_area_um2 * (px_per_um_thumb ** 2))
    if min_px_thumb > 0:
        log.info("removing objects < %d px", min_px_thumb)
        mask = remove_small_objects(mask, min_size=min_px_thumb)

    log.info("polygonizing mask")
    shapes_iter = features.shapes(mask.astype(np.uint8), mask=mask)
    polygons = []
    for geom, val in tqdm(shapes_iter, desc="polygonize", unit="poly", leave=False):
        if val != 1:
            continue
        poly = shape(geom)
        # Scale from thumbnail pixels to level-0 pixels
        poly = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
        polygons.append(poly)
    log.info("found %d tissue polygons", len(polygons))

    gdf = gpd.GeoDataFrame(
        {"tissue_id": list(range(len(polygons)))},
        geometry=polygons,
        crs=None,
    )
    return gdf
