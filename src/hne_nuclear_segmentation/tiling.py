"""Tile generation within tissue polygons."""
from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

from .slide import Slide

log = logging.getLogger("hne.tiling")


def generate_tiles(
    slide: Slide,
    tissue: gpd.GeoDataFrame,
    target_mpp: float,
    tile_size_model_px: int = 1024,
    overlap_fraction: float = 0.5,
    edge_fraction: float = 0.1,
) -> gpd.GeoDataFrame:
    """Generate tiles intersecting tissue.

    tile_size_model_px is the tile size *at the model's target MPP*.
    Tile bounds are returned in level-0 pixel coordinates.

    overlap_fraction controls stride: step = tile_size * (1 - overlap_fraction).
    edge_fraction controls the inner keep box: each side is inset by
    tile_size * edge_fraction, and only nuclei whose centroid lies inside that
    box are reported. With overlap_fraction > 2*edge_fraction adjacent keep
    boxes overlap, so tissue near tile borders is always covered by some
    tile's keep box — at the cost of duplicate detections, which downstream
    dedupe-by-union resolves.
    """
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0, 1)")
    if not (0.0 <= edge_fraction < 0.5):
        raise ValueError("edge_fraction must be in [0, 0.5)")

    scale = target_mpp / slide.mpp  # base px per model px
    tile_size_base = int(round(tile_size_model_px * scale))
    step_base = max(1, int(round(tile_size_base * (1.0 - overlap_fraction))))
    keep_margin_base = int(round(tile_size_base * edge_fraction))

    log.info(
        "tiling slide %dx%d: tile_base=%d step=%d keep_margin=%d (target_mpp=%.3f, slide_mpp=%.4f)",
        slide.width, slide.height, tile_size_base, step_base, keep_margin_base,
        target_mpp, slide.mpp,
    )
    log.info("building tissue union from %d polygons", len(tissue))
    tissue_union = unary_union(list(tissue.geometry)) if len(tissue) else None

    ys = list(range(0, slide.height, step_base))
    xs = list(range(0, slide.width, step_base))
    log.info("scanning %d x %d candidate tile grid", len(xs), len(ys))

    rows = []
    tid = 0
    for y0 in tqdm(ys, desc="tile rows", unit="row", leave=False):
        for x0 in xs:
            x1 = min(x0 + tile_size_base, slide.width)
            y1 = min(y0 + tile_size_base, slide.height)
            if x1 - x0 < tile_size_base // 2 or y1 - y0 < tile_size_base // 2:
                continue
            tile_geom = box(x0, y0, x1, y1)
            if tissue_union is not None and not tile_geom.intersects(tissue_union):
                continue
            keep = (
                x0 + keep_margin_base,
                y0 + keep_margin_base,
                x1 - keep_margin_base,
                y1 - keep_margin_base,
            )
            rows.append(
                {
                    "tile_id": tid,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "keep_x0": keep[0],
                    "keep_y0": keep[1],
                    "keep_x1": keep[2],
                    "keep_y1": keep[3],
                    "geometry": tile_geom,
                }
            )
            tid += 1

    log.info("generated %d tiles intersecting tissue", len(rows))
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    gdf.attrs["target_mpp"] = target_mpp
    gdf.attrs["tile_size_model_px"] = tile_size_model_px
    gdf.attrs["scale"] = scale
    return gdf
