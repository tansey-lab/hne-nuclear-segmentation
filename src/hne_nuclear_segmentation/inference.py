"""Batched inference over tiles."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.wkb import dumps as wkb_dumps
from tqdm import tqdm

from .models.base import Segmenter
from .slide import Slide

log = logging.getLogger("hne.inference")


def _project_polygon(poly: Polygon, scale: float, dx: float, dy: float) -> Polygon:
    # affine: [a, b, d, e, xoff, yoff] => x' = a*x + b*y + xoff, y' = d*x + e*y + yoff
    return affine_transform(poly, [scale, 0.0, 0.0, scale, dx, dy])


def _geo_metadata(model_name: str) -> dict:
    return {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon", "MultiPolygon"],
            }
        },
        "creator": {"library": "hne-nuclear-segmentation", "model": model_name},
    }


def _arrow_schema(model_name: str) -> pa.Schema:
    schema = pa.schema(
        [
            ("nucleus_id", pa.int64()),
            ("tile_id", pa.int64()),
            ("model_name", pa.string()),
            ("area_px", pa.float64()),
            ("geometry", pa.binary()),
        ]
    )
    return schema.with_metadata({b"geo": json.dumps(_geo_metadata(model_name)).encode()})


def run_inference(
    slide: Slide,
    tiles: gpd.GeoDataFrame,
    segmenter: Segmenter,
    batch_size: int = 8,
    progress: bool = True,
    out_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Run segmenter over tiles, return polygons in level-0 coords.

    When ``out_path`` is provided, results are streamed to a GeoParquet file
    incrementally (one row group per inference batch); only one batch worth of
    polygons is held in memory at a time. The returned GeoDataFrame is read
    back from disk. When ``out_path`` is None, behavior is the legacy
    accumulate-in-memory path.

    Note: ``slide.read_region_at_mpp`` uses pyvips ``access="random"`` (or the
    zarr backend's level slicing) so only the requested tile is materialized
    per call — the full slide is never loaded.
    """
    tile_records = tiles.to_dict("records")
    n_batches = (len(tile_records) + batch_size - 1) // batch_size
    log.info(
        "running %s over %d tiles (%d batches, batch_size=%d, target_mpp=%.3f)%s",
        segmenter.name, len(tile_records), n_batches, batch_size, segmenter.target_mpp,
        f" → streaming to {out_path}" if out_path else "",
    )

    iterator: Iterable = _batched(tile_records, batch_size)
    if progress:
        iterator = tqdm(iterator, total=n_batches, desc=f"infer/{segmenter.name}", unit="batch")

    writer: Optional[pq.ParquetWriter] = None
    in_mem_rows: list[dict] = []
    nucleus_id = 0
    total_kept = 0

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        schema = _arrow_schema(segmenter.name)
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    try:
        for batch in iterator:
            arrays: list[np.ndarray] = []
            scales: list[float] = []
            for t in batch:
                w = int(t["x1"] - t["x0"])
                h = int(t["y1"] - t["y0"])
                arr, scale = slide.read_region_at_mpp(
                    int(t["x0"]), int(t["y0"]), w, h, segmenter.target_mpp
                )
                arrays.append(arr)
                scales.append(scale)

            batch_polys = segmenter.predict_batch(arrays)

            batch_nuclei: list[dict] = []
            for t, polys, scale in zip(batch, batch_polys, scales):
                keep = box(t["keep_x0"], t["keep_y0"], t["keep_x1"], t["keep_y1"])
                for p in polys:
                    p_base = _project_polygon(p, scale, float(t["x0"]), float(t["y0"]))
                    if not p_base.is_valid or p_base.is_empty:
                        continue
                    if not keep.contains(p_base.centroid):
                        continue
                    batch_nuclei.append(
                        {
                            "nucleus_id": nucleus_id,
                            "tile_id": int(t["tile_id"]),
                            "model_name": segmenter.name,
                            "area_px": float(p_base.area),
                            "geometry": p_base,
                        }
                    )
                    nucleus_id += 1

            total_kept += len(batch_nuclei)

            if writer is not None:
                if batch_nuclei:
                    table = pa.table(
                        {
                            "nucleus_id": [r["nucleus_id"] for r in batch_nuclei],
                            "tile_id": [r["tile_id"] for r in batch_nuclei],
                            "model_name": [r["model_name"] for r in batch_nuclei],
                            "area_px": [r["area_px"] for r in batch_nuclei],
                            "geometry": [wkb_dumps(r["geometry"]) for r in batch_nuclei],
                        },
                        schema=writer.schema,
                    )
                    writer.write_table(table)
            else:
                in_mem_rows.extend(batch_nuclei)
    finally:
        if writer is not None:
            writer.close()

    log.info("inference done: %d nuclei kept", total_kept)

    if out_path is not None:
        return gpd.read_parquet(out_path)
    return gpd.GeoDataFrame(in_mem_rows, geometry="geometry", crs=None)


def _batched(seq, n):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
