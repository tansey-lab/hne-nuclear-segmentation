"""Consensus building between two segmentation outputs."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


def _overlap_components(geoms: list) -> list[list[int]]:
    """Union-find groups of geometries by pairwise intersection."""
    n = len(geoms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    if n:
        tree = STRtree(geoms)
        for i, g in enumerate(geoms):
            for j in tree.query(g):
                j = int(j)
                if j <= i:
                    continue
                if g.intersects(geoms[j]):
                    union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    return list(components.values())


def dedupe_overlapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge overlapping polygons within a single model's output by union.

    Tiles produced with overlap_fraction > 2*edge_fraction yield duplicate
    detections in their overlapping keep regions. Collapse each connected
    component of intersecting polygons into a single polygon (unary_union).
    nucleus_id is reassigned; tile_id of the first member is kept for
    traceability.
    """
    if gdf.empty:
        return gdf.copy()
    geoms = list(gdf.geometry)
    comps = _overlap_components(geoms)
    rows = []
    for new_id, idxs in enumerate(comps):
        merged = unary_union([geoms[i] for i in idxs]) if len(idxs) > 1 else geoms[idxs[0]]
        first = gdf.iloc[idxs[0]]
        row = {
            "nucleus_id": new_id,
            "tile_id": int(first["tile_id"]) if "tile_id" in gdf.columns else -1,
            "area_px": float(merged.area),
            "geometry": merged,
        }
        if "model_name" in gdf.columns:
            row["model_name"] = first["model_name"]
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def build_union(
    stardist_gdf: gpd.GeoDataFrame, cellpose_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Merge overlapping polygons from both models into single polygons.

    Each output row is a connected component (via overlap graph) of input polygons,
    geometry = unary_union of that component, models = sorted list of contributing
    model names.
    """
    a = stardist_gdf.assign(model_name="stardist")
    b = cellpose_gdf.assign(model_name="cellpose")
    combined = pd.concat([a, b], ignore_index=True)

    if combined.empty:
        return gpd.GeoDataFrame(
            columns=["consensus_id", "models", "n_sources", "area_px", "geometry"],
            geometry="geometry",
            crs=None,
        )

    geoms = list(combined.geometry)
    names = list(combined["model_name"])

    rows = []
    for cid, idxs in enumerate(_overlap_components(geoms)):
        merged = unary_union([geoms[i] for i in idxs]) if len(idxs) > 1 else geoms[idxs[0]]
        models = sorted({names[i] for i in idxs})
        rows.append(
            {
                "consensus_id": cid,
                "models": models,
                "n_sources": len(idxs),
                "area_px": float(merged.area),
                "geometry": merged,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def build_intersection(
    stardist_gdf: gpd.GeoDataFrame,
    cellpose_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Keep nuclei detected by BOTH models (greedy IoU match).

    Geometry of each output row is the union of the matched pair; this gives
    a shape that reflects both models' agreement on extent.
    """
    if stardist_gdf.empty or cellpose_gdf.empty:
        return gpd.GeoDataFrame(
            columns=[
                "consensus_id",
                "stardist_id",
                "cellpose_id",
                "iou",
                "area_px",
                "geometry",
            ],
            geometry="geometry",
            crs=None,
        )

    s_geoms = list(stardist_gdf.geometry)
    c_geoms = list(cellpose_gdf.geometry)
    s_ids = list(stardist_gdf["nucleus_id"])
    c_ids = list(cellpose_gdf["nucleus_id"])

    tree = STRtree(c_geoms)

    candidates: list[tuple[float, int, int, Polygon]] = []
    for i, sg in enumerate(s_geoms):
        for j in tree.query(sg):
            j = int(j)
            cg = c_geoms[j]
            if not sg.intersects(cg):
                continue
            inter = sg.intersection(cg).area
            if inter <= 0:
                continue
            uni = sg.area + cg.area - inter
            iou = inter / uni if uni > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, i, j, sg.union(cg)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_s, used_c = set(), set()
    rows = []
    cid = 0
    for iou, i, j, merged in candidates:
        if i in used_s or j in used_c:
            continue
        used_s.add(i)
        used_c.add(j)
        rows.append(
            {
                "consensus_id": cid,
                "stardist_id": int(s_ids[i]),
                "cellpose_id": int(c_ids[j]),
                "iou": float(iou),
                "area_px": float(merged.area),
                "geometry": merged,
            }
        )
        cid += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
