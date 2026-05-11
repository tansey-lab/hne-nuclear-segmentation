"""GeoParquet read/write helpers."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def write_geoparquet(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path, index=False)


def read_geoparquet(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)
