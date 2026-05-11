from unittest.mock import MagicMock

import geopandas as gpd
from shapely.geometry import box

from hne_nuclear_segmentation.tiling import generate_tiles


def _fake_slide(width=4096, height=4096, mpp=0.25):
    s = MagicMock()
    s.width = width
    s.height = height
    s.mpp = mpp
    return s


def test_tile_count_no_overlap():
    s = _fake_slide(2048, 2048, mpp=0.5)
    tissue = gpd.GeoDataFrame(geometry=[box(0, 0, 2048, 2048)], crs=None)
    # target_mpp == base_mpp -> scale 1, tile_size_base == tile_size_model_px
    tiles = generate_tiles(s, tissue, target_mpp=0.5, tile_size_model_px=512, overlap_fraction=0.0)
    assert len(tiles) == 16


def test_keep_box_inner_region():
    s = _fake_slide(2048, 2048, mpp=0.5)
    tissue = gpd.GeoDataFrame(geometry=[box(0, 0, 2048, 2048)], crs=None)
    tiles = generate_tiles(s, tissue, target_mpp=0.5, tile_size_model_px=1000, overlap_fraction=0.1)
    row = tiles.iloc[0]
    # keep box should be inset by 10% of tile size on each side
    assert row["keep_x0"] - row["x0"] == 100
    assert row["x1"] - row["keep_x1"] == 100


def test_scale_factor_applied_when_target_mpp_differs():
    s = _fake_slide(4096, 4096, mpp=0.25)
    tissue = gpd.GeoDataFrame(geometry=[box(0, 0, 4096, 4096)], crs=None)
    # target_mpp 0.5 -> scale 2 -> 1024 model px = 2048 base px
    tiles = generate_tiles(s, tissue, target_mpp=0.5, tile_size_model_px=1024, overlap_fraction=0.0)
    row = tiles.iloc[0]
    assert row["x1"] - row["x0"] == 2048
