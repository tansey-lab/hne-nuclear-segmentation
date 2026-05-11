import geopandas as gpd
from shapely.geometry import box

from hne_nuclear_segmentation.consensus import build_intersection, build_union


def _gdf(geoms, name):
    return gpd.GeoDataFrame(
        {"nucleus_id": list(range(len(geoms))), "model_name": name},
        geometry=geoms,
        crs=None,
    )


def test_union_merges_overlapping():
    s = _gdf([box(0, 0, 10, 10), box(100, 100, 110, 110)], "stardist")
    c = _gdf([box(5, 5, 15, 15), box(200, 200, 210, 210)], "cellpose")
    u = build_union(s, c)
    # (0..10) and (5..15) merge -> 1 component; the two isolated boxes -> 2 more
    assert len(u) == 3
    # The merged component should carry both model names
    merged = u[u["n_sources"] == 2].iloc[0]
    assert set(merged["models"]) == {"stardist", "cellpose"}


def test_intersection_matches_pairs_above_iou():
    s = _gdf([box(0, 0, 10, 10), box(100, 100, 110, 110)], "stardist")
    # First overlaps strongly with stardist[0]; second is alone
    c = _gdf([box(1, 1, 11, 11), box(500, 500, 510, 510)], "cellpose")
    out = build_intersection(s, c, iou_threshold=0.3)
    assert len(out) == 1
    assert out.iloc[0]["stardist_id"] == 0
    assert out.iloc[0]["cellpose_id"] == 0


def test_intersection_greedy_one_to_one():
    s = _gdf([box(0, 0, 10, 10), box(2, 2, 12, 12)], "stardist")
    c = _gdf([box(1, 1, 11, 11)], "cellpose")
    out = build_intersection(s, c, iou_threshold=0.1)
    assert len(out) == 1  # one cellpose can match only one stardist
