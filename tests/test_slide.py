"""Round-trip test for slide reader using a synthetic pyramid TIFF."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pyvips = pytest.importorskip("pyvips")
tifffile = pytest.importorskip("tifffile")

from hne_nuclear_segmentation.slide import Slide


def _make_synthetic_tiff(path: Path, size: int = 1024, mpp: float = 0.5) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    # Embed MPP via TIFF resolution (px/cm); 1/mpp px per µm -> *1e4 px per cm.
    res_per_cm = 1.0 / (mpp * 1e-4)
    tifffile.imwrite(
        path,
        img,
        photometric="rgb",
        tile=(256, 256),
        resolution=(res_per_cm, res_per_cm),
        resolutionunit="CENTIMETER",
    )


def test_slide_open_and_read(tmp_path: Path):
    p = tmp_path / "synth.tif"
    _make_synthetic_tiff(p, size=1024, mpp=0.5)
    slide = Slide.open(p)
    assert slide.width == 1024
    assert slide.height == 1024
    assert abs(slide.mpp - 0.5) < 0.05
    region = slide.read_region(0, 0, 256, 256)
    assert region.shape == (256, 256, 3)


def test_read_region_at_mpp_resamples(tmp_path: Path):
    p = tmp_path / "synth.tif"
    _make_synthetic_tiff(p, size=2048, mpp=0.25)
    slide = Slide.open(p)
    arr, scale = slide.read_region_at_mpp(0, 0, 1024, 1024, target_mpp=0.5)
    # scale = target/base = 0.5/0.25 = 2 -> output ~512x512
    assert abs(scale - 2.0) < 0.05
    assert 500 <= arr.shape[0] <= 520
    assert 500 <= arr.shape[1] <= 520
