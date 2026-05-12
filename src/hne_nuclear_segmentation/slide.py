"""WSI reader with pyvips and OME-Zarr backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._progress import attach_vips_progress

log = logging.getLogger("hne.slide")

_VIPS_DTYPE = {
    "uchar": np.uint8,
    "ushort": np.uint16,
    "float": np.float32,
}


class _Backend(Protocol):
    width: int
    height: int

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray: ...
    def thumbnail(self, scale: float) -> np.ndarray: ...


@dataclass
class Slide:
    path: Path
    backend: Any  # _Backend (Protocol can't be type-annotated as dataclass field cleanly)
    mpp: float

    @classmethod
    def open(cls, path: str | Path, mpp: float | None = None) -> "Slide":
        path = Path(path)
        if _looks_like_zarr(path):
            backend, detected_mpp = _ZarrBackend.open(path)
        else:
            backend, detected_mpp = _VipsBackend.open(path)
        effective_mpp = mpp if mpp is not None else detected_mpp
        if effective_mpp is None:
            raise ValueError(
                f"Cannot determine MPP for {path}; pass mpp= explicitly."
            )
        return cls(path=path, backend=backend, mpp=effective_mpp)

    @property
    def width(self) -> int:
        return self.backend.width

    @property
    def height(self) -> int:
        return self.backend.height

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.width - x)
        h = min(h, self.height - y)
        return self.backend.read_region(x, y, w, h)

    def read_region_at_mpp(
        self, x: int, y: int, w: int, h: int, target_mpp: float
    ) -> tuple[np.ndarray, float]:
        scale = target_mpp / self.mpp
        arr = self.read_region(x, y, w, h)
        if abs(scale - 1.0) > 1e-3:
            arr = _resize_rgb(arr, 1.0 / scale)
        return arr, scale

    def thumbnail(self, target_mpp: float) -> tuple[np.ndarray, float]:
        scale = target_mpp / self.mpp
        thumb = self.backend.thumbnail(scale)
        # Derive actual scale from produced thumbnail dims — vips thumbnail
        # honors a bounding box, so for portrait images the output may differ
        # from the requested width-based size.
        actual_scale = self.width / thumb.shape[1]
        return thumb, actual_scale


def _looks_like_zarr(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / ".zgroup").exists() or (path / ".zarray").exists()


def _resize_rgb(arr: np.ndarray, scale: float) -> np.ndarray:
    """Resize an HxWx3 uint8 array by `scale` (e.g. 0.5 = half size)."""
    from PIL import Image

    h, w = arr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


# ----------------------------- pyvips backend -----------------------------


class _VipsBackend:
    def __init__(self, image, path: Path):
        self.image = image
        self.path = path
        self.width = image.width
        self.height = image.height

    @classmethod
    def open(cls, path: Path):
        import pyvips

        log.info("opening slide via pyvips: %s", path)
        image = pyvips.Image.new_from_file(str(path), access="random")
        log.info("slide loaded: %d x %d, %d bands, %s", image.width, image.height, image.bands, image.format)
        mpp = _read_vips_mpp(image)
        if mpp is not None:
            log.info("detected MPP: %.4f µm/px", mpp)
        return cls(image, path), mpp

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        region = self.image.crop(x, y, w, h)
        return _vips_to_numpy(region)

    def _openslide_pyramid(self) -> list[tuple[int, float]] | None:
        """Return [(level_index, downsample), ...] from openslide fields, or None."""
        fields = set(self.image.get_fields())
        if "openslide.level-count" not in fields:
            return None
        try:
            n = int(self.image.get("openslide.level-count"))
        except (ValueError, TypeError):
            return None
        levels: list[tuple[int, float]] = []
        for i in range(n):
            key = f"openslide.level[{i}].downsample"
            if key not in fields:
                continue
            try:
                ds = float(self.image.get(key))
            except (ValueError, TypeError):
                continue
            levels.append((i, ds))
        return levels or None

    def _best_pyramid_level(self, scale: float) -> tuple[int, float] | None:
        """Highest-downsample level whose downsample <= scale (closest-but-higher-res than target)."""
        levels = self._openslide_pyramid()
        if not levels:
            return None
        best: tuple[int, float] | None = None
        for idx, ds in levels:
            if ds <= scale and (best is None or ds > best[1]):
                best = (idx, ds)
        return best

    def thumbnail(self, scale: float) -> np.ndarray:
        import pyvips

        thumb_w = max(1, int(round(self.width / scale)))
        log.info(
            "generating thumbnail: %d x %d -> width=%d (scale=%.2f)",
            self.width, self.height, thumb_w, scale,
        )

        chosen = self._best_pyramid_level(scale)
        if chosen is not None:
            level_idx, ds = chosen
            log.info(
                "using openslide pyramid level %d (downsample=%.2f, target_scale=%.2f) — fast path",
                level_idx, ds, scale,
            )
            level_img = pyvips.Image.new_from_file(
                str(self.path), level=level_idx, access="sequential"
            )
            log.info("level %d native size: %d x %d", level_idx, level_img.width, level_img.height)
            if level_img.width != thumb_w:
                resize_factor = thumb_w / level_img.width
                level_img = level_img.resize(resize_factor)
            attach_vips_progress(level_img, f"thumb L{level_idx}->w={thumb_w}")
            return _vips_to_numpy(level_img)

        thumb_h = max(1, int(round(self.height / scale)))
        log.info("no pyramid available; streaming full image for thumbnail (slow path)")
        thumb = pyvips.Image.thumbnail_image(self.image, thumb_w, height=thumb_h)
        attach_vips_progress(thumb, f"thumbnail w={thumb_w} h={thumb_h}")
        return _vips_to_numpy(thumb)


def _vips_to_numpy(image) -> np.ndarray:
    dtype = _VIPS_DTYPE[image.format]
    arr = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=dtype,
        shape=(image.height, image.width, image.bands),
    )
    if image.bands == 4:
        arr = arr[..., :3]
    return arr


def _read_vips_mpp(image) -> float | None:
    fields = set(image.get_fields())
    for key in ("openslide.mpp-x", "aperio.MPP"):
        if key in fields:
            try:
                return float(image.get(key))
            except (ValueError, TypeError):
                continue
    if "xres" in fields:
        xres = float(image.get("xres"))
        if xres > 0:
            return 1000.0 / xres
    return None


# ----------------------------- zarr backend -----------------------------


class _ZarrBackend:
    def __init__(self, levels: list, axes: tuple[int, int, int]):
        """levels: list of zarr arrays, level 0 first.
        axes: (c_axis, y_axis, x_axis) indices into the array shape.
        """
        self.levels = levels
        self.c_axis, self.y_axis, self.x_axis = axes
        shape = levels[0].shape
        self.height = shape[self.y_axis]
        self.width = shape[self.x_axis]

    @classmethod
    def open(cls, path: Path):
        import zarr

        node = zarr.open(str(path), mode="r")
        attrs = dict(node.attrs)
        levels = []
        axes_idx = (0, 1, 2)  # default (c, y, x)
        mpp = None

        if "multiscales" in attrs:
            ms = attrs["multiscales"][0]
            axes = ms.get("axes", [])
            axis_names = [a["name"].lower() if isinstance(a, dict) else str(a).lower() for a in axes]
            try:
                axes_idx = (
                    axis_names.index("c"),
                    axis_names.index("y"),
                    axis_names.index("x"),
                )
            except ValueError:
                axes_idx = (0, 1, 2)
            for ds in ms["datasets"]:
                levels.append(node[ds["path"]])
            # Read MPP from level-0 scale on the y axis (only if unit looks like microns)
            level0 = ms["datasets"][0]
            cts = level0.get("coordinateTransformations", [])
            for ct in cts:
                if ct.get("type") == "scale":
                    scale = ct["scale"]
                    y_scale = float(scale[axes_idx[1]])
                    unit = ""
                    if axes and isinstance(axes[axes_idx[1]], dict):
                        unit = axes[axes_idx[1]].get("unit", "").lower()
                    # Only trust scale if it's in microns AND not a placeholder 1.0/1.0
                    if "micro" in unit and y_scale != 1.0:
                        mpp = y_scale
                    break
        else:
            # plain array
            levels.append(node)
            # assume (c, y, x) if 3D
            if node.ndim == 3 and node.shape[0] in (1, 3, 4):
                axes_idx = (0, 1, 2)
            elif node.ndim == 3:
                axes_idx = (2, 0, 1)  # (y, x, c)
            else:
                raise ValueError(f"Unsupported zarr array ndim={node.ndim}")

        return cls(levels, axes_idx), mpp

    def _slice_to_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Reorder a (C,Y,X)-ish ndarray to (Y,X,C) uint8 RGB."""
        # Move axes to (y, x, c)
        arr = np.moveaxis(arr, [self.y_axis, self.x_axis, self.c_axis], [0, 1, 2])
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[..., :3]
        elif arr.shape[2] != 3:
            raise ValueError(f"Expected 1/3/4 channels, got {arr.shape[2]}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def _read_from_level(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        arr = self.levels[level_idx]
        slicer: list = [slice(None)] * arr.ndim
        slicer[self.y_axis] = slice(y, y + h)
        slicer[self.x_axis] = slice(x, x + w)
        data = arr[tuple(slicer)]
        return self._slice_to_rgb(np.asarray(data))

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return self._read_from_level(0, x, y, w, h)

    def thumbnail(self, scale: float) -> np.ndarray:
        # Pick best pyramid level: the deepest level whose downsampling <= requested scale
        # Each subsequent level is ~2x downsampled in the typical case.
        chosen = 0
        for i in range(1, len(self.levels)):
            level = self.levels[i]
            ds = self.levels[0].shape[self.y_axis] / level.shape[self.y_axis]
            if ds <= scale:
                chosen = i
            else:
                break
        level = self.levels[chosen]
        level_h = level.shape[self.y_axis]
        level_w = level.shape[self.x_axis]
        data = self._read_from_level(chosen, 0, 0, level_w, level_h)
        # If chosen level is still bigger than requested thumbnail, downscale further
        cur_scale = self.levels[0].shape[self.y_axis] / level_h
        if cur_scale < scale * 0.95:
            extra = cur_scale / scale
            data = _resize_rgb(data, extra)
        return data
