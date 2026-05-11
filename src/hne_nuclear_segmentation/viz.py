"""Visualizations: interactive HTML overview + full-res tile detail PDF.

`build_html_viewer` writes a single self-contained `visualization.html` with
the slide thumbnail and all per-layer polygons inlined as SVG paths in
level-0 coordinates, plus client-side pan/zoom.

`build_detail_pdf` writes `details.pdf` containing matplotlib panels of a few
high-density tiles rendered at full resolution with model polygons overlaid.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
from PIL import Image

from .slide import Slide

log = logging.getLogger("hne.viz")


# ---------------------------------------------------------------- HTML viewer

_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:#0e0e10; color:#eee; }
header { display:flex; flex-wrap:wrap; align-items:center; gap:.6rem 1.4rem;
  padding:.7rem 1rem; background:#17171a; border-bottom:1px solid #2a2a2e;
  position:sticky; top:0; z-index:10; }
header h1 { font-size:.95rem; font-weight:500; margin:0; }
header .layer { display:flex; align-items:center; gap:.35rem; font-size:.82rem; }
header .swatch { display:inline-block; width:.75rem; height:.75rem; border:1px solid #444; }
header .count { color:#999; font-size:.72rem; }
header label.opacity, header .mode { display:flex; align-items:center; gap:.4rem; font-size:.82rem; }
header input[type=range] { width:120px; }
header button { background:#26262b; color:#eee; border:1px solid #3a3a40;
  border-radius:4px; padding:.2rem .55rem; font-size:.82rem; cursor:pointer; }
header button:hover { background:#33333a; }
.meta { color:#888; font-size:.74rem; }
.hint { color:#666; font-size:.74rem; margin-left:auto; }
#viewport { position:relative; width:100vw; height:calc(100vh - 56px);
  overflow:hidden; background:#0a0a0c; cursor:grab; }
#viewport.dragging { cursor:grabbing; }
#wrap { position:absolute; left:0; top:0; transform-origin:0 0; will-change:transform; }
#wrap img { display:block; user-select:none; -webkit-user-drag:none; pointer-events:none; }
#overlay { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
#zoombadge { position:absolute; right:.6rem; bottom:.6rem;
  background:rgba(20,20,24,.85); border:1px solid #333; padding:.25rem .5rem;
  border-radius:4px; font-size:.74rem; color:#bbb; font-variant-numeric:tabular-nums; }
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  __LAYER_CONTROLS__
  <label class="opacity">opacity <input type="range" id="opacity" min="0" max="1" step="0.05" value="0.85"/></label>
  <label class="mode"><input type="checkbox" id="fillmode"/> fill</label>
  <button id="zoomIn" title="zoom in">+</button>
  <button id="zoomOut" title="zoom out">−</button>
  <button id="zoomReset" title="fit">fit</button>
  <span class="meta">__META__</span>
  <span class="hint">scroll = zoom · drag = pan · dbl-click = zoom</span>
</header>
<div id="viewport">
  <div id="wrap" style="width:__THUMB_W__px; height:__THUMB_H__px;">
    <img src="__THUMB_URI__" width="__THUMB_W__" height="__THUMB_H__" alt="thumb"/>
    <svg id="overlay" viewBox="0 0 __SLIDE_W__ __SLIDE_H__" preserveAspectRatio="none"
         width="__THUMB_W__" height="__THUMB_H__"></svg>
  </div>
  <div id="zoombadge">1.00×</div>
</div>
<script>
const LAYERS = __LAYERS_JSON__;
const overlay = document.getElementById("overlay");
const svgNS = "http://www.w3.org/2000/svg";
const paths = {};
for (const L of LAYERS) {
  const p = document.createElementNS(svgNS, "path");
  p.setAttribute("d", L.d);
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", L.color);
  p.setAttribute("stroke-width", "1.25");
  p.setAttribute("stroke-linejoin", "round");
  p.setAttribute("vector-effect", "non-scaling-stroke");
  p.style.display = L.default ? "" : "none";
  overlay.appendChild(p);
  paths[L.name] = p;
}
const opacity = document.getElementById("opacity");
overlay.style.opacity = opacity.value;
opacity.addEventListener("input", e => { overlay.style.opacity = e.target.value; });
document.getElementById("fillmode").addEventListener("change", e => {
  for (const L of LAYERS) {
    const p = paths[L.name];
    if (e.target.checked) { p.setAttribute("fill", L.color); p.setAttribute("fill-opacity", "0.35"); }
    else { p.setAttribute("fill", "none"); }
  }
});
for (const L of LAYERS) {
  const cb = document.getElementById("cb_" + L.name);
  cb.addEventListener("change", () => { paths[L.name].style.display = cb.checked ? "" : "none"; });
}
const viewport = document.getElementById("viewport");
const wrap = document.getElementById("wrap");
const badge = document.getElementById("zoombadge");
const THUMB_W = __THUMB_W__, THUMB_H = __THUMB_H__;
let view = { x: 0, y: 0, k: 1 };
function fit() {
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  const k = Math.min(vw / THUMB_W, vh / THUMB_H);
  view.k = k;
  view.x = (vw - THUMB_W * k) / 2;
  view.y = (vh - THUMB_H * k) / 2;
  apply();
}
function apply() {
  wrap.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`;
  badge.textContent = view.k.toFixed(2) + "×";
}
function zoomAt(cx, cy, factor) {
  const nk = Math.max(0.05, Math.min(80, view.k * factor));
  const f = nk / view.k;
  view.x = cx - (cx - view.x) * f;
  view.y = cy - (cy - view.y) * f;
  view.k = nk;
  apply();
}
viewport.addEventListener("wheel", e => {
  e.preventDefault();
  const r = viewport.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });
viewport.addEventListener("dblclick", e => {
  const r = viewport.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, e.shiftKey ? 0.5 : 2);
});
let drag = null;
viewport.addEventListener("pointerdown", e => {
  drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  viewport.setPointerCapture(e.pointerId);
  viewport.classList.add("dragging");
});
viewport.addEventListener("pointermove", e => {
  if (!drag) return;
  view.x = drag.vx + (e.clientX - drag.x);
  view.y = drag.vy + (e.clientY - drag.y);
  apply();
});
viewport.addEventListener("pointerup", () => { drag = null; viewport.classList.remove("dragging"); });
viewport.addEventListener("pointercancel", () => { drag = null; viewport.classList.remove("dragging"); });
document.getElementById("zoomIn").addEventListener("click", () => {
  const r = viewport.getBoundingClientRect();
  zoomAt(r.width / 2, r.height / 2, 1.5);
});
document.getElementById("zoomOut").addEventListener("click", () => {
  const r = viewport.getBoundingClientRect();
  zoomAt(r.width / 2, r.height / 2, 1 / 1.5);
});
document.getElementById("zoomReset").addEventListener("click", fit);
window.addEventListener("resize", () => apply());
fit();
</script>
</body></html>
"""


def _poly_to_svg_path(geom, tol: float) -> str:
    if geom is None or geom.is_empty:
        return ""
    g = geom.simplify(tol, preserve_topology=False) if tol > 0 else geom
    if g.is_empty:
        g = geom
    parts = []
    polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
    for p in polys:
        if p.is_empty:
            continue
        xs, ys = p.exterior.coords.xy
        if len(xs) < 3:
            continue
        s = [f"M{xs[0]:.1f} {ys[0]:.1f}"]
        s.extend(f"L{x:.1f} {y:.1f}" for x, y in zip(xs[1:], ys[1:]))
        s.append("Z")
        parts.append("".join(s))
    return "".join(parts)


def _layer_path(gdf: gpd.GeoDataFrame, tol: float) -> str:
    return "".join(_poly_to_svg_path(g, tol) for g in gdf.geometry)


def build_html_viewer(
    slide: Slide,
    out_dir: Path,
    *,
    out_path: Path | None = None,
    thumb_width: int = 1800,
    jpeg_quality: int = 80,
) -> Path:
    """Build a self-contained interactive HTML viewer from pipeline outputs in `out_dir`."""
    out_dir = Path(out_dir)
    out_path = Path(out_path) if out_path else out_dir / "visualization.html"

    scale = slide.width / thumb_width
    log.info(
        "rendering thumbnail at downsample=%.2f (L0 %dx%d -> target width %d)",
        scale, slide.width, slide.height, thumb_width,
    )
    thumb = slide.backend.thumbnail(scale)
    thumb_h, thumb_w = thumb.shape[:2]

    buf = io.BytesIO()
    Image.fromarray(thumb).save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    thumb_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    specs = [
        ("tissue",       "tissue.parquet",                 "#ff5577", False, 2.0),
        ("tiles",        "tiles.parquet",                  "#ffd700", False, 0.0),
        ("keep",         "tiles.parquet",                  "#ffeb3b", False, 0.0),
        ("stardist",     "nuclei_stardist.parquet",        "#3aa0ff", True,  0.4),
        ("cellpose",     "nuclei_cellpose.parquet",        "#ff8c1a", False, 0.4),
        ("union",        "consensus_union.parquet",        "#a26bff", False, 0.4),
        ("intersection", "consensus_intersection.parquet", "#2ecc71", False, 0.4),
    ]
    layers = []
    for name, fname, color, default_on, tol in specs:
        f = out_dir / fname
        if not f.exists():
            log.info("skipping missing layer %s (%s)", name, f)
            continue
        g = gpd.read_parquet(f)
        if name == "keep":
            from shapely.geometry import box

            g = g.assign(geometry=[
                box(r.keep_x0, r.keep_y0, r.keep_x1, r.keep_y1) for r in g.itertuples()
            ])
        log.info("layer %s: %d features", name, len(g))
        layers.append({
            "name": name, "color": color, "count": int(len(g)),
            "default": default_on, "d": _layer_path(g, tol=tol),
        })

    layer_controls = "\n".join(
        f'<label class="layer"><input type="checkbox" id="cb_{L["name"]}" '
        f'{"checked" if L["default"] else ""}/>'
        f'<span class="swatch" style="background:{L["color"]}"></span>'
        f'{L["name"]} <span class="count">({L["count"]:,})</span></label>'
        for L in layers
    )
    title = f"{slide.path.name} — segmentation overlay"
    n_nuclei = sum(L["count"] for L in layers if L["name"] not in ("tissue", "tiles"))
    meta = (f"L0 {slide.width}×{slide.height}px · thumb {thumb_w}×{thumb_h} · "
            f"{n_nuclei:,} total detections")

    html = (_HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__META__", meta)
        .replace("__THUMB_URI__", thumb_uri)
        .replace("__THUMB_W__", str(thumb_w))
        .replace("__THUMB_H__", str(thumb_h))
        .replace("__SLIDE_W__", str(slide.width))
        .replace("__SLIDE_H__", str(slide.height))
        .replace("__LAYER_CONTROLS__", layer_controls)
        .replace("__LAYERS_JSON__", json.dumps(layers, separators=(",", ":")))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("wrote %s (%.1f MiB)", out_path, out_path.stat().st_size / 1024 / 1024)
    return out_path


# ----------------------------------------------------------------- detail PDF

def _pick_dense_tiles(
    tiles: gpd.GeoDataFrame,
    nuclei: gpd.GeoDataFrame,
    n: int,
    *,
    min_separation_px: float = 4000.0,
) -> list[int]:
    """Pick the `n` tiles with the most nuclei, requiring spatial separation."""
    counts = nuclei.groupby("tile_id").size().sort_values(ascending=False)
    chosen: list[int] = []
    chosen_centers: list[tuple[float, float]] = []
    for tid in counts.index:
        row = tiles[tiles.tile_id == tid]
        if row.empty:
            continue
        r = row.iloc[0]
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if any(np.hypot(cx - x, cy - y) < min_separation_px for x, y in chosen_centers):
            continue
        chosen.append(int(tid))
        chosen_centers.append((cx, cy))
        if len(chosen) >= n:
            break
    return chosen


def _plot_polys_on_ax(ax, gdf: gpd.GeoDataFrame, bbox: tuple[int, int, int, int],
                     color: str, linewidth: float = 0.7,
                     keep_box: tuple[float, float, float, float] | None = None) -> int:
    """Plot polygon outlines (in L0 coords) on an axes whose extent is `bbox`.

    If `keep_box` is given, only polygons whose centroid lies inside that box
    are drawn — matches the in-pipeline rule that discards detections outside
    a tile's keep region, so neighboring tiles' overlap-zone polygons don't
    bleed into the panel.
    """
    x0, y0, x1, y1 = bbox
    clipped = gdf.cx[x0:x1, y0:y1]
    if keep_box is not None:
        import warnings

        kx0, ky0, kx1, ky1 = keep_box
        with warnings.catch_warnings():
            # Pixel coords carry a geographic CRS by default; harmless here.
            warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
            cents = clipped.geometry.centroid
        mask = (cents.x >= kx0) & (cents.x < kx1) & (cents.y >= ky0) & (cents.y < ky1)
        clipped = clipped[mask.values]
    if len(clipped) == 0:
        return 0
    for geom in clipped.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in polys:
            xs, ys = p.exterior.coords.xy
            ax.plot(np.asarray(xs) - x0, np.asarray(ys) - y0,
                    color=color, linewidth=linewidth, alpha=0.9)
    return len(clipped)


def build_detail_pdf(
    slide: Slide,
    out_dir: Path,
    *,
    out_path: Path | None = None,
    n_tiles: int = 2,
    crop_size: int | None = None,
) -> Path:
    """Render N high-density tiles at full resolution with per-model polygon overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = Path(out_dir)
    out_path = Path(out_path) if out_path else out_dir / "details.pdf"

    tiles = gpd.read_parquet(out_dir / "tiles.parquet")
    stardist = gpd.read_parquet(out_dir / "nuclei_stardist.parquet")
    cellpose = gpd.read_parquet(out_dir / "nuclei_cellpose.parquet")
    union_p = out_dir / "consensus_union.parquet"
    union = gpd.read_parquet(union_p) if union_p.exists() else None
    inter_p = out_dir / "consensus_intersection.parquet"
    intersection = gpd.read_parquet(inter_p) if inter_p.exists() else None

    # Use stardist tile_ids to rank density (covers both models reasonably).
    tile_ids = _pick_dense_tiles(tiles, stardist, n_tiles)
    log.info("detail PDF tiles: %s", tile_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for tid in tile_ids:
            t = tiles[tiles.tile_id == tid].iloc[0]
            x0, y0, x1, y1 = int(t.x0), int(t.y0), int(t.x1), int(t.y1)
            if crop_size is not None:
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                half = crop_size // 2
                x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(slide.width, x1); y1 = min(slide.height, y1)
            w, h = x1 - x0, y1 - y0
            log.info("rendering tile %d at L0: (%d,%d) %dx%d", tid, x0, y0, w, h)
            img = slide.read_region(x0, y0, w, h)
            bbox = (x0, y0, x1, y1)

            panels: list[tuple[str, gpd.GeoDataFrame | None, str]] = [
                ("H&E (full resolution)", None, ""),
                ("Stardist", stardist, "#1f77ff"),
                ("Cellpose", cellpose, "#ff8c1a"),
            ]
            if union is not None:
                panels.append(("Consensus union", union, "#a26bff"))
            if intersection is not None:
                panels.append(("Consensus intersection", intersection, "#2ecc71"))

            ncols = len(panels)
            fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 5.4))
            if ncols == 1:
                axes = [axes]
            fig.suptitle(
                f"{slide.path.name} — tile {tid} · L0 ({x0},{y0}) {w}×{h}px @ {slide.mpp:.3f} µm/px",
                fontsize=10,
            )
            keep_box = (float(t.keep_x0), float(t.keep_y0), float(t.keep_x1), float(t.keep_y1))
            kx0, ky0 = int(t.keep_x0) - x0, int(t.keep_y0) - y0
            kw, kh = int(t.keep_x1 - t.keep_x0), int(t.keep_y1 - t.keep_y0)
            for ax, (label, gdf, color) in zip(axes, panels):
                ax.imshow(img)
                n = 0
                if gdf is not None:
                    n = _plot_polys_on_ax(ax, gdf, bbox, color=color, linewidth=0.6,
                                          keep_box=keep_box)
                # Draw the keep-box: detections whose centroid falls outside this
                # inner rectangle are discarded to remove tile-edge duplicates,
                # so cells appear to be missing in the outer overlap band.
                from matplotlib.patches import Rectangle

                ax.add_patch(Rectangle(
                    (kx0, ky0), kw, kh,
                    fill=False, edgecolor="#ffeb3b", linewidth=0.9,
                    linestyle="--", alpha=0.9,
                ))
                ax.set_title(f"{label}" + (f" (n={n})" if gdf is not None else ""), fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            fig.text(
                0.5, 0.015,
                "yellow dashed box = keep region; detections outside are dropped to avoid tile-edge duplicates",
                ha="center", fontsize=7.5, color="#bbbb00",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    log.info("wrote %s (%.1f MiB)", out_path, out_path.stat().st_size / 1024 / 1024)
    return out_path
