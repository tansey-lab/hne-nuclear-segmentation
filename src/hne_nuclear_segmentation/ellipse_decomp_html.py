"""Render an interactive HTML viewer for the GMM-guided decomposition.

Usage (from a driver script)::

    from hne_nuclear_segmentation.ellipse_decomp_html import render_tile_html
    render_tile_html(polys, prior, out_path="tile.html", title="tile 123")

The page shows the polygons as SVG (color-coded by k_picked); hovering or
clicking a polygon shows a sidebar with area, solidity, gating decision,
k_max_by_area, k_picked, and a small BIC line chart (raw and modified).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from matplotlib.patches import Ellipse as MplEllipse

from .ellipse_decomp_gmm import (
    AreaPrior, DecompDiagnostics, decompose_with_gmm_verbose,
)
from .ellipse_decomp_recursive import decompose_polygon_recursive


_PALETTE = {1: "#2e8b57", 2: "#dc143c", 3: "#ff8c00",
            4: "#9370db", 5: "#4169e1", 6: "#daa520"}


def _ellipse_to_polyline(cx: float, cy: float, w: float, h: float,
                         angle_deg: float, n: int = 48) -> list[tuple[float, float]]:
    """Return n points along the boundary of a cv2-style ellipse."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rx, ry = w / 2.0, h / 2.0
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * x - s * y + cx
    yr = s * x + c * y + cy
    return list(zip(xr.tolist(), yr.tolist()))


def render_tile_html_recursive(
    polys: list[Polygon],
    prior: AreaPrior,
    out_path: str | Path,
    title: str = "recursive ellipse decomp",
    bbox: tuple[float, float, float, float] | None = None,
    solidity_threshold: float = 0.92,
    min_log_z: float = -2.0,
    max_depth: int = 4,
) -> Path:
    """Render recursive-decomposition results. Sidebar shows the split tree."""
    records = []
    for idx, p in enumerate(polys):
        try:
            ells, diag = decompose_polygon_recursive(
                p, prior, solidity_threshold=solidity_threshold,
                min_log_z=min_log_z, max_depth=max_depth,
            )
        except Exception:
            continue
        boundary = list(p.exterior.coords)
        ell_paths = [_ellipse_to_polyline(e.cx, e.cy, e.width, e.height, e.angle_deg)
                     for e in ells]
        steps = [{
            "depth": s.depth, "solidity": round(s.solidity, 3),
            "area": round(s.area, 0), "action": s.action,
            "rejection_reason": s.rejection_reason,
            "child_log_areas": (
                [round(x, 2) for x in s.child_log_areas]
                if s.child_log_areas else None),
        } for s in diag.steps]
        records.append({
            "id": idx,
            "boundary": boundary,
            "ellipses": ell_paths,
            "k": len(ells),
            "solidity": round(diag.solidity, 4),
            "area": round(diag.area, 1),
            "steps": steps,
        })

    if bbox is None:
        xs = [c[0] for r in records for c in r["boundary"]]
        ys = [c[1] for r in records for c in r["boundary"]]
        bbox = (min(xs), min(ys), max(xs), max(ys))

    k_counts: dict[int, int] = {}
    for r in records:
        k_counts[r["k"]] = k_counts.get(r["k"], 0) + 1
    total_ellipses = sum(len(r["ellipses"]) for r in records)

    min_log_area = prior.mu + min_log_z * prior.sigma
    payload = {
        "records": records,
        "bbox": list(bbox),
        "palette": _PALETTE,
        "prior": {"mu": prior.mu, "sigma": prior.sigma,
                  "median": float(np.exp(prior.mu)),
                  "min_area_2sd": float(np.exp(min_log_area)),
                  "n": prior.n},
        "params": {"solidity_threshold": solidity_threshold,
                   "min_log_z": min_log_z, "max_depth": max_depth},
        "summary": {"n_polys": len(records), "k_counts": k_counts,
                    "total_ellipses": total_ellipses},
        "title": title,
        "mode": "recursive",
    }

    class _NPEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    html = _HTML_TEMPLATE_RECURSIVE.replace(
        "__DATA__", json.dumps(payload, cls=_NPEncoder))
    out = Path(out_path)
    out.write_text(html)
    return out


def render_tile_html(
    polys: list[Polygon],
    prior: AreaPrior,
    out_path: str | Path,
    title: str = "ellipse decomp",
    bbox: tuple[float, float, float, float] | None = None,
    k_max: int = 6,
    solidity_threshold: float | None = None,
    prior_weight: float | None = None,
    min_area_frac: float | None = None,
) -> Path:
    """Run decomposition on every polygon and write a self-contained HTML."""
    kwargs: dict = {"k_max": k_max, "area_prior": prior}
    if solidity_threshold is not None:
        kwargs["solidity_threshold"] = solidity_threshold
    if prior_weight is not None:
        kwargs["prior_weight"] = prior_weight
    if min_area_frac is not None:
        kwargs["min_area_frac"] = min_area_frac

    records = []
    for idx, p in enumerate(polys):
        try:
            k, ells, diag = decompose_with_gmm_verbose(p, **kwargs)
        except Exception as e:
            continue
        boundary = list(p.exterior.coords)
        ell_paths = [_ellipse_to_polyline(e.cx, e.cy, e.width, e.height, e.angle_deg)
                     for e in ells]
        records.append({
            "id": idx,
            "boundary": boundary,
            "ellipses": ell_paths,
            "k": diag.k_picked,
            "k_requested": diag.k_requested,
            "k_max_by_area": diag.k_max_by_area,
            "solidity": round(diag.solidity, 4),
            "area": round(diag.area, 1),
            "gated_as_convex": diag.gated_as_convex,
            "raw_bic": [None if not np.isfinite(b) else round(float(b), 2)
                        for b in diag.raw_bic],
            "mod_bic": [None if not np.isfinite(b) else round(float(b), 2)
                        for b in diag.modified_bic],
        })

    if bbox is None:
        xs = [c[0] for r in records for c in r["boundary"]]
        ys = [c[1] for r in records for c in r["boundary"]]
        bbox = (min(xs), min(ys), max(xs), max(ys))

    k_counts: dict[int, int] = {}
    for r in records:
        k_counts[r["k"]] = k_counts.get(r["k"], 0) + 1
    total_ellipses = sum(len(r["ellipses"]) for r in records)

    payload = {
        "records": records,
        "bbox": list(bbox),
        "palette": _PALETTE,
        "prior": {"mu": prior.mu, "sigma": prior.sigma,
                  "median": float(np.exp(prior.mu)), "n": prior.n},
        "params": {
            "solidity_threshold": solidity_threshold,
            "prior_weight": prior_weight,
            "min_area_frac": min_area_frac,
            "k_max": k_max,
        },
        "summary": {"n_polys": len(records), "k_counts": k_counts,
                    "total_ellipses": total_ellipses},
        "title": title,
    }

    class _NPEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, cls=_NPEncoder))
    out = Path(out_path)
    out.write_text(html)
    return out


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ellipse decomp viewer</title>
<style>
  body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background: #1a1a1a; color: #ddd; }
  #wrap { display: flex; height: 100vh; }
  #stage { flex: 1; position: relative; overflow: hidden; background: #222; }
  #side { width: 360px; padding: 16px; background: #0f0f0f; overflow-y: auto;
          border-left: 1px solid #333; }
  #side h2 { margin: 0 0 8px; font-size: 14px; color: #aaa; font-weight: 500; }
  #side .pick-hint { color: #777; font-style: italic; font-size: 12px; }
  #side .field { display: flex; justify-content: space-between; margin: 4px 0;
                 font-size: 13px; font-variant-numeric: tabular-nums; }
  #side .field .label { color: #888; }
  #side .field .value { color: #eee; }
  #side .badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
                 font-size: 11px; font-weight: 600; }
  svg { display: block; }
  .poly { fill-opacity: 0.45; stroke: #000; stroke-width: 0.4px; cursor: pointer; }
  .poly:hover, .poly.locked { stroke: #fff; stroke-width: 1.4px; fill-opacity: 0.8; }
  .ellipse { fill: none; stroke: #fff; stroke-width: 1px; pointer-events: none; opacity: 0; }
  .ellipse.show { opacity: 0.9; }
  #bicchart { width: 100%; height: 180px; margin-top: 8px; background: #111;
              border: 1px solid #333; }
  #header { padding: 12px 16px; background: #0a0a0a; border-bottom: 1px solid #333;
            font-size: 12px; color: #aaa; }
  #header b { color: #eee; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            vertical-align: middle; margin-right: 4px; }
  .legend { font-size: 11px; color: #aaa; margin-top: 8px; }
</style>
</head>
<body>
<div id="header"></div>
<div id="wrap">
  <div id="stage"><svg id="map"></svg></div>
  <div id="side">
    <h2 id="side-title">Hover a polygon</h2>
    <div id="side-body"><div class="pick-hint">Move over any nucleus to see its diagnostics.</div></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const palette = DATA.palette;
const bbox = DATA.bbox;
const [bx0, by0, bx1, by1] = bbox;
const stage = document.getElementById("stage");
const svg = document.getElementById("map");
const side = document.getElementById("side-body");
const sideTitle = document.getElementById("side-title");

const header = document.getElementById("header");
const kSummary = Object.entries(DATA.summary.k_counts).sort((a,b)=>a[0]-b[0])
  .map(([k,c]) => `<span class="swatch" style="background:${palette[k]||'#888'}"></span>k=${k}: <b>${c}</b>`)
  .join(" &nbsp; ");
header.innerHTML = `<b>${DATA.title}</b> · ${DATA.summary.n_polys} polygons →
  <b>${DATA.summary.total_ellipses}</b> ellipses · prior median area
  <b>${DATA.prior.median.toFixed(0)}</b> px² (n=${DATA.prior.n}, σ_log=${DATA.prior.sigma.toFixed(2)})
  &nbsp; · &nbsp; ${kSummary}`;

function resize() {
  const r = stage.getBoundingClientRect();
  svg.setAttribute("width", r.width);
  svg.setAttribute("height", r.height);
  const w = bx1 - bx0, h = by1 - by0;
  svg.setAttribute("viewBox", `${bx0} ${by0} ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}
window.addEventListener("resize", resize);
resize();

// Draw polygons + ellipses
const recById = {};
for (const r of DATA.records) {
  recById[r.id] = r;
  const color = palette[r.k] || "#888";
  const pts = r.boundary.map(c => c.join(",")).join(" ");
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  poly.setAttribute("class", "poly");
  poly.setAttribute("points", pts);
  poly.setAttribute("fill", color);
  poly.dataset.id = r.id;
  svg.appendChild(poly);
  for (const epath of r.ellipses) {
    const ep = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    ep.setAttribute("class", "ellipse e-" + r.id);
    ep.setAttribute("points", epath.map(p => p.join(",")).join(" "));
    ep.setAttribute("stroke", "#fff");
    svg.appendChild(ep);
  }
}

// Flip Y (image coords go down).
svg.style.transform = "scaleY(-1)";
// Then we need to un-flip text via CSS... but ellipses/polygons have no text. OK.

let lockedId = null;
svg.addEventListener("mouseover", e => {
  if (lockedId !== null) return;
  const t = e.target.closest(".poly");
  if (!t) return;
  showRecord(recById[+t.dataset.id], t);
});
svg.addEventListener("click", e => {
  const t = e.target.closest(".poly");
  if (!t) {
    if (lockedId !== null) {
      document.querySelector(".poly.locked")?.classList.remove("locked");
      document.querySelectorAll(".ellipse.show").forEach(n => n.classList.remove("show"));
      lockedId = null;
    }
    return;
  }
  document.querySelector(".poly.locked")?.classList.remove("locked");
  t.classList.add("locked");
  lockedId = +t.dataset.id;
  showRecord(recById[lockedId], t);
});

function showRecord(r, polyEl) {
  document.querySelectorAll(".ellipse.show").forEach(n => n.classList.remove("show"));
  document.querySelectorAll(".e-" + r.id).forEach(n => n.classList.add("show"));
  sideTitle.innerHTML = `polygon #${r.id} ` +
    `<span class="badge" style="background:${palette[r.k]||'#888'};color:#000;">k = ${r.k}</span>`;

  const gateRow = r.gated_as_convex
    ? `<span style="color:#7fdf7f">yes → skipped GMM</span>`
    : `<span style="color:#dd9090">no → ran GMM</span>`;

  side.innerHTML = `
    <div class="field"><span class="label">area</span><span class="value">${r.area} px²</span></div>
    <div class="field"><span class="label">solidity</span><span class="value">${r.solidity}</span></div>
    <div class="field"><span class="label">convex gate</span><span class="value">${gateRow}</span></div>
    <div class="field"><span class="label">k_max by area</span><span class="value">${r.k_max_by_area}</span></div>
    <div class="field"><span class="label">k (GMM requested)</span><span class="value">${r.k_requested}</span></div>
    <div class="field"><span class="label">k (geom achieved)</span><span class="value">${r.k}${r.k !== r.k_requested ? ' <span style="color:#dd9090">⚠ splits rejected</span>' : ''}</span></div>
    <div class="field"><span class="label">ellipses fit</span><span class="value">${r.ellipses.length}</span></div>
    <svg id="bicchart"></svg>
    <div class="legend">
      <span style="color:#7aa6e0">● raw BIC</span> &nbsp;
      <span style="color:#ff9a55">● BIC + EB penalty</span>
    </div>
  `;
  drawBic(r);
}

function drawBic(r) {
  const chart = document.getElementById("bicchart");
  if (!chart) return;
  const w = chart.clientWidth, h = 180;
  chart.setAttribute("viewBox", `0 0 ${w} ${h}`);
  chart.innerHTML = "";

  const raw = r.raw_bic, mod = r.mod_bic;
  if (!raw || raw.length < 2) {
    chart.innerHTML = `<text x="${w/2}" y="${h/2}" fill="#666" font-size="12"
      text-anchor="middle">no BIC curve (k=1 fast path)</text>`;
    return;
  }
  const vals = [...raw, ...mod].filter(v => v !== null && isFinite(v));
  const vmin = Math.min(...vals), vmax = Math.max(...vals);
  const pad = (vmax - vmin) * 0.1 || 1;
  const lo = vmin - pad, hi = vmax + pad;
  const pl = 38, pr = 12, pt = 14, pb = 28;
  const xs = raw.map((_, i) => pl + (w - pl - pr) * i / (raw.length - 1));
  const yOf = v => pt + (h - pt - pb) * (hi - v) / (hi - lo);

  function path(arr) {
    return arr.map((v, i) => `${i === 0 ? "M" : "L"} ${xs[i]} ${yOf(v)}`).join(" ");
  }
  const rawPath = path(raw);
  const modPath = path(mod);

  // Axes
  for (let i = 0; i < raw.length; ++i) {
    const x = xs[i];
    chart.innerHTML += `<line x1="${x}" y1="${pt}" x2="${x}" y2="${h-pb}" stroke="#222"/>`;
    chart.innerHTML += `<text x="${x}" y="${h-pb+14}" fill="#888" font-size="10"
      text-anchor="middle">k=${i+1}</text>`;
  }
  chart.innerHTML += `<text x="4" y="${pt+8}" fill="#666" font-size="10">${hi.toFixed(0)}</text>`;
  chart.innerHTML += `<text x="4" y="${h-pb}" fill="#666" font-size="10">${lo.toFixed(0)}</text>`;

  // Curves
  chart.innerHTML += `<path d="${rawPath}" stroke="#7aa6e0" fill="none" stroke-width="1.6"/>`;
  chart.innerHTML += `<path d="${modPath}" stroke="#ff9a55" fill="none" stroke-width="1.6"
                       stroke-dasharray="3 2"/>`;
  // Dots
  for (let i = 0; i < raw.length; ++i) {
    chart.innerHTML += `<circle cx="${xs[i]}" cy="${yOf(raw[i])}" r="2.5" fill="#7aa6e0"/>`;
    chart.innerHTML += `<circle cx="${xs[i]}" cy="${yOf(mod[i])}" r="2.5" fill="#ff9a55"/>`;
  }
  // Highlight picked k
  const kIdx = r.k - 1;
  if (kIdx >= 0 && kIdx < raw.length) {
    chart.innerHTML += `<line x1="${xs[kIdx]}" y1="${pt}" x2="${xs[kIdx]}" y2="${h-pb}"
      stroke="#fff" stroke-width="0.8" stroke-dasharray="2 3"/>`;
  }
}
</script>
</body>
</html>
"""


_HTML_TEMPLATE_RECURSIVE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>recursive ellipse decomp viewer</title>
<style>
  body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background: #1a1a1a; color: #ddd; }
  #wrap { display: flex; height: calc(100vh - 50px); }
  #stage { flex: 1; position: relative; overflow: hidden; background: #222; }
  #side { width: 380px; padding: 16px; background: #0f0f0f; overflow-y: auto;
          border-left: 1px solid #333; }
  #side h2 { margin: 0 0 8px; font-size: 14px; color: #aaa; font-weight: 500; }
  #side .pick-hint { color: #777; font-style: italic; font-size: 12px; }
  #side .field { display: flex; justify-content: space-between; margin: 4px 0;
                 font-size: 13px; font-variant-numeric: tabular-nums; }
  #side .field .label { color: #888; }
  #side .field .value { color: #eee; }
  #side .badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
                 font-size: 11px; font-weight: 600; }
  svg { display: block; }
  .poly { fill-opacity: 0.45; stroke: #000; stroke-width: 0.4px; cursor: pointer; }
  .poly:hover, .poly.locked { stroke: #fff; stroke-width: 1.4px; fill-opacity: 0.8; }
  .ellipse { fill: none; stroke: #fff; stroke-width: 1px; pointer-events: none; opacity: 0; }
  .ellipse.show { opacity: 0.9; }
  #header { padding: 12px 16px; background: #0a0a0a; border-bottom: 1px solid #333;
            font-size: 12px; color: #aaa; }
  #header b { color: #eee; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            vertical-align: middle; margin-right: 4px; }
  .tree { margin-top: 10px; font-size: 12px; font-family: ui-monospace, monospace; }
  .tree .row { padding: 2px 0; }
  .tree .act-split { color: #ff9a55; }
  .tree .act-emit_convex { color: #7fdf7f; }
  .tree .act-emit_no_split { color: #dd9090; }
  .tree .act-emit_max_depth { color: #dd9090; }
  .tree .why { color: #777; }
</style>
</head>
<body>
<div id="header"></div>
<div id="wrap">
  <div id="stage"><svg id="map"></svg></div>
  <div id="side">
    <h2 id="side-title">Hover a polygon</h2>
    <div id="side-body"><div class="pick-hint">Move over any nucleus to see its decomposition tree.</div></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const palette = DATA.palette;
const [bx0, by0, bx1, by1] = DATA.bbox;
const stage = document.getElementById("stage");
const svg = document.getElementById("map");
const side = document.getElementById("side-body");
const sideTitle = document.getElementById("side-title");

const header = document.getElementById("header");
const kSummary = Object.entries(DATA.summary.k_counts).sort((a,b)=>a[0]-b[0])
  .map(([k,c]) => `<span class="swatch" style="background:${palette[k]||'#888'}"></span>k=${k}: <b>${c}</b>`)
  .join(" &nbsp; ");
header.innerHTML = `<b>${DATA.title}</b> · ${DATA.summary.n_polys} polygons →
  <b>${DATA.summary.total_ellipses}</b> ellipses · prior median area
  <b>${DATA.prior.median.toFixed(0)}</b> px² · min frag area (${DATA.params.min_log_z}σ)
  <b>${DATA.prior.min_area_2sd.toFixed(0)}</b> px² · solidity ≥
  <b>${DATA.params.solidity_threshold}</b>
  &nbsp; · &nbsp; ${kSummary}`;

function resize() {
  const r = stage.getBoundingClientRect();
  svg.setAttribute("width", r.width);
  svg.setAttribute("height", r.height);
  const w = bx1 - bx0, h = by1 - by0;
  svg.setAttribute("viewBox", `${bx0} ${by0} ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}
window.addEventListener("resize", resize);
resize();

const recById = {};
for (const r of DATA.records) {
  recById[r.id] = r;
  const color = palette[r.k] || "#888";
  const pts = r.boundary.map(c => c.join(",")).join(" ");
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  poly.setAttribute("class", "poly");
  poly.setAttribute("points", pts);
  poly.setAttribute("fill", color);
  poly.dataset.id = r.id;
  svg.appendChild(poly);
  for (const epath of r.ellipses) {
    const ep = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    ep.setAttribute("class", "ellipse e-" + r.id);
    ep.setAttribute("points", epath.map(p => p.join(",")).join(" "));
    ep.setAttribute("stroke", "#fff");
    svg.appendChild(ep);
  }
}
svg.style.transform = "scaleY(-1)";

let lockedId = null;
svg.addEventListener("mouseover", e => {
  if (lockedId !== null) return;
  const t = e.target.closest(".poly");
  if (!t) return;
  showRecord(recById[+t.dataset.id]);
});
svg.addEventListener("click", e => {
  const t = e.target.closest(".poly");
  if (!t) {
    if (lockedId !== null) {
      document.querySelector(".poly.locked")?.classList.remove("locked");
      document.querySelectorAll(".ellipse.show").forEach(n => n.classList.remove("show"));
      lockedId = null;
    }
    return;
  }
  document.querySelector(".poly.locked")?.classList.remove("locked");
  t.classList.add("locked");
  lockedId = +t.dataset.id;
  showRecord(recById[lockedId]);
});

function showRecord(r) {
  document.querySelectorAll(".ellipse.show").forEach(n => n.classList.remove("show"));
  document.querySelectorAll(".e-" + r.id).forEach(n => n.classList.add("show"));
  sideTitle.innerHTML = `polygon #${r.id} ` +
    `<span class="badge" style="background:${palette[r.k]||'#888'};color:#000;">${r.k} ellipses</span>`;
  const stepRows = r.steps.map(s => {
    const indent = "  ".repeat(s.depth);
    let why = "";
    if (s.action === "split" && s.child_log_areas) {
      const a = Math.exp(s.child_log_areas[0]).toFixed(0);
      const b = Math.exp(s.child_log_areas[1]).toFixed(0);
      why = `<span class="why"> → children ${a}, ${b} px²</span>`;
    } else if (s.rejection_reason) {
      why = `<span class="why"> (${s.rejection_reason})</span>`;
    }
    return `<div class="row"><span class="act-${s.action}">${indent}d${s.depth} sol=${s.solidity} area=${s.area} · ${s.action}</span>${why}</div>`;
  }).join("");
  side.innerHTML = `
    <div class="field"><span class="label">root area</span><span class="value">${r.area} px²</span></div>
    <div class="field"><span class="label">root solidity</span><span class="value">${r.solidity}</span></div>
    <div class="field"><span class="label">ellipses</span><span class="value">${r.k}</span></div>
    <h2 style="margin-top:14px">decision tree</h2>
    <div class="tree">${stepRows || '<div class="pick-hint">no steps</div>'}</div>
  `;
}
</script>
</body>
</html>
"""
