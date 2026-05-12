"""
Decompose a Shapely Polygon (assumed to be the union of several overlapping
ellipses) into its constituent ellipses using concavity-point splitting +
cv2.fitEllipse.

Pipeline:
  1. Sample the polygon boundary at roughly uniform arc-length spacing.
  2. Find concavity points: boundary points that lie far from the convex hull
     (these are the "pinch" points where two ellipses meet).
  3. Pair concavity points to draw split chords that subdivide the polygon
     into convex-ish fragments. Pairing is done greedily by a cost function
     that prefers (a) chords that stay inside the polygon, (b) short chords,
     (c) chords whose endpoints face each other (boundary normals roughly
     opposing).
  4. Recursively split: if a fragment still has strong concavities, split it
     again.
  5. For each leaf fragment, fit an ellipse with cv2.fitEllipse.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split as shapely_split


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Ellipse:
    cx: float
    cy: float
    width: float   # full length of axis (not semi-axis), matching cv2 convention
    height: float
    angle_deg: float  # rotation in degrees, cv2 convention

    def as_cv(self):
        return ((self.cx, self.cy), (self.width, self.height), self.angle_deg)


# ---------------------------------------------------------------------------
# Boundary sampling
# ---------------------------------------------------------------------------

def sample_boundary(poly: Polygon, spacing: float) -> np.ndarray:
    """Sample the exterior ring at ~uniform arc-length spacing.
    Returns an (N,2) array; the ring is NOT closed (first point != last)."""
    ring = poly.exterior
    length = ring.length
    n = max(int(np.ceil(length / spacing)), 16)
    ds = np.linspace(0.0, length, n, endpoint=False)
    pts = np.array([ring.interpolate(d).coords[0] for d in ds])
    return pts


# ---------------------------------------------------------------------------
# Concavity detection
# ---------------------------------------------------------------------------

def concavity_depths(boundary: np.ndarray) -> np.ndarray:
    """For each boundary point, compute its distance to the convex hull
    of the boundary. Points strictly inside the hull get positive depth;
    points on the hull get 0."""
    hull_idx = cv2.convexHull(boundary.astype(np.float32),
                              returnPoints=False).flatten()
    hull_idx = np.sort(hull_idx)
    hull_pts = boundary[hull_idx]

    # For each boundary point, distance to the hull polygon (signed by inside/outside).
    # All boundary points are inside-or-on the hull, so distance is non-negative.
    hull_poly = Polygon(hull_pts)
    depths = np.array([hull_poly.exterior.distance(Point(p)) for p in boundary])
    # Points that ARE hull vertices get depth 0 exactly.
    depths[hull_idx] = 0.0
    return depths


def find_concavity_points(boundary: np.ndarray,
                          min_depth: float,
                          min_separation: int) -> list[int]:
    """Return boundary indices that are local maxima of concavity depth and
    deeper than min_depth. min_separation is in *index units* along the
    boundary."""
    depths = concavity_depths(boundary)
    n = len(depths)
    if depths.max() < min_depth:
        return []

    # Local maxima in a circular array.
    cand = []
    for i in range(n):
        if depths[i] < min_depth:
            continue
        # window
        is_max = True
        for k in range(1, min_separation + 1):
            if depths[(i - k) % n] > depths[i] or depths[(i + k) % n] > depths[i]:
                is_max = False
                break
        if is_max:
            cand.append((i, depths[i]))

    # Sort by depth (deepest first), then NMS by separation.
    cand.sort(key=lambda t: -t[1])
    kept: list[int] = []
    for i, _ in cand:
        ok = True
        for j in kept:
            d = min(abs(i - j), n - abs(i - j))
            if d < min_separation:
                ok = False
                break
        if ok:
            kept.append(i)
    return sorted(kept)


# ---------------------------------------------------------------------------
# Pairing concavities into split chords
# ---------------------------------------------------------------------------

def _inward_normal(boundary: np.ndarray, i: int) -> np.ndarray:
    n = len(boundary)
    prev_p = boundary[(i - 1) % n]
    next_p = boundary[(i + 1) % n]
    tangent = next_p - prev_p
    # Inward normal: rotate tangent +90deg if polygon is CCW.
    # We'll figure out orientation from polygon signed area.
    nrm = np.array([-tangent[1], tangent[0]])
    nl = np.linalg.norm(nrm)
    return nrm / nl if nl > 0 else nrm


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def pair_concavities(boundary: np.ndarray,
                     concavity_idx: list[int],
                     poly: Polygon) -> list[tuple[int, int]]:
    """Greedy pairing of concavity points into split chords.
    Cost = chord_length * (1 + angle_penalty); reject chords not contained
    in the polygon."""
    if len(concavity_idx) < 2:
        return []

    ccw = _signed_area(boundary) > 0
    sign = 1.0 if ccw else -1.0

    pts = boundary[concavity_idx]
    normals = np.array([sign * _inward_normal(boundary, i) for i in concavity_idx])

    pairs_cost = []
    for a in range(len(concavity_idx)):
        for b in range(a + 1, len(concavity_idx)):
            pa, pb = pts[a], pts[b]
            chord = LineString([pa, pb])
            # Chord must lie inside polygon (allow tiny tolerance via buffer).
            if not poly.buffer(1e-6).contains(chord):
                continue
            v = pb - pa
            vlen = np.linalg.norm(v)
            if vlen < 1e-9:
                continue
            u = v / vlen
            # Normals should oppose chord direction from each endpoint:
            # normals[a] points roughly toward pb (i.e. along +u),
            # normals[b] points roughly toward pa (i.e. along -u).
            align = 0.5 * (np.dot(normals[a], u) + np.dot(normals[b], -u))
            # align in [-1,1]; we want it close to 1.
            angle_pen = (1.0 - align)  # 0 (perfect) to 2 (worst)
            cost = vlen * (1.0 + angle_pen)
            pairs_cost.append((cost, a, b))

    pairs_cost.sort(key=lambda t: t[0])

    used = set()
    chords: list[tuple[int, int]] = []
    for _, a, b in pairs_cost:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        chords.append((concavity_idx[a], concavity_idx[b]))
    return chords


# ---------------------------------------------------------------------------
# Splitting the polygon along chords
# ---------------------------------------------------------------------------

def split_polygon(poly: Polygon, chords: list[tuple[np.ndarray, np.ndarray]]
                  ) -> list[Polygon]:
    """Split polygon along each chord (LineString). Returns list of pieces."""
    pieces = [poly]
    for p1, p2 in chords:
        line = LineString([p1, p2])
        new_pieces = []
        for piece in pieces:
            # Extend the line slightly to ensure clean intersection.
            v = np.array(p2) - np.array(p1)
            vlen = np.linalg.norm(v)
            if vlen < 1e-9:
                new_pieces.append(piece)
                continue
            u = v / vlen
            eps = max(piece.length * 1e-4, 1e-6)
            ext_line = LineString([np.array(p1) - eps * u,
                                   np.array(p2) + eps * u])
            try:
                result = shapely_split(piece, ext_line)
                geoms = list(result.geoms) if hasattr(result, 'geoms') else [result]
                # Filter to polygons only.
                geoms = [g for g in geoms if isinstance(g, Polygon) and g.area > 1e-9]
                if len(geoms) >= 2:
                    new_pieces.extend(geoms)
                else:
                    new_pieces.append(piece)
            except Exception:
                new_pieces.append(piece)
        pieces = new_pieces
    return pieces


# ---------------------------------------------------------------------------
# Ellipse fitting
# ---------------------------------------------------------------------------

def fit_ellipse_to_polygon(poly: Polygon, boundary_spacing: float) -> Ellipse | None:
    """Fit an ellipse to a polygon's boundary via cv2.fitEllipse."""
    pts = sample_boundary(poly, boundary_spacing)
    if len(pts) < 5:
        return None
    cv_pts = pts.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), ang = cv2.fitEllipse(cv_pts)
    return Ellipse(cx=cx, cy=cy, width=w, height=h, angle_deg=ang)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def decompose_polygon_to_ellipses(
    poly: Polygon,
    boundary_spacing: float | None = None,
    min_concavity_depth: float | None = None,
    min_separation_frac: float = 0.05,
    max_depth: int = 3,
) -> list[Ellipse]:
    """Decompose a (possibly compound) polygon into ellipses.

    Parameters
    ----------
    poly : shapely Polygon
        Input shape (may be the union of several overlapping ellipses).
    boundary_spacing : float, optional
        Arc-length spacing for boundary sampling. Defaults to ~1/200 of the
        boundary length.
    min_concavity_depth : float, optional
        How deep a concavity must be to count. Defaults to 5% of sqrt(area).
    min_separation_frac : float
        Minimum index separation between two concavity peaks, as a fraction
        of the boundary sample count.
    max_depth : int
        Maximum recursive splitting depth.
    """
    if boundary_spacing is None:
        boundary_spacing = max(poly.exterior.length / 200.0, 1.0)
    if min_concavity_depth is None:
        min_concavity_depth = 0.05 * np.sqrt(poly.area)

    ellipses: list[Ellipse] = []
    stack: list[tuple[Polygon, int]] = [(poly, 0)]

    while stack:
        piece, depth = stack.pop()
        boundary = sample_boundary(piece, boundary_spacing)
        if len(boundary) < 8 or depth >= max_depth:
            e = fit_ellipse_to_polygon(piece, boundary_spacing)
            if e is not None:
                ellipses.append(e)
            continue

        min_sep = max(int(min_separation_frac * len(boundary)), 2)
        cidx = find_concavity_points(boundary, min_concavity_depth, min_sep)

        if len(cidx) < 2:
            e = fit_ellipse_to_polygon(piece, boundary_spacing)
            if e is not None:
                ellipses.append(e)
            continue

        chord_idx_pairs = pair_concavities(boundary, cidx, piece)
        if not chord_idx_pairs:
            e = fit_ellipse_to_polygon(piece, boundary_spacing)
            if e is not None:
                ellipses.append(e)
            continue

        chord_pts = [(boundary[i], boundary[j]) for i, j in chord_idx_pairs]
        sub_pieces = split_polygon(piece, chord_pts)
        if len(sub_pieces) <= 1:
            e = fit_ellipse_to_polygon(piece, boundary_spacing)
            if e is not None:
                ellipses.append(e)
            continue

        for sp in sub_pieces:
            stack.append((sp, depth + 1))

    return ellipses
