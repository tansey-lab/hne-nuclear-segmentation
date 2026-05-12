"""Recursive, solidity-gated chord-split decomposition.

Algorithm
---------
::

    decompose(poly):
        if solidity(poly) >= threshold:
            emit fit_ellipse(poly); return

        try the best concavity-chord split:
            if no usable chord, or either fragment is more than
            `min_log_z` * sigma below the prior log-area mean
            (i.e. an implausibly small "nucleus"):
                emit fit_ellipse(poly); return

        decompose(fragment_a)
        decompose(fragment_b)

This relies only on the chord-splitting primitives from ``ellipse_decomp``
plus the population-level log-normal ``AreaPrior`` defined in
``ellipse_decomp_gmm``. No mixture models involved.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Polygon

from .ellipse_decomp import (
    Ellipse,
    sample_boundary,
    concavity_depths,
    find_concavity_points,
    pair_concavities,
    split_polygon,
    fit_ellipse_to_polygon,
)
from .ellipse_decomp_gmm import AreaPrior, solidity


DEFAULT_SOLIDITY_THRESHOLD = 0.92
# Fragments whose log-area falls more than this many sigmas below the prior
# mean are treated as "not a nucleus" → that split is rejected.
DEFAULT_MIN_LOG_Z = -2.0
DEFAULT_MAX_DEPTH = 4


@dataclass
class SplitStep:
    """One node in the recursion tree."""
    depth: int
    solidity: float
    area: float
    action: str            # 'emit_convex' | 'emit_no_split' | 'emit_max_depth' | 'split'
    rejection_reason: str | None = None
    # If split: log-areas of the two children.
    child_log_areas: tuple[float, float] | None = None


@dataclass
class RecursiveDecompDiag:
    """Diagnostics for one top-level polygon."""
    solidity: float
    area: float
    steps: list[SplitStep] = field(default_factory=list)
    n_ellipses: int = 0


def _best_chord_split(
    piece: Polygon,
    prior: AreaPrior,
    min_log_z: float,
    min_separation_frac: float = 0.10,
) -> tuple[Polygon, Polygon, str] | tuple[None, None, str]:
    """Try to split ``piece`` along its best concavity chord.

    Considers chords between **all pairs** of detected concavity points
    (not just the greedy pairing). Among chords that lie inside the
    polygon and produce two fragments both above the min-area threshold,
    we pick the one with the most balanced area split — that gives the
    next recursion level the best chance of finding further valid
    splits, instead of peeling off slivers.

    Returns ``(child_a, child_b, "ok")`` on success, or
    ``(None, None, reason)`` if no acceptable chord was found.
    """
    from shapely.geometry import LineString

    boundary_spacing = max(piece.exterior.length / 200.0, 1.0)
    bnd = sample_boundary(piece, boundary_spacing)
    if len(bnd) < 8:
        return None, None, "boundary too small"

    depths = concavity_depths(bnd)
    if depths.max() < 1.0:
        return None, None, "no concavity"

    min_depth = max(0.5, 0.02 * np.sqrt(piece.area))
    min_sep = max(int(min_separation_frac * len(bnd)), 2)
    cidx = find_concavity_points(bnd, min_depth, min_sep)
    if len(cidx) < 2:
        return None, None, "fewer than 2 concavity points"

    min_log_area = prior.mu + min_log_z * prior.sigma
    poly_buf = piece.buffer(1e-6)

    candidates: list[tuple[float, Polygon, Polygon]] = []
    rejected_for_size = False
    rejected_outside = False
    for ii in range(len(cidx)):
        for jj in range(ii + 1, len(cidx)):
            i, j = cidx[ii], cidx[jj]
            chord = LineString([bnd[i], bnd[j]])
            if not poly_buf.contains(chord):
                rejected_outside = True
                continue
            sub = split_polygon(piece, [(bnd[i], bnd[j])])
            if len(sub) != 2:
                continue
            a, b = sub
            if a.area <= 0 or b.area <= 0:
                continue
            if np.log(a.area) < min_log_area or np.log(b.area) < min_log_area:
                rejected_for_size = True
                continue
            # Score: balance is min/max area; higher is better. Weight by
            # combined depth so deep pinches beat shallow boundary jitter
            # at similar balance.
            balance = min(a.area, b.area) / max(a.area, b.area)
            depth_bonus = float(depths[i] + depths[j]) / (2.0 * depths.max())
            score = balance + 0.25 * depth_bonus
            candidates.append((score, a, b))

    if not candidates:
        if rejected_for_size:
            return None, None, "all candidate splits produced an implausibly-small fragment"
        if rejected_outside:
            return None, None, "no chord stays inside polygon"
        return None, None, "no valid chord"

    candidates.sort(key=lambda t: -t[0])
    _, a, b = candidates[0]
    return a, b, "ok"


def decompose_polygon_recursive(
    poly: Polygon,
    area_prior: AreaPrior,
    solidity_threshold: float = DEFAULT_SOLIDITY_THRESHOLD,
    min_log_z: float = DEFAULT_MIN_LOG_Z,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[Ellipse], RecursiveDecompDiag]:
    """Recursively decompose a polygon into ellipses.

    Returns ``(ellipses, diagnostics)``.
    """
    diag = RecursiveDecompDiag(
        solidity=solidity(poly), area=float(poly.area),
    )

    def emit(piece: Polygon) -> list[Ellipse]:
        boundary_spacing = max(piece.exterior.length / 200.0, 1.0)
        e = fit_ellipse_to_polygon(piece, boundary_spacing)
        return [e] if e is not None else []

    def recurse(piece: Polygon, depth: int) -> list[Ellipse]:
        sol = solidity(piece)
        if sol >= solidity_threshold:
            diag.steps.append(SplitStep(
                depth=depth, solidity=sol, area=float(piece.area),
                action="emit_convex",
            ))
            return emit(piece)
        if depth >= max_depth:
            diag.steps.append(SplitStep(
                depth=depth, solidity=sol, area=float(piece.area),
                action="emit_max_depth",
            ))
            return emit(piece)

        a, b, reason = _best_chord_split(piece, area_prior, min_log_z)
        if a is None:
            diag.steps.append(SplitStep(
                depth=depth, solidity=sol, area=float(piece.area),
                action="emit_no_split", rejection_reason=reason,
            ))
            return emit(piece)

        diag.steps.append(SplitStep(
            depth=depth, solidity=sol, area=float(piece.area),
            action="split",
            child_log_areas=(float(np.log(a.area)), float(np.log(b.area))),
        ))
        return recurse(a, depth + 1) + recurse(b, depth + 1)

    ellipses = recurse(poly, 0)
    diag.n_ellipses = len(ellipses)
    return ellipses, diag
