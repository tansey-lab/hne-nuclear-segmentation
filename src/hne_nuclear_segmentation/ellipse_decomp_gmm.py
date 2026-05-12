"""GMM-guided ellipse decomposition.

Pipeline for a polygon that may be the union of overlapping ellipses:

  0. **Convexity gate.** If ``area / convex_hull_area`` exceeds a threshold,
     treat the polygon as a single ellipse and skip the GMM step entirely.
     Only clearly non-convex shapes proceed.
  1. **Rasterize** the polygon interior to a pixel point cloud.
  2. **Fit GMMs** for k = 1..k_max and pick the best k via the elbow of a
     modified BIC curve (kneed.KneeLocator). The modification is an
     **empirical-Bayes** penalty: each component's effective area
     (weight * polygon_area) is scored against a log-normal prior learned
     from the population of *convex* nuclei. This prevents BIC from adding
     components that explain elongated uniform density with implausibly
     small lobes.
  3. **Iteratively chord-split** the polygon, each round splitting the
     sub-piece whose boundary has the deepest concavity, until we have k
     leaves (or no further split is possible).
  4. **Fit** a cv2 ellipse to each leaf.

The chord-splitting primitives are reused from ``ellipse_decomp``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

from .ellipse_decomp import (
    Ellipse,
    sample_boundary,
    concavity_depths,
    find_concavity_points,
    pair_concavities,
    split_polygon,
    fit_ellipse_to_polygon,
)


# ---------------------------------------------------------------------------
# Convexity gate + empirical-Bayes area prior
# ---------------------------------------------------------------------------

DEFAULT_SOLIDITY_THRESHOLD = 0.90
# Threshold for *fitting* the area prior — keep it strict so the corpus of
# "true single nuclei" is uncontaminated. Different from the gate above,
# which can be more permissive.
DEFAULT_PRIOR_FIT_SOLIDITY = 0.95
# A polygon with area A can plausibly hold at most
#   k_max_by_area = floor(A / (MIN_AREA_FRAC * prior_median))
# nuclei, allowing 1 - MIN_AREA_FRAC overlap.
DEFAULT_MIN_AREA_FRAC = 0.7
DEFAULT_PRIOR_WEIGHT = 5.0


def solidity(poly: Polygon) -> float:
    """area / convex_hull.area. 1.0 = perfectly convex."""
    hull = poly.convex_hull
    if hull.area <= 0:
        return 1.0
    return float(poly.area / hull.area)


@dataclass
class AreaPrior:
    """Log-normal prior over single-nucleus area (in pixels^2)."""
    mu: float       # mean of log(area)
    sigma: float    # std of log(area)
    n: int          # how many convex shapes the prior was fit on

    def neg_log_pdf(self, area: float) -> float:
        """Negative log-pdf (in nats) for an area under the log-normal prior."""
        if area <= 0:
            return 1e6
        log_a = np.log(area)
        z = (log_a - self.mu) / self.sigma
        # log p(a) = -log(a) - log(sigma) - 0.5 log(2pi) - 0.5 z^2
        return float(log_a + np.log(self.sigma) + 0.5 * np.log(2 * np.pi)
                     + 0.5 * z * z)


def fit_area_prior(polys: Iterable[Polygon],
                   solidity_threshold: float = DEFAULT_PRIOR_FIT_SOLIDITY,
                   ) -> AreaPrior:
    """Fit a log-normal prior to the areas of polygons whose solidity is
    above ``solidity_threshold``. These are the "trustworthy single nuclei"."""
    areas = [p.area for p in polys
             if p.area > 0 and solidity(p) >= solidity_threshold]
    if len(areas) < 5:
        raise ValueError(
            f"Only {len(areas)} convex polygons (solidity>={solidity_threshold}) "
            "— not enough to fit an area prior.")
    log_a = np.log(np.array(areas))
    return AreaPrior(mu=float(log_a.mean()),
                     sigma=float(log_a.std(ddof=1)),
                     n=len(areas))


# ---------------------------------------------------------------------------
# Rasterize polygon interior to a pixel cloud
# ---------------------------------------------------------------------------

def polygon_pixels(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the polygon and return ``(pts, weights)``.

    ``pts``: (N, 2) pixel-center coords inside the polygon.

    ``weights``: per-pixel weight from the Euclidean distance transform, i.e.
    the distance from that pixel to the polygon boundary. Weights are
    normalized so their mean is 1 (this keeps the "effective sample size"
    used in BIC on the same scale as the unweighted version, so the
    complexity penalty stays comparable).

    Weighting by distance-to-boundary makes the input look like a real
    density: a single nucleus shows a unimodal peak at its center, two
    overlapping nuclei show a bimodal density with a valley along the
    pinch, etc. — the structure GMM is actually designed for.
    """
    minx, miny, maxx, maxy = poly.bounds
    x0, y0 = int(np.floor(minx)), int(np.floor(miny))
    x1, y1 = int(np.ceil(maxx)), int(np.ceil(maxy))
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    transform = Affine.translation(x0, y0) * Affine.scale(1, 1)
    mask = rasterize(
        [(poly, 1)], out_shape=(h, w), transform=transform,
        fill=0, all_touched=False, dtype=np.uint8,
    )
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        # Sub-pixel polygon: fall back to boundary samples, equal weight.
        pts = np.array(poly.exterior.coords)[:-1]
        return pts, np.ones(len(pts))

    # Distance transform; boundary pixels get weight ~0.5, interior peaks
    # at the medial axis of each lobe.
    dist = distance_transform_edt(mask)
    pts = np.column_stack([cols + x0 + 0.5, rows + y0 + 0.5])
    weights = dist[rows, cols].astype(float)
    # Replace any zero weights (shouldn't happen but be safe) with a small
    # epsilon so GMM doesn't ignore boundary pixels entirely.
    weights = np.maximum(weights, 0.25)
    weights *= len(weights) / weights.sum()  # mean -> 1
    return pts, weights


# ---------------------------------------------------------------------------
# BIC + kneed to pick k
# ---------------------------------------------------------------------------

_UPSAMPLE_SCALE = 4  # repetition factor for weighted-fit emulation


def _expand_by_weight(pts: np.ndarray, weights: np.ndarray,
                      scale: int = _UPSAMPLE_SCALE) -> np.ndarray:
    """Replicate each point by ``round(weight * scale)`` (min 1).

    Emulates ``sample_weight`` for GMMs whose ``fit`` doesn't support it
    (sklearn < ~1.9). Weights are assumed normalized to mean 1, so the
    expanded array has ~``scale * len(pts)`` rows.
    """
    counts = np.maximum(np.round(weights * scale).astype(int), 1)
    return np.repeat(pts, counts, axis=0)


def _weighted_bic(gm: GaussianMixture, pts: np.ndarray,
                  weights: np.ndarray) -> float:
    """BIC for a GMM fit on weight-expanded data, scored against the
    original weighted points.

    ``Σ w_i log p(x_i)`` is the weighted log-likelihood; with weights
    normalized to mean 1, ``Σ w_i = N`` so the complexity penalty uses
    ``log(N)`` exactly like the unweighted BIC. This keeps BIC values
    comparable to the prior penalty's natural scale.
    """
    log_probs = gm.score_samples(pts)
    log_lik = float((log_probs * weights).sum())
    n_eff = float(weights.sum())
    return -2.0 * log_lik + gm._n_parameters() * np.log(max(n_eff, 2.0))


def pick_k_by_bic(
    pts: np.ndarray,
    poly_area: float,
    k_max: int = 6,
    area_prior: AreaPrior | None = None,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    weights: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Fit GMM for k=1..k_max, return
    ``(best_k, raw_bic, modified_bic, k_max_by_area)``.

    Two regularizers on top of standard BIC:

    * **Area-cap (hard).** ``k_max_by_area = floor(poly_area / (min_area_frac
      * prior_median))``. We never consider k above this cap.
    * **Empirical-Bayes penalty (soft).** Each component's effective area
      (``weight_i * poly_area``) gets a log-normal NLL under the prior;
      total NLL is multiplied by ``prior_weight`` and added to BIC.

    The elbow is found with kneed.KneeLocator on the modified curve.
    """
    n = len(pts)
    k_max = min(k_max, max(1, n // 6))

    # Area-based hard cap.
    k_max_by_area = k_max
    if area_prior is not None:
        median = float(np.exp(area_prior.mu))
        k_max_by_area = max(1, int(np.floor(poly_area / (min_area_frac * median))))
        k_max = min(k_max, k_max_by_area)

    if k_max < 2:
        return 1, np.array([0.0]), np.array([0.0]), k_max_by_area

    ks = list(range(1, k_max + 1))
    raw_bics: list[float] = []
    mod_bics: list[float] = []
    for k in ks:
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type='full',
                random_state=0, reg_covar=1e-4,
                n_init=2, max_iter=200,
            )
            if weights is not None:
                expanded = _expand_by_weight(pts, weights)
                gm.fit(expanded)
                raw = _weighted_bic(gm, pts, weights)
            else:
                gm.fit(pts)
                raw = gm.bic(pts)
            raw_bics.append(raw)
            mod = raw
            if area_prior is not None:
                comp_areas = gm.weights_ * poly_area
                penalty = sum(area_prior.neg_log_pdf(a) for a in comp_areas)
                mod += prior_weight * penalty
            mod_bics.append(mod)
        except Exception:
            raw_bics.append(np.inf)
            mod_bics.append(np.inf)
    raw_arr = np.array(raw_bics)
    mod_arr = np.array(mod_bics)

    if mod_arr[0] <= mod_arr.min() + 1e-6:
        return 1, raw_arr, mod_arr, k_max_by_area

    try:
        kl = KneeLocator(
            ks, mod_arr, curve='convex', direction='decreasing',
            S=1.0, interp_method='interp1d',
        )
        knee = kl.knee
    except Exception:
        knee = None

    if knee is None:
        knee = int(ks[int(np.argmin(mod_arr))])
    return int(knee), raw_arr, mod_arr, k_max_by_area


# ---------------------------------------------------------------------------
# Iterative single-chord splitting toward a target piece count
# ---------------------------------------------------------------------------

def _piece_concavity_score(piece: Polygon, boundary_spacing: float) -> float:
    bnd = sample_boundary(piece, boundary_spacing)
    if len(bnd) < 8:
        return 0.0
    return float(concavity_depths(bnd).max())


def _split_once(piece: Polygon, boundary_spacing: float,
                min_piece_area: float = 0.0,
                min_separation_frac: float = 0.10) -> list[Polygon]:
    """Make ONE chord split on a piece using the best concavity pair.

    Tries chords in greedy-best order and accepts the first one that
    produces exactly two pieces both with area >= ``min_piece_area``.
    Returns ``[piece]`` if no acceptable chord exists.
    """
    bnd = sample_boundary(piece, boundary_spacing)
    if len(bnd) < 8:
        return [piece]
    if concavity_depths(bnd).max() < 1e-6:
        return [piece]

    min_depth = max(0.5, 0.02 * np.sqrt(piece.area))
    min_sep = max(int(min_separation_frac * len(bnd)), 2)
    cidx = find_concavity_points(bnd, min_depth, min_sep)
    if len(cidx) < 2:
        return [piece]

    chord_idx_pairs = pair_concavities(bnd, cidx, piece)
    if not chord_idx_pairs:
        return [piece]

    for i, j in chord_idx_pairs:
        sub = split_polygon(piece, [(bnd[i], bnd[j])])
        if len(sub) < 2:
            continue
        if min_piece_area > 0 and min(s.area for s in sub) < min_piece_area:
            continue
        return sub
    return [piece]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

@dataclass
class DecompDiagnostics:
    """Per-polygon diagnostics from ``decompose_with_gmm_verbose``."""
    solidity: float
    area: float
    gated_as_convex: bool
    k_picked: int          # ellipses actually produced (=k_requested when geometry cooperates)
    k_requested: int       # k chosen by (modified) BIC
    k_max_by_area: int
    raw_bic: np.ndarray
    modified_bic: np.ndarray


def decompose_with_gmm(
    poly: Polygon,
    k_max: int = 6,
    area_prior: AreaPrior | None = None,
    solidity_threshold: float = DEFAULT_SOLIDITY_THRESHOLD,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
) -> tuple[int, list[Ellipse]]:
    """Pick k via (modified) BIC + kneed, then iteratively chord-split."""
    k, ells, _ = decompose_with_gmm_verbose(
        poly, k_max=k_max, area_prior=area_prior,
        solidity_threshold=solidity_threshold, prior_weight=prior_weight,
        min_area_frac=min_area_frac,
    )
    return k, ells


def decompose_with_gmm_verbose(
    poly: Polygon,
    k_max: int = 6,
    area_prior: AreaPrior | None = None,
    solidity_threshold: float = DEFAULT_SOLIDITY_THRESHOLD,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
) -> tuple[int, list[Ellipse], DecompDiagnostics]:
    """Like ``decompose_with_gmm`` but also returns ``DecompDiagnostics``.

    Use this when building visualizations / debugging — the diagnostics
    object carries the BIC curves and gating decisions for the polygon.
    """
    sol = solidity(poly)
    pts, weights = polygon_pixels(poly)
    boundary_spacing = max(poly.exterior.length / 200.0, 1.0)

    if len(pts) < 10:
        e = fit_ellipse_to_polygon(poly, boundary_spacing)
        diag = DecompDiagnostics(
            solidity=sol, area=float(poly.area), gated_as_convex=False,
            k_picked=1, k_requested=1, k_max_by_area=1,
            raw_bic=np.array([0.0]), modified_bic=np.array([0.0]),
        )
        return 1, ([e] if e is not None else []), diag

    # Convexity gate.
    if sol >= solidity_threshold:
        e = fit_ellipse_to_polygon(poly, boundary_spacing)
        diag = DecompDiagnostics(
            solidity=sol, area=float(poly.area), gated_as_convex=True,
            k_picked=1, k_requested=1, k_max_by_area=1,
            raw_bic=np.array([0.0]), modified_bic=np.array([0.0]),
        )
        return 1, ([e] if e is not None else []), diag

    k, raw_bic, mod_bic, k_max_by_area = pick_k_by_bic(
        pts, poly_area=float(poly.area), k_max=k_max,
        area_prior=area_prior, prior_weight=prior_weight,
        min_area_frac=min_area_frac, weights=weights,
    )

    # Minimum area for any chord-split piece: same physical reasoning as
    # k_max_by_area — a sub-piece below this is implausibly small for a
    # nucleus.
    min_piece_area = (min_area_frac * float(np.exp(area_prior.mu))
                      if area_prior is not None else 0.0)

    pieces: list[Polygon] = [poly]
    safety = 0
    while len(pieces) < k and safety < k * 3:
        safety += 1
        scores = [_piece_concavity_score(p, boundary_spacing) for p in pieces]
        order = np.argsort(scores)[::-1]
        if scores[order[0]] < 1.0:
            break
        progressed = False
        for idx in order:
            sub = _split_once(pieces[idx], boundary_spacing,
                              min_piece_area=min_piece_area)
            if len(sub) >= 2:
                pieces = pieces[:idx] + sub + pieces[idx + 1:]
                progressed = True
                break
        if not progressed:
            break

    ellipses: list[Ellipse] = []
    for p in pieces:
        e = fit_ellipse_to_polygon(p, max(p.exterior.length / 200.0, 1.0))
        if e is None:
            continue
        # Reject grossly over-sized fits: an ellipse whose area is more than
        # ~1.6x its source polygon is fitting an arc/elongated shape badly
        # and would only confuse downstream consumers.
        ell_area = np.pi * (e.width / 2.0) * (e.height / 2.0)
        if ell_area > 1.6 * p.area:
            continue
        ellipses.append(e)

    # Report the k we actually achieved geometrically, not what GMM asked for.
    k_achieved = max(len(ellipses), 1) if ellipses else 0
    diag = DecompDiagnostics(
        solidity=sol, area=float(poly.area), gated_as_convex=False,
        k_picked=k_achieved, k_requested=k, k_max_by_area=k_max_by_area,
        raw_bic=raw_bic, modified_bic=mod_bic,
    )
    return k_achieved, ellipses, diag
