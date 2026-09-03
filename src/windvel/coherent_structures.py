#!/usr/bin/env python3
"""Coherent updraft and downdraft detection.

Two definitions, both computed so they can be compared rather than chosen blind:

  threshold    connected gates where |w| exceeds a fixed speed, kept if their
               physical AREA reaches a floor. Area, not a gate count: in an RHI
               the rays diverge, so 1 degree spans 87 m at 5 km and 520 m at
               30 km, and a fixed gate count is a sixth of the physical area
               near the radar that it is far out.

  persistence  merge-tree segmentation of w/sigma. Sweeping down from the
               highest value, each local maximum starts a component; where two
               components meet is a saddle, and the weaker one dies there. A
               core survives only if it stands persistence_min sigma ABOVE that
               saddle and its outline is drawn at the saddle level, so an 
               object's extent is set by where it stops being distinct from 
               its neighbour.

               sigma is the per-gate retrieval error, so the same 8 m/s is
               4 sigma near vertical and 0.2 sigma at 5 degrees elevation.

References
----------
Edelsbrunner, Letscher, and Zomorodian (2002): Topological persistence and
simplification, Discrete Comput. Geom., 28, 511-533.
Chazal, F., L. J. Guibas, S. Y. Oudot, and P. Skraba (2013): Persistence-based
clustering in Riemannian manifolds, JACM, 60(6) -- the ToMATo algorithm
followed here.
Grimaud, M. (1992) / Vincent, L. (1993): the same quantity in mathematical
morphology, under the names dynamics and h-maxima.
Rosolowsky, E. W., et al. (2008): Structural analysis of molecular clouds:
Dendrograms, ApJ, 679, 1338 -- astrodendro's min_delta is this persistence
cut and min_npix the area floor.
Sousbie, T. (2011): The persistent cosmic web and its filamentary structure,
MNRAS, 414, 350 -- persistence filtering of noisy fields with the threshold
in units of the noise sigma, as here.


Pure array in, array out: nothing here imports the retrieval.

Connectivity is 8-connected THROUGHOUT (a gate touches its diagonal
neighbours): the two detectors are meant to differ only in their definition
of a structure (contrast vs speed), the same reason their area floors match,
so connectivity must not be a second, silent difference between them.

No parameter defaults: every threshold is passed explicitly by the caller
from config, so there is one copy of each constant. `sigma=None` in the core
functions is a mode switch (rank by |w| instead of |w|/sigma), not a
threshold.
"""
import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

__all__ = [
    "EIGHT_CONNECTED",
    "TRUNC_BOT",
    "TRUNC_FAR",
    "TRUNC_GAP",
    "TRUNC_NEAR",
    "TRUNC_TOP",
    "cell_area_km2",
    "object_core",
    "object_is_usable",
    "object_truncation",
    "persistence",
    "threshold",
]

# The one connectivity, for every labelling, dilation and cleanup pass here.
EIGHT_CONNECTED = np.ones((3, 3), bool)


def cell_area_km2(dk, ak):
    """Area of each gate from the geometry, not from a pixel count.

    Parameters
    ----------
    dk, ak : numpy.ndarray, shape (nrays, ngates)
        Horizontal distance and altitude of each gate, km.

    Returns
    -------
    numpy.ndarray, shape (nrays, ngates)
        Area of each gate, km^2.

    Notes
    -----
    In an RHI the mesh is curvilinear: spacing along the beam is fixed but
    rays diverge, so 1 deg is ~87 m at 5 km and ~520 m at 30 km. The area is
    the magnitude of the Jacobian of (distance, altitude) with respect to
    (ray, gate) index,

        A = | d(dist)/d(ray) * d(alt)/d(gate) - d(dist)/d(gate) * d(alt)/d(ray) |

    with the partials from `numpy.gradient`.
    """
    dd_r, dd_g = np.gradient(np.asarray(dk, float))
    da_r, da_g = np.gradient(np.asarray(ak, float))
    return np.abs(dd_r * da_g - dd_g * da_r)



# --------------------------------------------------------------- threshold
def threshold(w, valid, dk, ak, w_min, min_area_km2):
    """Connected |w| above a fixed speed, kept if large enough in AREA.

    Parameters
    ----------
    w : numpy.ndarray, shape (nrays, ngates)
        Vertical velocity, m/s, NaN where not retrieved.
    valid : numpy.ndarray of bool
        Gates eligible for detection.
    dk, ak : numpy.ndarray
        Gate distance and altitude, km (see `cell_area_km2`).
    w_min : float
        Speed a gate must exceed, m/s.
    min_area_km2 : float
        Area floor for a connected component, km^2.

    Returns
    -------
    up, down : numpy.ndarray of int32
        Labels, numbered from 1 within each sign; 0 is no object.

    Notes
    -----
    Components are 8-connected. Components below the area floor are dropped
    and the count logged.
    """
    base = valid & np.isfinite(w)
    area = cell_area_km2(dk, ak)
    out = []
    n_small = 0
    for sign in (+1, -1):
        m = base & ((w > w_min) if sign > 0 else (w < -w_min))
        lab, n = ndimage.label(m, structure=EIGHT_CONNECTED)
        keep, nxt = np.zeros_like(lab), 0
        for i in range(1, n + 1):
            k = lab == i
            if float(area[k].sum()) < min_area_km2:
                n_small += 1
                continue
            nxt += 1
            keep[k] = nxt
        out.append(keep)
    if n_small:
        logger.debug('threshold detector: %d component(s) below %g km^2 '
                     'dropped', n_small, min_area_km2)
    return out[0], out[1]


# ------------------------------------------------------------- persistence
def _merge_tree(f, valid):
    """Superlevel-set merge tree by union-find.

    Sweeping down from the highest value, each local maximum starts a component
    and every pixel joins the component it first touches. Where a pixel connects
    two components that is a saddle: the younger branch (lower peak) dies there,
    and its persistence is peak minus saddle. NaN and invalid pixels are never
    traversed, so nothing merges across an echo-free gap.

    Returns (joined, peak, death, parent) -- joined is the component each pixel
    first joined; the rest are keyed by component id.
    """
    ny, nx = f.shape
    ok = valid & np.isfinite(f)
    order = np.argsort(np.where(ok, f, -np.inf), axis=None)[::-1]
    order = order[: int(ok.sum())]

    joined = np.full(f.size, -1, np.int32)
    uf, peak, death, parent = [], [], [], []

    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        return a

    flat = f.ravel()
    for p in order:
        y, x = divmod(int(p), nx)
        roots = set()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),      # 8-connectivity,
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):   # like the rest
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx:
                q = yy * nx + xx
                if joined[q] >= 0:
                    roots.add(find(joined[q]))
        if not roots:
            c = len(uf)
            uf.append(c); peak.append(flat[p]); death.append(-np.inf); parent.append(-1)
            joined[p] = c
            continue
        roots = sorted(roots, key=lambda r: -peak[r])
        keep = roots[0]
        for r in roots[1:]:                                    # saddle
            death[r] = flat[p]
            parent[r] = keep
            uf[r] = keep
        joined[p] = keep
    return (joined.reshape(f.shape), np.array(peak), np.array(death),
            np.array(parent))


def _largest_connected(k, f):
    """The piece of a labelled region that actually contains its peak.

    The merge tree assigns every pixel to the component it first joined, but the
    drawn extent then masks that assignment at a level and nothing checks that
    what survives is still contiguous. Pixels can keep the label while the
    pixels linking them to the core fall below the cut, so an object is drawn in
    pieces scattered across the field.
    """
    lab, n = ndimage.label(k, structure=EIGHT_CONNECTED)
    if n <= 1:
        return k
    vals = np.where(k, f, -np.inf)
    return lab == lab[np.unravel_index(np.argmax(vals), vals.shape)]


def _surrounded(child, parent):
    """Does `parent` lie on every side of `child`?

    This is what separates a core INSIDE another from a core BESIDE it, and the
    two need separating: an enclosed maximum is a dip within one structure,
    while an adjacent one is a boundary between two, and only the first should
    merge.

    Look outward from the child in the four directions and ask whether
    the parent is there. A nested core has its parent above, below, left and
    right; a core lying alongside another has it on one side only.

    Each direction is tested over the BAND spanning the child, not along a
    single ray through its centre. 
    """
    ys, xs = np.where(child)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sides = (parent[:y0, x0:x1 + 1].any(), parent[y1 + 1:, x0:x1 + 1].any(),
             parent[y0:y1 + 1, :x0].any(), parent[y0:y1 + 1, x1 + 1:].any())
    return all(sides)


def _merge_enclosed(keep, f):
    """Absorb objects that lie wholly INSIDE another into their encloser.

    A core beside another, the two meeting at a saddle, is a genuine split: two
    structures with a boundary between them. A core enclosed by another's
    territory is not -- the saddle there is a dip within one structure, not a
    border between two. 

    A child merges only if its parent is on all four sides of it; see
    _surrounded. Sharing a long boundary is NOT enough, or two cores lying side
    by side would merge as well, and that split is worth keeping.
    """
    ids = [i for i in np.unique(keep) if i > 0]
    if len(ids) < 2:
        return keep
    out = keep.copy()
    # smallest first: an object absorbed into its parent takes its own children
    for i in sorted(ids, key=lambda j: int((keep == j).sum())):
        k = out == i
        if not k.any():
            continue
        ring = ndimage.binary_dilation(k, EIGHT_CONNECTED) & ~k
        nb = out[ring]
        nb = nb[nb > 0]
        if not nb.size:
            continue
        parent = int(np.bincount(nb).argmax())
        if parent == i:
            continue
        if _surrounded(k, out == parent):
            out[k] = parent
    return out



def persistence(w, valid, dk, ak, sigma, persistence_min,
                floor_sigma, min_area_km2):
    """Merge-tree segmentation of w/sigma, boundaries at saddles.

    Parameters
    ----------
    w : numpy.ndarray, shape (nrays, ngates)
        Vertical velocity, m/s, NaN where not retrieved.
    valid : numpy.ndarray of bool
        Gates eligible for detection.
    dk, ak : numpy.ndarray
        Gate distance and altitude, km (see `cell_area_km2`).
    sigma : numpy.ndarray
        Per-gate 1-sigma error on ``w``, m/s.
    persistence_min : float
        A core survives if its peak stands this many sigma above the saddle
        where it merges into a stronger neighbour.
    floor_sigma : float
        Lowest level, in sigma, at which any object is outlined.
    min_area_km2 : float
        Area floor for a finished object, km^2.

    Returns
    -------
    up, down : numpy.ndarray of int32
        Labels, numbered from 1 within each sign; 0 is no object.

    Notes
    -----
    The field segmented is S = w / sigma, so the same 8 m/s is 4 sigma near
    vertical and 0.2 sigma at 5 degrees elevation. Sweeping down from the
    highest value of S, each local maximum starts a component; where two
    components meet is a saddle, and the weaker one dies there with
    persistence = peak - saddle (Edelsbrunner et al. 2002; the ToMATo
    clustering of Chazal et al. 2013). A surviving core is outlined at the
    level max(saddle, floor_sigma), reduced to the connected piece holding
    its peak, and cores lying wholly inside another are absorbed into it.
    Area is judged last, on the finished objects. Components are
    8-connected. Dropped objects are counted and logged.

    References
    ----------
    Edelsbrunner, Letscher & Zomorodian (2002), Discrete Comput. Geom. 28, 511.
    Chazal, Guibas, Oudot & Skraba (2013), JACM 60(6).
    """
    S = np.asarray(w, float) / np.asarray(sigma, float)
    area = cell_area_km2(dk, ak)
    base = valid & np.isfinite(S)

    out = []
    n_faint = n_empty = n_small = 0
    for sign in (+1, -1):
        f = S if sign > 0 else -S
        joined, peak, death, parent = _merge_tree(f, base)
        if not peak.size:
            out.append(np.zeros(f.shape, np.int32))
            continue
        d = np.where(np.isfinite(death), death, floor_sigma)
        pers = peak - d

        # a component is absorbed by its parent unless it stands far enough
        # above the saddle where it merges
        surv = pers >= persistence_min
        target = np.arange(peak.size)
        for c in np.argsort(-peak):                    # parents before children
            if not surv[c] and parent[c] >= 0:
                target[c] = target[parent[c]]
        lab = np.where(joined >= 0, target[np.clip(joined, 0, None)], -1)

        keep, nxt = np.zeros(f.shape, np.int32), 0
        for c in np.unique(lab[lab >= 0]):
            if not surv[c]:
                n_faint += 1
                continue
            # outer boundary: never below the level where this object stops
            # being distinct, and never below the physical floor
            cut = max(d[c], floor_sigma)
            k = (lab == c) & (f >= cut) & base
            if not k.any():
                n_empty += 1
                continue
            k = _largest_connected(k, f)
            nxt += 1
            keep[k] = nxt

        keep = _merge_enclosed(keep, f)

        # area is judged last, on what each object finally is: absorbing an
        # enclosed core changes the parent's extent, and dropping fragments
        # changes everyone's
        final, nxt = np.zeros(f.shape, np.int32), 0
        for c in [i for i in np.unique(keep) if i > 0]:
            k = keep == c
            if float(area[k].sum()) < min_area_km2:
                n_small += 1
                continue
            nxt += 1
            final[k] = nxt
        keep = final
        out.append(keep)
    if n_faint or n_empty or n_small:
        logger.debug('persistence detector: %d component(s) absorbed below '
                     '%g sigma, %d empty at their cut, %d below %g km^2',
                     n_faint, persistence_min, n_empty, n_small, min_area_km2)
    return out[0], out[1]


# ------------------------------------------------------------- usability
def object_core(k, w, core_peak_fraction, core_min_gates, sigma=None):
    """The object's core: the connected patch of strongest gates around its peak.

    The core always EXISTS. A persistence object stands at least
    persistence_min sigma above the saddle where it merges, so it has a peak by
    construction; a threshold object is a connected run above a wind speed, so it
    has one too. 

    Starting at core_peak_fraction of the peak, the level is lowered until the
    connected component containing the peak gate reaches core_min_gates. A

    If the whole object is smaller than core_min_gates, the object is its core.

    Ranked by |w| / sigma when sigma is given, not by |w|. An object detected on
    significance must have its core located on significance too, or the two
    disagree. Ranking by speed put the core on the artefact and condemned the 
    real structure.
    """
    field = np.abs(w) if sigma is None else np.abs(w) / np.asarray(sigma, float)
    aw = np.where(k, field, np.nan)
    if not np.isfinite(aw).any():
        return np.zeros_like(k, dtype=bool)
    pk = np.unravel_index(np.nanargmax(aw), aw.shape)
    peak = aw[pk]
    n_obj = int(np.isfinite(aw).sum())
    if n_obj <= core_min_gates:
        return k & np.isfinite(aw)

    core = None
    for frac in np.linspace(core_peak_fraction, 0.0, 33):
        cand = k & (aw >= frac * peak)
        cl, _ = ndimage.label(cand, structure=EIGHT_CONNECTED)
        core = cl == cl[pk]
        if int(core.sum()) >= core_min_gates:
            break
    return core


def object_is_usable(lab, label, w, area, usable, min_area_fraction,
                     core_peak_fraction, core_min_gates, sigma=None):
    """Is one labelled object measured well enough to quote?

    Two tests, and both must pass, because their failure modes are opposite.

      area      at least min_area_fraction of the object's AREA lies inside the
                usable domain. An object whose core sits just inside the range
                limit but whose remaining bulk sprawls beyond it is built mostly
                from broad-beam gates, and every property quoted from it -- area,
                depth, mean w -- would be dominated by the untrusted part.

      core      the object's core lies inside. The core is what defines the
                object: its identity, its intensity, and for the persistence
                detector its very existence rest on it. An object whose core is
                outside is anchored on data we do not trust, however much of its
                area happens to fall inside.

    The core is grown to core_min_gates around the peak rather than cut at a
    fixed level, and is located by |w| / sigma rather than by wind speed when
    sigma is given (see object_core), so a small core is never itself a reason to
    reject: the object passed a detector that required a peak, so a peak is
    there. A majority of the core's area must be inside, rather than all of it,
    or the test becomes hostage to a single gate at a different threshold.

    Parameters
    ----------
    lab : numpy.ndarray of int
        Label field from `threshold` or `persistence`.
    label : int
        The object to judge.
    w : numpy.ndarray
        Vertical velocity, m/s.
    area : numpy.ndarray
        Gate areas, km^2 (see `cell_area_km2`).
    usable : numpy.ndarray of bool
        The trusted domain.
    min_area_fraction : float
        Fraction of the object's area that must lie inside ``usable``.
    core_peak_fraction, core_min_gates : float, int
        Passed to `object_core`.
    sigma : numpy.ndarray, optional
        Rank the core by |w|/sigma when given.

    Returns
    -------
    usable : bool
    diagnostics : dict
        ``area_fraction``, ``core_gates``, ``core_inside``, ``core_fraction``.
    """
    k = lab == label
    a_tot = float(area[k].sum())
    if a_tot <= 0:
        return False, dict(area_fraction=np.nan, core_gates=0, core_inside=False)
    frac = float(area[k & usable].sum()) / a_tot

    core = object_core(k, w, core_peak_fraction, core_min_gates, sigma=sigma)
    n_core = int(core.sum())
    a_core = float(area[core].sum())
    core_inside = a_core > 0 and float(area[core & usable].sum()) > 0.5 * a_core

    ok = bool(frac >= min_area_fraction and core_inside)
    return ok, dict(area_fraction=frac, core_gates=n_core,
                    core_inside=bool(core_inside),
                    core_fraction=(float(area[core & usable].sum()) / a_core
                                   if a_core > 0 else np.nan))


# ------------------------------------------------------------- truncation
# Which edge of the SAMPLING an object runs into. Bits, because an object can
# reach more than one.
TRUNC_TOP  = 1   # the highest-elevation ray of the sweep
TRUNC_BOT  = 2   # the lowest-elevation ray
TRUNC_NEAR = 4   # the first range gate
TRUNC_FAR  = 8   # the last range gate
TRUNC_GAP  = 16  # an unretrieved hole INSIDE the echo, bracketed by data on
#                  both sides along the beam. Holes that open out to cloud top
#                  are NOT counted: the object ends there because the cloud
#                  does, and near-top bins routinely fail to anchor, so flagging
#                  them would mark most of the anvil for no informative reason.


def object_truncation(k, elev, hole=None):
    """Does this object run off the edge of what the radar sampled?

    An object touching the top of the scan is real but INCOMPLETELY OBSERVED:
    it continues above the highest beam, and its area, depth and peak are all
    lower bounds. 

    Recorded separately from usability, and never folded into it, because the
    two answer different questions: usable asks whether the measurement can be
    trusted, truncated asks whether the object was wholly seen. An object can be
    either, both or neither, and a count of updrafts wants the truncated ones
    included while a mean updraft area wants them excluded. Collapsing the two
    would make that choice unrecoverable.

    The scan edge is not the only way to lose part of an object. `hole` marks
    INTERIOR gates where the retrieval produced nothing -- a dead altitude bin
    with data above and below it, printing as a horizontal white stripe -- and an
    object abutting one of those is cut off just as surely, it simply happens in
    the middle of the field rather than at its edge. The caller decides what
    counts as interior; holes that open out to cloud top should not be passed,
    since the object ends there because the cloud does.

    Parameters
    ----------
    k : numpy.ndarray of bool
        The object's gates.
    elev : numpy.ndarray
        Per-gate (or per-ray) elevation for the sweep, used only to find
        which ray is the top and which the bottom, since ray order is not
        guaranteed.
    hole : numpy.ndarray of bool, optional
        Interior unretrieved gates (see above).

    Returns
    -------
    int
        Bitmask of TRUNC_*; 0 means the object is wholly inside the sampled
        volume and abuts no hole in it.
    """
    if not k.any():
        return 0
    rays = np.asarray(elev, float)[:, 0] if np.ndim(elev) == 2 else np.asarray(elev, float)
    rows = np.where(k.any(axis=1))[0]
    cols = np.where(k.any(axis=0))[0]
    out = 0
    if int(np.argmax(rays)) in rows:
        out |= TRUNC_TOP
    if int(np.argmin(rays)) in rows:
        out |= TRUNC_BOT
    if cols.min() == 0:
        out |= TRUNC_NEAR
    if cols.max() == k.shape[1] - 1:
        out |= TRUNC_FAR
    if hole is not None and np.any(hole):
        # one gate of slack: the object's own boundary sits next to the hole,
        # not on it, so touching is tested after a single dilation
        if np.any(k & ndimage.binary_dilation(np.asarray(hole, bool),
                                              EIGHT_CONNECTED)):
            out |= TRUNC_GAP
    return int(out)
