"""Tests for the coherent-object detectors and the usability rule.

Synthetic fields only: each test states the structure it builds and the number
of objects that structure ought to contain, so a failure says which property
broke rather than that a number moved.

"""
import numpy as np
import pytest

from windvel.coherent_structures import (
    TRUNC_BOT,
    TRUNC_FAR,
    TRUNC_GAP,
    TRUNC_NEAR,
    TRUNC_TOP,
    cell_area_km2,
    object_core,
    object_is_usable,
    object_truncation,
    persistence,
    threshold,
)

NY, NX = 40, 40


def _grid(dx=0.25, dz=0.25):
    """A uniform mesh, so area is trivially dx*dz per cell."""
    dk = np.broadcast_to(np.arange(NX) * dx, (NY, NX)).copy()
    ak = np.broadcast_to((np.arange(NY) * dz).reshape(-1, 1), (NY, NX)).copy()
    return dk, ak


def _blob(cy, cx, amp, width=3.0):
    y, x = np.mgrid[0:NY, 0:NX]
    return amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * width ** 2))


def test_area_comes_from_the_geometry_not_the_gate_count():
    """Diverging rays: the same gate count is a larger area further out."""
    dk = np.broadcast_to(np.linspace(1, 30, NX), (NY, NX)).copy()
    ak = np.broadcast_to(np.linspace(0, 15, NY).reshape(-1, 1), (NY, NX)).copy()
    # widen the mesh with range, as an RHI does
    dk = dk * (1 + 0.02 * np.arange(NY).reshape(-1, 1))
    a = cell_area_km2(dk, ak)
    assert a[0, -1] > a[0, 0]


def test_threshold_finds_updraft_and_downdraft():
    dk, ak = _grid()
    w = _blob(12, 12, 12.0) - _blob(28, 28, 12.0)
    up, dn = threshold(w, np.ones_like(w, bool), dk, ak, w_min=6.0,
                       min_area_km2=0.1)
    assert up.max() == 1 and dn.max() == 1


def test_threshold_rejects_on_area_not_on_gate_count():
    dk, ak = _grid()
    w = _blob(20, 20, 12.0, width=1.2)          # small but intense
    big = threshold(w, np.ones_like(w, bool), dk, ak, w_min=6.0,
                    min_area_km2=0.1)[0].max()
    small = threshold(w, np.ones_like(w, bool), dk, ak, w_min=6.0,
                      min_area_km2=50.0)[0].max()
    assert big == 1 and small == 0


def test_persistence_splits_two_cores_over_a_low_saddle():
    dk, ak = _grid()
    sigma = np.full((NY, NX), 2.0)
    w = np.maximum(_blob(15, 12, 20.0), _blob(15, 28, 14.0))
    up, _ = persistence(w, np.ones_like(w, bool), dk, ak, sigma,
                        persistence_min=3.0, floor_sigma=1.0, min_area_km2=0.01)
    assert up.max() == 2


def test_persistence_merges_them_when_the_cut_is_raised():
    """The weaker core is absorbed once the cut exceeds its own prominence.

    Note the cut applies to the surviving root too, which never merges and so
    is measured against the floor: raise it far enough and everything goes,
    which is why the value here is chosen between the two prominences rather
    than made arbitrarily large. With these blobs the saddle sits at 0.29 sigma,
    so the prominences are 9.00 and 6.71 and the cut goes between them.
    """
    dk, ak = _grid()
    sigma = np.full((NY, NX), 2.0)
    w = np.maximum(_blob(15, 12, 20.0), _blob(15, 28, 14.0))
    up, _ = persistence(w, np.ones_like(w, bool), dk, ak, sigma,
                        persistence_min=8.0, floor_sigma=1.0, min_area_km2=0.01)
    assert up.max() == 1


def test_persistence_is_a_contrast_not_a_speed():
    """Doubling sigma halves w/sigma, so a core can stop being significant
    without its wind speed changing at all."""
    dk, ak = _grid()
    w = np.maximum(_blob(15, 12, 20.0), _blob(15, 28, 14.0))
    good = persistence(w, np.ones_like(w, bool), dk, ak,
                       np.full((NY, NX), 1.0), persistence_min=3.0,
                       floor_sigma=1.0, min_area_km2=0.01)[0].max()
    noisy = persistence(w, np.ones_like(w, bool), dk, ak,
                        np.full((NY, NX), 12.0), persistence_min=3.0,
                        floor_sigma=1.0, min_area_km2=0.01)[0].max()
    assert good > noisy


# ------------------------------------------------------------- usability

def _one_object(usable_cols):
    """One blob, with `usable_cols` marking the columns inside the domain."""
    dk, ak = _grid()
    w = _blob(20, 20, 12.0)
    lab = (w > 4.0).astype(np.int32)
    usable = np.zeros_like(lab, bool)
    usable[:, usable_cols] = True
    return lab, w, cell_area_km2(dk, ak), usable


def test_object_usable_when_wholly_inside():
    lab, w, area, usable = _one_object(slice(None))
    ok, info = object_is_usable(lab, 1, w, area, usable, min_area_fraction=0.7,
                                core_peak_fraction=0.8, core_min_gates=3)
    assert ok and info['area_fraction'] == pytest.approx(1.0)


def test_object_not_usable_when_wholly_outside():
    lab, w, area, usable = _one_object(slice(0, 0))
    ok, _ = object_is_usable(lab, 1, w, area, usable, min_area_fraction=0.7,
                                core_peak_fraction=0.8, core_min_gates=3)
    assert not ok


def test_area_fraction_alone_does_not_carry_it():
    """Most of the area inside, but the core outside: the properties you would
    quote come from the part that is not trusted."""
    lab, w, area, usable = _one_object(slice(0, 18))   # core at column 20
    ok, info = object_is_usable(lab, 1, w, area, usable,
                                min_area_fraction=0.0,
                                core_peak_fraction=0.8, core_min_gates=3)
    assert not info['core_inside'] and not ok


def test_core_alone_does_not_carry_it_either():
    lab, w, area, usable = _one_object(slice(18, 23))  # only the core inside
    ok, info = object_is_usable(lab, 1, w, area, usable, min_area_fraction=0.7,
                                core_peak_fraction=0.8, core_min_gates=3)
    assert info['core_inside'] and info['area_fraction'] < 0.7 and not ok


def test_a_peaked_object_still_gets_a_core():
    """A sharp object -- 30 m/s at the tip, 5 around it -- has nothing else
    above 0.8 * peak. A fixed cut would return one gate and reject the object
    for being peaked, which is backwards: it passed a detector that REQUIRED a
    peak. The level is lowered until the core reaches the size asked for."""
    dk, ak = _grid()
    w = np.zeros((NY, NX))
    w[20, 20] = 30.0
    w[20, 21] = w[21, 20] = w[19, 20] = w[20, 19] = 5.0
    lab = (w > 1.0).astype(np.int32)
    ok, info = object_is_usable(lab, 1, w, cell_area_km2(dk, ak),
                                np.ones_like(lab, bool), min_area_fraction=0.7,
                                core_peak_fraction=0.8, core_min_gates=3)
    assert info['core_gates'] >= 3
    assert info['core_inside'] and ok


def test_a_tiny_object_is_its_own_core():
    dk, ak = _grid()
    w = np.zeros((NY, NX))
    w[20, 20], w[20, 21] = 12.0, 9.0
    lab = (w > 1.0).astype(np.int32)
    core = object_core(lab == 1, w, core_peak_fraction=0.8, core_min_gates=3)
    assert int(core.sum()) == 2


def test_the_core_is_the_strongest_part_not_the_whole_object():
    dk, ak = _grid()
    w = _blob(20, 20, 12.0, width=5.0)
    lab = (w > 2.0).astype(np.int32)
    core = object_core(lab == 1, w, core_peak_fraction=0.8, core_min_gates=3)
    assert 3 <= int(core.sum()) < int((lab == 1).sum())
    assert np.nanmin(np.where(core, w, np.nan)) > np.nanmin(np.where(lab == 1, w, np.nan))


def test_core_is_located_by_significance_not_by_speed():
    """An object detected on significance must have its core found the same way.

    Two peaks: a fast one where the error bar is huge (near the ground, where
    1/sin(elevation) inflates everything) and a slower one where it is small.
    In m/s the fast peak wins; in units of sigma the slower one does, and that
    is the gate the object's prominence actually came from.
    """
    dk, ak = _grid()
    w = np.maximum(_blob(4, 20, 37.0), _blob(30, 20, 12.0))
    sigma = np.where(np.arange(NY).reshape(-1, 1) < 15, 30.0, 3.0)
    sigma = np.broadcast_to(sigma, w.shape).copy()
    lab = (w > 2.0).astype(np.int32)

    by_speed = object_core(lab == 1, w, core_peak_fraction=0.8, core_min_gates=3)
    by_sigma = object_core(lab == 1, w, core_peak_fraction=0.8,
                           core_min_gates=3, sigma=sigma)
    row_speed = np.median(np.where(by_speed)[0])
    row_sigma = np.median(np.where(by_sigma)[0])
    assert row_speed < 15      # the fast, badly measured peak
    assert row_sigma > 15      # the slower, well measured one


# ------------------------------------------------------------- truncation

def _elev_grid():
    """Elevation rising with row index, as an RHI sweep."""
    return np.broadcast_to(np.linspace(1.0, 76.0, NY).reshape(-1, 1), (NY, NX)).copy()


def test_object_inside_the_scan_is_not_truncated():
    k = np.zeros((NY, NX), bool)
    k[10:20, 10:20] = True
    assert object_truncation(k, _elev_grid()) == 0


def test_object_reaching_the_top_beam_is_flagged():
    """The case that matters here: the sweep stops at 76 degrees, so anything
    directly above the radar is sliced off and its depth is a lower bound."""
    k = np.zeros((NY, NX), bool)
    k[NY - 6:, 10:20] = True
    assert object_truncation(k, _elev_grid()) & TRUNC_TOP


def test_the_edges_are_reported_separately():
    k = np.zeros((NY, NX), bool)
    k[NY - 3:, :3] = True                       # top beam AND first gate
    got = object_truncation(k, _elev_grid())
    assert got & TRUNC_TOP and got & TRUNC_NEAR
    assert not got & TRUNC_BOT and not got & TRUNC_FAR


def test_ray_order_does_not_matter():
    """Ray order is not guaranteed, so the top is found from the elevations
    themselves rather than from the row index."""
    k = np.zeros((NY, NX), bool)
    k[:5, 10:20] = True
    rising = _elev_grid()
    falling = rising[::-1].copy()
    assert object_truncation(k, rising) & TRUNC_BOT
    assert object_truncation(k, falling) & TRUNC_TOP


def test_object_abutting_an_internal_gap_is_truncated():
    """A hole in the middle of the field cuts an object off just as surely as
    the edge of the scan does. Over six scans of 20220818 this is the commoner
    case: 39% of persistence objects touch a gap against 16% reaching an edge."""
    k = np.zeros((NY, NX), bool)
    k[10:20, 10:20] = True
    hole = np.zeros((NY, NX), bool)
    hole[20:22, 10:20] = True                 # a dead altitude bin just above
    assert object_truncation(k, _elev_grid()) == 0
    assert object_truncation(k, _elev_grid(), hole) & TRUNC_GAP


def test_a_distant_gap_does_not_count():
    k = np.zeros((NY, NX), bool)
    k[10:20, 10:20] = True
    hole = np.zeros((NY, NX), bool)
    hole[30:32, 30:35] = True
    assert not object_truncation(k, _elev_grid(), hole) & TRUNC_GAP


def test_edge_and_gap_are_reported_separately():
    k = np.zeros((NY, NX), bool)
    k[NY - 5:, 10:20] = True
    hole = np.zeros((NY, NX), bool)
    hole[NY - 7:NY - 5, 10:20] = True          # immediately below the object
    got = object_truncation(k, _elev_grid(), hole)
    assert got & TRUNC_TOP and got & TRUNC_GAP


def test_an_enclosed_core_is_absorbed_but_a_neighbouring_one_is_not():
    """The distinction the merge rule turns on. A core INSIDE another is a dip
    within one structure; a core BESIDE another is a boundary between two."""
    dk, ak = _grid()
    sigma = np.full((NY, NX), 2.0)

    # a broad basin with a second maximum buried inside it
    nested = _blob(20, 20, 16.0, width=9.0) + _blob(20, 20, 6.0, width=1.5)
    n_nested = persistence(nested, np.ones_like(nested, bool), dk, ak, sigma,
                           persistence_min=2.0, floor_sigma=1.0, min_area_km2=0.01)[0].max()

    # two maxima side by side, each with its own outward flank
    beside = np.maximum(_blob(20, 12, 16.0, width=3.0),
                        _blob(20, 28, 15.0, width=3.0))
    n_beside = persistence(beside, np.ones_like(beside, bool), dk, ak, sigma,
                           persistence_min=2.0, floor_sigma=1.0, min_area_km2=0.01)[0].max()

    assert n_nested == 1
    assert n_beside == 2


def test_an_object_is_drawn_in_one_piece():
    """A label surviving where it is not contiguous with its own core is
    drawn as scattered fragments -- ten of them for one object on 20220818."""
    from scipy import ndimage as ndi
    dk, ak = _grid()
    sigma = np.full((NY, NX), 2.0)
    w = np.maximum(_blob(12, 12, 20.0), _blob(28, 28, 15.0))
    up, _ = persistence(w, np.ones_like(w, bool), dk, ak, sigma,
                        persistence_min=2.0, floor_sigma=1.0, min_area_km2=0.01)
    for i in range(1, int(up.max()) + 1):
        _, n = ndi.label(up == i, structure=np.ones((3, 3), bool))
        assert n == 1, f'object {i} drawn in {n} pieces'


# ---------------------------------------------------------------------------
# The pipeline's config for the detectors. All keys required, no defaults.
# ---------------------------------------------------------------------------

def test_every_detector_config_key_is_required():
    """Regression against drifted defaults: code that falls back to
    threshold_min_area_km2=1.0 and persistence_min_sigma
    =3.0 while every config said 0.1 and 2.0 -- a config missing a key
    silently ran on the stale copy."""
    from windvel.calculate_windvel import detect_coherent_objects

    class _EmptyRadar:
        nsweeps = 0

    from _configs import coherent_objects

    w = np.zeros((1, 1))
    sig = np.ones((1, 1))
    use = np.ones((1, 1), bool)

    with pytest.raises(KeyError):
        detect_coherent_objects(_EmptyRadar(), {}, w, sig, use)
    full = coherent_objects()
    for sub, keys in full.items():
        for key in keys:
            cfg = {'coherent_objects': coherent_objects()}
            del cfg['coherent_objects'][sub][key]
            with pytest.raises(KeyError, match=key):
                detect_coherent_objects(_EmptyRadar(), cfg, w, sig, use)
