"""Regression test for the summary panel's edge-anchor profiles.

Pinned bug (found by the user on 20220818 213532 sweep 2):
`_edge_profiles` took the nearest and furthest gate of each altitude bin
across the WHOLE SWEEP, ignoring object identity. Every anchor, bin and
interpolation in the retrieval belongs to ONE object in ONE sweep, so
pooling paired the near edge of one cloud with the far edge of another tens
of km away, and where a cloud ended the trace stepped to a different cloud
and read as wind shear. The panel is now per object.
"""
import numpy as np

from windvel.plot_windvel import _edge_profiles

NRAYS, NGATES = 4, 40
BIN_M = 300.0


def _two_clouds():
    """Two objects in the SAME altitude bins at very different ranges.

    Object 1 sits at 5-8 km range with hv rising 2 -> 3 across it; object 2
    at 40-43 km with hv 20 -> 21. Pooled, a bin's nearest gate belongs to
    object 1 and its furthest to object 2, so the old code reported a
    fictitious 2 -> 21 m/s span for a single 'bin'.
    """
    sr = np.zeros((NRAYS, NGATES))
    alt = np.zeros((NRAYS, NGATES))
    hv = np.full((NRAYS, NGATES), np.nan)
    ids = np.zeros((NRAYS, NGATES), dtype=int)
    for j in range(NRAYS):                       # one altitude bin per ray
        z = 1000.0 + j * BIN_M
        for i in range(10):                      # object 1: near, hv 2..3
            sr[j, i] = 5000.0 + i * 300.0
            alt[j, i] = z
            hv[j, i] = 2.0 + i / 9.0
            ids[j, i] = 1
        for i in range(20, 30):                  # object 2: far, hv 20..21
            sr[j, i] = 40000.0 + (i - 20) * 300.0
            alt[j, i] = z
            hv[j, i] = 20.0 + (i - 20) / 9.0
            ids[j, i] = 2
    return hv, sr, alt, ids


def test_each_object_gets_its_own_profile():
    hv, sr, alt, ids = _two_clouds()
    prof = _edge_profiles(hv, sr, alt, BIN_M, ids)
    assert set(prof) == {1, 2}


def test_an_objects_anchors_come_only_from_its_own_gates():
    """The heart of the bug: object 1's near edge must never be paired with
    object 2's far edge."""
    hv, sr, alt, ids = _two_clouds()
    prof = _edge_profiles(hv, sr, alt, BIN_M, ids)
    z1, near1, far1 = prof[1]
    z2, near2, far2 = prof[2]
    assert np.allclose(near1, 2.0) and np.allclose(far1, 3.0)
    assert np.allclose(near2, 20.0) and np.allclose(far2, 21.0)
    # and nothing spans the two clouds
    assert far1.max() < 10.0 and near2.min() > 10.0


def test_bins_are_anchored_at_each_objects_own_base():
    """The retrieval bins from the object's base, so the panel must too --
    otherwise the drawn profile is not the interpolated one."""
    hv, sr, alt, ids = _two_clouds()
    alt[:, 20:30] += 5000.0                      # lift object 2 by 5 km
    prof = _edge_profiles(hv, sr, alt, BIN_M, ids)
    assert prof[2][0].min() > prof[1][0].max()   # separate altitude ranges
    assert np.allclose(np.diff(prof[1][0]), BIN_M / 1000.0)


def test_gates_outside_any_object_are_ignored():
    hv, sr, alt, ids = _two_clouds()
    hv[:, 35] = 99.0                             # a retrieved gate, id 0
    sr[:, 35] = 60000.0
    alt[:, 35] = 1000.0
    prof = _edge_profiles(hv, sr, alt, BIN_M, ids)
    assert all(np.all(f < 50.0) for _, _, f in prof.values())


def test_a_single_object_still_gives_one_profile():
    hv, sr, alt, ids = _two_clouds()
    ids[ids == 2] = 0
    prof = _edge_profiles(hv, sr, alt, BIN_M, ids)
    assert set(prof) == {1}
