"""Tests for calculate_vertical_velocity.

The distance rule: w is computed for every object gate
within vertical_velocity.compute.max_horizontal_distance_km, PER GATE and in
HORIZONTAL distance. It replaced an all-or-nothing per-object test (>= 80% of
gates within a slant-range threshold, the fraction hardcoded) that denied w
to a whole object -- near gates included -- when too much of it sat far out.
"""
import numpy as np
import pytest

from windvel.calculate_windvel import calculate_vertical_velocity

NRAYS, NGATES = 3, 10
SIN45 = np.sin(np.deg2rad(45.0))


class StubRadar:
    def __init__(self, fields, elevation):
        self.fields = fields
        self.elevation = {'data': elevation}


def _setup(ids=None):
    """45/45/3-degree rays; slant range 20..38 km; flat earth (alt 0), so
    horizontal distance == slant range. Uniform vr=5, sed=2, hv=4:
    w = vr/sin(e) + sed - hv = 5.0711 on the 45-degree rays."""
    fields = {
        'vr': {'data': np.full((NRAYS, NGATES), 5.0)},
        'sedimentation_velocity': {'data': np.full((NRAYS, NGATES), 2.0)},
        'horizontal_velocity': {'data': np.full((NRAYS, NGATES), 4.0)},
        'local_object_ids': {'data': (np.ones((NRAYS, NGATES), dtype=int)
                                      if ids is None else ids)},
    }
    radar = StubRadar(fields, np.array([45.0, 45.0, 3.0]))
    sr_map = np.tile(20.0 + 2.0 * np.arange(NGATES), (NRAYS, 1))  # 20..38 km
    alt_map = np.zeros((NRAYS, NGATES))
    return radar, sr_map, alt_map


def _cfg(**compute):
    from _configs import vertical_velocity, vv_compute
    return {'input_variables': {'vr_var': 'vr'},
            'vertical_velocity': vertical_velocity(compute=vv_compute(**compute))}


def test_gates_inside_the_horizontal_cut_get_w_gates_beyond_do_not():
    radar, sr, alt = _setup()
    w = calculate_vertical_velocity(radar, _cfg(), sr, alt)
    expected = 5.0 / SIN45 + 2.0 - 4.0
    assert np.allclose(w[0, :8], expected)       # 20..34 km: inside
    assert np.all(np.isnan(w[0, 8:]))            # 36, 38 km: beyond 35


def test_an_object_mostly_beyond_the_cut_still_gets_its_near_gates():
    """Regression against an all-or-nothing 80% rule: 2 of this object's 4 gates
    lie beyond the cut, and the near two must be retrieved anyway."""
    ids = np.zeros((NRAYS, NGATES), dtype=int)
    ids[0, 6:] = 1                               # gates at 32, 34, 36, 38 km
    radar, sr, alt = _setup(ids)
    w = calculate_vertical_velocity(radar, _cfg(), sr, alt)
    assert np.all(np.isfinite(w[0, 6:8]))        # 32, 34 km retrieved
    assert np.all(np.isnan(w[0, 8:]))            # 36, 38 km not
    assert np.all(np.isnan(w[0, :6]))            # outside the object


def test_the_cut_is_horizontal_distance_not_slant_range():
    """Lift a ray to altitude: slant 36 km at 12 km altitude is ~33.9 km
    horizontal, so a gate that FAILS a slant test passes the horizontal one."""
    radar, sr, alt = _setup()
    alt[1, :] = 12000.0                          # 12 km altitude on ray 1
    w = calculate_vertical_velocity(radar, _cfg(), sr, alt)
    assert np.isfinite(w[1, 8])                  # slant 36 -> hdist 33.9 km
    assert np.isnan(w[0, 8])                     # same slant at 0 altitude: out


def test_low_elevation_rays_are_floored():
    radar, sr, alt = _setup()
    w = calculate_vertical_velocity(radar, _cfg(), sr, alt)
    assert np.all(np.isnan(w[2, :]))             # the 3-degree ray


def test_an_unknown_key_is_refused_by_the_schema():
    """A per-object slant-range key does not exist; a config carrying one
    must fail loudly at load, never run with a different meaning."""
    from windvel.config import RETRIEVAL_SCHEMA, validate
    cfg = _cfg()['vertical_velocity']
    cfg['compute']['distance_thresh'] = 40
    with pytest.raises(ValueError, match='distance_thresh'):
        validate(cfg, RETRIEVAL_SCHEMA['vertical_velocity'], 'vertical_velocity')


def test_the_distance_key_is_required():
    radar, sr, alt = _setup()
    cfg = _cfg()
    del cfg['vertical_velocity']['compute']['max_horizontal_distance_km']
    with pytest.raises(KeyError):
        calculate_vertical_velocity(radar, cfg, sr, alt)
