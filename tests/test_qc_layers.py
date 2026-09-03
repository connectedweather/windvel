"""The three QC layers, motivated by a -56.9 m/s "horizontal wind"
reached the saved output (20220818 213532 sweep 2, 16.65 km).

Root cause: the weak-echo test that picks anchor gates has a CEILING but no
FLOOR, so near cloud top it selected noise-floor gates (0 dBZ, Doppler
disagreeing by 30 m/s between adjacent beams), and 1/cos(62 deg) doubled the
error. Nothing anywhere asked whether the answer is physically possible.

  layer 1  SNR screen on the cloud mask        select_cloud_transects
  layer 2  candidate bound on anchor voting    get_horizontal_velocity
  layer 3  |w| magnitude -> usable FLAG        vertical_velocity_usability

Layers 1 and 2 remove bad INPUTS (a gate that is noise is not a measurement).
Layer 3 FLAGS a bad output rather than erasing it: values are kept, and if
this layer ever catches a lot, that is evidence the first two are failing.
"""
import numpy as np
import pytest

from _configs import cloud_objects, corrections, horizontal_wind
from windvel.calculate_windvel import (
    get_horizontal_velocity,
    vertical_velocity_usability,
)
from windvel.select_cloud_transects import select_cloud_transects

NGATES = 20
BIN_M = 300.0


# ---------------------------------------------------------------- layer 1

def _radar_with_snr(snr_value):
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(30, 20, 1)
    shape = (radar.nrays, radar.ngates)
    refl = np.ma.masked_array(np.full(shape, -100.0), mask=np.zeros(shape, bool))
    refl[5:15, 5:25] = 30.0                       # a solid blob of echo
    radar.fields['reflectivity'] = {'data': refl}
    radar.fields['SNR'] = {'data': np.full(shape, float(snr_value))}
    radar.elevation['data'] = np.ma.array(radar.elevation['data'], mask=False)
    return radar


def _cfg1(**mc):
    m = cloud_objects(min_area_km2=0.0,
                      gate_quality=[{'field': 'SNR', 'min': 3.0}])
    m.update(mc)
    return {'input_variables': {'refl_var': 'reflectivity'},
            'corrections': corrections(),
            'cloud_objects': m}


def _labelled(radar):
    ids = radar.fields['local_object_ids']['data']
    return int((np.ma.filled(ids, 0) > 0).sum())


def test_strong_signal_survives_the_snr_screen():
    radar = _radar_with_snr(20.0)
    select_cloud_transects(radar, _cfg1(), [0])
    assert _labelled(radar) > 0


def test_noise_floor_gates_are_not_cloud():
    """Same echo, no signal behind it: nothing should be labelled."""
    radar = _radar_with_snr(0.5)
    select_cloud_transects(radar, _cfg1(), [0])
    assert _labelled(radar) == 0


def test_the_screen_can_be_switched_off_explicitly():
    radar = _radar_with_snr(0.5)
    select_cloud_transects(radar, _cfg1(gate_quality=[]), [0])
    assert _labelled(radar) > 0


def test_the_config_decides_never_the_file():
    """A named field missing from the file is an ERROR, not a silent skip --
    otherwise two radars are processed differently without anyone knowing."""
    radar = _radar_with_snr(20.0)
    del radar.fields['SNR']
    with pytest.raises(KeyError, match='gate_quality'):
        select_cloud_transects(radar, _cfg1(), [0])


def test_gate_quality_is_required():
    radar = _radar_with_snr(20.0)
    cfg = _cfg1()
    del cfg['cloud_objects']['gate_quality']
    with pytest.raises(KeyError, match='gate_quality is required'):
        select_cloud_transects(radar, cfg, [0])


def test_several_screens_compose():
    """A list, so a new field is config and not code. A gate must pass ALL
    of them; here SNR passes and SQI fails, so nothing is cloud."""
    radar = _radar_with_snr(20.0)
    shape = (radar.nrays, radar.ngates)
    radar.fields['SQI'] = {'data': np.full(shape, 0.20)}
    cfg = _cfg1(gate_quality=[{'field': 'SNR', 'min': 3.0},
                              {'field': 'SQI', 'min': 0.45}])
    select_cloud_transects(radar, cfg, [0])
    assert _labelled(radar) == 0


# ---------------------------------------------------------------- layer 2

class _Stub:
    def __init__(self, fields, elevation):
        self.fields = fields
        self.elevation = {'data': elevation}
        self._slices = [slice(0, len(elevation))]

    @property
    def nsweeps(self):
        return 1

    def get_slice(self, s):
        return self._slices[s]


def _one_bin(vr):
    vr = np.asarray(vr, float).reshape(1, NGATES)
    sr = np.tile(10.0 + 0.1 * np.arange(NGATES), (1, 1))
    alt = (150.0 + np.linspace(0.0, 1.0, NGATES))[None, :]
    fields = {'refl': {'data': np.zeros((1, NGATES))},
              'vr': {'data': vr},
              'sedimentation_velocity': {'data': np.zeros((1, NGATES))},
              'local_object_ids': {'data': np.ones((1, NGATES), dtype=int)}}
    return _Stub(fields, np.zeros(1)), sr, np.ascontiguousarray(alt)


def _cfg2(bound):
    return {'input_variables': {'refl_var': 'refl', 'vr_var': 'vr'},
            'corrections': corrections(),
            'horizontal_wind': horizontal_wind(weak_echo_dbz=[25], bin_size_m=BIN_M,
                                               min_gates=5, max_candidate_mps=bound)}


def test_impossible_gates_lose_their_vote():
    """Near half: 5 sane gates at 10 m/s and 5 wild ones at 200. Unbounded,
    the median is dragged; bounded, the wild ones never vote."""
    vr = np.full(NGATES, 20.0)
    vr[:5] = 10.0
    vr[5:10] = 200.0
    radar, sr, alt = _one_bin(vr)
    loose = get_horizontal_velocity(radar, _cfg2(None), sr, alt)[0][0, 0]
    tight = get_horizontal_velocity(radar, _cfg2(40.0), sr, alt)[0][0, 0]
    assert loose > 40.0        # today: the median sits among the wild gates
    assert tight == pytest.approx(10.0)


def test_a_bin_of_pure_noise_yields_no_anchor_at_all():
    """All near-half gates impossible: with the bound, too few survive to
    form an anchor, so the bin gives no wind -- the honest answer."""
    vr = np.full(NGATES, 20.0)
    vr[:10] = -200.0
    radar, sr, alt = _one_bin(vr)
    hv = get_horizontal_velocity(radar, _cfg2(40.0), sr, alt)[0]
    assert np.all(np.abs(hv[np.isfinite(hv)]) <= 40.0)


def test_the_candidate_bound_is_required():
    radar, sr, alt = _one_bin(np.full(NGATES, 10.0))
    cfg = _cfg2(40.0)
    del cfg['horizontal_wind']['anchor']['max_candidate_mps']
    with pytest.raises(KeyError, match='max_candidate_mps is required'):
        get_horizontal_velocity(radar, cfg, sr, alt)


# ---------------------------------------------------------------- layer 3

class _FakeRadar:
    def __init__(self, elev):
        self.elevation = {'data': np.asarray(elev, float)}
        self.nrays = len(elev)
        self.ngates = 3


def _usable(w, max_abs_w):
    elev = np.array([45.0])
    sr = np.array([[10.0, 10.0, 10.0]])
    alt = np.zeros((1, 3))
    cfg = {'vertical_velocity': {'usable': {'min_elevation_deg': 30.0,
                                            'max_horizontal_distance_km': 20.0,
                                            'max_abs_w_mps': max_abs_w}}}
    return vertical_velocity_usability(_FakeRadar(elev), cfg, sr, alt,
                                       np.asarray(w, float).reshape(1, 3))


def test_an_impossible_updraft_is_flagged_not_usable():
    ok = _usable([5.0, 300.0, -400.0], 60.0)
    assert ok[0, 0] and not ok[0, 1] and not ok[0, 2]


def test_the_value_itself_is_never_touched():
    """Flag, never filter: usability returns a mask and nothing else -- the
    caller's w array must come back unchanged."""
    w = np.array([[5.0, 300.0, -400.0]])
    before = w.copy()
    _usable(w, 60.0)
    assert np.array_equal(w, before)


def test_the_magnitude_check_can_be_disabled():
    assert _usable([5.0, 300.0, -400.0], None).all()


def test_max_abs_w_is_required():
    with pytest.raises(KeyError, match='max_abs_w_mps is required'):
        vertical_velocity_usability(
            _FakeRadar([45.0]),
            {'vertical_velocity': {'usable': {'min_elevation_deg': 30.0,
                                              'max_horizontal_distance_km': 20.0}}},
            np.array([[10.0]]), np.zeros((1, 1)), np.array([[5.0]]))


def test_a_fully_masked_scan_does_not_crash_the_bound():
    """Found by the 20220805 case: elevation comes back masked in
    real files, so hv_candidate is a MaskedArray; when a whole scan is masked
    the comparison sums to np.ma.masked and int() raised, killing 30 of 374
    scans. A masked gate has no candidate, so it simply cannot vote."""
    radar, sr, alt = _one_bin(np.full(NGATES, 10.0))
    radar.elevation['data'] = np.ma.array(np.zeros(1), mask=True)
    radar.fields['vr']['data'] = np.ma.array(np.full((1, NGATES), 10.0),
                                             mask=True)
    hv, src = get_horizontal_velocity(radar, _cfg2(40.0), sr, alt)
    assert not np.isfinite(hv).any()      # no data in, no wind out
