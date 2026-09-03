"""Regression tests for RHI sweep selection.

Two bugs are pinned here (both found in review):

1. `extract_rhi_sweep_indices`'s except branch fell through without
   returning, so ONE file with an undecodable sweep_mode killed every later
   file in the run. Failure must mean "skip this file" ([]), never a crash.

2. `select_cloud_transects` re-derived the sweep list with a DIFFERENT
   rule — elevations non-decreasing, sweep_mode ignored — overwriting the
   driver's list. A PPI sweep has constant elevation, which passes a
   non-decreasing test, so mixed-mode files would have had PPI sweeps
   processed as RHIs. There is now ONE definition, owned by
   `extract_rhi_sweep_indices`: sweep_mode == 'rhi' AND non-decreasing
   elevations.

The sweep list is a RETURN VALUE passed
explicitly to every consumer — it never travels inside cfg as
'rhi_sweep_indices', where it outlived each file and hid its consumers.
"""
import numpy as np
import pytest

from windvel.utils import extract_rhi_sweep_indices, is_valid_elevation_sweep


class FakeRadar:
    """The two attributes sweep selection actually touches."""

    def __init__(self, modes, elevs):
        self.sweep_mode = {'data': np.array([m.encode() for m in modes])}
        self._elevs = [np.asarray(e, dtype=float) for e in elevs]
        self.nsweeps = len(self._elevs)

    def get_elevation(self, sweep):
        return self._elevs[sweep]


def test_ascending_rhi_is_selected():
    r = FakeRadar(['rhi'], [[1.0, 2.0, 3.0]])
    assert extract_rhi_sweep_indices(r) == [0]


def test_ppi_is_excluded_despite_passing_the_elevation_test():
    # Constant elevation passes the non-decreasing check -- which is exactly
    # why sweep_mode must also be tested. This is bug 2's failure mode.
    r = FakeRadar(['azimuth_surveillance'], [[2.0, 2.0, 2.0]])
    assert is_valid_elevation_sweep(r, 0)          # the trap...
    assert extract_rhi_sweep_indices(r) == []      # ...does not spring


def test_descending_rhi_is_dropped_current_policy():
    # Pins the CURRENT policy: descending RHIs are excluded, not reversed.
    # (Known limitation -- on an up-down scanner this drops half the rays.)
    r = FakeRadar(['rhi', 'rhi'], [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert extract_rhi_sweep_indices(r) == [0]


def test_mixed_mode_file_selects_only_the_rhis():
    r = FakeRadar(['rhi', 'azimuth_surveillance', 'rhi'],
                  [[1, 2, 3], [2, 2, 2], [0, 1, 2]])
    assert extract_rhi_sweep_indices(r) == [0, 2]


def test_failure_returns_an_empty_list_not_a_crash():
    # Bug 1's modern form: sweep_mode['data'] of None makes decoding raise;
    # the file must be skipped ([]), not the run killed.
    r = FakeRadar([], [])
    r.sweep_mode = {'data': None}
    assert extract_rhi_sweep_indices(r) == []


def test_sweep_mode_decoding_handles_all_three_encodings():
    # str rows, bytes rows, and per-character masked byte arrays (the
    # CFRadial padded-char layout the original decoder expects).
    r = FakeRadar(['rhi'], [[1, 2], [1, 2], [1, 2]])
    chars = np.ma.masked_array(
        np.array([b'r', b'h', b'i', b' ']), mask=[False, False, False, True])
    r.sweep_mode = {'data': np.array(['rhi', 'x'], dtype=object)}
    r.sweep_mode['data'] = [chars, b'rhi\x00', 'rhi ']
    r._elevs = [np.array([1.0, 2.0])] * 3
    r.nsweeps = 3
    assert extract_rhi_sweep_indices(r) == [0, 1, 2]


def _clear_air_radar():
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(20, 15, 2)
    nrays, ngates = radar.nrays, radar.ngates
    refl = np.ma.masked_array(np.full((nrays, ngates), -100.0),
                              mask=np.zeros((nrays, ngates), bool))
    # assigned directly: add_field can degrade an all-False-mask MaskedArray
    # to a plain ndarray, and the module requires masked reflectivity
    radar.fields['reflectivity'] = {'data': refl}
    # real CFRadial reads yield masked coordinate arrays; the empty test
    # radar gives plain ndarrays, so mask them the way a real file would be
    radar.elevation['data'] = np.ma.array(radar.elevation['data'], mask=False)
    return radar


def test_select_cloud_transects_takes_the_sweep_list_explicitly():
    """The sweep list is an argument, never read from
    or written into cfg — calling the old cfg-carried way must fail, and a
    cfg-smuggled list must be ignored in favour of the explicit one."""
    from _configs import cloud_objects, corrections
    from windvel.select_cloud_transects import select_cloud_transects
    cfg = {
        'input_variables': {'refl_var': 'reflectivity'},
        'corrections': corrections(),
        'cloud_objects': cloud_objects(min_area_km2=0.0),
    }
    with pytest.raises(TypeError):
        select_cloud_transects(_clear_air_radar(), cfg)    # old API refused

    radar = _clear_air_radar()
    cfg['rhi_sweep_indices'] = [0]         # a smuggled list changes nothing
    select_cloud_transects(radar, cfg, [1])
    assert cfg['rhi_sweep_indices'] == [0]  # cfg is configuration, untouched
