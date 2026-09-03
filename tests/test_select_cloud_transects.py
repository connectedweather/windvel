"""Regression test for the policy: clouds truncated by
the top of the scan are KEPT and labelled, not discarded. (The old
remove_cutoff_clouds filter deleted any cloud whose echo reached the scan
top over half its gates; the coherent-object detectors downstream already
treat truncation as a flag, not a death sentence.)
"""
import numpy as np
import pytest

from _configs import cloud_objects, corrections


def _rhi_radar_with_top_touching_blob():
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(30, 20, 1)  # 20 rays x 30 gates
    nrays, ngates = radar.nrays, radar.ngates
    # ascending elevations, masked as a real CFRadial read would give them
    radar.elevation['data'] = np.ma.array(
        np.linspace(0.0, 76.0, nrays), mask=False)

    refl = np.full((nrays, ngates), -100.0)
    # a blob of echo that extends through the HIGHEST-elevation rays --
    # i.e. a cloud cut off by the top of the scan
    refl[8:, 5:16] = 10.0
    radar.fields['reflectivity'] = {
        'data': np.ma.masked_array(refl, mask=np.zeros_like(refl, bool))}
    return radar


def test_cloud_reaching_scan_top_is_kept_and_labelled():
    from windvel.select_cloud_transects import select_cloud_transects

    radar = _rhi_radar_with_top_touching_blob()
    cfg = {
        'input_variables': {'refl_var': 'reflectivity'},
        'corrections': corrections(),
        'cloud_objects': cloud_objects(min_area_km2=0.0),
    }
    select_cloud_transects(radar, cfg, [0])

    ids = radar.fields['local_object_ids']['data']
    ids = np.ma.filled(ids, 0)
    assert (ids > 0).any(), 'top-truncated cloud must be labelled, not dropped'
    # and specifically inside the blob
    assert (ids[8:, 5:16] > 0).any()


def test_a_pixel_count_key_is_refused_by_the_schema():
    # a pixel count is not an area: a config carrying contour_area_threshold
    # must fail loudly at load, not silently run with a different meaning.
    from windvel.config import RETRIEVAL_SCHEMA, validate

    co = cloud_objects()
    co['contour_area_threshold'] = 1000
    with pytest.raises(ValueError, match='contour_area_threshold'):
        validate(co, RETRIEVAL_SCHEMA['cloud_objects'], 'cloud_objects')


def test_area_floor_is_physical_km2_not_pixels():
    # Two blobs with the same PIXEL count sit at different ranges; the floor
    # must judge them by true area from the gate geometry, keeping the far
    # one (rays diverge, so far pixels cover more km^2) when the threshold
    # is set between the two areas.
    from windvel.coherent_structures import cell_area_km2
    from windvel.select_cloud_transects import select_cloud_transects

    radar = _rhi_radar_with_top_touching_blob()
    nrays, ngates = radar.nrays, radar.ngates
    refl = np.full((nrays, ngates), -100.0)
    refl[2:8, 2:6] = 10.0      # near blob, 6x4 px
    refl[2:8, 24:28] = 10.0    # far blob, 6x4 px -- same pixel count
    radar.fields['reflectivity'] = {
        'data': np.ma.masked_array(refl, mask=np.zeros_like(refl, bool))}

    x, y, z = radar.get_gate_x_y_z(0)
    gate_area = cell_area_km2(np.hypot(x, y) / 1000.0, z / 1000.0)
    near = float(gate_area[2:8, 2:6].sum())
    far = float(gate_area[2:8, 24:28].sum())
    assert far > near          # sanity: rays diverge with range

    cfg = {
        'input_variables': {'refl_var': 'reflectivity'},
        'corrections': corrections(),
        'cloud_objects': cloud_objects(min_area_km2=(near + far) / 2.0),
    }
    select_cloud_transects(radar, cfg, [0])

    ids = np.ma.filled(radar.fields['local_object_ids']['data'], 0)
    assert (ids[2:8, 24:28] > 0).any(), 'far blob is over the km^2 floor'
    assert not (ids[2:8, 2:6] > 0).any(), 'near blob is under it'
