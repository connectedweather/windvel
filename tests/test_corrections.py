"""Tests for the corrections stage: `apply_corrections` writes a new field
and never touches the delivered one; `working_refl_field` is the one rule
deciding which reflectivity the stages read; a stage asked to read a
corrected field that is not there fails loudly.
"""
import numpy as np
import pytest

from _configs import attenuation, cloud_objects, corrections, input_variables
from windvel.corrections import (
    CORRECTED_REFL_FIELD,
    apply_corrections,
    working_refl_field,
)
from windvel.errors import InputFieldError

NRAYS, NGATES = 20, 30


def _cfg(correct=False):
    a = (attenuation(correct=True, relation='x_andsager', liquid_only=False)
         if correct else attenuation())
    return {'corrections': corrections(attenuation_cfg=a),
            'input_variables': input_variables()}


def _radar(dbz=45.0):
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(NGATES, NRAYS, 1)
    radar.elevation['data'] = np.ma.array(
        np.linspace(0.0, 76.0, NRAYS), mask=False)
    refl = np.full((NRAYS, NGATES), dbz)
    radar.fields['reflectivity'] = {
        'data': np.ma.masked_array(refl, mask=np.zeros_like(refl, bool))}
    return radar


def test_working_field_is_refl_var_when_off():
    assert working_refl_field(_cfg(correct=False)) == 'reflectivity'


def test_working_field_is_the_corrected_field_when_on():
    assert working_refl_field(_cfg(correct=True)) == CORRECTED_REFL_FIELD


def test_apply_off_adds_nothing_and_touches_nothing():
    radar = _radar()
    before = radar.fields['reflectivity']['data'].copy()
    apply_corrections(radar, _cfg(correct=False))
    assert set(radar.fields) == {'reflectivity'}
    assert np.array_equal(radar.fields['reflectivity']['data'], before)


def test_apply_on_adds_the_corrected_field_and_keeps_the_raw():
    radar = _radar()
    raw_before = radar.fields['reflectivity']['data'].copy()
    apply_corrections(radar, _cfg(correct=True))

    assert CORRECTED_REFL_FIELD in radar.fields
    assert np.array_equal(radar.fields['reflectivity']['data'], raw_before)

    corr = np.asarray(radar.fields[CORRECTED_REFL_FIELD]['data'])
    raw = np.asarray(raw_before)
    assert np.all(corr >= raw)             # attenuation is only ever added back
    assert np.any(corr > raw)              # and 45 dBZ along a ray does add
    # the field's comment names its source field and the relation
    comment = radar.fields[CORRECTED_REFL_FIELD]['comment']
    assert "corrected from 'reflectivity'" in comment


def test_apply_on_without_the_input_field_raises():
    radar = _radar()
    del radar.fields['reflectivity']
    with pytest.raises(InputFieldError, match='refl_var'):
        apply_corrections(radar, _cfg(correct=True))


def test_a_stage_refuses_a_correction_that_was_not_applied():
    """The config says corrected, the radar has no corrected field: the
    stage must fail naming the missing field, never quietly fall back to
    the delivered reflectivity."""
    from windvel.select_cloud_transects import select_cloud_transects
    radar = _radar()
    cfg = _cfg(correct=True)
    cfg['cloud_objects'] = cloud_objects()
    with pytest.raises(InputFieldError, match=CORRECTED_REFL_FIELD):
        select_cloud_transects(radar, cfg, [0])
