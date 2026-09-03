"""Tests for the three fall-speed methods and the freezing-level resolution.

The reflectivity/temperature method exists for files with no particle id, so
these check the branch table directly -- including the boundaries, which in the
original C matched no branch and silently produced a fall speed of zero.
"""
import numpy as np
import pytest

from _configs import corrections, environment, object_selection, sedimentation
from windvel.calculate_windvel import (
    GIANGRANDE_DARWIN2026_FORMULAS,
    GIANGRANDE_OKLAHOMA2013_FORMULAS,
    ZT_Z_CONVECTIVE,
    ZT_Z_GRAUPEL,
    compute_sed_vel_zt,
    resolve_freezing_level,
)


def _radar_with_one_object():
    """A minimal RHI radar carrying refl, PID, and one labelled object."""
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(20, 10, 1)
    nrays, ngates = radar.nrays, radar.ngates
    refl = np.ma.masked_array(np.full((nrays, ngates), 30.0),
                              mask=np.zeros((nrays, ngates), bool))
    pid = np.full((nrays, ngates), 3, dtype=int)          # Light_Rain
    ids = np.zeros((nrays, ngates), dtype=int)
    ids[2:8, 3:12] = 1
    radar.fields['reflectivity'] = {'data': refl}
    radar.fields['PID'] = {'data': pid}
    radar.fields['local_object_ids'] = {
        'data': np.ma.masked_array(ids, mask=(ids == 0))}
    return radar


def _cfg_sed(**selection):
    return {
        'input_variables': {'refl_var': 'reflectivity', 'pid_var': 'PID'},
        'corrections': corrections(),
        'sedimentation': sedimentation(),
        'environment': environment(),
        'object_selection': object_selection(**selection),
    }


def test_renamed_sedimentation_methods_are_refused():
    """Both PID tables are Giangrande first-author; keyed
    on different axes -- one by author, one by site -- so the names
    did not tell them apart. An old config now names a method that does not
    exist, and must be given the new name AND told the coefficients did not
    move; otherwise a re-run reads as a science change when it is a rename."""
    from windvel.calculate_windvel import SED_METHODS, SED_METHODS_RENAMED, get_sed_vel
    radar = _radar_with_one_object()
    assert SED_METHODS_RENAMED, 'the retired names must stay listed'
    for old, new in SED_METHODS_RENAMED.items():
        cfg = _cfg_sed()
        cfg['sedimentation'] = sedimentation(method=old)
        with pytest.raises(ValueError, match=new):
            get_sed_vel(radar, cfg, [0])
        assert new in SED_METHODS, f'{new!r} must be a live method'
        assert old not in SED_METHODS, f'{old!r} must not still resolve'


def test_get_sed_vel_returns_provenance_notes():
    """The output must say what happened: method, which reflectivity field,
    freezing-level source -- never discarded."""
    from windvel.calculate_windvel import get_sed_vel
    vel, notes = get_sed_vel(_radar_with_one_object(), _cfg_sed(), [0])
    assert notes['sedimentation_method'] == 'pid_giangrande_oklahoma2013'
    assert notes['reflectivity_field'] == 'reflectivity'
    assert notes['freezing_level_source'] == 'not needed (PID method)'
    assert bool((~vel.mask[2:8, 3:12]).any())      # object gates retrieved
    assert bool(vel.mask[0, 0])                    # background masked


def test_object_criteria_no_longer_filter_inside_get_sed_vel():
    """Selection lives in ONE place (apply_object_selection_criteria, run
    first). The embedded copy in get_sed_vel -- which had already drifted,
    never learning min_cloud_depth_km -- is gone."""
    from windvel.calculate_windvel import get_sed_vel
    vel, _ = get_sed_vel(_radar_with_one_object(),
                         _cfg_sed(min_span_km=1e6), [0])
    assert bool((~vel.mask[2:8, 3:12]).any())


def test_apply_object_selection_criteria_zeroes_failing_objects():
    from windvel.calculate_windvel import (
        apply_object_selection_criteria,
        make_sr_alt_maps,
    )
    radar = _radar_with_one_object()
    sr_map, alt_map = make_sr_alt_maps(radar)
    cfg = _cfg_sed(min_top_km=1e6)
    apply_object_selection_criteria(radar, cfg, sr_map, alt_map)
    ids = np.ma.filled(radar.fields['local_object_ids']['data'], 0)
    assert not (ids > 0).any()


def _v(dbz, warm):
    out = compute_sed_vel_zt(np.array([[dbz]], dtype=float),
                             np.array([[warm]], dtype=bool))
    return float(out[0, 0]) if not np.ma.is_masked(out[0, 0]) else np.nan


def test_warm_rain_is_discontinuous_at_the_convective_threshold():
    """The convective branch uses the SMALLER coefficient (2.65 against 3.15),
    so crossing Z = 40 the fall speed DROPS by about 1.2 m/s even though the
    reflectivity rose. That is a property of the rule, not of the atmosphere:
    the two relations were fitted to different populations and never joined up.
    Recorded here so the step is a known feature rather than a surprise."""
    below, above = _v(40.0, True), _v(40.0001, True)
    assert above < below
    assert 1.0 < (below - above) < 1.4


def test_cold_branches_are_snow_graupel_and_convective():
    assert _v(20.0, False) < 2.0                 # snow, slow
    assert 2.0 < _v(36.0, False) < 6.0           # graupel
    assert _v(50.0, False) > 6.0                 # convective core


def test_every_boundary_gets_a_value():
    """The original C left Z of exactly 33 or 40 unmatched, keeping Vt = 0 --
    which lands in w as a 3-7 m/s error rather than as missing data."""
    for dbz in (ZT_Z_GRAUPEL, ZT_Z_CONVECTIVE):
        for warm in (True, False):
            got = _v(dbz, warm)
            assert np.isfinite(got) and got > 0, (dbz, warm)


def test_nothing_is_ever_silently_zero():
    dbz = np.linspace(-20, 60, 200)
    for warm in (True, False):
        v = compute_sed_vel_zt(dbz[None, :], np.full((1, dbz.size), warm))
        assert not np.any(np.isclose(v.filled(np.nan), 0.0))


def test_invalid_reflectivity_is_masked_not_guessed():
    out = compute_sed_vel_zt(np.array([[np.nan]]), np.array([[True]]))
    assert np.ma.is_masked(out[0, 0])


def test_the_three_tables_disagree_most_over_ice():
    """Ice is where the methods part company: a constant against a relation."""
    Z = 10 ** (20.0 / 10.0)
    g = GIANGRANDE_OKLAHOMA2013_FORMULAS[12](20.0, Z)      # Ice_Crystals
    d = GIANGRANDE_DARWIN2026_FORMULAS[12](20.0, Z)
    assert g == pytest.approx(2.0)
    assert d < 1.5
    assert abs(g - d) > abs(GIANGRANDE_OKLAHOMA2013_FORMULAS[3](20.0, Z)
                            - GIANGRANDE_DARWIN2026_FORMULAS[3](20.0, Z))


# ------------------------------------------------- freezing level resolution

class _R:
    def __init__(self, fields=None):
        self.fields = fields or {}


def test_melting_level_puts_warm_air_below_it():
    alt = np.array([[1000.0, 4000.0, 6000.0, 12000.0]])
    warm, how = resolve_freezing_level(
        _R(), {'environment': environment(melting_level_km=4.89)}, alt)
    assert list(warm[0]) == [True, True, False, False]
    assert '4.89' in how


def test_a_temperature_field_in_the_file_wins():
    t = np.array([[10.0, -10.0]])
    r = _R({'T': {'data': t}})
    warm, how = resolve_freezing_level(
        r, {'environment': environment(temperature_field='T')},
        np.zeros_like(t))
    assert list(warm[0]) == [True, False] and 'temperature field' in how


def test_a_cached_profile_beats_the_constant():
    alt = np.array([[1000.0, 9000.0]])
    cfg = {'environment': environment(
        temperature_profile={'height_km': [0, 5, 10],
                             'temperature_c': [25, 0, -40]},
        melting_level_km=1.0)}
    warm, how = resolve_freezing_level(_R(), cfg, alt)
    assert list(warm[0]) == [True, False] and 'profile' in how


def test_no_source_raises_rather_than_assuming():
    from windvel.errors import ConfigError
    with pytest.raises(KeyError):                       # keys absent
        resolve_freezing_level(_R(), {'environment': {}}, np.zeros((1, 2)))
    with pytest.raises(ConfigError, match='no temperature source'):   # all null
        resolve_freezing_level(
            _R(), {'environment': environment(melting_level_km=None)},
            np.zeros((1, 2)))


def test_sounding_bounds_are_required_when_a_sounding_dir_is_set(tmp_path):
    from windvel.errors import MissingConfigKeyError
    cfg = {'environment': environment(sounding_dir=str(tmp_path))}
    with pytest.raises(MissingConfigKeyError, match='max_sounding_age_hours'):
        resolve_freezing_level(_R(), cfg, np.zeros((1, 2)))
