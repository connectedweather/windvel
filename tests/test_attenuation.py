"""Tests for the optional attenuation correction.

The correction exists for reflectivity that is not already corrected. Most of
what is tested here is its refusal to guess: the coefficients are band specific,
and a wrong band is an error of a factor of five or more, so every path that
would let one through silently is checked.

"""
import numpy as np
import pytest

from _configs import attenuation, corrections, environment
from windvel.corrections import correct_attenuation
from windvel.errors import ConfigError, MissingConfigKeyError

NRAYS, NGATES = 4, 50
C_MADE_UP = dict(band='C', a=2.7e-5, b=0.78)      # an inline relation


class _Radar:
    def __init__(self, ghz=9.4):
        self.range = {'data': np.arange(NGATES) * 250.0}
        self.instrument_parameters = {'frequency': {'data': np.array([ghz * 1e9])}}


def _cfg(**over):
    a = attenuation(correct=True, relation='x_andsager',
                    require_band_match=True, liquid_only=False,
                    max_correction_db=10.0)
    a.update(over)
    return {'corrections': corrections(attenuation_cfg=a)}


def _refl(dbz=45.0):
    return np.full((NRAYS, NGATES), dbz)


def test_correct_false_returns_the_field_untouched():
    r, note = correct_attenuation(_Radar(), _cfg(correct=False), _refl(),
                                  np.zeros((NRAYS, NGATES)))
    assert np.array_equal(r, _refl()) and 'not applied' in note


def test_the_section_and_the_correct_key_are_required():
    """Off is a decision the config makes, never one an absent section
    makes for it."""
    with pytest.raises(MissingConfigKeyError):
        correct_attenuation(_Radar(), {}, _refl(), np.zeros((NRAYS, NGATES)))
    with pytest.raises(MissingConfigKeyError, match=r'attenuation\.correct'):
        correct_attenuation(_Radar(), {'corrections': {'attenuation': {}}},
                            _refl(), np.zeros((NRAYS, NGATES)))


@pytest.mark.parametrize('key', ['relation', 'require_band_match',
                                 'liquid_only', 'max_correction_db'])
def test_every_key_is_required_when_correcting(key):
    cfg = _cfg()
    del cfg['corrections']['attenuation'][key]
    with pytest.raises(MissingConfigKeyError, match=key):
        correct_attenuation(_Radar(), cfg, _refl(), np.zeros((NRAYS, NGATES)))


def test_correction_accumulates_along_the_ray():
    """Two-way path integral: a gate further out sits behind more rain."""
    out, _ = correct_attenuation(_Radar(), _cfg(), _refl(),
                                 np.zeros((NRAYS, NGATES)))
    added = out[0] - 45.0
    assert added[0] < added[10] < added[-1]
    assert np.all(np.diff(added) >= 0)


def test_correction_is_capped():
    out, _ = correct_attenuation(_Radar(), _cfg(max_correction_db=2.0),
                                 _refl(55.0), np.zeros((NRAYS, NGATES)))
    assert np.nanmax(out - 55.0) == pytest.approx(2.0)


def test_wrong_band_is_refused():
    """The error this guard exists for: an X-band relation on a C-band radar."""
    with pytest.raises(ValueError, match='C-band'):
        correct_attenuation(_Radar(ghz=5.6), _cfg(), _refl(),
                            np.zeros((NRAYS, NGATES)))


def test_the_band_guard_can_be_overridden_deliberately():
    out, _ = correct_attenuation(_Radar(ghz=5.6), _cfg(require_band_match=False),
                                 _refl(), np.zeros((NRAYS, NGATES)))
    assert np.nanmax(out) > 45.0


def test_a_matching_band_is_accepted():
    out, note = correct_attenuation(_Radar(ghz=5.6), _cfg(relation=C_MADE_UP),
                                    _refl(), np.zeros((NRAYS, NGATES)))
    assert np.nanmax(out) > 45.0 and 'C-band' in note


def test_no_relation_named_raises_rather_than_guessing():
    with pytest.raises(ConfigError):
        correct_attenuation(_Radar(), _cfg(relation=None), _refl(),
                            np.zeros((NRAYS, NGATES)))
    with pytest.raises(ConfigError):
        correct_attenuation(_Radar(), _cfg(relation='nonesuch'), _refl(),
                            np.zeros((NRAYS, NGATES)))


def test_liquid_only_stops_at_the_melting_level():
    """A rain relation integrated up through an anvil manufactures a large
    correction out of nothing."""
    alt = np.broadcast_to(np.linspace(0, 12000, NGATES), (NRAYS, NGATES)).copy()
    cfg = _cfg(liquid_only=True)
    cfg['environment'] = environment(melting_level_km=4.0)
    out, _ = correct_attenuation(_Radar(), cfg, _refl(), alt)
    added = out[0] - 45.0
    warm = alt[0] / 1000.0 <= 4.0
    # it keeps growing through the rain and then holds flat above it
    assert added[warm][-1] > added[warm][0]
    assert np.allclose(np.diff(added[~warm]), 0.0)


def test_the_three_published_fits_barely_differ():
    """Under 3% apart, against a factor of five between bands."""
    Z = 10 ** (5.0)
    vals = [1.375e-4 * Z ** 0.778, 1.355e-4 * Z ** 0.781, 1.379e-4 * Z ** 0.779]
    assert (max(vals) - min(vals)) / np.mean(vals) < 0.03


def test_first_gate_gets_no_correction():
    """The integral is EXCLUSIVE: a gate is corrected for the loss on the way
    to it, not for its own contribution -- so the first gate adds nothing.
    (A cumsum that includes the gate itself is the bug this pins.)"""
    out, _ = correct_attenuation(_Radar(), _cfg(), _refl(),
                                 np.zeros((NRAYS, NGATES)))
    assert out[0, 0] == pytest.approx(45.0)
    assert out[0, 1] > 45.0


def test_nonuniform_gate_spacing_is_refused():
    """The path integral assumes one spacing per ray; staggered gates must
    fail loudly, not integrate silently wrong."""
    r = _Radar()
    r.range = {'data': np.concatenate([np.arange(25) * 250.0,
                                       6250.0 + np.arange(25) * 500.0])}
    with pytest.raises(ValueError, match='not uniform'):
        correct_attenuation(r, _cfg(), _refl(), np.zeros((NRAYS, NGATES)))


def test_note_reports_the_freezing_level_source():
    """liquid_only resolves a freezing level; the note must say from where,
    so the output file can carry the provenance."""
    alt = np.broadcast_to(np.linspace(0, 12000, NGATES), (NRAYS, NGATES)).copy()
    cfg = _cfg(liquid_only=True)
    cfg['environment'] = environment(melting_level_km=4.0)
    _, note = correct_attenuation(_Radar(), cfg, _refl(), alt)
    assert 'melting_level_km = 4' in note
