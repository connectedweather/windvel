"""Corrections to the input variables: step 0, before any stage.

`apply_corrections` writes each enabled correction as a NEW field on the
radar object; the delivered fields are never modified, and both are saved.
`working_refl_field` names the reflectivity field the stages read — the one
resolution rule, read by the object contour, the weak-echo anchor screen
and the fall speed alike.
"""

import logging
from typing import Dict

import numpy as np

from .config import require, section
from .errors import ConfigError, InputDataError, InputFieldError
from .soundings import resolve_freezing_level
from .tables import attenuation_relation
from .utils import as_float_nan, make_sr_alt_maps

logger = logging.getLogger(__name__)

__all__ = [
    "CORRECTED_REFL_FIELD",
    "apply_corrections",
    "correct_attenuation",
    "working_refl_field",
]

# The corrected-reflectivity field name, pinned: files already written carry
# it, so renaming it would orphan the field in every output on disk.
CORRECTED_REFL_FIELD = 'reflectivity_corrected'


def correct_attenuation(radar, cfg: Dict, refl: np.ndarray, alt_map: np.ndarray):
    """Add back the two-way path attenuation, for data that is not already corrected.

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config; ``corrections.attenuation`` is read. ``correct``
        is REQUIRED; when true, so are ``relation`` (a name from
        `windvel.tables.ATTENUATION_RELATIONS` or an inline ``{band, a, b}``),
        ``require_band_match``, ``liquid_only`` and ``max_correction_db``.
    refl : numpy.ndarray
        Reflectivity, dBZ, NaN where missing.
    alt_map : numpy.ndarray
        Gate altitude, m (for the freezing level when ``liquid_only``).

    Returns
    -------
    corrected : numpy.ndarray
        Reflectivity with the correction added, dBZ.
    note : str
        What was applied, for the output field's attributes.

    Notes
    -----
    At C and X band the beam loses power passing through rain, so a gate
    behind a cell reads lower than it should: it can pass a weak-echo
    screen while holding rain, shrink an object at the reflectivity
    contour, and bias the fall speed low -- and since dw/dVsed = 1 exactly,
    straight into w. The fall-speed relations were built at S band, where
    the loss is negligible.

    The specific attenuation is the power law Ah = a * Z^b (dB/km one way,
    Z linear), from the reflectivity itself; accumulating it along the ray
    and doubling gives the two-way correction to add at each gate:

        dZ(r) = 2 * sum_{r' < r} a * Z(r')^b * dr

    exclusive of the gate's own contribution.

    Three things this does NOT do, all of them deliberate:

      it does not guess the coefficients. a and b are band specific -- the
      same rain attenuates about five times more at X band than at C, and a
      hundred times more than at S -- so there is no default. Name a
      relation from the package table, or give one inline, or the
      retrieval raises rather than inventing one.

      it does not run above the melting level, where the relation does not
      hold: ice attenuates far less than rain, and integrating a rain
      relation up through an anvil produces a large correction out of
      nothing.

      it does not trust itself without limit. The estimate is
      self-referential: it reads attenuation from an already-attenuated Z,
      so it under-corrects, and the shortfall compounds along the ray. Past
      a few dB that shortfall is the dominant error and the total is capped
      rather than extrapolated; the number of gates at the cap is logged.

    References
    ----------
    Battan, L. J. (1973), Radar Observation of the Atmosphere, Univ. Chicago
    Press -- the k = a Z^b specific-attenuation form.
    """
    acfg = section(section(cfg, 'corrections'), 'attenuation')
    s = 'corrections.attenuation'
    if not bool(require(acfg, s, 'correct',
                        'true to add back path attenuation, false to leave '
                        'the reflectivity as delivered')):
        return refl, 'not applied'

    chosen = require(acfg, s, 'relation',
                     'a name from windvel.tables.ATTENUATION_RELATIONS, or '
                     'an inline {band, a, b}')
    if chosen is None:
        raise ConfigError(
            f'{s}.correct is true but {s}.relation is null. Name one of the '
            'package relations or give one inline. There is no default: a and '
            'b are band specific and the wrong band is an error of a factor of '
            'five or more.')
    rel = attenuation_relation(chosen)
    name = rel['name']
    a, b = float(rel['a']), float(rel['b'])
    band = str(rel['band']).upper()
    require_band = bool(require(acfg, s, 'require_band_match',
                                'refuse a relation for a different band'))
    liquid_only = bool(require(acfg, s, 'liquid_only',
                               'apply below the melting level only'))
    cap = float(require(acfg, s, 'max_correction_db',
                        'cap on the two-way total, dB'))

    # does the relation match the radar it is being applied to?
    freq = None
    ip = getattr(radar, 'instrument_parameters', None) or {}
    if 'frequency' in ip:
        f = float(np.asarray(ip['frequency']['data']).ravel()[0])
        freq = f / 1e9 if f > 1e6 else f
    if freq is not None and band:
        actual = ('S' if freq < 4 else 'C' if freq < 8 else 'X' if freq < 12
                  else '?')
        if actual != band and require_band:
            raise ConfigError(
                '%s.relation %r is a %s-band relation but this radar '
                'is %s-band (%.2f GHz). Applying it would be wrong by roughly a '
                'factor of five per band step. Choose a %s-band relation, or set '
                '%s.require_band_match to false if you mean it.'
                % (s, name, band, actual, freq, actual, s))

    warm = np.ones(refl.shape, dtype=bool)
    fl_note = ''
    if liquid_only:
        warm, fl_src = resolve_freezing_level(radar, cfg, alt_map)
        fl_note = f'; liquid-only below freezing level from {fl_src}'

    rng_km = np.asarray(radar.range['data'], dtype=float) / 1000.0
    if rng_km.size > 1:
        d = np.diff(rng_km)
        dr = float(np.median(d))
        # the path integral below assumes one gate spacing per ray; on a
        # staggered-gate radar it would be silently wrong, so refuse instead
        if float(np.ptp(d)) > 1e-3 * abs(dr):
            raise InputDataError(
                'gate spacing is not uniform (%.4g..%.4g km); the attenuation '
                'path integral assumes a single spacing' % (d.min(), d.max()))
    else:
        dr = 0.0

    z_lin = 10.0 ** (np.clip(np.asarray(refl, dtype=float), -30.0, 70.0) / 10.0)
    ah = np.where(np.isfinite(refl) & warm, a * z_lin ** b, 0.0)
    # loss accumulated ON THE WAY TO each gate -- exclusive of the gate's own
    # contribution -- then doubled for the two-way path
    two_way = 2.0 * (np.cumsum(ah, axis=1) - ah) * dr
    n_capped = int((two_way > cap).sum())
    two_way = np.clip(two_way, 0.0, cap)
    if n_capped:
        logger.info('attenuation correction: %d gates capped at %g dB',
                    n_capped, cap)

    out = np.asarray(refl, dtype=float) + np.where(np.isfinite(refl), two_way, 0.0)
    hit = float(np.nanmax(two_way)) if two_way.size else 0.0
    return out, (f'{name} (a={a:g}, b={b:g}, {band}-band), '
                 f'up to {hit:.1f} dB added, cap {cap:g} dB{fl_note}')


def apply_corrections(radar, cfg: Dict):
    """Apply the enabled corrections, writing each as a new field.

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config; ``corrections`` decides what runs,
        ``input_variables.refl_var`` names the field to correct.

    Returns
    -------
    radar : pyart.core.Radar
        The same object. With ``corrections.attenuation.correct`` false,
        untouched; with it true, `CORRECTED_REFL_FIELD` is added and the
        delivered field is left as it came.

    Raises
    ------
    InputFieldError
        When ``input_variables.refl_var`` names a field the file lacks.
    """
    acfg = section(section(cfg, 'corrections'), 'attenuation')
    if not bool(require(acfg, 'corrections.attenuation', 'correct',
                        'true to add back path attenuation, false to leave '
                        'the reflectivity as delivered')):
        return radar

    refl_var = require(section(cfg, 'input_variables'), 'input_variables',
                       'refl_var')
    if refl_var not in radar.fields:
        raise InputFieldError(
            f'input_variables.refl_var names {refl_var!r}, which is not in '
            f'the file; fields present: {sorted(radar.fields)}')

    _, alt_map = make_sr_alt_maps(radar)
    refl = as_float_nan(radar.fields[refl_var]['data'])
    corrected, note = correct_attenuation(radar, cfg, refl, alt_map)
    logger.info('corrections: attenuation applied to %r -> %r (%s)',
                refl_var, CORRECTED_REFL_FIELD, note)
    radar.add_field(CORRECTED_REFL_FIELD, {
        'long_name': 'Reflectivity, path attenuation added back',
        'units': 'dBZ',
        'comment': f'corrected from {refl_var!r}: {note}',
        'data': np.ma.masked_invalid(corrected),
    }, replace_existing=True)
    return radar


def working_refl_field(cfg: Dict) -> str:
    """The reflectivity field name the stages read; the one resolution rule.

    Parameters
    ----------
    cfg : dict
        The whole config; ``corrections.attenuation.correct`` and
        ``input_variables.refl_var`` are read.

    Returns
    -------
    str
        `CORRECTED_REFL_FIELD` when the attenuation correction is on,
        ``input_variables.refl_var`` when it is off.
    """
    acfg = section(section(cfg, 'corrections'), 'attenuation')
    if bool(require(acfg, 'corrections.attenuation', 'correct',
                    'true to add back path attenuation, false to leave '
                    'the reflectivity as delivered')):
        return CORRECTED_REFL_FIELD
    return require(section(cfg, 'input_variables'), 'input_variables',
                   'refl_var')
