"""Physics tables the config chooses from by name.

A table is science, not a site setting: it belongs in the package, cited,
in one copy, and a config names the entry it wants. A config may also give
an entry inline (the same keys) for a radar these tables do not cover --
see `attenuation_relation` and `vertical_error_table`.
"""

from collections.abc import Mapping

import numpy as np

from .errors import ConfigError

__all__ = [
    "ATTENUATION_RELATIONS",
    "VERTICAL_ERROR_TABLES",
    "attenuation_relation",
    "vertical_error_table",
]

# Specific attenuation k = a * Z^b, dB/km one way, Z linear (mm^6 m^-3).
# a and b are band specific: the same rain attenuates about five times more
# at X band than at C, so a relation is never applied to another band
# without the config saying so (sedimentation.attenuation.require_band_match).
ATTENUATION_RELATIONS = {
    'x_andsager': {'band': 'X', 'a': 1.375e-4, 'b': 0.778,
                   'source': 'Andsager drop shape relation'},
    'x_keenan':   {'band': 'X', 'a': 1.355e-4, 'b': 0.781,
                   'source': 'Keenan drop shape relation'},
    'x_minimum':  {'band': 'X', 'a': 1.379e-4, 'b': 0.779,
                   'source': 'minimum-difference fit'},
}

# The elevation-dependent anchor error dVh1 (m/s) of the vertical-velocity
# error model (calculate_vertical_velocity_error), tabulated against
# elevation (deg, increasing). The table's empirical source is a comparison
# of the edge anchors against radiosondes on the CHIVO radar during
# TRACER 2022.
VERTICAL_ERROR_TABLES = {
    'chivo_tracer_2022': {'elevation_deg': [0, 30, 40, 50, 60, 70, 90],
                          'dVh1': [3, 3, 4, 6, 9, 11, 13]},
}


def _pick(value, table: Mapping, what: str, keys: tuple) -> dict:
    """A named entry of `table`, or an inline mapping holding `keys`."""
    if isinstance(value, str):
        if value not in table:
            raise ConfigError(
                f'{what} {value!r} is not a known name; choose one of '
                f'{sorted(table)} or give the entry inline with keys {keys}')
        return dict(table[value], name=value)
    if isinstance(value, Mapping):
        missing = [k for k in keys if k not in value]
        if missing:
            raise ConfigError(
                f'inline {what} is missing {missing}; it needs {keys}')
        return dict(value, name='inline')
    raise ConfigError(
        f'{what} must be a name from {sorted(table)} or an inline mapping '
        f'with keys {keys}, got {value!r}')


def attenuation_relation(value) -> dict:
    """The attenuation relation a config asks for.

    Parameters
    ----------
    value : str or Mapping
        A key of `ATTENUATION_RELATIONS`, or ``{band, a, b}`` inline.

    Returns
    -------
    dict
        ``band``, ``a``, ``b`` (and ``source`` when known) plus ``name``.
    """
    return _pick(value, ATTENUATION_RELATIONS, 'attenuation relation',
                 ('band', 'a', 'b'))


def vertical_error_table(value) -> tuple:
    """The dVh1(elevation) table a config asks for.

    Parameters
    ----------
    value : str or Mapping
        A key of `VERTICAL_ERROR_TABLES`, or ``{elevation_deg, dVh1}``
        inline.

    Returns
    -------
    elevation_deg, dVh1 : numpy.ndarray
        Same length, at least two points, elevation strictly increasing.

    Raises
    ------
    ConfigError
        On an unknown name or an inconsistent table.
    """
    t = _pick(value, VERTICAL_ERROR_TABLES, 'dVh1 table',
              ('elevation_deg', 'dVh1'))
    e_tab = np.asarray(t['elevation_deg'], dtype=float)
    v_tab = np.asarray(t['dVh1'], dtype=float)
    if e_tab.size != v_tab.size or e_tab.size < 2:
        raise ConfigError(f'dVh1 table {t["name"]!r}: elevation_deg and dVh1 '
                          'must be the same length and hold at least two points')
    if np.any(np.diff(e_tab) <= 0):
        raise ConfigError(f'dVh1 table {t["name"]!r}: elevation_deg must '
                          'increase monotonically')
    return e_tab, v_tab
