"""The retrieval: sedimentation, horizontal wind, vertical velocity, structures.

Stages run in the order `calculate_windvel` calls them. Each stage takes the
config and reads its own section up front, so a bad config fails before any
sweep is touched; each writes its result back onto the radar object as a
field whose attributes say how it was made.
"""

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, Dict, List

import numpy as np

from . import coherent_structures as CS
from .config import require, section
from .corrections import correct_attenuation, working_refl_field
from .errors import ConfigError, InputFieldError
from .soundings import resolve_freezing_level
from .tables import vertical_error_table
from .utils import (
    FILL_VALUE,
    altitude_bin_edges,
    as_float_nan,
    gate_dist_alt_km,
    ids_as_int,
    make_sr_alt_maps,
    project_to_radial_component,
    scrub_fill,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GIANGRANDE_DARWIN2026_FORMULAS",
    "GIANGRANDE_OKLAHOMA2013_FORMULAS",
    "HV_SRC_INTERPOLATED",
    "HV_SRC_MEASURED",
    "HV_SRC_MIRRORED",
    "HV_SRC_NONE",
    "HV_SRC_SMOOTHED",
    "OUTPUT_FIELDS",
    "SED_METHODS",
    "SED_METHODS_RENAMED",
    "RepairSettings",
    "apply_object_selection_criteria",
    "calculate_vertical_velocity",
    "calculate_vertical_velocity_error",
    "calculate_windvel",
    "compute_sed_vel",
    "compute_sed_vel_zt",
    "correct_attenuation",
    "detect_coherent_objects",
    "get_horizontal_velocity",
    "get_sed_vel",
    "resolve_freezing_level",
    "resolve_repair_settings",
    "smooth_edge_profile",
    "vertical_velocity_usability",
]

# =============================================================================
# The fields this package adds to the radar object
# =============================================================================
# THE single source of truth for what a windvel output contains, in creation
# order. Extract-mode saving and the quick-look panel both consume this list;
# a new output field is not finished until it is added here.

OUTPUT_FIELDS = [
    'reflectivity_corrected',                  # apply_corrections; present only
                                               #   when a correction is on
    'local_object_ids',                        # select_cloud_transects
    'object_boundaries',                       # select_cloud_transects
    'sedimentation_velocity',
    'horizontal_velocity',
    'horizontal_velocity_source',
    'vertical_velocity',
    'vertical_velocity_error',
    'vertical_velocity_flag',
    'coherent_objects_threshold',
    'coherent_objects_threshold_truncated',
    'coherent_objects_threshold_flag',
    'coherent_objects_persistence',
    'coherent_objects_persistence_truncated',
    'coherent_objects_persistence_flag',
]

# =============================================================================
# Horizontal-velocity provenance codes
# =============================================================================
# ONE vocabulary, used at two levels. First per EDGE, while building the
# per-altitude-bin edge means; then per GATE, written to the
# `horizontal_velocity_source` field. A gate is interpolated between two edges
# and reports the WEAKEST rung of the two -- see the collapse in
# get_horizontal_velocity, which is the only place the two levels meet.
#
#   NONE      rung 0  no estimate. The edge stays NaN, and because a bin needs
#                     BOTH edges finite to be interpolated, no gate in it is
#                     written either -- so NONE never reaches the collapse.
#   MEASURED  rung 1  in-bin statistic over enough weak-echo gates to stand alone
#   MIRRORED  rung 2  innermost mirror: a NEAR edge with nothing usable takes
#                     the median of the far half's innermost gates, and the far
#                     anchor is rebuilt from the gates left over. Only the near
#                     edge is ever mirrored, and only against an already
#                     MEASURED far edge, so at gate level this code always means
#                     near-mirrored / far-measured -- there is no ambiguity for
#                     the collapse to lose.
#   INTERPOLATED rung 3  a small interior gap in one edge's profile, bounded
#                     by finite anchors above AND below, filled linearly in
#                     altitude (fill_interpolate_max_gap_bins). The weakest
#                     estimating rung: no gate in the bin supplied the value.
#   SMOOTHED          the profile spike correction rewrote the value,
#                     whatever rung produced it originally. Outranks the rest.
#
# Codes 3 and 4 are reserved and never assigned, so outputs from every
# version of this package stay comparable; the numbering is pinned by
# test_provenance_codes_are_pinned.
HV_SRC_NONE         = 0
HV_SRC_MEASURED     = 1
HV_SRC_MIRRORED     = 2
HV_SRC_SMOOTHED     = 5
HV_SRC_INTERPOLATED = 6

# =============================================================================
# Fall-speed relations
# =============================================================================
# Reflectivity-weighted hydrometeor fall speed Vt by PID class, Vt = a * Z^b
# with Z linear (mm^6 m^-3) unless stated. The science -- where each number
# comes from and how the tables compare -- is in the docstrings of
# `compute_sed_vel` and `compute_sed_vel_zt`.
#
# Both PID tables are Giangrande first-author, so the SITE is the part of the
# key that tells them apart, and the part that decides which to use: a
# continental mid-latitude drop population is not a tropical maritime one.
# Any table added here takes the same AUTHOR_SITE+YEAR form of key.

OK13_RAIN_A, OK13_RAIN_B = 3.15, 0.098      # Oklahoma 2013 rain classes
OK13_ICE_MPS = 2.0                          # Oklahoma 2013 ice and snow: constant
GRAUPEL_OFFSET_MPS, GRAUPEL_Z_REF_DBZ = 2.2, 33.0   # the offset square-root form

DW26_RAIN_A, DW26_RAIN_B = 2.59, 0.1        # Darwin 2026 convective rain row
DW26_GRAUPEL_A, DW26_GRAUPEL_B = 1.03, 0.19  # Darwin 2026 "faster" mixed/graupel
DW26_SNOW_A, DW26_SNOW_B = 0.43, 0.19        # Darwin 2026 fixed-b snow midpoint


def _power_law(a: float, b: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Vt = a * Z^b, as a (dbz, Z) callable like every other table entry."""
    return lambda dbz, Z: a * Z**b


def _graupel_sqrt(dbz, Z):
    """The offset square-root graupel form of Giangrande et al. (2013)."""
    return GRAUPEL_OFFSET_MPS + np.sqrt(10**((dbz - GRAUPEL_Z_REF_DBZ)/10.0))


def _constant(v: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return lambda dbz, Z: v


_OK13_RAIN = _power_law(OK13_RAIN_A, OK13_RAIN_B)
_OK13_ICE = _constant(OK13_ICE_MPS)
_DW26_RAIN = _power_law(DW26_RAIN_A, DW26_RAIN_B)
_DW26_GRAUPEL = _power_law(DW26_GRAUPEL_A, DW26_GRAUPEL_B)
_DW26_SNOW = _power_law(DW26_SNOW_A, DW26_SNOW_B)
_NO_FALL_SPEED = _constant(np.nan)

# PID classes 1..18, in the order the classifier numbers them.
GIANGRANDE_OKLAHOMA2013_FORMULAS: Dict[int, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    1:  _NO_FALL_SPEED,   # Cloud
    2:  _OK13_RAIN,       # Drizzle
    3:  _OK13_RAIN,       # Light_Rain
    4:  _OK13_RAIN,       # Moderate_Rain
    5:  _OK13_RAIN,       # Heavy_Rain
    6:  _OK13_RAIN,       # Hail
    7:  _OK13_RAIN,       # Rain_Hail_Mixture
    8:  _graupel_sqrt,    # Graupel_Small_Hail
    9:  _graupel_sqrt,    # Graupel_Rain
    10: _OK13_ICE,        # Dry_Snow
    11: _OK13_ICE,        # Wet_Snow
    12: _OK13_ICE,        # Ice_Crystals
    13: _OK13_ICE,        # Irreg_Ice_Crystals
    14: _OK13_ICE,        # Supercooled_Liquid_Droplets
    15: _NO_FALL_SPEED,   # Flying_Insects
    16: _NO_FALL_SPEED,   # Second_Trip
    17: _NO_FALL_SPEED,   # Ground_Clutter
    18: _NO_FALL_SPEED,   # Receiver_Saturation
}

GIANGRANDE_DARWIN2026_FORMULAS: Dict[int, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    1:  _NO_FALL_SPEED,   # Cloud
    2:  _DW26_RAIN,       # Drizzle
    3:  _DW26_RAIN,       # Light_Rain
    4:  _DW26_RAIN,       # Moderate_Rain
    5:  _DW26_RAIN,       # Heavy_Rain
    6:  _DW26_RAIN,       # Hail
    7:  _DW26_RAIN,       # Rain_Hail_Mixture
    8:  _DW26_GRAUPEL,    # Graupel_Small_Hail
    9:  _DW26_GRAUPEL,    # Graupel_Rain
    10: _DW26_SNOW,       # Dry_Snow
    11: _DW26_SNOW,       # Wet_Snow
    12: _DW26_SNOW,       # Ice_Crystals
    13: _DW26_SNOW,       # Irreg_Ice_Crystals
    14: _DW26_SNOW,       # Supercooled_Liquid_Droplets
    15: _NO_FALL_SPEED,   # Flying_Insects
    16: _NO_FALL_SPEED,   # Second_Trip
    17: _NO_FALL_SPEED,   # Ground_Clutter
    18: _NO_FALL_SPEED,   # Receiver_Saturation
}

# The reflectivity / temperature method (no PID): see compute_sed_vel_zt.
ZT_RAIN_CONV, ZT_RAIN_STRAT = 2.65, OK13_RAIN_A
ZT_RAIN_B = OK13_RAIN_B
ZT_SNOW_A, ZT_SNOW_B = 0.37, 0.19
ZT_Z_CONVECTIVE, ZT_Z_GRAUPEL = 40.0, GRAUPEL_Z_REF_DBZ

# PID classes whose sqrt graupel form is replaced by the rain relation above
# GRAUPEL_RAIN_OVERRIDE_DBZ -- see compute_sed_vel.
GRAUPEL_PID_CLASSES = (8, 9)
GRAUPEL_RAIN_OVERRIDE_DBZ = 50.0
GRAUPEL_RAIN_OVERRIDE_CLASS = 2

# The air-density correction applied to every fall speed: see get_sed_vel.
DENSITY_SCALE_HEIGHT_KM = 8.5
DENSITY_EXPONENT = 0.4

SED_METHODS = ('pid_giangrande_oklahoma2013',
               'pid_giangrande_darwin2026',
               'reflectivity_temperature')

# Superseded method names -> current ones. A config naming one of these
# fails with the current name and is told the COEFFICIENTS are identical, so
# nobody re-runs a campaign looking for a science difference that is not
# there.
SED_METHODS_RENAMED = {
    'pid_giangrande2013': 'pid_giangrande_oklahoma2013',
    'pid_darwin2026':     'pid_giangrande_darwin2026',
}


def compute_sed_vel_zt(dbz: np.ndarray, above_freezing: np.ndarray) -> np.ma.MaskedArray:
    """Fall speed from reflectivity and the sign of temperature, without a PID.

    Parameters
    ----------
    dbz : numpy.ndarray
        Reflectivity, dBZ. NaN or masked where unknown.
    above_freezing : numpy.ndarray of bool
        True where T >= 0 degC. Only the sign of the temperature enters, so a
        freezing level is enough -- a full temperature field is not required.

    Returns
    -------
    numpy.ma.MaskedArray
        Fall speed, m/s, positive downward. Anything unclassifiable is masked
        rather than set to a plausible-looking number.

    Notes
    -----
    Modified from Giangrande et al. (2013); With Z linear:

        T >= 0, Z > 40 dBZ    2.65 * Z^0.098                convective rain
        T >= 0, Z <= 40       3.15 * Z^0.098                stratiform rain
        T <  0, Z > 40        2.65 * Z^0.098                convective core
        T <  0, 33 < Z <= 40  2.2 + sqrt(10^((dBZ-33)/10))  graupel
        T <  0, Z <= 33       0.37 * Z^0.19                 snow

    References
    ----------
    Giangrande, S. E., et al. (2013), J. Appl. Meteor. Climatol., 52,
    2278-2295.
    """
    dbz_ma = np.ma.masked_invalid(np.asarray(dbz, dtype=float))
    Z = 10 ** (dbz_ma / 10.0)
    warm = np.asarray(above_freezing, dtype=bool)

    out = np.ma.masked_all(dbz_ma.shape, dtype=float)
    conv = dbz_ma > ZT_Z_CONVECTIVE
    graupel = (~warm) & (dbz_ma > ZT_Z_GRAUPEL) & ~conv
    snow = (~warm) & (dbz_ma <= ZT_Z_GRAUPEL)
    strat = warm & ~conv

    out[conv] = (ZT_RAIN_CONV * Z**ZT_RAIN_B)[conv]
    out[strat] = (ZT_RAIN_STRAT * Z**ZT_RAIN_B)[strat]
    out[graupel] = _graupel_sqrt(dbz_ma, Z)[graupel]
    out[snow] = (ZT_SNOW_A * Z**ZT_SNOW_B)[snow]
    out.mask |= dbz_ma.mask
    return out


# =============================================================================
# Helpers
# =============================================================================
def _hdist_km_from_sr_alt(sr_map_km: np.ndarray, alt_map_m: np.ndarray) -> np.ndarray:
    """Horizontal distance (km) from slant range and altitude."""
    z_km = alt_map_m / 1000.0
    h2 = np.maximum(sr_map_km**2 - z_km**2, 0.0)
    return np.sqrt(h2)


def _mask_invalid(a):
    """Return masked array with mask on NaNs/inf; no sentinel values."""
    return np.ma.masked_invalid(a.astype(float, copy=False))


def _field(radar, name: str):
    """A field's data, or a loud error naming the field the config asked for."""
    if name not in radar.fields:
        raise InputFieldError(
            f'field {name!r} is not in this file (has: {sorted(radar.fields)})')
    return radar.fields[name]['data']


# =============================================================================
# Sedimentation
# =============================================================================
def compute_sed_vel(dbz: np.ndarray, pid: np.ndarray,
                    formulas: Dict[int, Callable]) -> np.ma.MaskedArray:
    """Fall speed per gate from a particle identification and reflectivity.

    Parameters
    ----------
    dbz : numpy.ndarray
        Reflectivity, dBZ.
    pid : numpy.ndarray of int
        Particle-id class per gate, keyed as in the tables.
    formulas : dict
        ``{pid_class: f(dbz, Z)}``, one of `GIANGRANDE_OKLAHOMA2013_FORMULAS`
        or `GIANGRANDE_DARWIN2026_FORMULAS`.

    Returns
    -------
    numpy.ma.MaskedArray
        Fall speed, m/s, positive downward; masked where the class has no
        relation or the reflectivity is missing.

    Notes
    -----
    Vt enters the retrieval as w = (Vr + Vt*sin(e) - Vh*cos(e)) / sin(e), so
    dw/dVt = 1 exactly: an error in Vt is an equal and opposite error in w,
    at every elevation.

    OKLAHOMA 2013: rain follows a power law in Z,
    Vt = 3.15 * Z^0.098; graupel and small hail the offset square-root form
    Vt = 2.2 + sqrt(10^((dBZ - 33)/10)); ice and snow are held at a constant
    2 m/s regardless of reflectivity. Derived by matching precipitation-mode
    Doppler against collocated UHF profiler vertical velocities.

    DARWIN 2026: a single power law Vt = a * Z^b
    throughout, including for ice and snow. From the paper's Table 1:

        rain      a=2.59, b=0.1   the CONVECTIVE rain row (stratiform rows
                                  are a=3.0-3.3 at the same b; this retrieval
                                  targets convection)
        graupel   a=1.03, b=0.19  the best-fit "faster" mixed/graupel row
                                  (R^2=0.52; a "slower" alternative a=0.70,
                                  b=0.25 exists). Applicable for
                                  25 <~ Z <~ 45 dBZ only; the paper puts the
                                  graupel-rain intersection near 40-45 dBZ
                                  for Darwin.
        snow/ice  a=0.43, b=0.19  midpoint of the fixed-b=0.19 snow range
                                  [0.41, 0.45] across 4.5-8.5 km (the paper's
                                  own best fit prefers b=0.1 with
                                  a=[0.62, 0.72])

    Rain/snow rows applicable over 0 < Z <~ 50 dBZ. All relations are at
    sea-level pressure; the density correction in `get_sed_vel` supplies the
    altitude adjustment. How the tables compare, in m/s, before that
    correction (Z-T is `compute_sed_vel_zt`):

        dBZ   rain: 2013 / 2026 / Z-T   graupel: 2013 / 2026   ice: 2013 / 2026 / Z-T
          0        3.15 / 2.59 / 3.15         2.22 / 1.03           2.00 / 0.43 / 0.37
         20        4.95 / 4.10 / 4.95         2.42 / 2.47           2.00 / 1.03 / 0.89
         40        7.77 / 6.51 / 6.54         4.44 / 5.93           2.00 / 2.47 / 2.14

    GRAUPEL OVERRIDE: gates the PID calls graupel or graupel-rain (classes 8,
    9) above 50 dBZ take the RAIN relation instead. This reads as a guard
    against the 2013 square-root form running away past its fitted range
    (~15 m/s at 55 dBZ); it is uncited, has no config knob, and also fires
    under the Darwin table whose graupel relation is already tame. Kept as
    inherited pending an A/B on strong-core scans; the number of gates it
    touches is logged.

    References
    ----------
    Giangrande, S. E., et al. (2013): A summary of convective-core vertical
    velocity properties using ARM UHF wind profilers in Oklahoma, J. Appl.
    Meteor. Climatol., 52, 2278-2295.
    Giangrande, S. E., C. R. Williams, and A. Protat (2026): Dual-frequency
    profiler study of hydrometeor fall speeds in tropical deep convection,
    EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-856.
    """
    dbz_ma = np.ma.masked_invalid(dbz)
    Z      = 10 ** (dbz_ma / 10.0)

    pid_mod = pid.copy()
    override = np.isin(pid_mod, GRAUPEL_PID_CLASSES) & (dbz_ma > GRAUPEL_RAIN_OVERRIDE_DBZ)
    pid_mod[override] = GRAUPEL_RAIN_OVERRIDE_CLASS
    n_override = int(np.ma.filled(override, False).sum())
    if n_override:
        logger.info('graupel > %g dBZ override: %d gates take the rain '
                    'relation', GRAUPEL_RAIN_OVERRIDE_DBZ, n_override)

    valid = ~dbz_ma.mask
    vel   = np.full(dbz.shape, np.nan, dtype=float)
    for code, func in formulas.items():
        m = valid & (pid_mod == code)
        if m.any():
            vel[m] = func(dbz_ma.data[m], Z.data[m])
    return np.ma.masked_invalid(vel)


def get_sed_vel(radar, cfg: Dict, sweeps):
    """Fall speed for the gates of the surviving cloud objects.

    Parameters
    ----------
    radar : pyart.core.Radar
        Carrying ``local_object_ids`` from `select_cloud_transects` after
        `apply_object_selection_criteria` has zeroed the rejected objects.
    cfg : dict
        The whole config. ``sedimentation.method`` (one of `SED_METHODS`)
        and the input variable names are REQUIRED. Reads the corrected
        reflectivity when ``corrections.attenuation.correct`` is true,
        ``input_variables.refl_var`` when it is false.
    sweeps : list of int
        The RHI sweeps to compute for.

    Returns
    -------
    velocity : numpy.ma.MaskedArray
        Fall speed, m/s, positive downward, masked outside the objects.
    notes : dict
        Provenance strings -- method, reflectivity field read, freezing-
        level source -- for the caller to write into the output field's
        attributes, so a file always says what happened.

    Notes
    -----
    Object selection is NOT done here: `apply_object_selection_criteria`
    runs first and zeroes the ids of rejected objects, so masking on
    ``local_object_ids <= 0`` is all the selection this needs.

    Every fall speed is scaled for air density,

        Vt(z) = Vt0 * (rho0 / rho(z))^0.4,   rho0 / rho ~= exp(z / H)

    with an isothermal scale height H = 8.5 km: the (rho0/rho)^0.4 form of
    Foote & du Toit (1969).

    References
    ----------
    Foote, G. B., and P. S. du Toit (1969): Terminal velocity of raindrops
    aloft, J. Appl. Meteor., 8, 249-253.
    """
    ivars = section(cfg, 'input_variables')
    refl_name = working_refl_field(cfg)
    refl = as_float_nan(_field(radar, refl_name))
    sel = ids_as_int(_field(radar, 'local_object_ids'))

    scfg = section(cfg, 'sedimentation')
    method = str(require(scfg, 'sedimentation', 'method',
                         'one of %r' % (list(SED_METHODS),)))
    if method in SED_METHODS_RENAMED:
        raise ConfigError(
            'sedimentation.method %r is now called %r. The COEFFICIENTS ARE '
            'UNCHANGED -- only the key is, because both PID tables are '
            'Giangrande first-author and are told apart by site. Update the '
            'config; results will be identical.'
            % (method, SED_METHODS_RENAMED[method]))
    if method not in SED_METHODS:
        raise ConfigError('sedimentation.method must be one of %r, got %r'
                          % (list(SED_METHODS), method))

    _, alt_map = make_sr_alt_maps(radar)

    notes = {'sedimentation_method': method,
             'reflectivity_field': refl_name}

    if method == 'reflectivity_temperature':
        warm, fl_src = resolve_freezing_level(radar, cfg, alt_map)
        notes['freezing_level_source'] = fl_src
        sed = compute_sed_vel_zt(refl, warm)
    else:
        pid = _field(radar, require(ivars, 'input_variables', 'pid_var'))
        table = (GIANGRANDE_OKLAHOMA2013_FORMULAS if method == 'pid_giangrande_oklahoma2013'
                 else GIANGRANDE_DARWIN2026_FORMULAS)
        sed = compute_sed_vel(refl, pid, table)
        notes['freezing_level_source'] = 'not needed (PID method)'
    sed.mask |= (sel <= 0)

    vel_corr = np.ma.masked_all_like(sed)
    for s in sweeps:
        slc = radar.get_slice(sweep=s)
        _, _, z = radar.get_gate_x_y_z(sweep=s)
        dens = np.exp((z/1000.0) / DENSITY_SCALE_HEIGHT_KM) ** DENSITY_EXPONENT
        vel_corr[slc] = sed[slc] * dens

    logger.info('sedimentation: %s', notes)
    return vel_corr, notes


# =============================================================================
# Horizontal velocity
# =============================================================================
@dataclass(frozen=True)
class RepairSettings:
    """The horizontal-wind anchor and repair settings, resolved from config.

    One object, built once by `resolve_repair_settings`, read by the
    retrieval AND stamped into the output file's attributes -- so the file
    can never claim a setting the retrieval did not use.
    """
    fill_mirror: bool
    fill_interpolate_max_gap_bins: int
    min_edge_gates: int
    anchor_max_elev_deg: float | None
    anchor_robust: bool
    max_candidate_mps: float | None
    spike_passes: List[int]
    spike_shear_per_km: float
    spike_k_mad: float
    spike_max_frac: float
    min_bin_gates: int
    refl_ladder_dbz: List[float]
    bin_size_m: float


def resolve_repair_settings(cfg: Dict) -> RepairSettings:
    """Read every horizontal-wind setting out of the config, once.

    Parameters
    ----------
    cfg : dict
        The whole config. Every key of ``horizontal_wind`` and its
        ``anchor`` and ``spike_correction`` subsections is REQUIRED.

    Returns
    -------
    RepairSettings

    Raises
    ------
    MissingConfigKeyError
        On any absent key.
    ConfigError
        On a malformed weak-echo ladder or a non-positive spike half-width.

    Notes
    -----
    What each setting decides (config key in brackets):

    anchor_max_elev_deg   [anchor.max_elevation_deg]
                          hv_candidate = u + w*tan(elev), so on a steep beam
                          the retrieval cannot separate horizontal wind from
                          vertical motion. Above ~65 deg tan(elev) > 2.1 and
                          a modest updraft reads as a large apparent wind.
                          null disables the screen.
    anchor_robust         [anchor.robust] median instead of mean, so a few
                          contaminated gates cannot drag the anchor.
    max_candidate_mps     [anchor.max_candidate_mps]
                          a gate whose implied horizontal wind is physically
                          impossible is not a wind measurement, so it gets
                          no vote (QC layer 2). 
    min_edge_gates        [anchor.min_gates] the sample an anchor needs; five
                          gates is not a sample.
    spike_passes          [spike_correction.passes]
                          half-widths of the profile spike correction, run
                          in turn; cleaning the obvious spikes at +-3 lets
                          the +-5 fit find its line through better points.
                          An empty list disables the correction.
    spike_shear_per_km    [spike_correction.shear_per_km] physical floor on
                          the residual threshold, m/s per km -- invariant to
                          the bin thickness.
    spike_k_mad           [spike_correction.k_mad] adaptive term: this many
                          robust sigmas of the
                          profile's own residuals. The threshold is the
                          LARGER of the two, so neither a clean profile nor
                          a genuinely sheared one is judged by the wrong
                          standard.
    spike_max_frac        [spike_correction.max_fraction] refuse to rewrite
                          more than this fraction of a profile: past about a
                          quarter the correction is no longer fixing a
                          profile, it is inventing one.
    min_bin_gates         [min_bin_gates] altitude bins holding fewer gates
                          than this are
                          not retrieved at all; near cloud top a bin can
                          fall to a handful of gates where an anchor is
                          noise. 0 disables the floor.
    fill_interpolate_max_gap_bins
                          [fill_interpolate_max_gap_bins] widest run of
                          empty bins in one edge's profile that rung 3
                          fills by linear interpolation in altitude. Only
                          interior gaps, bounded by finite anchors above
                          AND below -- never the profile's ends, where a
                          one-sided fill would be extrapolation. Each edge
                          on its own. 0 disables the rung.
    refl_ladder_dbz       [anchor.weak_echo_dbz] one or two thresholds,
                          strict first: take the cleanest gates available,
                          and fall back to the looser threshold only where
                          the strict one cannot muster min_gates.
    bin_size_m            [bin_size_m] altitude bin thickness.
    """
    hw = section(cfg, 'horizontal_wind')
    s = 'horizontal_wind'
    anchor = section(hw, 'anchor')
    spike = section(hw, 'spike_correction')

    ladder = require(anchor, s + '.anchor', 'weak_echo_dbz',
                     'one or two dBZ thresholds, strict first')
    if (not isinstance(ladder, (list, tuple)) or not 1 <= len(ladder) <= 2
            or not all(isinstance(v, (int, float)) for v in ladder)):
        raise ConfigError(f'{s}.anchor.weak_echo_dbz must be a list of one or '
                          f'two numbers, strict first; got {ladder!r}')
    ladder = [float(v) for v in ladder]

    spike_passes = require(spike, s + '.spike_correction', 'passes',
                           'list of half-widths, [] to disable')
    spike_passes = [int(h) for h in (spike_passes or [])]
    if any(h < 1 for h in spike_passes):
        raise ConfigError(f"{s}.spike_correction.passes must be positive "
                          f"half-widths, got {spike_passes!r}")
    max_elev = require(anchor, s + '.anchor', 'max_elevation_deg',
                       'steepest beam that may vote on an anchor, or null')
    max_cand = require(anchor, s + '.anchor', 'max_candidate_mps',
                       'largest |hv| a gate may imply and still vote, or null')

    gap_bins = int(require(hw, s, 'fill_interpolate_max_gap_bins',
                           'widest interior gap (bins) filled by linear '
                           'interpolation, 0 = off'))
    if gap_bins < 0:
        raise ConfigError(f'{s}.fill_interpolate_max_gap_bins must be >= 0, '
                          f'got {gap_bins}')

    return RepairSettings(
        fill_mirror=bool(require(hw, s, 'fill_mirror', 'fill an empty near '
                                 'edge by the innermost mirror')),
        fill_interpolate_max_gap_bins=gap_bins,
        min_edge_gates=int(require(anchor, s + '.anchor', 'min_gates')),
        anchor_max_elev_deg=None if max_elev is None else float(max_elev),
        anchor_robust=bool(require(anchor, s + '.anchor', 'robust',
                                   'median (true) or mean (false) statistic')),
        max_candidate_mps=None if max_cand is None else float(max_cand),
        spike_passes=spike_passes,
        spike_shear_per_km=float(require(spike, s + '.spike_correction', 'shear_per_km')),
        spike_k_mad=float(require(spike, s + '.spike_correction', 'k_mad')),
        spike_max_frac=float(require(spike, s + '.spike_correction', 'max_fraction')),
        min_bin_gates=int(require(hw, s, 'min_bin_gates', '0 disables')),
        refl_ladder_dbz=ladder,
        bin_size_m=float(require(hw, s, 'bin_size_m', 'altitude bin thickness, m')),
    )


def get_horizontal_velocity(
    radar, cfg: dict,
    sr_map: np.ndarray,
    alt_map: np.ndarray,
):
    """Per-object, per-altitude-bin horizontal wind from the edge anchors.

    Parameters
    ----------
    radar : pyart.core.Radar
        Carrying reflectivity, radial velocity, ``sedimentation_velocity``
        and ``local_object_ids``.
    cfg : dict
        The whole config; see `resolve_repair_settings` for the keys read.
    sr_map, alt_map : numpy.ndarray, shape (nrays, ngates)
        Slant range (km) and altitude (m) of every gate.

    Returns
    -------
    hv : numpy.ndarray of float, shape (nrays, ngates)
        Horizontal wind, NaN where unavailable.
    hv_src : numpy.ndarray of int8, shape (nrays, ngates)
        Per-gate provenance, one of the HV_SRC_* codes. A gate reports the
        weakest rung of the two edges it was interpolated between.

    Notes
    -----
    The candidate wind at every gate is the Doppler velocity with the fall
    speed removed, projected onto the horizontal,

        hv_candidate = (Vr + Vsed * sin(e)) / cos(e),

    which equals the true horizontal wind only where the air is not moving
    vertically -- in weak echo at the cloud's edges. Each object (one
    contour in ONE sweep) is cut into altitude bins from its own base; in
    each bin the gates are split at the median slant range into a near and a
    far half, and each half yields one anchor by the first rung of the
    ladder it can satisfy:

      1. MEASURED  in-bin median (mean if anchor_robust is false) of the
                   weak-echo gates in that half, requiring at least
                   min_edge_gates of them and excluding gates steeper than
                   anchor_max_elev_deg or beyond max_candidate_mps.
      2. MIRRORED  innermost mirror, NEAR edge only: the median of the far
                   half's innermost min_edge_gates gates becomes the near
                   anchor, and the far anchor is rebuilt from the gates
                   left over, so both ends of the bin are real gates at
                   opposite ends of the measured span. Requires the far
                   half to hold at least 2*min_edge_gates measured gates.
                   An empty FAR edge is never filled from the near side.
      3. INTERPOLATED  a run of at most fill_interpolate_max_gap_bins
                   empty bins with a finite anchor above AND below is
                   filled linearly in altitude, each edge on its own.
                   Never at the profile's ends: a one-sided fill would
                   be extrapolation, a guess rather than an estimate.
      0. NONE      left NaN; the bin produces no output.

    The assembled edge profiles then pass through ONE spike correction
    (`smooth_edge_profile`, half-windows from spike_passes); a rewritten bin
    reports SMOOTHED. MIRRORED values never anchor anything else. Every gate
    of a bin with two finite anchors is then interpolated linearly in slant
    range between them.

    Objects are keyed by sweep as well as id: `select_cloud_transects`
    numbers the contours of each sweep from 1, and sweep 0 and sweep 2 both
    contain an 'object 1' that are different clouds at different azimuths.
    Pooling them would assert that the cross-beam wind is identical in two
    different scans.

    Bins killed by min_bin_gates, objects with no finite altitude, and the
    per-code gate tally are logged for every file.
    """
    st = resolve_repair_settings(cfg)
    ivars = section(cfg, 'input_variables')
    refl = as_float_nan(_field(radar, working_refl_field(cfg)))
    vr   = as_float_nan(_field(radar, require(ivars, 'input_variables', 'vr_var')))
    elev = radar.elevation['data']
    cos_el = np.cos(np.deg2rad(elev)).reshape(-1, 1)

    sed_rad = as_float_nan(project_to_radial_component(
        radar, field_name='sedimentation_velocity', trig='sin'))

    with np.errstate(invalid='ignore', divide='ignore'):
        vr_sans_sed  = vr + sed_rad
        # PLAIN array with NaN for missing, like vr and sed_rad above: real
        # files hand back a MASKED elevation, so cos_el and everything derived
        # from it carries a mask, and on a fully masked scan `int(mask.sum())`
        # raises MaskError. Missing data must read as NaN here, not as a mask.
        hv_candidate = np.ma.filled(vr_sans_sed / cos_el, np.nan)

    refl_ladder = st.refl_ladder_dbz
    bin_size_m  = st.bin_size_m

    fill_mirror     = st.fill_mirror
    fill_gap_bins   = st.fill_interpolate_max_gap_bins
    min_gates       = st.min_edge_gates
    anchor_max_elev = st.anchor_max_elev_deg
    _anchor_stat    = np.nanmedian if st.anchor_robust else np.nanmean
    anchor_max_cand = st.max_candidate_mps
    spike_passes    = st.spike_passes
    spike_shear_max = st.spike_shear_per_km
    spike_k_mad     = st.spike_k_mad
    spike_max_frac  = st.spike_max_frac
    min_bin_gates   = st.min_bin_gates

    # Per-gate elevation screen for rung-1 anchors, broadcast to (nrays, ngates).
    # Applied ONLY to anchor formation: steep gates still receive an interpolated
    # wind and still get a vertical velocity, they just do not get a vote on what
    # the layer's wind is.
    if anchor_max_elev is None:
        anchor_elev_ok = np.ones(vr.shape, dtype=bool)
    else:
        anchor_elev_ok = np.broadcast_to(
            np.asarray(elev).reshape(-1, 1) <= anchor_max_elev, vr.shape)

    # QC layer 2: a gate whose implied horizontal wind is impossible gets no
    # vote (see max_candidate_mps). Like the elevation screen this affects
    # ANCHOR FORMATION only -- the gate still receives an interpolated wind
    # and a vertical velocity.
    if anchor_max_cand is None:
        cand_ok = np.ones(vr.shape, dtype=bool)
    else:
        # np.ma.filled, because hv_candidate inherits a mask from the
        # elevation array and a FULLY masked comparison sums to np.ma.masked,
        # which int() refuses. A masked gate has no candidate, so it cannot
        # vote either way -- False is correct there.
        with np.errstate(invalid='ignore'):
            cand_ok = np.ma.filled(np.abs(hv_candidate) <= anchor_max_cand,
                                   False)
        _finite = np.ma.filled(np.isfinite(hv_candidate), False)
        _n_wild = int((_finite & ~cand_ok).sum())
        if _n_wild:
            logger.info(
                'anchor candidate bound (|hv| > %g m/s): %d gates barred from '
                'voting on an anchor', anchor_max_cand, _n_wild)

    # objects, keyed by sweep as well as id (see Notes)
    cloud_ids = ids_as_int(_field(radar, 'local_object_ids'))

    nrays, ngates = vr.shape
    hv = np.full((nrays, ngates), np.nan, dtype=float)
    hv_src = np.full((nrays, ngates), HV_SRC_NONE, dtype=np.int8)

    cloud_ids = cloud_ids.copy()
    for _s in range(radar.nsweeps):
        _sl = radar.get_slice(_s)
        _blk = cloud_ids[_sl]
        _blk[_blk > 0] += 1000 * (_s + 1)
        cloud_ids[_sl] = _blk

    obj_ids = np.unique(cloud_ids)
    obj_ids = obj_ids[obj_ids > 0]

    n_obj_no_alt = n_bins_thin = 0

    for obj in obj_ids:
        obj_mask = (cloud_ids == obj)
        if not obj_mask.any():
            continue

        z_obj = alt_map[obj_mask]
        if not np.isfinite(z_obj).any():
            n_obj_no_alt += 1
            continue

        zmin = float(np.nanmin(z_obj))
        zmax = float(np.nanmax(z_obj))
        if not (np.isfinite(zmin) and np.isfinite(zmax)):
            n_obj_no_alt += 1
            continue

        edges = altitude_bin_edges(zmin, zmax, bin_size_m)
        nbins = max(0, edges.size - 1)
        if nbins == 0:
            n_obj_no_alt += 1
            continue

        v0 = np.full(nbins, np.nan, dtype=float)  # near
        v1 = np.full(nbins, np.nan, dtype=float)  # far
        zc = 0.5 * (edges[:-1] + edges[1:])

        # per-edge provenance for this object's bins (see HV_SRC_* above)
        prov0 = np.full(nbins, HV_SRC_NONE, dtype=np.int8)
        prov1 = np.full(nbins, HV_SRC_NONE, dtype=np.int8)

        # raw per-edge statistics and gate counts
        raw0 = np.full(nbins, np.nan, dtype=float)
        raw1 = np.full(nbins, np.nan, dtype=float)
        cnt0 = np.zeros(nbins, dtype=int)
        cnt1 = np.zeros(nbins, dtype=int)
        # innermost-mirror candidates, filled during rung 1
        inner0 = np.full(nbins, np.nan, dtype=float)
        inner1 = np.full(nbins, np.nan, dtype=float)

        # ----- RUNG 1: in-bin edge means -----
        # Strictly in-bin. A half that cannot muster min_gates weak-echo gates
        # is left empty for the ladder below to fill; it is NOT back-filled from
        # the whole object, which would mix every altitude of the cloud into a
        # single layer.
        bin_dead = np.zeros(nbins, dtype=bool)
        for i, (lo, hi) in enumerate(pairwise(edges)):
            bm = obj_mask & (alt_map >= lo) & (alt_map < hi)
            if not bm.any(): continue
            if int(bm.sum()) < min_bin_gates:
                bin_dead[i] = True          # too thin to retrieve; see min_bin_gates
                n_bins_thin += 1
                continue

            sr_bin = sr_map[bm]
            center_sr = np.nanmedian(sr_bin)
            if not np.isfinite(center_sr): continue

            Lbin = bm & (sr_map <= center_sr)
            Rbin = bm & (sr_map >= center_sr)

            base_ok = np.isfinite(hv_candidate) & anchor_elev_ok & cand_ok

            # Each edge walks the ladder independently and stops at the first
            # threshold that gives it min_gates: an edge is never relaxed
            # because the OTHER edge needed it.
            Lok = Rok = None
            for _th in refl_ladder:
                _ok = base_ok & (refl < _th)
                if Lok is None or int(Lok.sum()) < min_gates:
                    Lok = Lbin & _ok
                if Rok is None or int(Rok.sum()) < min_gates:
                    Rok = Rbin & _ok

            cnt0[i], cnt1[i] = int(Lok.sum()), int(Rok.sum())
            if cnt0[i] > 0:
                raw0[i] = float(_anchor_stat(hv_candidate[Lok]))
            if cnt1[i] > 0:
                raw1[i] = float(_anchor_stat(hv_candidate[Rok]))

            if cnt0[i] >= min_gates:
                v0[i] = raw0[i]; prov0[i] = HV_SRC_MEASURED
            if cnt1[i] >= min_gates:
                v1[i] = raw1[i]; prov1[i] = HV_SRC_MEASURED

            # Prepare the innermost fallback for a near edge with nothing of its
            # own. Computed here, where the gate masks still exist, and applied
            # at rung 2, so the ladder stays in one place.
            if (fill_mirror and cnt0[i] < min_gates
                    and cnt1[i] >= 2 * min_gates):
                _d = sr_map[Rok]
                _v = hv_candidate[Rok]
                _order = np.argsort(_d)
                inner0[i] = float(_anchor_stat(_v[_order[:min_gates]]))
                # the gates NOT used for the near anchor become the far anchor,
                # so the same gate never sets both ends of the same bin
                inner1[i] = float(_anchor_stat(_v[_order[min_gates:]]))

        # ----- RUNG 2: fill from the other edge of the SAME bin -----
        # Innermost form: a near edge with nothing of its own takes the median
        # of the far half's innermost gates rather than the far half as a
        # whole, and the far anchor is rebuilt from the gates left over. Both
        # ends of the bin then come from real gates at opposite ends of the
        # measured span, instead of one value copied across.
        if fill_mirror:
            _use = ((prov0 == HV_SRC_NONE) & np.isfinite(inner0)
                    & np.isfinite(inner1) & (prov1 == HV_SRC_MEASURED))
            v0[_use] = inner0[_use]
            prov0[_use] = HV_SRC_MIRRORED
            # note this rung fills the NEAR edge only. The far edge is where the
            # weak-echo gates are, so a bin with no far anchor has nothing to
            # mirror FROM and is left empty rather than filled from the near
            # side -- filling a far edge from a near one would copy a value
            # across the full width of the bin in the wrong direction.
            # the far anchor is REPLACED, not left as the whole-half median: the
            # gates that supplied the near end must not also set the far end
            v1[_use] = inner1[_use]

        # ----- RUNG 3: interpolate across small interior gaps, per edge -----
        # A run of at most fill_interpolate_max_gap_bins empty bins,
        # with a finite anchor BOTH above and below, is filled linearly in
        # altitude -- each edge on its own, so the two sides of the cloud
        # never mix. Never at the profile's top or bottom: with an anchor on
        # one side only that would be extrapolation, a guess rather than an
        # estimate. A bin killed later by min_bin_gates cannot be resurrected
        # here (the bin_dead sweep below runs after every rung).
        if fill_gap_bins > 0:
            for _v, _prov in ((v0, prov0), (v1, prov1)):
                _fin = np.where(np.isfinite(_v))[0]
                for _a, _b in pairwise(_fin):
                    if not 1 <= _b - _a - 1 <= fill_gap_bins:
                        continue
                    _mid = np.arange(_a + 1, _b)
                    _frac = (zc[_mid] - zc[_a]) / (zc[_b] - zc[_a])
                    _v[_mid] = _v[_a] + _frac * (_v[_b] - _v[_a])
                    _prov[_mid] = HV_SRC_INTERPOLATED

        # ----- profile spike correction: the single spike test, run last -----
        # On the assembled profiles, so it sees every rung's output and judges
        # the edge as a whole. A rewritten bin reports SMOOTHED rather than
        # keeping the provenance of the value it replaced: the number is the
        # product of a fit through its neighbours and should not claim to be
        # an observation.
        if spike_passes:
            # zc is in METRES here; the shear threshold is per km, so convert
            # rather than silently scaling the tolerance by 1000
            _zc_km = zc / 1000.0
            for _v, _prov in ((v0, prov0), (v1, prov1)):
                for _half in spike_passes:
                    # Judge the profile on its POPULATED bins only. An empty bin
                    # is not a missing measurement so much as a layer with no
                    # weak-echo gates to anchor on -- concentrated in the
                    # convective core and at cloud top. Leaving those in place
                    # as NaN would shrink the window to one or two points
                    # exactly there, so the test would go quiet in the anvil,
                    # which is where it is most needed. The cost is that a
                    # window may span a gap and fit its line through points
                    # that are not immediate neighbours; the threshold then
                    # adapts, because the shear floor is scaled by the median
                    # spacing of the points actually present.
                    #
                    # The window is taken over bins where BOTH edges have a
                    # value, i.e. bins that actually produce output. A bin with
                    # only one edge filled contributes nothing to the field, so
                    # letting it into one edge's profile and not the other's
                    # would judge the two edges against different vertical
                    # grids.
                    _idx = np.where(np.isfinite(v0) & np.isfinite(v1))[0]
                    if _idx.size < 2 * _half + 1:
                        continue
                    _corr, _fl = smooth_edge_profile(
                        z_km=_zc_km[_idx], v=_v[_idx], half=_half,
                        shear_max=spike_shear_max, k_mad=spike_k_mad,
                        max_frac=spike_max_frac)
                    _moved = _fl & np.isfinite(_corr) \
                        & (np.abs(_corr - _v[_idx]) > 1e-9)
                    _v[_idx] = _corr
                    _prov[_idx[_moved]] = HV_SRC_SMOOTHED

        # Whatever the ladder did, a bin below the gate floor is not retrieved.
        # Enforced here rather than in each rung so no rung can resurrect it.
        if bin_dead.any():
            v0[bin_dead] = np.nan; v1[bin_dead] = np.nan
            prov0[bin_dead] = HV_SRC_NONE; prov1[bin_dead] = HV_SRC_NONE

        # ---------- interpolate across gates with final edges ----------
        for i, (lo, hi) in enumerate(pairwise(edges)):
            if not (np.isfinite(v0[i]) and np.isfinite(v1[i])): continue
            bm = obj_mask & (alt_map >= lo) & (alt_map < hi)
            if not bm.any(): continue

            # a gate reports the weakest rung of the two edges it sits between,
            # and SMOOTHED outranks the lot: whatever rung produced the value
            # originally, a fit has since replaced it
            if HV_SRC_SMOOTHED in (prov0[i], prov1[i]):
                src_code = HV_SRC_SMOOTHED
            elif HV_SRC_INTERPOLATED in (prov0[i], prov1[i]):
                src_code = HV_SRC_INTERPOLATED
            elif HV_SRC_MIRRORED in (prov0[i], prov1[i]):
                src_code = HV_SRC_MIRRORED
            else:
                src_code = HV_SRC_MEASURED

            dvals = sr_map[bm]
            dmin, dmax = float(np.nanmin(dvals)), float(np.nanmax(dvals))
            span = dmax - dmin
            if not np.isfinite(span) or span <= 0:
                hv[bm] = v0[i]
            else:
                frac = (dvals - dmin) / span
                hv[bm] = v0[i] + frac * (v1[i] - v0[i])
            hv_src[bm] = src_code

    in_obj = cloud_ids > 0
    tally = {name: int((hv_src[in_obj] == code).sum()) for name, code in (
        ('none', HV_SRC_NONE), ('measured', HV_SRC_MEASURED),
        ('mirrored', HV_SRC_MIRRORED), ('smoothed', HV_SRC_SMOOTHED),
        ('interpolated', HV_SRC_INTERPOLATED))}
    logger.info('horizontal wind: %d objects; gates by provenance %s; '
                '%d objects skipped (no finite altitude); %d bins below '
                'min_bin_gates=%d not retrieved',
                obj_ids.size, tally, n_obj_no_alt, n_bins_thin, min_bin_gates)
    return hv, hv_src


def smooth_edge_profile(z_km, v, half, shear_max, k_mad, max_frac):
    """Remove points that do not fit their own vertical trend, one at a time.

    Parameters
    ----------
    z_km : numpy.ndarray
        Bin centre altitudes, km, increasing.
    v : numpy.ndarray
        The edge profile, m/s; NaN where empty. Not modified.
    half : int
        Half-width of the fitting window, in bins.
    shear_max : float
        Physical floor on the residual threshold, m/s per km.
    k_mad : float
        Adaptive term: this many robust sigmas of the residuals.
    max_frac : float
        Largest fraction of the profile that may be rewritten.

    Returns
    -------
    corrected : numpy.ndarray
        The profile with flagged interior points replaced.
    flagged : numpy.ndarray of bool
        Which points failed the test, replaced or not.

    Notes
    -----
    For each point a straight line is fitted through the surviving points
    within +-half bins, EXCLUDING the point itself, and the residual taken.
    A genuine shear layer is a trend and leaves a small residual; a spike
    leaves a large one. The threshold is

        tol = max(shear_max * dz, k_mad * 1.4826 * MAD(residuals))

    with dz the median bin spacing, so a clean profile is judged tightly and
    a noisy one is not shredded, and the physical floor does not depend on
    the bin thickness.

    Points are removed greedily, worst first, refitting after each removal:
    a run of two or three bad bins masks itself if they are all judged at
    once against a line fitted through themselves.

    A flagged point is replaced by linear interpolation between the nearest
    surviving points above and below -- bounded by them, so it cannot
    overshoot, which is the failure mode of one-sided extrapolation.

    Points beyond the outermost survivors are LEFT AS RETRIEVED: there is no
    bracketing evidence for them either way, and dropping them costs about
    half a kilometre of anvil top per profile. They are still reported as
    flagged so the caller can treat them as low confidence.
    """
    v = np.asarray(v, dtype=float).copy()
    n = v.size
    flagged = np.zeros(n, dtype=bool)
    if n < 2 * half + 1:
        return v, flagged
    alive = np.isfinite(v)
    dz = float(np.median(np.diff(z_km))) if n > 1 else 0.3
    floor = shear_max * abs(dz)

    for _ in range(int(max_frac * n) + 1):
        res = np.full(n, np.nan)
        for i in range(n):
            if not alive[i]:
                continue
            lo, hi = max(0, i - half), min(n, i + half + 1)
            idx = np.array([k for k in range(lo, hi) if k != i and alive[k]])
            if idx.size < 2:
                continue
            if idx.size >= 3:
                a, b = np.polyfit(z_km[idx], v[idx], 1)
                pred = a * z_km[i] + b
            else:
                pred = float(np.mean(v[idx]))
            res[i] = v[i] - pred
        good = np.isfinite(res)
        if not good.any():
            break
        sigma = 1.4826 * float(np.median(np.abs(res[good] - np.median(res[good]))))
        tol = max(floor, k_mad * sigma)
        worst = int(np.nanargmax(np.abs(np.where(good, res, np.nan))))
        if not np.isfinite(res[worst]) or abs(res[worst]) <= tol:
            break
        if flagged.sum() + 1 > max_frac * n:
            break
        flagged[worst] = True
        alive[worst] = False

    if flagged.any():
        keep = np.where(alive & np.isfinite(v))[0]
        if keep.size >= 2:
            inside = flagged & (np.arange(n) > keep.min()) & (np.arange(n) < keep.max())
            if inside.any():
                v[inside] = np.interp(z_km[inside], z_km[keep], v[keep])
    return v, flagged


# =============================================================================
# Vertical velocity
# =============================================================================
def calculate_vertical_velocity(radar, cfg: Dict, sr_map: np.ndarray,
                                alt_map: np.ndarray) -> np.ndarray:
    """w = (Vr + Vsed*sin e - Vh*cos e) / sin e, per gate.

    Parameters
    ----------
    radar : pyart.core.Radar
        Carrying the radial velocity, ``sedimentation_velocity``,
        ``horizontal_velocity`` and ``local_object_ids``.
    cfg : dict
        The whole config. ``vertical_velocity.compute.max_horizontal_distance_km``
        and ``min_elevation_deg`` are REQUIRED.
    sr_map, alt_map : numpy.ndarray
        Slant range (km) and altitude (m) of every gate.

    Returns
    -------
    numpy.ndarray of float, shape (nrays, ngates)
        Vertical velocity, m/s, positive up; NaN where not computed.

    Notes
    -----
    Computed for every object gate within
    vertical_velocity.compute.max_horizontal_distance_km of the radar -- a
    PER-GATE horizontal-distance cut, not an all-or-nothing test on the
    object, so a near gate is never denied a w because its cloud's far end
    lies beyond the limit. Rays at or below min_elevation_deg are NaN across
    the board: the 1/sin(e) division diverges there and the arithmetic stops
    meaning anything. Gates excluded by either rule are counted and logged.
    """
    vcrit = section(section(cfg, 'vertical_velocity'), 'compute')
    s = 'vertical_velocity.compute'
    max_hdist = float(require(vcrit, s, 'max_horizontal_distance_km',
                              'km, per gate: w exists inside this distance'))
    elev_floor = float(require(vcrit, s, 'min_elevation_deg',
                               'degrees; rays at or below it get no w'))

    # The elevation keeps the file's own dtype here (float32 in CFRadial):
    # sin(e) computed in float64 instead shifts w by ~1e-5 m/s, and every
    # output on disk shares this arithmetic. Upgrading the precision is a
    # deliberate, separate decision.
    elev = radar.elevation['data']
    elev_f = (np.ma.filled(elev, np.nan) if isinstance(elev, np.ma.MaskedArray)
              else np.asarray(elev, dtype=float))
    sin_el = np.sin(np.deg2rad(elev_f)).reshape(-1, 1)

    ivars = section(cfg, 'input_variables')
    vr0      = as_float_nan(_field(radar, require(ivars, 'input_variables', 'vr_var')))
    sed_rad0 = as_float_nan(project_to_radial_component(radar, field_name='sedimentation_velocity', trig='sin'))
    hor_rad0 = as_float_nan(project_to_radial_component(radar, field_name='horizontal_velocity',   trig='cos'))

    for a in (vr0, sed_rad0, hor_rad0):
        scrub_fill(a)

    rad_vert = vr0 + sed_rad0 - hor_rad0

    cloud_ids = ids_as_int(_field(radar, 'local_object_ids'))

    hdist = _hdist_km_from_sr_alt(sr_map, alt_map)
    ok = (cloud_ids > 0) & (hdist <= max_hdist)
    n_far = int(((cloud_ids > 0) & ~(hdist <= max_hdist)).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        vert = np.where(ok, rad_vert / sin_el, np.nan)

    low = elev_f <= elev_floor
    n_low = int((ok & low[:, None]).sum())
    vert[low, :] = np.nan
    logger.info('vertical velocity: %d object gates beyond %g km horizontal '
                'and %d on rays at or below %g deg elevation not computed',
                n_far, max_hdist, n_low, elev_floor)
    return vert


def calculate_vertical_velocity_error(radar, cfg: Dict) -> np.ndarray:
    """Per-gate 1-sigma uncertainty on the retrieved vertical velocity.

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config. Every key of ``vertical_velocity.error`` is
        REQUIRED: ``dVr``, ``dVsed``, ``dVh2`` (m/s), and ``dVh1_table`` --
        a name from `windvel.tables.VERTICAL_ERROR_TABLES` or an inline
        ``{elevation_deg, dVh1}``.

    Returns
    -------
    numpy.ndarray of float, shape (nrays, ngates)
        1-sigma error on w, m/s; NaN at an elevation of exactly 0.

    Notes
    -----
    w is recovered from the radial velocity by removing the fall speed and
    the projection of the horizontal wind, then dividing by sin(elevation):

        w = (Vr + Vsed*sin(e) - Vh*cos(e)) / sin(e)

    so the three input errors enter with very different weights:

        dw/dVr   = 1 / sin(e)      1.0 at zenith, 2.0 at 30 deg, 11.5 at 5 deg
        dw/dVsed = 1               unamplified, the fall speed is already radial
        dw/dVh   = cot(e)          0 at zenith, 1.7 at 30 deg, 11.4 at 5 deg

    Adding them in quadrature, with the horizontal wind contributing through
    its two edge anchors:

        dW = sqrt[ (dVr/sin e)^2 + dVsed^2 + (dVh1^2 + dVh2^2) * cot^2(e) ]

    dVh1 is the larger, elevation-dependent term: the anchor error grows
    with height because the retrieval has less to work with there, and it
    is tabulated against elevation rather than assumed constant. dVh2 and
    dVsed are scalars.

    A function of ELEVATION ONLY. It deliberately does not depend on how a
    given gate's horizontal wind was filled -- a measured anchor and a
    mirrored one report the same sigma. That makes the error field a
    property of the viewing geometry, comparable between fill methods and
    between campaigns, rather than a self-assessment that changes whenever
    the fill rule changes. `horizontal_velocity_source` records provenance
    separately, so the two can be combined downstream by anyone who wants
    that.

    Note what this implies near the horizon: below about 10 degrees the two
    amplified terms dominate and dW exceeds any plausible w, which is the
    quantitative form of the argument for restricting analysis to high
    elevation.
    """
    ecfg = section(section(cfg, 'vertical_velocity'), 'error')
    s = 'vertical_velocity.error'
    d_vr = float(require(ecfg, s, 'dVr', 'radial velocity error, m/s'))
    d_sed = float(require(ecfg, s, 'dVsed', 'fall speed error, m/s'))
    d_vh2 = float(require(ecfg, s, 'dVh2', 'second anchor error, m/s'))
    e_tab, v_tab = vertical_error_table(
        require(ecfg, s, 'dVh1_table', 'a table name or {elevation_deg, dVh1}'))

    elev = np.asarray(radar.elevation['data'], dtype=float).reshape(-1, 1)
    # the table is defined on 0-90; fold the far side of zenith back onto it so a
    # sweep that passes 90 degrees does not fall off the end of the interpolation
    e_fold = np.where(elev > 90.0, 180.0 - elev, elev)
    d_vh1 = np.interp(e_fold, e_tab, v_tab)

    rad = np.deg2rad(e_fold)
    sin_e = np.sin(rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sin = 1.0 / sin_e
        cot = np.cos(rad) / sin_e
        err = np.sqrt((d_vr * inv_sin) ** 2 + d_sed ** 2
                      + (d_vh1 ** 2 + d_vh2 ** 2) * cot ** 2)
    err[~np.isfinite(err)] = np.nan          # elevation of exactly 0: undefined
    return np.broadcast_to(err, (radar.nrays, radar.ngates)).copy()


def vertical_velocity_usability(radar, cfg: Dict, sr_map: np.ndarray,
                                alt_map: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Where is the retrieved w worth using? Returned as a flag, not a filter.

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config. Every key of ``vertical_velocity.usable`` is
        REQUIRED: ``min_elevation_deg``, ``max_horizontal_distance_km``
        (``null`` disables the range limit), ``max_abs_w_mps`` (``null``
        disables the magnitude check).
    sr_map, alt_map : numpy.ndarray
        Slant range (km) and altitude (m) of every gate.
    w : numpy.ndarray
        The retrieved vertical velocity, m/s. Not modified.

    Returns
    -------
    numpy.ndarray of bool, shape (nrays, ngates)
        True where w is usable.

    Notes
    -----
    Three independent limits, for three different reasons:

      elevation   w = (...) / sin(e), so every error is amplified by
                  at lower elevations. Below the threshold
                  the retrieval still produces numbers, they simply are not
                  worth much. Compare `calculate_vertical_velocity_error`,
                  which says the same thing continuously.

      range       beyond the threshold the beam is wide enough that a gate
                  is no longer a point measurement -- 1 degree spans 87 m
                  at 5 km and 520 m at 30 km -- and the anchors that set
                  the horizontal wind are averaged over that width. Nothing
                  in the error budget captures this, which is why it has to
                  be a geometric cut. Disabled (null), beam broadening goes
                  unrepresented in the output, bounded only by the compute
                  cut (vertical_velocity.compute.max_horizontal_distance_km
                  decides where w EXISTS; this key decides where it is
                  TRUSTED).

      magnitude   QC layer 3: an updraft of some hundreds of m/s is not a
                  measurement, it is a retrieval that ran on bad inputs.
                  The value is FLAGGED, never NaN'd -- and if the two
                  upstream screens (the gate-quality cloud mask and the
                  anchor candidate bound) are working, this one should
                  catch almost nothing. If it starts catching a lot, that is
                  the signal that they are not, and erasing the evidence
                  would hide it.

    Deliberately NOT applied to `vertical_velocity`: the retrieved values
    are kept as they are, and this field says where to trust them. Masking
    would conflate 'outside the trusted domain' with 'no retrieval
    possible', and on write to netCDF the masked values would become fill
    and be gone for good.
    """
    ucfg = section(section(cfg, 'vertical_velocity'), 'usable')
    s = 'vertical_velocity.usable'
    max_hdist = require(ucfg, s, 'max_horizontal_distance_km',
                        'km, or null to disable the range limit')
    min_elev = float(require(ucfg, s, 'min_elevation_deg', 'degrees'))
    max_abs_w = require(ucfg, s, 'max_abs_w_mps',
                        'largest |w| that can be a measurement, m/s, or null')

    elev = np.asarray(radar.elevation['data'], dtype=float).reshape(-1, 1)
    ok = np.broadcast_to(elev > min_elev, sr_map.shape).copy()
    if max_hdist is not None:
        # a gate whose distance cannot be computed is not demonstrably inside
        # the domain, so it is excluded rather than quietly passed
        ok &= _hdist_km_from_sr_alt(sr_map, alt_map) <= float(max_hdist)
    if max_abs_w is not None:
        w = np.asarray(w, dtype=float)
        with np.errstate(invalid='ignore'):
            wild = np.isfinite(w) & (np.abs(w) > float(max_abs_w))
        ok &= ~wild
        n = int((wild & np.isfinite(w)).sum())
        if n:
            logger.info('w magnitude check (|w| > %g m/s): %d gates flagged '
                        'not usable (values kept)', float(max_abs_w), n)
    return ok


# =============================================================================
# Object selection
# =============================================================================
_SELECTION_CRITERIA = (
    ('min_span_km', 'smallest horizontal span along the RHI plane, km'),
    ('max_span_km', 'largest horizontal span, km'),
    ('min_top_km', 'lowest acceptable cloud top, km'),
    ('min_depth_km', 'smallest top - base, km'),
)


def apply_object_selection_criteria(radar, cfg, sr_map, alt_map):
    """Zero the ids of cloud objects that fail any selection criterion.

    Parameters
    ----------
    radar : pyart.core.Radar
        Carrying ``local_object_ids``; modified in place.
    cfg : dict
        The whole config. All four keys of ``object_selection`` are
        REQUIRED, ``null`` meaning the criterion is off: ``min_span_km``,
        ``max_span_km``, ``min_top_km``, ``min_depth_km``.
    sr_map, alt_map : numpy.ndarray
        Slant range (km) and altitude (m) of every gate.

    Returns
    -------
    pyart.core.Radar
        The same object. Rejected objects have id 0; the field stays an INT
        masked array (no NaNs written into it) and the mask is unchanged.

    Notes
    -----
    Span is the horizontal extent in the RHI plane, top and base the highest
    and lowest gate altitude. The count rejected by each criterion is
    logged.
    """
    cid = radar.fields['local_object_ids']['data']
    if isinstance(cid, np.ma.MaskedArray):
        ids  = cid.data.copy()
        mask = np.array(cid.mask, copy=True)
    else:
        ids  = np.array(cid, copy=True)
        mask = np.zeros_like(ids, dtype=bool)

    hdist = _hdist_km_from_sr_alt(sr_map, alt_map)  # km

    vcrit = section(cfg, 'object_selection')
    limits = {k: require(vcrit, 'object_selection', k, why + ', or null')
              for k, why in _SELECTION_CRITERIA}
    minspan = limits['min_span_km']
    maxspan = limits['max_span_km']
    mintop  = limits['min_top_km']
    mindep  = limits['min_depth_km']

    rejected = {k: 0 for k, _ in _SELECTION_CRITERIA}
    obj_ids = np.unique(ids[~mask])
    n_obj = 0
    for obj in obj_ids:
        if obj <= 0:
            continue
        m = (ids == obj) & (~mask)
        if not m.any():
            continue
        n_obj += 1

        span_km  = float(np.nanmax(hdist[m]) - np.nanmin(hdist[m]))
        top_km   = float(np.nanmax(alt_map[m]) / 1000.0)
        base_km  = float(np.nanmin(alt_map[m]) / 1000.0)
        depth_km = top_km - base_km

        reject = False
        if (minspan is not None) and (span_km  < minspan):
            reject = True; rejected['min_span_km'] += 1
        if (maxspan is not None) and (span_km  > maxspan):
            reject = True; rejected['max_span_km'] += 1
        if (mintop  is not None) and (top_km   < mintop ):
            reject = True; rejected['min_top_km'] += 1
        if (mindep  is not None) and (depth_km < mindep ):
            reject = True; rejected['min_depth_km'] += 1

        if reject:
            ids[m] = 0

    logger.info('object selection: %d objects, rejected by criterion %s',
                n_obj, rejected)

    radar.fields['local_object_ids']['data'] = np.ma.array(ids, mask=mask)
    return radar


# =============================================================================
# Coherent objects
# =============================================================================
def detect_coherent_objects(radar, cfg: Dict, w: np.ndarray, sigma: np.ndarray,
                            usable: np.ndarray):
    """Label coherent updrafts and downdrafts, per sweep, on the UNMASKED w.

    Parameters
    ----------
    radar : pyart.core.Radar
        Carrying ``local_object_ids``.
    cfg : dict
        The whole config. Every key of ``coherent_objects.threshold``,
        ``.persistence`` and ``.usable`` is REQUIRED (see
        `coherent_structures`).
    w : numpy.ndarray
        Vertical velocity, m/s, NaN where not retrieved.
    sigma : numpy.ndarray
        Its 1-sigma error, m/s.
    usable : numpy.ndarray of bool
        The trusted domain (`vertical_velocity_usability`).

    Returns
    -------
    lab_t, flag_t, trunc_t, lab_p, flag_p, trunc_p : numpy.ndarray
        Labels (int32, positive updraft / negative downdraft), usable flag
        (bool) and truncation bitmask (int8) for the threshold detector,
        then the same three for the persistence detector.

    Notes
    -----
    Run on every retrieved gate rather than on the usable subset: a
    structure is a physical object, and clipping it at the edge of the
    trusted domain would split one updraft into two or truncate it at a
    boundary that has nothing to do with its extent. Where an object sits
    relative to that domain is recorded afterwards, as a flag.

    Sweeps are labelled independently -- an RHI sweep is a plane, and two
    sweeps are different planes, so a structure cannot be connected across
    them -- and the labels are then offset so that every object in the file
    has its own id. Sweeps with no retrieved w are skipped and counted.
    """
    # Read up front so a bad config fails before any sweep runs.
    co = section(cfg, 'coherent_objects')
    tcfg = section(co, 'threshold')
    pcfg = section(co, 'persistence')
    ucfg = section(co, 'usable')
    thr_w = float(require(tcfg, 'coherent_objects.threshold', 'w_mps'))
    thr_area = float(require(tcfg, 'coherent_objects.threshold', 'min_area_km2'))
    pers_min = float(require(pcfg, 'coherent_objects.persistence', 'min_sigma'))
    pers_floor = float(require(pcfg, 'coherent_objects.persistence', 'floor_sigma'))
    pers_area = float(require(pcfg, 'coherent_objects.persistence', 'min_area_km2'))
    u_area_frac = float(require(ucfg, 'coherent_objects.usable', 'min_area_fraction'))
    u_core_frac = float(require(ucfg, 'coherent_objects.usable', 'core_peak_fraction'))
    u_core_gates = int(require(ucfg, 'coherent_objects.usable', 'core_min_gates'))

    lab_t = np.zeros(w.shape, dtype=np.int32)
    lab_p = np.zeros(w.shape, dtype=np.int32)
    flag_t = np.zeros(w.shape, dtype=bool)
    flag_p = np.zeros(w.shape, dtype=bool)
    trunc_t = np.zeros(w.shape, dtype=np.int8)
    trunc_p = np.zeros(w.shape, dtype=np.int8)
    next_up = next_dn = 1
    n_empty = 0

    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        if not np.isfinite(w[sl]).any():
            n_empty += 1
            continue
        dk, ak = gate_dist_alt_km(radar, s)
        valid = np.isfinite(w[sl])

        up_t, dn_t = CS.threshold(
            w[sl], valid, dk, ak,
            w_min=thr_w, min_area_km2=thr_area)
        up_p, dn_p = CS.persistence(
            w[sl], valid, dk, ak, sigma[sl],
            persistence_min=pers_min, floor_sigma=pers_floor,
            min_area_km2=pers_area)

        area = CS.cell_area_km2(dk, ak)
        # Both detectors are judged the same way, per OBJECT: an object is a
        # single physical thing, so either it is measured well enough to quote
        # or it is not. 
        elev_sweep = np.broadcast_to(
            np.asarray(radar.elevation['data'])[sl].reshape(-1, 1), up_t.shape)
        # holes in the retrieval INSIDE the echo: gates belonging to a cloud
        # object for which no w could be computed. An object abutting one of
        # these is cut off in the middle of the field rather than at its edge.
        #
        # Only INTERIOR holes count -- ones with retrieved gates on both sides
        # along the beam. A hole at cloud top has nothing beyond it to be cut
        # off from: the object ends there because the cloud does, which is not
        # a gap in the observation but the observation finding the top. Bins
        # near cloud top routinely fail to anchor, so counting those would flag
        # most of the anvil for a reason that carries no information.
        _cid = ids_as_int(_field(radar, 'local_object_ids'))
        _hole = (~np.isfinite(w[sl])) & (_cid[sl] > 0)
        _ok = np.isfinite(w[sl])
        _below = np.cumsum(_ok, axis=1) > 0                    # data nearer along the ray
        _above = np.cumsum(_ok[:, ::-1], axis=1)[:, ::-1] > 0  # data further along it
        hole_sweep = _hole & _below & _above
        for lab_out, flag_out, trunc_out, up, dn in (
                (lab_t, flag_t, trunc_t, up_t, dn_t),
                (lab_p, flag_p, trunc_p, up_p, dn_p)):
            sub = np.zeros(up.shape, dtype=np.int32)
            keep = np.zeros(up.shape, dtype=bool)
            cut = np.zeros(up.shape, dtype=np.int8)
            for src, sign in ((up, +1), (dn, -1)):
                for i in range(1, int(src.max()) + 1):
                    k = src == i
                    if not k.any():
                        continue
                    if sign > 0:
                        gid, next_up = next_up, next_up + 1
                    else:
                        gid, next_dn = next_dn, next_dn + 1
                    sub[k] = sign * gid
                    ok, _ = CS.object_is_usable(
                        src, i, w[sl], area, usable[sl], sigma=sigma[sl],
                        min_area_fraction=u_area_frac,
                        core_peak_fraction=u_core_frac,
                        core_min_gates=u_core_gates)
                    if ok:
                        keep[k] = True
                    cut[k] = CS.object_truncation(k, elev_sweep, hole_sweep)
            lab_out[sl] = sub
            flag_out[sl] = keep
            trunc_out[sl] = cut

    if n_empty:
        logger.info('coherent objects: %d of %d sweeps had no retrieved w '
                    'and were skipped', n_empty, radar.nsweeps)
    return lab_t, flag_t, trunc_t, lab_p, flag_p, trunc_p


# =============================================================================
# Full wind-velocity pipeline
# =============================================================================
def calculate_windvel(radar, cfg: Dict, sweeps):
    """Run every retrieval stage and write the results onto the radar.

    Parameters
    ----------
    radar : pyart.core.Radar
        After `select_cloud_transects`, so ``local_object_ids`` exists.
    cfg : dict
        The whole config; each stage reads its own section.
    sweeps : list of int
        The per-file RHI sweep list from `extract_rhi_sweep_indices`,
        passed explicitly -- the config never carries run-state.

    Returns
    -------
    pyart.core.Radar
        The same object, carrying every field in `OUTPUT_FIELDS`.

    Notes
    -----
    Stages, in order:

    1. Geometry, then object selection. Which clouds are worth retrieving
       is decided FIRST and in ONE place (`apply_object_selection_criteria`
       zeroes the ids of rejected objects); every later stage computes only
       for the survivors.
    2. Sedimentation, for the surviving objects only. The provenance notes
       become attributes of the output field: the file says which method,
       what attenuation, and where the freezing level came from.
    3. Horizontal velocity, with its per-gate provenance published because
       it cannot be derived from anything else in the output and carries
       the central caveat on this retrieval: whether a gate rests on a
       measurement or on an inference. The settings stamped into the field
       are the ones the retrieval used (`resolve_repair_settings`).
    4. Vertical velocity, its uncertainty (its own field rather than a
       mask, so a threshold on signal-to-noise is a decision the user makes
       and can change), and its usability flag (a flag, never a filter).
    5. Coherent updrafts and downdrafts, by two definitions, computed on
       the UNMASKED w so that no structure is clipped by the domain
       boundary.
    """
    sr_map, alt_map = make_sr_alt_maps(radar)
    radar = apply_object_selection_criteria(radar, cfg, sr_map, alt_map)

    sed, sed_notes = get_sed_vel(radar, cfg, sweeps)
    sed_ma = _mask_invalid(as_float_nan(sed))
    radar.add_field('sedimentation_velocity', {
        '_FillValue': FILL_VALUE,                # metadata only; not inserted into data
        'long_name': 'Sedimentation velocity',
        'units': 'm/s',
        **sed_notes,
        'data': _mask_invalid(sed_ma)
    }, replace_existing=True)

    hor, hor_src = get_horizontal_velocity(radar, cfg, sr_map, alt_map)
    scrub_fill(hor)
    hor_ma = _mask_invalid(hor)
    radar.add_field('horizontal_velocity', {
        '_FillValue': FILL_VALUE,
        'long_name': 'Horizontal wind velocity',
        'units': 'm/s',
        'data': _mask_invalid(hor_ma)
    }, replace_existing=True)

    st = resolve_repair_settings(cfg)
    radar.add_field('horizontal_velocity_source', {
        'long_name': 'Horizontal velocity provenance',
        'standard_name': 'horizontal_velocity_source',
        'units': '1',
        'flag_values': np.array([HV_SRC_NONE, HV_SRC_MEASURED, HV_SRC_MIRRORED,
                                 HV_SRC_SMOOTHED, HV_SRC_INTERPOLATED],
                                dtype='int16'),
        'flag_meanings': 'none measured mirrored smoothed interpolated',
        'comment': ('0 no estimate; 1 both bin edges measured in-bin; '
                    '2 near edge filled by the innermost mirror (the far '
                    "half's innermost gates set the near anchor, the rest "
                    'rebuild the far anchor); '
                    '5 at least one edge was rewritten by the profile spike '
                    'correction, and is a fit through its neighbours rather '
                    'than the rung it originally came from; '
                    '6 at least one edge filled by linear interpolation '
                    'across a small interior gap of its own profile '
                    '(fill_interpolate_max_gap_bins). '
                    'Codes 3 and 4 are retired and never reused. '
                    'A gate reports the WEAKEST rung of the two edges it sits '
                    'between, so the code is a floor on trustworthiness'),
        # Which rungs were enabled, and how, stamped into the file so the
        # codes above can be interpreted without the config that made them.
        # From the SAME resolved settings the retrieval read.
        'fill_mirror': str(st.fill_mirror),
        'fill_interpolate_max_gap_bins': str(st.fill_interpolate_max_gap_bins),
        'min_edge_gates': str(st.min_edge_gates),
        'spike_passes': str(list(st.spike_passes)),
        'reflectivity_ladder_dbz': str(list(st.refl_ladder_dbz)),
        'data': np.ma.masked_array(hor_src.astype('int16'), mask=False)
    }, replace_existing=True)

    vert = calculate_vertical_velocity(radar, cfg, sr_map, alt_map)
    scrub_fill(vert)
    vert_ma = _mask_invalid(vert)
    radar.add_field('vertical_velocity', {
        '_FillValue': FILL_VALUE,
        'long_name': 'Vertical wind velocity',
        'units': 'm/s',
        'data': _mask_invalid(vert_ma)
    }, replace_existing=True)

    verr = calculate_vertical_velocity_error(radar, cfg)
    radar.add_field('vertical_velocity_error', {
        'long_name': 'One-sigma uncertainty on vertical velocity',
        'standard_name': 'vertical_velocity_error',
        'units': 'm/s',
        'comment': ('quadrature sum of the radial-velocity, sedimentation and '
                    'horizontal-wind errors, weighted by 1/sin(elevation), 1 and '
                    'cot(elevation) respectively; a function of elevation only, '
                    'independent of how the horizontal wind was filled'),
        'data': _mask_invalid(verr)
    }, replace_existing=True)

    usable = vertical_velocity_usability(radar, cfg, sr_map, alt_map, vert)
    radar.add_field('vertical_velocity_flag', {
        'long_name': 'Vertical velocity usable flag',
        'standard_name': 'vertical_velocity_flag',
        'units': '1',
        'flag_values': np.array([0, 1], dtype='int16'),
        'flag_meanings': 'not_usable usable',
        'comment': ('1 where elevation exceeds '
                    'vertical_velocity.usable.min_elevation_deg AND horizontal '
                    'distance is within max_horizontal_distance_km AND |w| is '
                    'within max_abs_w_mps (a value beyond that is a retrieval '
                    'on bad inputs, not a measurement). '
                    'vertical_velocity is NOT masked by this field'),
        'data': np.ma.masked_array(usable.astype('int16'), mask=False)
    }, replace_existing=True)

    lab_t, flag_t, trunc_t, lab_p, flag_p, trunc_p = detect_coherent_objects(
        radar, cfg, vert, verr, usable)
    for name, lab, flag, trunc, how in (
            ('threshold', lab_t, flag_t, trunc_t,
             'connected |w| above coherent_objects.threshold.w_mps, kept if '
             'its area reaches threshold.min_area_km2'),
            ('persistence', lab_p, flag_p, trunc_p,
             'merge-tree segmentation of w/sigma: a core survives if it stands '
             'persistence.min_sigma above the saddle at which it merges into a '
             'stronger neighbour, and is outlined at that saddle')):
        radar.add_field(f'coherent_objects_{name}', {
            'long_name': f'Coherent updraft and downdraft labels ({name})',
            'standard_name': f'coherent_objects_{name}',
            'units': '1',
            'comment': (f'{how}. Positive ids are updrafts, negative ids '
                        'downdrafts, 0 is neither. Labelled per sweep on the '
                        'unmasked vertical velocity; ids are unique within the '
                        'file'),
            'data': np.ma.masked_array(lab.astype('int32'), mask=False)
        }, replace_existing=True)
        gate_or_object = ('per OBJECT, identically for both detectors: at '
                          'least coherent_objects.usable.min_area_fraction of '
                          "the object's area lies inside the usable domain AND "
                          'the majority of its core does. The core is the '
                          'connected patch of strongest gates around the peak, '
                          'grown until it reaches core_min_gates -- a small '
                          'core is never itself a reason to reject, since the '
                          'object passed a detector that required a peak')
        radar.add_field(f'coherent_objects_{name}_truncated', {
            'long_name': f'Coherent object truncation flags ({name})',
            'standard_name': f'coherent_objects_{name}_truncated',
            'units': '1',
            'flag_masks': np.array([CS.TRUNC_TOP, CS.TRUNC_BOT, CS.TRUNC_NEAR,
                                    CS.TRUNC_FAR, CS.TRUNC_GAP], dtype='int16'),
            'flag_meanings': ('highest_elevation_ray lowest_elevation_ray '
                              'first_range_gate last_range_gate '
                              'unretrieved_gap'),
            'comment': ('0 where the object lies wholly inside the sampled '
                        'volume and abuts no hole in it. Non-zero means it runs '
                        'off the edge of the scan, or abuts an unretrieved gap '
                        'within the echo, and is incompletely observed, so its area, depth '
                        'and peak are LOWER BOUNDS. Recorded separately from '
                        'the usable flag, which asks a different question: '
                        'usable is about whether the measurement can be '
                        'trusted, truncated about whether the object was wholly '
                        'seen. Count truncated objects when counting updrafts; '
                        'exclude them when averaging their size'),
            'data': np.ma.masked_array(trunc.astype('int16'), mask=False)
        }, replace_existing=True)

        radar.add_field(f'coherent_objects_{name}_flag', {
            'long_name': f'Coherent object usable flag ({name})',
            'standard_name': f'coherent_objects_{name}_flag',
            'units': '1',
            'flag_values': np.array([0, 1], dtype='int16'),
            'flag_meanings': 'not_usable usable',
            'comment': f'1 where usable, judged {gate_or_object}',
            'data': np.ma.masked_array(flag.astype('int16'), mask=False)
        }, replace_existing=True)

    return radar
