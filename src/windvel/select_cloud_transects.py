"""Cloud object identification: which echo the retrieval works on.

Contours of reflectivity above a threshold, per sweep, kept by physical
area, numbered largest first, and outlined.
"""

import logging
from typing import List

import cv2
import numpy as np
import pyart

from .coherent_structures import cell_area_km2
from .config import require, section
from .corrections import working_refl_field
from .errors import ConfigError, InputFieldError
from .utils import (
    FILL_VALUE,
    add_field_with_sweep,
    as_float_nan,
    gate_dist_alt_km,
    get_cloud_boundaries,
)

logger = logging.getLogger(__name__)

__all__ = ["get_contours", "label_cloud_regions", "select_cloud_transects"]


# =============================================================================
# Region labeling
# =============================================================================
def label_cloud_regions(
    shape: tuple[int, int],
    contours: List[np.ndarray],
) -> np.ma.MaskedArray:
    """Fill each contour polygon with its 1-based index.

    Parameters
    ----------
    shape : tuple of int
        (nrays, ngates) of the sweep.
    contours : list of numpy.ndarray
        OpenCV contours, in the order they are to be numbered.

    Returns
    -------
    numpy.ma.MaskedArray of int32
        Label per gate; 0 (masked) outside every contour.
    """
    labels = np.zeros(shape, dtype=np.int32)
    for i, contour in enumerate(contours, start=1):
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(labels, [pts], color=i)
    mask = labels == 0
    return np.ma.masked_array(labels, mask=mask)


# =============================================================================
# Contour extraction
# =============================================================================
def get_contours(
    refl2d: np.ndarray,
    threshold: float,
) -> List[np.ndarray]:
    """External contours of echo at or above a reflectivity threshold.

    Parameters
    ----------
    refl2d : numpy.ndarray
        Reflectivity of one sweep, dBZ, NaN where missing.
    threshold : float
        dBZ.

    Returns
    -------
    list of numpy.ndarray
        OpenCV contours. Size filtering happens in the caller, by PHYSICAL
        area from the gate geometry -- a pixel count means a different area
        at every range and on every radar.
    """
    mask = (refl2d >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# =============================================================================
# Main transect selection pipeline
# =============================================================================
def select_cloud_transects(radar, cfg: dict, sweeps) -> 'pyart.core.Radar':
    """Identify and label cloud transects in RHI sweeps, then outline them.

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config. REQUIRED under ``cloud_objects``:
        ``reflectivity_threshold_dbz``, ``min_area_km2``, ``gate_quality``
        (a list of ``{field, min}`` screens, ``[]`` for none),
        ``boundary_thickness_gates`` and ``exclude_lowest_rays``; under
        ``input_variables``: ``refl_var``. The contour reads the corrected
        reflectivity when ``corrections.attenuation.correct`` is true.
    sweeps : list of int
        The RHI sweeps to label.

    Returns
    -------
    pyart.core.Radar
        The same object, carrying ``local_object_ids`` (per sweep, numbered
        from 1, largest object first) and ``object_boundaries``.

    Notes
    -----
    QC layer 1, the gate-quality screens: a reflectivity threshold asks "is
    the echo strong enough"; at long range a gate can clear it on noise
    alone. Radar files carry fields built to answer the question properly
    -- SNR (is there signal), SQI/NCP (is the Doppler coherent), RHOHV (is
    it meteorological) -- so the config names any of them with a minimum,
    and a gate that fails any screen is not cloud. A LIST rather than named
    keys, so a new field or a new radar is config and not code. The fields
    answer DIFFERENT questions: SNR is about echo presence, while SQI is
    about velocity trust -- screening on SQI removes a gate from the cloud
    entirely, including its reflectivity, and scattered removals punch
    interior holes that the coherent-object TRUNC_GAP flag will then
    report. Measure before enabling it.

    The CONFIG decides, never the file: an empty list means no screen, and
    a named field missing from a file is an ERROR. Auto-detecting "use it
    if present" would silently process two radars differently and make
    their results incomparable.

    Gates removed by each screen and contours dropped by the area floor are
    counted and logged.
    """
    mc = section(cfg, 'cloud_objects')
    refl_name    = working_refl_field(cfg)
    refl_thresh  = require(mc, 'cloud_objects', 'reflectivity_threshold_dbz', 'dBZ')
    min_area_km2 = float(require(mc, 'cloud_objects', 'min_area_km2',
                                 'true-area floor for a cloud object, km^2'))
    shell = int(require(mc, 'cloud_objects', 'boundary_thickness_gates',
                        'width of the object outline, gates'))
    low_rays = int(require(mc, 'cloud_objects', 'exclude_lowest_rays',
                           'lowest rays of each sweep kept out of the '
                           'outline, 0 for none'))
    if refl_name not in radar.fields:
        raise InputFieldError(
            f'reflectivity field {refl_name!r} is not on the radar '
            f'(has: {sorted(radar.fields)})')

    screens = []
    for spec in (require(mc, 'cloud_objects', 'gate_quality',
                         'a list of {field, min} screens applied to cloud '
                         'gates, e.g. [{field: SNR, min: 3.0}], or [] for '
                         'none') or []):
        try:
            fname, fmin = spec['field'], float(spec['min'])
        except (TypeError, KeyError, ValueError) as err:
            raise ConfigError(
                'each cloud_objects.gate_quality entry needs a field name '
                f'and a numeric min; got {spec!r}') from err
        if fname not in radar.fields:
            raise InputFieldError(
                f'cloud_objects.gate_quality names {fname!r}, which is not '
                f'in this file (has: {sorted(radar.fields)}). Fix the config '
                'or drop the entry.')
        screens.append((fname, fmin))

    n_dropped = {name: 0 for name, _ in screens}
    n_small = 0

    for sweep in sweeps:
        slc    = radar.get_slice(sweep=sweep)
        refl   = radar.fields[refl_name]['data'][slc].filled(np.nan)

        for fname, fmin in screens:
            q = as_float_nan(radar.fields[fname]['data'][slc])
            bad = np.isfinite(refl) & ~(q >= fmin)      # a NaN quality fails
            n_dropped[fname] += int(bad.sum())
            refl = np.where(bad, np.nan, refl)

        # keep the contours covering enough PHYSICAL area
        dk, ak = gate_dist_alt_km(radar, sweep)
        gate_area = cell_area_km2(dk, ak)
        sized = []
        for cnt in get_contours(refl, refl_thresh):
            m = np.zeros(refl.shape, dtype=np.uint8)
            cv2.fillPoly(m, [cnt.reshape(-1, 1, 2).astype(np.int32)], 1)
            area = float(gate_area[m.astype(bool)].sum())
            if area >= min_area_km2:
                sized.append((area, cnt))
            else:
                n_small += 1
        sized.sort(key=lambda t: t[0], reverse=True)   # largest = id 1
        contours = [c for _, c in sized]

        labels = label_cloud_regions(refl.shape, contours)
        add_field_with_sweep(
            radar, sweep,
            'local_object_ids',
            {
                '_FillValue': 0,
                'long_name': 'Local cloud object IDs',
                'units': '#',
                'data': labels
            }
        )

    boundaries = get_cloud_boundaries(
        radar,
        rhi_sweep_indices=sweeps,
        cloud_id_field='local_object_ids',
        shell_thickness=shell,
        exclude_lowest_rays=low_rays,
    )

    radar.add_field(
        'object_boundaries',
        {
            '_FillValue': FILL_VALUE,
            'long_name': 'Cloud object boundaries',
            'units': '',
            'data': boundaries
        },
        replace_existing=True
    )

    for fname, fmin in screens:
        logger.info('gate-quality screen (%s < %g): %d gates removed from the '
                    'cloud mask', fname, fmin, n_dropped[fname])
    logger.info('cloud objects: %d contour(s) below %g km^2 dropped',
                n_small, min_area_km2)

    return radar
