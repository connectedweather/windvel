"""Radar-object helpers shared by every stage of the retrieval.

Geometry (slant range, altitude, horizontal distance, altitude bins), the
masked-array conventions the stages agree on, sweep selection, and per-sweep
field writing. Nothing here knows about the science; each helper exists so
the stages share ONE implementation of the thing.
"""

import logging
from typing import Dict

import numpy as np
from pyart.core import Radar
from scipy.ndimage import binary_erosion, generate_binary_structure

logger = logging.getLogger(__name__)

__all__ = [
    "FILL_VALUE",
    "add_field_with_sweep",
    "altitude_bin_edges",
    "as_float_nan",
    "extract_rhi_sweep_indices",
    "gate_dist_alt_km",
    "get_cloud_boundaries",
    "ids_as_int",
    "is_valid_elevation_sweep",
    "make_sr_alt_maps",
    "project_to_radial_component",
    "scrub_fill",
]

# The netCDF fill value written into every float output field, and scrubbed
# back to NaN whenever a field is read for arithmetic.
FILL_VALUE = -9999


# =============================================================================
# Array conventions
# =============================================================================
def as_float_nan(arr) -> np.ndarray:
    """A plain float array, with NaN wherever the input is masked.

    Parameters
    ----------
    arr : array_like or numpy.ma.MaskedArray
        Any field data.

    Returns
    -------
    numpy.ndarray
        ``float`` dtype; masked elements become NaN. Never a masked array,
        so downstream arithmetic cannot hit MaskError on a fully masked scan.
    """
    if isinstance(arr, np.ma.MaskedArray):
        return arr.astype(float).filled(np.nan)
    return np.asarray(arr, dtype=float)


def ids_as_int(arr) -> np.ndarray:
    """Object-id field as a plain int array, with 0 (no object) where masked.

    Parameters
    ----------
    arr : array_like or numpy.ma.MaskedArray
        An integer label field such as ``local_object_ids``.

    Returns
    -------
    numpy.ndarray
        ``int`` dtype; masked elements are 0.
    """
    if isinstance(arr, np.ma.MaskedArray):
        return arr.filled(0).astype(int)
    return np.asarray(arr).astype(int)


def scrub_fill(a: np.ndarray) -> np.ndarray:
    """Turn any lingering `FILL_VALUE` in a float array into NaN, in place.

    Parameters
    ----------
    a : numpy.ndarray
        Float array; modified in place.

    Returns
    -------
    numpy.ndarray
        The same array.
    """
    a[np.isclose(a, FILL_VALUE, atol=1e-6)] = np.nan
    return a


# =============================================================================
# Geometry
# =============================================================================
def make_sr_alt_maps(radar):
    """Slant range and altitude of every gate, for the whole radar.

    Parameters
    ----------
    radar : pyart.core.Radar

    Returns
    -------
    sr_map : numpy.ndarray, shape (nrays, ngates)
        Slant range in km.
    alt_map : numpy.ndarray, shape (nrays, ngates)
        Altitude above the radar in m, from Py-ART's 4/3-earth geometry.
    """
    nrays, ngates = radar.nrays, radar.ngates
    sr_map = np.empty((nrays, ngates), dtype=float)
    alt_map = np.empty((nrays, ngates), dtype=float)
    for s in range(radar.nsweeps):
        sl = radar.get_slice(sweep=s)
        x, y, z = radar.get_gate_x_y_z(sweep=s)
        sr_map[sl] = np.sqrt(x*x + y*y + z*z) / 1000.0
        alt_map[sl] = z
    return sr_map, alt_map


def gate_dist_alt_km(radar, sweep: int):
    """Horizontal distance and altitude of every gate in one sweep, in km.

    The (distance, altitude) mesh the coherent-object detectors, the cloud
    selector's area floor and the summary panels all work on -- one
    definition, so an area computed in one is the area drawn in another.

    Parameters
    ----------
    radar : pyart.core.Radar
    sweep : int

    Returns
    -------
    dk : numpy.ndarray, shape (nrays_in_sweep, ngates)
        Horizontal distance from the radar, km.
    ak : numpy.ndarray, shape (nrays_in_sweep, ngates)
        Altitude above the radar, km.
    """
    x, y, z = radar.get_gate_x_y_z(sweep)
    return np.sqrt(x ** 2 + y ** 2) / 1000.0, z / 1000.0


def altitude_bin_edges(zmin: float, zmax: float, bin_size: float) -> np.ndarray:
    """Edges of the altitude bins an object is retrieved on.

    Anchored at the object's own base, ``zmin``, and extended to cover
    ``zmax``. The retrieval and the summary panel bin the same way because
    they both call this.

    Parameters
    ----------
    zmin, zmax : float
        Lowest and highest gate altitude of the object (any one unit).
    bin_size : float
        Bin thickness, same unit.

    Returns
    -------
    numpy.ndarray
        Monotonic edges; ``edges.size - 1`` bins.
    """
    return np.arange(zmin, zmax + bin_size, bin_size)


# =============================================================================
# Sweep selection
# =============================================================================
def _sweep_mode_strings(radar):
    """Decode radar.sweep_mode['data'] rows to plain strings.

    CFRadial files carry sweep_mode variously as byte strings, str, or
    per-character (possibly masked) byte arrays; handle all three.
    """
    out = []
    for row in radar.sweep_mode['data']:
        if isinstance(row, bytes):
            s = row.decode('utf-8', 'ignore')
        elif isinstance(row, str):
            s = row
        else:
            vals = row.compressed() if hasattr(row, 'compressed') else row
            s = ''.join(v.decode('utf-8', 'ignore') if isinstance(v, bytes)
                        else str(v) for v in vals)
        out.append(s.strip('\x00').strip())
    return out


def is_valid_elevation_sweep(radar, sweep_idx: int) -> bool:
    """Are the elevations in a sweep non-decreasing?

    Descending RHI sweeps are thereby excluded -- a known limitation: on an
    up-down scanning radar this drops half the rays. Reversing rather than
    rejecting them is the eventual fix.

    Parameters
    ----------
    radar : pyart.core.Radar
    sweep_idx : int

    Returns
    -------
    bool
        False also when the elevations cannot be read at all; that case is
        logged so the dropped sweep is never silent.
    """
    try:
        elevs = radar.get_elevation(sweep=sweep_idx)
        return bool(np.all(np.diff(elevs) >= 0))
    except Exception:
        logger.exception('sweep %d: elevations unreadable; treating as '
                         'invalid', sweep_idx)
        return False


def extract_rhi_sweep_indices(radar):
    """The sweeps the retrieval applies to.

    THE one definition, used everywhere: a sweep is selected when its
    sweep_mode is 'rhi' AND its elevations are non-decreasing
    (`is_valid_elevation_sweep`). Both tests are required: the elevation
    test alone admits PPI sweeps (constant elevation passes a non-decreasing
    check), and the mode test alone admits descending RHIs the retrieval
    does not handle. Dropped RHI sweeps are logged, never silent.

    A per-file RESULT, not configuration: the list is returned and passed
    explicitly to every consumer, so the config never carries run-state.

    Parameters
    ----------
    radar : pyart.core.Radar

    Returns
    -------
    list of int
        Selected sweep indices. On failure the error is logged and ``[]``
        is returned, so the caller skips the file rather than crashing the
        run.
    """
    try:
        modes = _sweep_mode_strings(radar)
        rhi = [i for i, mode in enumerate(modes) if mode == 'rhi']
        ascending = [i for i in rhi if is_valid_elevation_sweep(radar, i)]
        dropped = sorted(set(rhi) - set(ascending))
        if dropped:
            logger.warning(
                'dropping %d RHI sweep(s) with non-ascending elevations: %s',
                len(dropped), dropped)
        return ascending
    except Exception:
        logger.exception('could not determine RHI sweeps; selecting none')
        return []


# =============================================================================
# Per-sweep field writing
# =============================================================================
def add_field_with_sweep(
    radar: Radar,
    sweep: int,
    field_name: str,
    field_dict: Dict,
) -> Radar:
    """Add or update a field on a single sweep of a Py-ART Radar object.

    Parameters
    ----------
    radar : Radar
        The Py-ART Radar instance to modify.
    sweep : int
        Index of the sweep to populate (0-based).
    field_name : str
        Name under which to add the field.
    field_dict : dict
        Dictionary containing at least the key 'data' whose value is a
        numpy.ma.MaskedArray of shape (nrays, nrange), plus any metadata.

    Returns
    -------
    Radar
        The modified Radar object (same instance, returned for chaining).
    """
    ntime = len(radar.time['data'])
    nrange = len(radar.range['data'])
    if not (0 <= sweep < radar.nsweeps):
        raise IndexError(f"Sweep index {sweep} out of range [0..{radar.nsweeps-1}]")
    if 'data' not in field_dict:
        raise KeyError("field_dict must contain a 'data' key")
    sweep_data = field_dict['data']
    if not isinstance(sweep_data, np.ma.MaskedArray):
        raise TypeError("field_dict['data'] must be a numpy.ma.MaskedArray")

    if field_name in radar.fields:
        field_data = radar.fields[field_name]['data'].copy()
    else:
        field_data = np.ma.masked_array(
            data=np.zeros((ntime, nrange)),
            mask=False,
            fill_value=0
        )

    sl = radar.get_slice(sweep)
    field_data.data[sl, :] = sweep_data.data
    field_data.mask[sl, :] = sweep_data.mask

    new_field = field_dict.copy()
    new_field['data'] = field_data
    radar.add_field(field_name, new_field, replace_existing=True)
    return radar


# =============================================================================
# Radial projection
# =============================================================================
def project_to_radial_component(
    radar,
    field_name: str,
    trig: str = 'sin',
    sweeps: list[int] = None,
    fill_value: float = np.nan
) -> np.ndarray:
    """Project a radar field onto the radial by sin or cos of elevation.

    Parameters
    ----------
    radar : Radar
        Py-ART Radar object with the specified field and elevation.
    field_name : str
        Name of the radar field to project (must exist in radar.fields).
    trig : {'sin', 'cos'}, optional
        Trigonometric function to apply to the elevation angles.
    sweeps : sequence of int, optional
        List of sweep indices to include. If None, all rays are included.
    fill_value : scalar, optional
        Value to assign outside the selected sweeps (default: NaN).

    Returns
    -------
    proj : ndarray, shape (n_rays, n_gates)
        Projected field values across all rays and gates.
        Values outside the specified sweeps are set to `fill_value`.
    """
    data = radar.fields[field_name]['data']
    elev_deg = radar.elevation['data']

    if trig == 'sin':
        factor = np.sin(np.deg2rad(elev_deg))
    elif trig == 'cos':
        factor = np.cos(np.deg2rad(elev_deg))
    else:
        raise ValueError("`trig` must be either 'sin' or 'cos'")

    proj = factor[:, None] * data

    if sweeps is not None:
        ray_mask = np.zeros(radar.nrays, dtype=bool)
        for sw in sweeps:
            ray_mask[radar.get_slice(sw)] = True
        proj[~ray_mask[:, None]] = fill_value

    return proj


# =============================================================================
# Cloud object boundaries
# =============================================================================
def get_cloud_boundaries(
    radar,
    rhi_sweep_indices: list[int],
    cloud_id_field: str = 'local_object_ids',
    shell_thickness: int = 1,
    *,
    exclude_lowest_rays: int,
) -> np.ndarray:
    """Boolean mask of the outer shell of every cloud object.

    Parameters
    ----------
    radar : Radar
    rhi_sweep_indices : list of int
        Sweeps to process; other sweeps stay False.
    cloud_id_field : str, optional
        Name of the integer label field.
    shell_thickness : int, optional
        Shell width in gates (erosion iterations, 4-connected).
    exclude_lowest_rays : int
        The lowest-elevation rays of each sweep are never marked as
        boundary, whatever the labels there say -- the shell is not
        trusted where the beam grazes the ground. 0 marks every ray.

    Returns
    -------
    numpy.ndarray, shape (nrays, ngates), dtype bool
        True on the shell of each object.
    """
    boundary_full = np.zeros((radar.nrays, radar.ngates), dtype=bool)
    struct = generate_binary_structure(2, 1)

    for s in rhi_sweep_indices:
        sl = radar.get_slice(s)
        labels = ids_as_int(radar.get_field(field_name=cloud_id_field, sweep=s))

        sweep_boundary = np.zeros_like(labels, dtype=bool)
        for obj_id in np.unique(labels):
            if obj_id <= 0:
                continue
            obj_mask = (labels == obj_id)
            if not obj_mask.any():
                continue

            eroded = binary_erosion(obj_mask, structure=struct, iterations=shell_thickness)
            boundary = obj_mask & ~eroded

            # rays, not gates: the lowest-elevation rows of the sweep,
            # not the gates nearest the radar
            if exclude_lowest_rays > 0:
                boundary[:exclude_lowest_rays, :] = False

            sweep_boundary |= boundary

        boundary_full[sl, :] = sweep_boundary

    return boundary_full
