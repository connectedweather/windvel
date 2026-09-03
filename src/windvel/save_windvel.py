"""Writing the retrieval to CFRadial, atomically."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyart

from . import __version__
from .calculate_windvel import OUTPUT_FIELDS
from .config import require, section
from .errors import ConfigError
from .utils import FILL_VALUE

__all__ = ["SAVE_MODES", "save_windvel_file", "validate_save_mode"]

SAVE_MODES = ('full', 'extract')

# Superseded mode names -> what to use instead. 'append' always wrote every
# field on the radar object and never appended to an existing file.
_SAVE_MODES_RENAMED = {'append': 'full'}


def validate_save_mode(mode: str) -> str:
    """The one check on a save-mode name, shared by the CLI and the saver.

    Parameters
    ----------
    mode : str

    Returns
    -------
    str
        ``mode``, when it is one of `SAVE_MODES`.

    Raises
    ------
    ConfigError
        On a superseded name (saying what replaced it) or an unknown one.
    """
    if mode in _SAVE_MODES_RENAMED:
        raise ConfigError(
            f"save mode '{mode}' was renamed '{_SAVE_MODES_RENAMED[mode]}': it "
            "always wrote every field on the radar object and never appended "
            "to an existing file. Use 'full' or 'extract'.")
    if mode not in SAVE_MODES:
        raise ConfigError(f"Unsupported save mode '{mode}'; "
                          f"save_mode must be one of {SAVE_MODES}")
    return mode


def _atomic_write_cfradial(outfile, radar, include_fields=None):
    """Write CFRadial via a sibling .tmp then os.replace.

    An interrupted write leaves the target either absent or its previous
    complete self -- never half-written. Both save modes go through here.
    """
    outfile = Path(outfile)
    fd, tmp = tempfile.mkstemp(
        prefix=outfile.name + '.', suffix='.tmp', dir=str(outfile.parent)
    )
    os.close(fd)
    try:
        if include_fields is None:
            pyart.io.write_cfradial(tmp, radar)
        else:
            pyart.io.write_cfradial(tmp, radar, include_fields=include_fields)
        os.replace(tmp, outfile)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _normalize_field_dtypes(radar):
    """Make every field netCDF-writable, whatever the save mode.

    Bool arrays (object_boundaries) are not a netCDF primitive and unsigned
    types need an in-range fill. Writability is a property of the data, not
    of which fields are kept, so this runs for both modes.
    """
    for fdict in radar.fields.values():
        data = fdict['data']
        dt = getattr(data, 'dtype', None)
        if dt is None:
            continue
        if dt == bool:
            fdict['data'] = data.astype('int16')
            fdict['_FillValue'] = np.int16(FILL_VALUE)
        elif dt.kind == 'u':
            fdict['_FillValue'] = dt.type(np.iinfo(dt).max)


def save_windvel_file(outfile, radar, cfg, mode):
    """Write the retrieval to CFRadial.

    Parameters
    ----------
    outfile : str or Path
        Target path; parent directories are created.
    radar : pyart.core.Radar
    cfg : dict
        The resolved config; ``input_variables`` (its values name the input
        fields kept in extract mode) and ``provenance`` (stamped into the
        file) are REQUIRED.
    mode : {'full', 'extract'}
        'full' writes every field on the radar object; 'extract' only the
        configured input variables plus the computed windvel fields
        (`OUTPUT_FIELDS`, the single canonical list).

    Returns
    -------
    Path
        ``outfile``.

    Notes
    -----
    Every output file records what made it, as global attributes:
    ``windvel_version``, and from ``provenance`` the ``retrieval_file``,
    ``retrieval_version``, ``retrieval_overrides`` (JSON) and ``site_file``
    -- so two files can be told apart from their metadata alone.
    """
    validate_save_mode(mode)
    prov = section(cfg, 'provenance')
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    _normalize_field_dtypes(radar)
    radar.metadata['windvel_version'] = __version__
    for key in ('retrieval_file', 'retrieval_version', 'site_file'):
        radar.metadata[f'windvel_{key}'] = str(require(prov, 'provenance', key))
    radar.metadata['windvel_retrieval_overrides'] = json.dumps(
        require(prov, 'provenance', 'retrieval_overrides'), sort_keys=True)

    if mode == 'full':
        _atomic_write_cfradial(outfile, radar)
    else:
        ivars = section(cfg, 'input_variables')
        keep = []
        for fld in ivars.values():
            if fld and fld not in keep:
                keep.append(fld)
        for fld in OUTPUT_FIELDS:
            if fld in radar.fields and fld not in keep:
                keep.append(fld)
        _atomic_write_cfradial(outfile, radar, include_fields=keep)

    return outfile

