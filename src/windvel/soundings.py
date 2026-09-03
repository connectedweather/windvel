#!/usr/bin/env python3
"""Find and read the sounding nearest a radar file, from a local directory.

Nothing here touches the network. The retrieval reads whatever soundings are
already on disk, so the same input file always gives the same output -- a
retrieval that fetches at run time gives different answers on different days for
reasons that have nothing to do with the radar, and leaves no record of which
happened. Use tools/fetch_soundings.py to populate the directory beforehand.

Two layouts are understood, because both are common:

  per-launch   one profile per file, ARM-style: temperature in 'tdry' (degC) or
               'temp', height in 'alt' (m) or 'height', position in 'lat'/'lon',
               launch time from 'base_time' or the file name.
  collection   many profiles in one file, indexed by a sounding dimension, as in
               the TRACER sounding archive: 'Sounding_Temperature',
               'Sounding_Height', 'Sounding_Latitude', 'Sounding_Longitude' and
               'Sounding_Datetime_Year' and friends.

Anything else raises with what it looked for, rather than guessing at variable
names and silently returning a wrong profile.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import require, section
from .errors import ConfigError, InputDataError
from .utils import as_float_nan

logger = logging.getLogger(__name__)

__all__ = ["MIN_PROFILE_LEVELS", "find_nearest", "great_circle_km",
           "read_profiles", "resolve_freezing_level"]

EARTH_R_KM = 6371.0
_TEMP_NAMES = ('tdry', 'temp', 'temperature', 'air_temperature', 't')
_ALT_NAMES = ('alt', 'height', 'altitude', 'gpheight', 'z')

# A profile with fewer finite levels than this is not a sounding.
MIN_PROFILE_LEVELS = 10


def great_circle_km(lat1, lon1, lat2, lon2):
    """Distance over the sphere, km (haversine).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array_like
        Degrees.

    Returns
    -------
    float or numpy.ndarray
        km, on a sphere of radius `EARTH_R_KM`; good enough at
        sounding-to-radar ranges.
    """
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _first(ds, names):
    lower = {str(v).lower(): v for v in ds.variables}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _time_from_name(path):
    """Last resort: a YYYYMMDD.HHMM or YYYYMMDD_HHMMSS stamp in the file name."""
    m = re.search(r'(\d{8})[._-]?(\d{4,6})', Path(path).name)
    if not m:
        return None
    d, t = m.group(1), m.group(2).ljust(6, '0')
    try:
        return datetime(int(d[:4]), int(d[4:6]), int(d[6:]),
                        int(t[:2]), int(t[2:4]), int(t[4:6]),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def read_profiles(path):
    """Every profile in one file.

    Parameters
    ----------
    path : str or Path
        A netCDF file in either layout (see the module docstring).

    Returns
    -------
    list of dict
        Each with ``time`` (aware UTC), ``lat``, ``lon``, ``z_km``, ``t_c``
        (finite levels only) and ``source``. Profiles in a collection with
        an unreadable time or fewer than `MIN_PROFILE_LEVELS` finite levels
        are skipped and counted in the log.

    Raises
    ------
    InputDataError
        When a per-launch file has no recognisable temperature/height
        variables, too few levels, or no launch time.
    """
    from netCDF4 import Dataset
    out = []
    with Dataset(path) as ds:
        if 'Sounding_Temperature' in ds.variables:          # collection layout
            g = lambda n: np.asarray(ds.variables[n][:])
            T, Z = g('Sounding_Temperature'), g('Sounding_Height')
            lat, lon = g('Sounding_Latitude'), g('Sounding_Longitude')
            parts = {k: g('Sounding_Datetime_' + k)
                     for k in ('Year', 'Month', 'Day', 'Hour', 'Min')}
            n_bad_time = n_short = 0
            for j in range(T.shape[1]):
                try:
                    when = datetime(int(parts['Year'][0, j]),
                                    int(parts['Month'][0, j]),
                                    int(parts['Day'][0, j]),
                                    int(parts['Hour'][0, j]),
                                    int(parts['Min'][0, j]),
                                    tzinfo=timezone.utc)
                except (ValueError, IndexError):
                    n_bad_time += 1
                    continue
                z, t = np.asarray(Z[:, j], float), np.asarray(T[:, j], float)
                ok = np.isfinite(z) & np.isfinite(t)
                if ok.sum() < MIN_PROFILE_LEVELS:
                    n_short += 1
                    continue
                out.append(dict(time=when, lat=float(np.nanmedian(lat[:, j])),
                                lon=float(np.nanmedian(lon[:, j])),
                                z_km=z[ok] / 1000.0, t_c=t[ok], source=str(path)))
            if n_bad_time or n_short:
                logger.info('%s: %d profile(s) with an unreadable time and %d '
                            'with fewer than %d levels skipped', path,
                            n_bad_time, n_short, MIN_PROFILE_LEVELS)
            return out

        tname, zname = _first(ds, _TEMP_NAMES), _first(ds, _ALT_NAMES)
        if tname is None or zname is None:
            raise InputDataError(
                f'{path}: no sounding found. Looked for a collection layout '
                f'(Sounding_Temperature/Sounding_Height) and for per-launch '
                f'variables named {_TEMP_NAMES} and {_ALT_NAMES}.')
        t = np.asarray(ds.variables[tname][:], float).squeeze()
        z = np.asarray(ds.variables[zname][:], float).squeeze()
        when = None
        if 'base_time' in ds.variables:
            when = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                seconds=float(np.asarray(ds.variables['base_time'][:]).ravel()[0]))
        when = when or _time_from_name(path)
        lat = lon = np.nan
        for nm, key in (('lat', 'lat'), ('latitude', 'lat'),
                        ('lon', 'lon'), ('longitude', 'lon')):
            if nm in ds.variables:
                v = float(np.nanmedian(np.asarray(ds.variables[nm][:], float)))
                if key == 'lat':
                    lat = v
                else:
                    lon = v
        ok = np.isfinite(z) & np.isfinite(t)
        if ok.sum() < MIN_PROFILE_LEVELS or when is None:
            raise InputDataError(
                f'{path}: profile too short, or no launch time '
                f'(no base_time and no stamp in the file name).')
        out.append(dict(time=when, lat=lat, lon=lon, z_km=z[ok] / 1000.0,
                        t_c=t[ok], source=str(path)))
    return out


def find_nearest(sounding_dir, when, lat, lon, max_age_hours, max_distance_km):
    """The closest usable sounding, or None with a reason.

    Parameters
    ----------
    sounding_dir : str or Path
        Directory of .nc/.cdf/.nc4 sounding files.
    when : datetime
        Aware UTC time of the radar scan.
    lat, lon : float
        Radar position, degrees.
    max_age_hours : float
        Largest |launch - scan| accepted.
    max_distance_km : float
        Largest launch-to-radar distance accepted.

    Returns
    -------
    profile : dict or None
        As from `read_profiles`; nearest first, then most recent.
    note : str
        Which sounding, or why none.

    Notes
    -----
    Time and distance are both bounded and both fall THROUGH rather than
    failing: past those limits a balloon is not evidence about this radar's
    freezing level, and the caller should drop to its next source. A
    sounding 400 km away across a front is worse than an honest constant.
    Every profile rejected, and every unreadable file, is counted and
    logged.
    """
    d = Path(sounding_dir)
    if not d.is_dir():
        return None, f'sounding_dir {sounding_dir!r} is not a directory'
    files = sorted([p for p in d.iterdir()
                    if p.suffix.lower() in ('.nc', '.cdf', '.nc4')])
    if not files:
        return None, f'no .nc files in {sounding_dir!r}'

    best, best_key, skipped = None, None, []
    n_old = n_far = 0
    for f in files:
        try:
            profs = read_profiles(f)
        except Exception as exc:                     # one bad file is not fatal
            skipped.append(f'{f.name}: {exc}')
            continue
        for p in profs:
            age = abs((p['time'] - when).total_seconds()) / 3600.0
            if age > max_age_hours:
                n_old += 1
                continue
            dist = (great_circle_km(lat, lon, p['lat'], p['lon'])
                    if np.isfinite(p['lat']) and np.isfinite(p['lon']) else 0.0)
            if dist > max_distance_km:
                n_far += 1
                continue
            key = (dist, age)                # nearest first, then most recent
            if best_key is None or key < best_key:
                best, best_key = p, key
    logger.info('soundings in %s: %d profile(s) beyond %g h, %d beyond %g km, '
                '%d file(s) unreadable', sounding_dir, n_old, max_age_hours,
                n_far, max_distance_km, len(skipped))
    for line in skipped:
        logger.warning('unreadable sounding file: %s', line)
    if best is None:
        why = (f'no sounding within {max_age_hours:g} h and '
               f'{max_distance_km:g} km of {when:%Y-%m-%d %H:%M}Z')
        if skipped:
            why += f' ({len(skipped)} file(s) unreadable: {skipped[0]})'
        return None, why
    dist, age = best_key
    return best, (f'sounding {best["time"]:%Y-%m-%d %H:%M}Z '
                  f'({age:+.1f} h, {dist:.0f} km) from {Path(best["source"]).name}')


def _radar_time(radar):
    """The file's start time, as an aware UTC datetime."""
    units = str(radar.time.get('units', ''))
    m = re.search(r'since\s+(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})',
                  units)
    if not m:
        raise InputDataError('cannot read the radar start time from time units '
                             f'{units!r}; needed to choose a sounding')
    base = datetime(*[int(g) for g in m.groups()], tzinfo=timezone.utc)
    return base + timedelta(seconds=float(np.asarray(radar.time['data']).ravel()[0]))


def resolve_freezing_level(radar, cfg, alt_map):
    """Where is the 0 degC level, and how do we know?

    Parameters
    ----------
    radar : pyart.core.Radar
    cfg : dict
        The whole config; the site's ``environment`` section is read. Its
        four source keys are all REQUIRED, ``null`` meaning "not this
        source": ``temperature_field``, ``sounding_dir``,
        ``temperature_profile``, ``melting_level_km``. When ``sounding_dir``
        is set, ``max_sounding_age_hours`` and ``max_sounding_distance_km``
        are required too.
    alt_map : numpy.ndarray
        Gate altitude, m.

    Returns
    -------
    above_freezing : numpy.ndarray of bool
        True where T >= 0 degC.
    provenance : str
        Which source answered, written into the output so a file always
        says where its freezing level came from.

    Notes
    -----
    Tried in order, so the most direct source available wins and the answer
    is always attributable:

      1. a per-gate temperature field already in the file -- most accurate,
         because it is already matched to the radar geometry, and it needs no
         freezing level at all
      2. the nearest sounding in environment.sounding_dir, bounded in both
         age and distance. Read from disk, never fetched: a retrieval that
         depends on a remote server gives different answers on different
         days for reasons that have nothing to do with the radar, and leaves
         no record of which happened. tools/fetch_soundings.py fills the
         directory
      3. a cached temperature profile supplied directly in the config
      4. environment.melting_level_km -- explicit, always available, honest
         about being an assumption
      5. nothing. Raise, rather than pick a default: a wrong freezing level
         is a systematic 1-2 m/s error through the whole ice region, and a
         systematic error that never announces itself is the worst kind
    """
    scfg = section(cfg, 'environment')
    tname = require(scfg, 'environment', 'temperature_field',
                    'name of a per-gate temperature field in the file, or null')
    sdir = require(scfg, 'environment', 'sounding_dir',
                   'directory of sounding files to search, or null')
    prof_cfg = require(scfg, 'environment', 'temperature_profile',
                       '{height_km: [...], temperature_c: [...]}, or null')
    ml = require(scfg, 'environment', 'melting_level_km',
                 'the assumed 0 degC height in km, or null')

    if tname and tname in radar.fields:
        tval = as_float_nan(radar.fields[tname]['data'])
        return tval >= 0.0, f'temperature field {tname!r} in the file'

    sounding_note = None
    if sdir:
        max_age = float(require(scfg, 'environment', 'max_sounding_age_hours',
                                'required when sounding_dir is set'))
        max_dist = float(require(scfg, 'environment', 'max_sounding_distance_km',
                                 'required when sounding_dir is set'))
        when = _radar_time(radar)
        prof, note = find_nearest(
            sdir, when,
            float(np.asarray(radar.latitude['data']).ravel()[0]),
            float(np.asarray(radar.longitude['data']).ravel()[0]),
            max_age_hours=max_age, max_distance_km=max_dist)
        if prof is not None:
            zc = np.asarray(prof['z_km'], float)
            tc = np.asarray(prof['t_c'], float)
            order = np.argsort(zc)
            tgate = np.interp(alt_map / 1000.0, zc[order], tc[order])
            return tgate >= 0.0, note
        # fall through to the next source, and say why this one gave nothing
        sounding_note = note

    if prof_cfg:
        zc = np.asarray(prof_cfg['height_km'], dtype=float)
        tc = np.asarray(prof_cfg['temperature_c'], dtype=float)
        ok = np.isfinite(zc) & np.isfinite(tc)
        zc, tc = zc[ok], tc[ok]
        order = np.argsort(zc)
        tgate = np.interp(alt_map / 1000.0, zc[order], tc[order])
        return tgate >= 0.0, 'cached temperature profile'

    if ml is not None:
        note = f'melting_level_km = {float(ml):g}'
        if sounding_note:
            note += f" (no sounding used: {sounding_note})"
        return alt_map / 1000.0 <= float(ml), note

    raise ConfigError(
        "no temperature source is set: every environment source key is "
        "null (temperature_field, sounding_dir, temperature_profile, "
        "melting_level_km). Refusing to assume one -- a wrong freezing level "
        "is a systematic error through the whole ice region.")
