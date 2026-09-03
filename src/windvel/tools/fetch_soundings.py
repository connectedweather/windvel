#!/usr/bin/env python3
"""Download radiosonde profiles for a place and a period, once, to a directory.

Deliberately separate from the retrieval. calculate_windvel reads soundings from
a directory and never fetches: a retrieval that depends on a remote server gives
different answers on different days for reasons that have nothing to do with the
radar, and leaves no record of which happened. This script is the deliberate act
that fills the directory; run it when the data changes, not when you process.

Source is NOAA IGRA v2 -- static files over HTTPS, global, documented fixed-width
format, 1905 to present. The University of Wyoming archive was tried first and
rejected: the same request returned 352 kB once and nothing twice, which is
rate limiting, and a utility should not be built on that.

Note IGRA holds the operational network only. A field campaign's own soundings
are usually closer and more frequent -- for the CHIVO radar at Houston the
nearest IGRA station is 211 km away while the TRACER launches were 7 km away --
so prefer campaign data where it exists and use this for everywhere else.

Usage
    python tools/fetch_soundings.py --lat 29.67 --lon -95.06 \
        --start 20220818 --end 20220819 --out soundings/
    python tools/fetch_soundings.py --station USM00072240 --start 20220801 \
        --end 20220831 --out soundings/

Writes one netCDF per launch, named <station>_<YYYYmmdd_HHMM>.nc, holding the
variables windvel/soundings.py expects: alt (m), tdry (degC), lat, lon and
base_time. Station files are cached under --cache so a second run is free.
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..soundings import MIN_PROFILE_LEVELS, great_circle_km

IGRA = 'https://www.ncei.noaa.gov/pub/data/igra'
STATION_LIST = f'{IGRA}/igra2-station-list.txt'
STATION_DATA = f'{IGRA}/data/data-por/{{sid}}-data.txt.zip'
MISSING = (-9999, -8888)


def _get(url, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def stations(cache: Path):
    """The IGRA station table: id, name, position, period of record."""
    f = cache / 'igra2-station-list.txt'
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(_get(STATION_LIST))
    out = []
    for line in f.read_text(errors='replace').splitlines():
        try:
            out.append(dict(id=line[0:11].strip(), lat=float(line[12:20]),
                            lon=float(line[21:30]), name=line[41:71].strip(),
                            first=int(line[72:76]), last=int(line[77:81])))
        except ValueError:
            continue
    return out


def nearest(cache, lat, lon, year, radius_km, limit):
    """Stations within radius that were reporting in the year requested.

    At most ``limit`` of them, nearest first; how many were within the
    radius but cut by the limit is printed.
    """
    out = []
    n_period = 0
    for s in stations(cache):
        if not (s['first'] <= year <= s['last']):
            n_period += 1
            continue
        d = float(great_circle_km(lat, lon, s['lat'], s['lon']))
        if d <= radius_km:
            out.append((d, s))
    chosen = [s for _, s in sorted(out, key=lambda r: r[0])[:limit]]
    print(f'  {len(out)} station(s) within {radius_km:g} km, '
          f'{len(out) - len(chosen)} beyond --max-stations dropped, '
          f'{n_period} not reporting in {year}')
    return chosen


def parse_igra(text, start, end):
    """IGRA2 records into profiles. Heights in m, temperature in degC.

    Fixed-width, one header line per launch (marked '#') followed by NUMLEV data
    records. Temperature is stored in tenths of a degree and height in metres,
    with -9999 and -8888 as missing.
    """
    profiles, cur = [], None
    n_short = 0
    for line in text.splitlines():
        if line.startswith('#'):
            if cur and len(cur['z']) >= MIN_PROFILE_LEVELS:
                profiles.append(cur)
            elif cur:
                n_short += 1
            try:
                when = datetime(int(line[13:17]), int(line[18:20]),
                                int(line[21:23]), max(int(line[24:26]), 0) % 24,
                                tzinfo=timezone.utc)
            except ValueError:
                cur = None
                continue
            cur = (dict(time=when, id=line[1:12].strip(), z=[], t=[],
                        lat=float(line[55:62]) / 1e4, lon=float(line[63:71]) / 1e4)
                   if start <= when.date() <= end else None)
            continue
        if cur is None:
            continue
        try:
            gph, temp = int(line[16:21]), int(line[22:27])
        except ValueError:
            continue
        if gph in MISSING or temp in MISSING:
            continue
        cur['z'].append(float(gph))
        cur['t'].append(temp / 10.0)
    if cur and len(cur['z']) >= MIN_PROFILE_LEVELS:
        profiles.append(cur)
    elif cur:
        n_short += 1
    if n_short:
        print(f'  {n_short} launch(es) in the period with fewer than '
              f'{MIN_PROFILE_LEVELS} levels dropped')
    return profiles


def write_profile(prof, outdir: Path):
    from netCDF4 import Dataset
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{prof['id']}_{prof['time']:%Y%m%d_%H%M}.nc"
    path = outdir / name
    z = np.asarray(prof['z'], float)
    t = np.asarray(prof['t'], float)
    order = np.argsort(z)
    with Dataset(path, 'w') as ds:
        ds.createDimension('level', z.size)
        for nm, val, units, long_name in (
                ('alt', z[order], 'm', 'geopotential height'),
                ('tdry', t[order], 'degC', 'dry bulb temperature')):
            v = ds.createVariable(nm, 'f4', ('level',))
            v[:] = val
            v.units = units
            v.long_name = long_name
        for nm, val, units in (('lat', prof['lat'], 'degrees_north'),
                               ('lon', prof['lon'], 'degrees_east'),
                               ('base_time', prof['time'].timestamp(),
                                'seconds since 1970-01-01 00:00:00 UTC')):
            v = ds.createVariable(nm, 'f8')
            v[:] = val
            v.units = units
        ds.station = prof['id']
        ds.source = 'NOAA IGRA v2, fetched by tools/fetch_soundings.py'
        ds.history = f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} created'
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lat', type=float)
    ap.add_argument('--lon', type=float)
    ap.add_argument('--station', help='IGRA id, e.g. USM00072240; skips the search')
    ap.add_argument('--start', required=True, help='YYYYMMDD')
    ap.add_argument('--end', required=True, help='YYYYMMDD')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cache', type=Path, default=Path.home() / '.cache' / 'igra')
    ap.add_argument('--radius-km', type=float, default=300.0)
    ap.add_argument('--max-stations', type=int, default=2)
    a = ap.parse_args()

    start = datetime.strptime(a.start, '%Y%m%d').date()
    end = datetime.strptime(a.end, '%Y%m%d').date()
    if end < start:
        ap.error('--end is before --start')

    if a.station:
        chosen = [dict(id=a.station, name='(given)', lat=np.nan, lon=np.nan)]
    else:
        if a.lat is None or a.lon is None:
            ap.error('give --station, or both --lat and --lon')
        chosen = nearest(a.cache, a.lat, a.lon, start.year,
                         a.radius_km, a.max_stations)
        if not chosen:
            print(f'no IGRA station within {a.radius_km:g} km reporting in '
                  f'{start.year}. Widen --radius-km, or use campaign soundings '
                  f'if any exist -- they are usually much closer.', file=sys.stderr)
            return 1
        for s in chosen:
            print(f"  {s['id']}  {s['name']:32s} "
                  f"{float(great_circle_km(a.lat, a.lon, s['lat'], s['lon'])):5.0f} km")

    total = 0
    for s in chosen:
        f = a.cache / f"{s['id']}-data.txt.zip"
        if not f.exists():
            print(f"  downloading {s['id']} (period of record, ~100 MB) ...")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(_get(STATION_DATA.format(sid=s['id'])))
        with zipfile.ZipFile(io.BytesIO(f.read_bytes())) as z:
            text = z.read(z.namelist()[0]).decode('utf-8', 'replace')
        profs = parse_igra(text, start, end)
        for p in profs:
            write_profile(p, a.out)
        print(f"  {s['id']}: {len(profs)} launches written to {a.out}")
        total += len(profs)
    if not total:
        print('nothing in that period for the station(s) chosen.', file=sys.stderr)
        return 1
    print(f'{total} soundings in {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
