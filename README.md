# windvel

Horizontal and vertical wind retrieval from RHI radar scans, and the coherent
updraft/downdraft structures found in the result.

A scanning radar measures only the velocity along its beam, so a vertical
scan mixes the horizontal and vertical wind. windvel separates them from a
single radar, and publishes a per-gate uncertainty and the provenance of
every wind value alongside the result.

## What it needs

Input is **CFRadial** files holding RHI sweeps; other formats have to be
converted first. Each file must carry a reflectivity and a Doppler velocity
field, named in the site file.

The two `pid_*` fall-speed methods also need a particle-identification
field, and the shipped default is one of them
(`pid_giangrande_oklahoma2013`). If your files have no PID, set
`sedimentation.method: reflectivity_temperature`, which branches on
reflectivity and the sign of temperature instead — that needs a freezing
level, from the site file's `environment` section.

## Install

    uv sync --extra dev        # creates .venv from uv.lock (the primary way)

or, without uv: `pip install -e ".[dev]"`. Python >= 3.10. Then:

    cp examples/retrieval.yaml examples/site.example.yaml .
    mv site.example.yaml site_myradar.yaml     # fill in the seven keys below
    uv run windvel --config site_myradar.yaml

Two files: `retrieval.yaml` (every threshold and method choice — shared by
every site so results stay comparable) and `site_<radar>.yaml` (paths, field
names, freezing-level sources, run policy, and which retrieval file to use).
Every key is documented in place. The paths in the site file are
placeholders; the program fails loudly on anything missing rather than
assuming a value.

Seven keys are yours to set; the rest of the site file ships with values
that work:

    paths:
      input_data_path: /data/myradar/rhi/    # a file, or a directory
                                             #   searched recursively
      file_pattern: "cfrad.*_RHI.nc"         # prefix*suffix on the file
                                             #   NAME -- not a glob
      output_dir: /data/myradar/windvel/
      log_dir: /data/myradar/windvel/logs/

    input_variables:                         # whatever your files call them
      refl_var: reflectivity
      vr_var: velocity
      pid_var: PID

`--config` is the only argument. `environment.melting_level_km` already
carries a value, so the freezing-level ladder answers out of the box; point
`temperature_field` or `sounding_dir` at something better when you have it.

## From Python

The CLI is a thin wrapper over the same public functions, so a run can be
driven from a script:

```python
import pyart
import windvel

cfg    = windvel.load_config('site_myradar.yaml')
radar  = pyart.io.read_cfradial('cfrad.20220818_212930_RHI.nc')
sweeps = windvel.extract_rhi_sweep_indices(radar)

radar = windvel.select_cloud_transects(radar, cfg, sweeps)
radar = windvel.calculate_windvel(radar, cfg, sweeps)

w    = radar.fields['vertical_velocity']['data']
err  = radar.fields['vertical_velocity_error']['data']
good = radar.fields['vertical_velocity_flag']['data'] == 1

windvel.save_windvel_file('out.nc', radar, cfg, mode='extract')
```

`load_config` resolves both files into one mapping, so the site file still
supplies the field names and the retrieval file the thresholds; the stage
functions take that mapping and the sweep list as arguments and never read a
path themselves.

The example assumes `corrections.attenuation.correct` is `false`, which is
what ships. With corrections switched on, use the CLI — the corrections
stage runs ahead of the others and is not part of the published surface.

Everything in `windvel.__all__` is public; anything else is internal and may
change without notice.

## What it does

In an RHI the radial velocity mixes the horizontal and vertical wind. At low
elevation the beam is nearly horizontal, so those gates see mostly horizontal
wind; the retrieval uses that to estimate the wind layer by layer, then removes
it to leave the vertical motion.

    corrections        an optional attenuation correction, written as a new
                       field that every later stage reads
    cloud objects      contiguous echo above a reflectivity threshold, labelled
                       per sweep; the ones worth retrieving picked once by size
    fall speed         from a particle id, or from reflectivity and the sign of
                       temperature where no particle id exists
    horizontal wind    300 m altitude bins, one anchor per side of each bin
                       from its weak-echo gates; a near side with none is
                       mirrored from the bin's own far half, a short gap in a
                       profile is interpolated, then a two-pass spike
                       correction; every gate records which rung produced it
    vertical velocity  w = (Vr + Vsed*sin e - Vh*cos e) / sin e
    error              one sigma per gate, from the viewing geometry
    usability          a flag, never a filter: values are kept everywhere
    objects            two definitions, an absolute threshold and a
                       persistence measure in units of the error

See `docs/PIPELINE.md` for the detail, and the docstrings for why each choice
was made — most of them record a measurement rather than a preference.

## What it writes

One output file per input, carrying the retrieval's fields beside the ones
it read:

    sedimentation_velocity      fall speed, positive downward
    horizontal_velocity         wind in the RHI plane, positive away from
                                the radar
    horizontal_velocity_source  which rung produced each gate
    vertical_velocity           kept everywhere it can be computed
    vertical_velocity_error     one sigma, from the viewing geometry
    vertical_velocity_flag      1 where the value is trusted
    coherent_objects_*          labels, usable flag and truncation mask,
                                for each of the two detectors

Each file also records the package version, the config files that made it
and any overrides, so a result can be traced back to the code and the
numbers behind it.

`vertical_velocity` is deliberately **not** masked outside the trusted
domain. Below about 10 degrees elevation the 1/sin(e) amplification puts the
error past any plausible w, and the field will hold large apparent updrafts
there. Select on `vertical_velocity_flag`, or threshold on
`vertical_velocity_error`, rather than reading the raw field.

## Layout

    src/windvel/
      cli.py                   the windvel command: read, retrieve, save, plot
      config.py                the two-file config, validated once at load
      corrections.py           the corrections stage
      select_cloud_transects.py  labels contiguous echo into objects, per sweep
      calculate_windvel.py     the retrieval: fall speed to usability
      coherent_structures.py   the two object detectors and their verdicts
      soundings.py             nearest sounding from a local directory
      tables.py                the physics tables the config chooses by name
      errors.py                the package's exception types
      plot_windvel.py          the eight-panel summary figure
      save_windvel.py, utils.py
      tools/fetch_soundings.py fills a sounding directory from NOAA IGRA
                               (windvel-fetch-soundings)
    examples/retrieval.yaml       the retrieval definition, documented
    examples/site.example.yaml    the per-site template, documented
    docs/PIPELINE.md              how the retrieval works, stage by stage
    tests/                        synthetic fixtures, no data required

## Tests

    uv run ruff check src tests && uv run pytest -q

## Licence

BSD 3-Clause — see `LICENSE`.
