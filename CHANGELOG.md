# Changelog

Notable changes to the windvel package.

Newest first. Each release is a `## <version> — <YYYY-MM-DD>` heading;
unreleased work accumulates under Unreleased until it is tagged.

## 1.0.0 — 2026-09-03

The first release. Every output file carries a `windvel_version` global
attribute.

### Configuration

- Two files, resolved once by `windvel.config.load_config`.
  `retrieval.yaml` holds every threshold and method choice, shared by every
  site so results stay comparable (`corrections`, `cloud_objects`,
  `object_selection`, `sedimentation`, `horizontal_wind`,
  `vertical_velocity`, `coherent_objects`) and carries a `retrieval_version`
  label. `site_<radar>.yaml` holds `paths`, `input_variables`,
  `environment` (the freezing-level ladder) and `run` (`save_mode`,
  `overwrite_existing`, `plot_summary`), names the retrieval file, and may
  carry `retrieval_overrides` -- dotted keys that must exist, are logged one
  per line, and are stamped into every output. Both carry
  `config_version: 1`.
- Every key is required and an unknown key is refused, so a misspelt key
  can never be silently ignored; retired keys are refused by name. Run
  state (the sweep list) never travels in the config.
- Physics tables live in `windvel.tables` and are chosen by name:
  `corrections.attenuation.relation` (`x_andsager`, `x_keenan`,
  `x_minimum`) and `vertical_velocity.error.dVh1_table`
  (`chivo_tracer_2022`); an inline mapping with the same keys is accepted
  for a radar the tables do not cover.
- Output files carry `windvel_retrieval_file`, `windvel_retrieval_version`,
  `windvel_retrieval_overrides` (JSON) and `windvel_site_file`; a run
  freezes its resolved config to `config_used.yaml`.

### The retrieval

- Corrections as stage 0 (`windvel.corrections`): an optional attenuation
  correction written as a NEW field, `reflectivity_corrected`, that every
  reflectivity consumer -- the object contour, the weak-echo anchor screen,
  the fall speed -- reads while it is on; the delivered field is never
  modified. The relation is band specific and never guessed.
- Cloud objects by true area from the gate geometry (`min_area_km2`), with
  per-gate quality screens (`gate_quality`) decided by the config, never
  the file; scan-top-truncated clouds are kept. Which objects are retrieved
  is decided once, by size, before any stage runs.
- Fall speed by three methods (`pid_giangrande_oklahoma2013`,
  `pid_giangrande_darwin2026`, `reflectivity_temperature`), density
  corrected; the freezing level from a per-gate field, a local sounding
  directory (`windvel-fetch-soundings` fills it from NOAA IGRA), a cached
  profile or an explicit level -- never a silent default.
- Horizontal wind from per-object, per-sweep, per-altitude-bin edge
  anchors: MEASURED, then the innermost MIRROR for an empty near edge, then
  INTERPOLATED across interior gaps of at most
  `fill_interpolate_max_gap_bins`, else none; then a two-pass spike
  correction with a physical shear floor. Provenance per gate in
  `horizontal_velocity_source` (1 measured, 2 mirrored, 6 interpolated,
  5 smoothed, 0 none), with the settings the retrieval used stamped into
  the field.
- Vertical velocity per gate within a horizontal-distance cut and above an
  elevation floor; the one-sigma error as its own field, a function of
  elevation only; usability as a flag, never a filter (elevation, optional
  range, |w| cap).
- Coherent objects saved as fields, by two detectors (an absolute threshold
  and a persistence measure on w/sigma; 8-connected throughout), each with
  a per-object usable flag and a five-bit truncation mask.
- Every skip, cap or drop is counted and logged; run logs are UTC.

### Library

- Errors are the package's own types: `WindvelError`, `ConfigError` (a
  ValueError), `MissingConfigKeyError` (a ConfigError and a KeyError),
  `InputFieldError` (a KeyError), `InputDataError` (a ValueError).
- Every library module declares `__all__`; the package surface is
  `windvel.__all__`. Science documentation lives in NumPy-style docstrings
  (`Notes` / `References`) on the stage functions.
- The summary figure widens its vertical-velocity colorbar to ±40 m/s when
  any |w| in the sweep exceeds ±20 m/s.
