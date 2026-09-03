"""The config: two files, one resolved mapping.

    site_<radar>.yaml     what is true HERE: paths, field names, the
                          freezing-level ladder, run policy, and which
                          retrieval file to use (plus any overrides to it)
    retrieval.yaml        what the retrieval IS: every threshold and method
                          choice, shared by every site so results stay
                          comparable

`load_config(site_file)` reads both, applies the site's ``retrieval_overrides``
(each must name an existing leaf), validates every section against the
schema -- a missing key and an unknown key are both loud -- and returns one
flat mapping of sections plus a ``provenance`` section that says which files
and overrides made it. The CLI freezes that mapping to ``config_used.yaml``
and the saver stamps the provenance into every output file.

Library functions take the resolved mapping and read their own section with
`section` / `require`, so a hand-built dict in a test is checked the same
way as a file.
"""

from collections.abc import Mapping
from pathlib import Path

import yaml

from .errors import ConfigError, MissingConfigKeyError

__all__ = [
    "CONFIG_VERSION",
    "RETRIEVAL_SCHEMA",
    "SITE_SCHEMA",
    "dotted_get",
    "dotted_set",
    "load_config",
    "refuse_retired",
    "require",
    "resolve",
    "section",
    "validate",
]

CONFIG_VERSION = 1

# A schema is a nested mapping: a leaf is REQUIRED (required) or OPTIONAL;
# a mapping is a subsection. ``null`` is always a legal value of a leaf;
# what it means is documented on the key.
REQUIRED, OPTIONAL = 'required', 'optional'

RETRIEVAL_SCHEMA = {
    'cloud_objects': {
        'reflectivity_threshold_dbz': REQUIRED,
        'min_area_km2': REQUIRED,
        'gate_quality': REQUIRED,
        'boundary_thickness_gates': REQUIRED,
        'exclude_lowest_rays': REQUIRED,
    },
    'object_selection': {
        'min_span_km': REQUIRED,
        'max_span_km': REQUIRED,
        'min_top_km': REQUIRED,
        'min_depth_km': REQUIRED,
    },
    'corrections': {
        'attenuation': {
            'correct': REQUIRED,
            'relation': REQUIRED,
            'require_band_match': REQUIRED,
            'liquid_only': REQUIRED,
            'max_correction_db': REQUIRED,
        },
    },
    'sedimentation': {
        'method': REQUIRED,
    },
    'horizontal_wind': {
        'bin_size_m': REQUIRED,
        'fill_mirror': REQUIRED,
        'fill_interpolate_max_gap_bins': REQUIRED,
        'min_bin_gates': REQUIRED,
        'anchor': {
            'weak_echo_dbz': REQUIRED,
            'min_gates': REQUIRED,
            'max_elevation_deg': REQUIRED,
            'max_candidate_mps': REQUIRED,
            'robust': REQUIRED,
        },
        'spike_correction': {
            'passes': REQUIRED,
            'shear_per_km': REQUIRED,
            'k_mad': REQUIRED,
            'max_fraction': REQUIRED,
        },
    },
    'vertical_velocity': {
        'compute': {
            'max_horizontal_distance_km': REQUIRED,
            'min_elevation_deg': REQUIRED,
        },
        'error': {
            'dVr': REQUIRED,
            'dVsed': REQUIRED,
            'dVh2': REQUIRED,
            'dVh1_table': REQUIRED,
        },
        'usable': {
            'min_elevation_deg': REQUIRED,
            'max_horizontal_distance_km': REQUIRED,
            'max_abs_w_mps': REQUIRED,
        },
    },
    'coherent_objects': {
        'threshold': {'w_mps': REQUIRED, 'min_area_km2': REQUIRED},
        'persistence': {'min_sigma': REQUIRED, 'floor_sigma': REQUIRED,
                        'min_area_km2': REQUIRED},
        'usable': {'min_area_fraction': REQUIRED, 'core_peak_fraction': REQUIRED,
                   'core_min_gates': REQUIRED},
    },
}

SITE_SCHEMA = {
    'paths': {
        'input_data_path': REQUIRED,
        'file_pattern': REQUIRED,
        'output_dir': REQUIRED,
        'log_dir': REQUIRED,
    },
    'input_variables': {
        'refl_var': REQUIRED,
        'vr_var': REQUIRED,
        'pid_var': REQUIRED,
    },
    'environment': {
        'temperature_field': REQUIRED,
        'sounding_dir': REQUIRED,
        'max_sounding_age_hours': OPTIONAL,      # required when sounding_dir is set
        'max_sounding_distance_km': OPTIONAL,    # required when sounding_dir is set
        'temperature_profile': REQUIRED,
        'melting_level_km': REQUIRED,
    },
    'run': {
        'save_mode': REQUIRED,
        'overwrite_existing': REQUIRED,
        'plot_summary': REQUIRED,
    },
}

# Top-level keys of a v0.1 flat config, used only to recognise one and say so.
_V01_MARKERS = ('masking_criteria', 'horizontal_velocity_repair',
                'vertical_velocity_criteria', 'defaults', 'modules')


# ----------------------------------------------------------------- access
def section(cfg: Mapping, name: str) -> Mapping:
    """The config section ``name``, which must be present and be a mapping.

    Parameters
    ----------
    cfg : Mapping
        A resolved config, or any one of its sections.
    name : str
        Section name, e.g. ``'horizontal_wind'``.

    Returns
    -------
    Mapping
        ``cfg[name]``.

    Raises
    ------
    MissingConfigKeyError
        If the section is absent or is not a mapping (``null`` included).
    """
    if name not in cfg:
        raise MissingConfigKeyError(
            f"config section '{name}' is required and must hold keys")
    value = cfg[name]
    if not isinstance(value, Mapping):
        raise MissingConfigKeyError(
            f"config section '{name}' must hold keys, got {value!r}")
    return value


def require(cfg_section: Mapping, section_name: str, key: str,
            why: str = "") -> object:
    """The value of a required key, or a loud error naming it.

    Parameters
    ----------
    cfg_section : Mapping
        One section of the config (see `section`).
    section_name : str
        Its dotted name, for the message.
    key : str
        The key that must be present. ``null`` is a present value and is
        returned as ``None`` -- keys that accept ``null`` to mean "off" say
        so in their own documentation.
    why : str, optional
        What the key decides, appended to the message.

    Returns
    -------
    object
        ``cfg_section[key]``, unconverted.

    Raises
    ------
    MissingConfigKeyError
        If the key is absent. There is never a hidden default.
    """
    if key not in cfg_section:
        msg = f"{section_name}.{key} is required and has no default"
        if why:
            msg += f": {why}"
        raise MissingConfigKeyError(msg)
    return cfg_section[key]


def refuse_retired(cfg_section: Mapping, section_name: str,
                   retired: Mapping[str, str]) -> None:
    """Fail on config keys the code does not honour.

    A retired key left in a config would be ignored, and the run would
    silently mean something other than what the config says. Each retired
    key maps to the message that tells the user what replaced it.

    Parameters
    ----------
    cfg_section : Mapping
        One section of the config.
    section_name : str
        Its name, for the message.
    retired : Mapping[str, str]
        ``{retired_key: what to do instead}``.

    Raises
    ------
    ConfigError
        Naming every retired key found, and the replacement for each.
    """
    found = sorted(k for k in retired if k in cfg_section)
    if found:
        lines = [f"{section_name}.{k}: {retired[k]}" for k in found]
        raise ConfigError(
            "config keys the code does not honour; update the config:\n  "
            + "\n  ".join(lines))


def dotted_get(cfg: Mapping, dotted: str):
    """``cfg['a']['b']['c']`` for ``'a.b.c'``; MissingConfigKeyError if absent."""
    node = cfg
    walked = []
    for part in dotted.split('.'):
        walked.append(part)
        if not isinstance(node, Mapping) or part not in node:
            raise MissingConfigKeyError(
                f"config key {'.'.join(walked)!r} does not exist")
        node = node[part]
    return node


def dotted_set(cfg: dict, dotted: str, value) -> None:
    """Set an EXISTING leaf ``'a.b.c'`` to ``value``, in place.

    Raises
    ------
    MissingConfigKeyError
        If the path does not exist.
    ConfigError
        If the path names a section rather than a leaf.
    """
    *parents, leaf = dotted.split('.')
    node = cfg
    for p in parents:
        node = dotted_get(node, p)
    if leaf not in node:
        raise MissingConfigKeyError(f"config key {dotted!r} does not exist")
    if isinstance(node[leaf], Mapping):
        raise ConfigError(f"config key {dotted!r} is a section, not a value; "
                          "override its keys one by one")
    node[leaf] = value


# ------------------------------------------------------------- validation
def validate(cfg: Mapping, schema: Mapping, where: str = '') -> None:
    """Every required key present, no unknown key, at every level.

    Parameters
    ----------
    cfg : Mapping
        The mapping to check.
    schema : Mapping
        `RETRIEVAL_SCHEMA` or `SITE_SCHEMA` (or one subsection of either).
    where : str
        Dotted prefix for messages.

    Raises
    ------
    MissingConfigKeyError
        On the first required key that is absent.
    ConfigError
        On keys the schema does not know -- a misspelt key would otherwise
        be ignored and the run would silently use something else.
    """
    prefix = where + '.' if where else ''
    unknown = sorted(k for k in cfg if k not in schema)
    if unknown:
        raise ConfigError(
            f"unknown config key(s) {[prefix + k for k in unknown]}; "
            f"{where or 'the top level'} accepts {sorted(schema)}")
    for key, rule in schema.items():
        if isinstance(rule, Mapping):
            validate(section(cfg, key), rule, prefix + key)
        elif rule == REQUIRED and key not in cfg:
            raise MissingConfigKeyError(
                f"{prefix}{key} is required and has no default")


# --------------------------------------------------------------- loading
def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ConfigError(f'{path}: not a mapping of sections')
    return dict(data)


def _check_version(data: Mapping, path) -> None:
    if any(k in data for k in _V01_MARKERS):
        raise ConfigError(
            f'{path} is a v0.1 single-file config (it has '
            f'{[k for k in _V01_MARKERS if k in data]}). windvel 1.0.0 reads a '
            'site file that points at a retrieval file; see '
            'examples/site.example.yaml and examples/retrieval.yaml')
    if 'config_version' not in data:
        raise MissingConfigKeyError(f'{path}: config_version is required')
    if data['config_version'] != CONFIG_VERSION:
        raise ConfigError(f'{path}: config_version {data["config_version"]!r} '
                          f'is not {CONFIG_VERSION}')


def resolve(site: Mapping, retrieval: Mapping, site_file='', retrieval_file='') -> dict:
    """One config from a site mapping and a retrieval mapping.

    Parameters
    ----------
    site : Mapping
        The site file's content: ``config_version``, ``retrieval``,
        ``retrieval_overrides`` and the `SITE_SCHEMA` sections.
    retrieval : Mapping
        The retrieval file's content: ``config_version``,
        ``retrieval_version`` and the `RETRIEVAL_SCHEMA` sections.
    site_file, retrieval_file : str, optional
        Names for the ``provenance`` section and messages.

    Returns
    -------
    dict
        The retrieval sections (with overrides applied), the site sections,
        ``config_version``, and ``provenance`` = ``{site_file,
        retrieval_file, retrieval_version, retrieval_overrides}``.

    Raises
    ------
    ConfigError, MissingConfigKeyError
        On any missing, unknown or retired key, or an override that names
        no existing leaf.
    """
    site = dict(site)
    retrieval = dict(retrieval)
    for name, data in ((site_file or 'site', site),
                       (retrieval_file or 'retrieval', retrieval)):
        _check_version(data, name)

    retrieval_version = require(retrieval, 'retrieval', 'retrieval_version',
                                'a label you bump when the science changes')
    retrieval_body = {k: v for k, v in retrieval.items()
                      if k not in ('config_version', 'retrieval_version')}
    validate(retrieval_body, RETRIEVAL_SCHEMA)

    require(site, 'site', 'retrieval', 'path of the retrieval file')
    overrides = require(site, 'site', 'retrieval_overrides',
                        '{dotted.key: value}, or {} for none') or {}
    if not isinstance(overrides, Mapping):
        raise ConfigError('site retrieval_overrides must be a mapping of '
                          'dotted.key: value')
    site_body = {k: v for k, v in site.items()
                 if k not in ('config_version', 'retrieval', 'retrieval_overrides')}
    validate(site_body, SITE_SCHEMA)
    env = site_body['environment']
    if env['sounding_dir']:
        for k in ('max_sounding_age_hours', 'max_sounding_distance_km'):
            require(env, 'environment', k, 'required when sounding_dir is set')

    import copy
    body = copy.deepcopy(retrieval_body)
    for dotted, value in overrides.items():
        dotted_set(body, str(dotted), value)   # must name an existing leaf

    resolved = {'config_version': CONFIG_VERSION}
    resolved.update(body)
    resolved.update(copy.deepcopy(site_body))
    resolved['provenance'] = {
        'site_file': str(site_file),
        'retrieval_file': str(retrieval_file),
        'retrieval_version': str(retrieval_version),
        'retrieval_overrides': {str(k): v for k, v in overrides.items()},
    }
    return resolved


def load_config(site_file) -> dict:
    """Read a site file and the retrieval file it names; return `resolve` of them.

    Parameters
    ----------
    site_file : str or Path
        ``site_<radar>.yaml``. Its ``retrieval`` key is a path relative to
        the site file's directory (or absolute).

    Returns
    -------
    dict
        See `resolve`.
    """
    site_path = Path(site_file)
    site = _read_yaml(site_path)
    _check_version(site, site_path)
    rel = require(site, 'site', 'retrieval', 'path of the retrieval file')
    retrieval_path = Path(rel)
    if not retrieval_path.is_absolute():
        retrieval_path = (site_path.parent / retrieval_path).resolve()
    retrieval = _read_yaml(retrieval_path)
    return resolve(site, retrieval, site_file=str(site_path),
                   retrieval_file=str(retrieval_path))
