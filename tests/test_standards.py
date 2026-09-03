"""The coding standards, as behaviour.

Required keys fail loudly and name themselves; unknown keys are refused;
the output file's settings are the retrieval's settings; log lines are in
UTC; every stage shares one geometry; the two-file config resolves to one
mapping with its provenance.
"""
import json
import logging
import time

import numpy as np
import pytest
import yaml

from _configs import full, retrieval, site
from windvel.calculate_windvel import (
    calculate_windvel,
    resolve_repair_settings,
)
from windvel.config import (
    RETRIEVAL_SCHEMA,
    SITE_SCHEMA,
    dotted_get,
    dotted_set,
    load_config,
    refuse_retired,
    require,
    resolve,
    section,
    validate,
)
from windvel.errors import (
    ConfigError,
    InputDataError,
    InputFieldError,
    MissingConfigKeyError,
    WindvelError,
)
from windvel.select_cloud_transects import select_cloud_transects
from windvel.tables import attenuation_relation, vertical_error_table
from windvel.utils import gate_dist_alt_km

# ---------------------------------------------------------------- config

def test_section_and_require_name_what_is_missing():
    with pytest.raises(MissingConfigKeyError, match="section 'a'"):
        section({}, 'a')
    with pytest.raises(MissingConfigKeyError, match='must hold keys'):
        section({'a': None}, 'a')
    with pytest.raises(MissingConfigKeyError, match=r'a\.b is required'):
        require({}, 'a', 'b')
    with pytest.raises(MissingConfigKeyError, match='because'):
        require({}, 'a', 'b', 'because')
    assert require({'b': None}, 'a', 'b') is None      # null is a value
    assert require({'b': 3}, 'a', 'b') == 3


def test_refuse_retired_names_every_key_and_its_replacement():
    with pytest.raises(ConfigError) as exc:
        refuse_retired({'old': 1, 'older': 2, 'fine': 3}, 'sec',
                       {'old': 'use new', 'older': 'use newer'})
    msg = str(exc.value)
    assert 'sec.old: use new' in msg and 'sec.older: use newer' in msg
    assert 'fine' not in msg
    refuse_retired({'fine': 3}, 'sec', {'old': 'use new'})   # nothing to refuse


def test_the_error_types_are_catchable_by_base_and_by_builtin():
    assert issubclass(MissingConfigKeyError, ConfigError)
    assert issubclass(MissingConfigKeyError, KeyError)
    assert issubclass(MissingConfigKeyError, ValueError)
    assert issubclass(InputDataError, WindvelError)
    assert issubclass(InputDataError, ValueError)
    assert issubclass(InputFieldError, KeyError)
    assert str(MissingConfigKeyError('plain')) == 'plain'


# ---------------------------------------------------------------- schema

def _leaves(schema, prefix=''):
    for k, rule in schema.items():
        if isinstance(rule, dict):
            yield from _leaves(rule, prefix + k + '.')
        else:
            yield prefix + k


def test_the_complete_retrieval_and_site_validate():
    r = retrieval()
    del r['config_version'], r['retrieval_version']
    validate(r, RETRIEVAL_SCHEMA)
    s = site()
    for k in ('config_version', 'retrieval', 'retrieval_overrides'):
        del s[k]
    validate(s, SITE_SCHEMA)


@pytest.mark.parametrize('dotted', sorted(_leaves(RETRIEVAL_SCHEMA)))
def test_every_retrieval_leaf_is_required(dotted):
    r = retrieval()
    del r['config_version'], r['retrieval_version']
    *parents, leaf = dotted.split('.')
    node = r
    for p in parents:
        node = node[p]
    del node[leaf]
    with pytest.raises(MissingConfigKeyError, match=leaf):
        validate(r, RETRIEVAL_SCHEMA)


def test_an_unknown_key_is_refused_anywhere():
    r = retrieval()
    del r['config_version'], r['retrieval_version']
    r['horizontal_wind']['anchor']['min_edge_gates'] = 15     # the v0.1 name
    with pytest.raises(ConfigError, match=r'horizontal_wind\.anchor\.min_edge_gates'):
        validate(r, RETRIEVAL_SCHEMA)
    r = retrieval()
    del r['config_version'], r['retrieval_version']
    r['masking_criteria'] = {}
    with pytest.raises(ConfigError, match='masking_criteria'):
        validate(r, RETRIEVAL_SCHEMA)


# --------------------------------------------------------------- resolve

def test_resolve_merges_and_records_provenance():
    cfg = resolve(site(), retrieval(), site_file='s.yaml', retrieval_file='r.yaml')
    assert cfg['config_version'] == 1
    assert cfg['horizontal_wind']['anchor']['min_gates'] == 15
    assert cfg['run']['save_mode'] == 'full'
    assert cfg['provenance'] == {'site_file': 's.yaml', 'retrieval_file': 'r.yaml',
                                 'retrieval_version': 'test',
                                 'retrieval_overrides': {}}


def test_an_override_changes_exactly_the_named_leaf_and_is_recorded():
    s = site(retrieval_overrides={'horizontal_wind.spike_correction.k_mad': 0.0})
    cfg = resolve(s, retrieval())
    assert cfg['horizontal_wind']['spike_correction']['k_mad'] == 0.0
    assert cfg['horizontal_wind']['spike_correction']['max_fraction'] == 0.25
    assert cfg['provenance']['retrieval_overrides'] == {
        'horizontal_wind.spike_correction.k_mad': 0.0}


def test_an_override_must_name_an_existing_leaf():
    with pytest.raises(MissingConfigKeyError, match='k_madd'):
        resolve(site(retrieval_overrides={'horizontal_wind.spike_correction.k_madd': 0}),
                retrieval())
    with pytest.raises(ConfigError, match='is a section'):
        resolve(site(retrieval_overrides={'horizontal_wind.anchor': {}}), retrieval())


def test_a_v01_flat_config_is_recognised_and_refused():
    old = site()
    old['masking_criteria'] = {'reflectivity_threshold': -20}
    with pytest.raises(ConfigError, match=r'v0\.1'):
        resolve(old, retrieval())


def test_config_version_is_required_and_checked():
    s = site()
    del s['config_version']
    with pytest.raises(MissingConfigKeyError, match='config_version'):
        resolve(s, retrieval())
    r = retrieval()
    r['config_version'] = 2
    with pytest.raises(ConfigError, match='config_version'):
        resolve(site(), r)


def test_sounding_bounds_are_required_only_with_a_sounding_dir(tmp_path):
    s = site()
    s['environment']['sounding_dir'] = str(tmp_path)
    with pytest.raises(MissingConfigKeyError, match='max_sounding_age_hours'):
        resolve(s, retrieval())
    s['environment'].update(max_sounding_age_hours=12, max_sounding_distance_km=150)
    resolve(s, retrieval())


def test_dotted_access():
    cfg = {'a': {'b': {'c': 1}}}
    assert dotted_get(cfg, 'a.b.c') == 1
    dotted_set(cfg, 'a.b.c', 2)
    assert cfg['a']['b']['c'] == 2
    with pytest.raises(MissingConfigKeyError, match=r'a\.x'):
        dotted_get(cfg, 'a.x.c')


def test_load_config_reads_the_two_files(tmp_path):
    (tmp_path / 'retrieval.yaml').write_text(yaml.safe_dump(retrieval()))
    s = site(retrieval_overrides={'cloud_objects.min_area_km2': 1.5})
    (tmp_path / 'site_x.yaml').write_text(yaml.safe_dump(s))
    cfg = load_config(tmp_path / 'site_x.yaml')
    assert cfg['cloud_objects']['min_area_km2'] == 1.5
    assert cfg['provenance']['retrieval_file'] == str(tmp_path / 'retrieval.yaml')
    assert cfg['provenance']['site_file'] == str(tmp_path / 'site_x.yaml')


def test_the_shipped_examples_load(tmp_path):
    """examples/retrieval.yaml + site.example.yaml resolve as shipped."""
    from pathlib import Path
    ex = Path(__file__).resolve().parent.parent / 'examples'
    cfg = load_config(ex / 'site.example.yaml')
    assert cfg['provenance']['retrieval_version']
    assert cfg['horizontal_wind']['anchor']['weak_echo_dbz'] == [25, 30]


# ---------------------------------------------------------------- tables

def test_physics_tables_by_name_or_inline():
    rel = attenuation_relation('x_andsager')
    assert rel['band'] == 'X' and rel['name'] == 'x_andsager'
    inline = attenuation_relation({'band': 'C', 'a': 1e-5, 'b': 0.8})
    assert inline['name'] == 'inline'
    with pytest.raises(ConfigError, match='nonesuch'):
        attenuation_relation('nonesuch')
    with pytest.raises(ConfigError, match="missing \\['b'\\]"):
        attenuation_relation({'band': 'C', 'a': 1e-5})
    e, v = vertical_error_table('chivo_tracer_2022')
    assert e.size == v.size == 7
    with pytest.raises(ConfigError, match='increase'):
        vertical_error_table({'elevation_deg': [0, 50, 40], 'dVh1': [1, 2, 3]})


# ------------------------------------------------------- the whole pipeline

def test_resolve_repair_settings_reads_the_ladder():
    cfg = full()
    st = resolve_repair_settings(cfg)
    assert st.refl_ladder_dbz == [25.0, 30.0]
    assert st.bin_size_m == 300.0
    cfg['horizontal_wind']['anchor']['weak_echo_dbz'] = [30]
    assert resolve_repair_settings(cfg).refl_ladder_dbz == [30.0]
    cfg['horizontal_wind']['anchor']['weak_echo_dbz'] = [1, 2, 3]
    with pytest.raises(ConfigError, match='weak_echo_dbz'):
        resolve_repair_settings(cfg)


def _rhi_radar():
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    radar = pyart_testing.make_empty_rhi_radar(40, 30, 1)
    nrays, ngates = radar.nrays, radar.ngates
    radar.elevation['data'] = np.ma.array(np.linspace(2.0, 80.0, nrays),
                                          mask=False)
    refl = np.full((nrays, ngates), -100.0)
    refl[5:, 4:30] = 35.0
    refl[5:, 4:8] = 10.0          # weak echo at the near edge
    refl[5:, 26:30] = 10.0        # and the far edge
    ok = np.zeros_like(refl, bool)
    radar.fields['reflectivity'] = {'data': np.ma.masked_array(refl, mask=ok)}
    radar.fields['vr'] = {'data': np.ma.masked_array(
        np.full((nrays, ngates), 5.0), mask=ok)}
    radar.fields['PID'] = {'data': np.full((nrays, ngates), 3, dtype=int)}
    return radar


def test_output_stamps_the_settings_the_retrieval_used():
    """The provenance field's attributes come from the SAME resolved settings
    the retrieval read, so a config with min_gates 7 can never be stamped
    as 15 or passes [3, 5] stamped for a run that used [2]."""
    radar = _rhi_radar()
    cfg = full()
    cfg['cloud_objects']['min_area_km2'] = 0.0
    cfg['horizontal_wind']['anchor']['min_gates'] = 7
    cfg['horizontal_wind']['spike_correction']['passes'] = [2]
    cfg['horizontal_wind']['fill_mirror'] = False
    select_cloud_transects(radar, cfg, [0])
    calculate_windvel(radar, cfg, [0])
    src = radar.fields['horizontal_velocity_source']
    assert src['min_edge_gates'] == '7'
    assert src['spike_passes'] == '[2]'
    assert src['fill_mirror'] == 'False'
    assert src['reflectivity_ladder_dbz'] == '[25.0, 30.0]'


def test_saved_file_carries_the_config_provenance(tmp_path):
    import netCDF4

    from windvel.save_windvel import save_windvel_file
    radar = _rhi_radar()
    cfg = full()
    cfg['provenance']['retrieval_overrides'] = {'cloud_objects.min_area_km2': 0.0}
    cfg['provenance']['retrieval_version'] = '1.0'
    out = save_windvel_file(tmp_path / 'o.nc', radar, cfg, mode='full')
    with netCDF4.Dataset(out) as ds:
        assert ds.windvel_retrieval_version == '1.0'
        assert ds.windvel_site_file == 'site_test.yaml'
        assert json.loads(ds.windvel_retrieval_overrides) == {
            'cloud_objects.min_area_km2': 0.0}


def test_a_missing_input_field_is_an_input_field_error():
    radar = _rhi_radar()
    cfg = full(refl_var='nonesuch')
    with pytest.raises(InputFieldError, match='nonesuch'):
        select_cloud_transects(radar, cfg, [0])


# ------------------------------------------------------------- geometry

def test_every_stage_shares_one_gate_geometry():
    radar = _rhi_radar()
    x, y, z = radar.get_gate_x_y_z(0)
    dk, ak = gate_dist_alt_km(radar, 0)
    assert np.array_equal(dk, np.sqrt(x ** 2 + y ** 2) / 1000.0)
    assert np.array_equal(ak, z / 1000.0)


# ------------------------------------------------------------------- UTC

def test_run_log_lines_are_utc():
    from windvel.cli import utc_log_formatter
    fmt = utc_log_formatter()
    rec = logging.LogRecord('windvel', logging.INFO, __file__, 1, 'hi', (), None)
    rec.created = 0.0                     # the epoch, 1970-01-01T00:00:00Z
    line = fmt.format(rec)
    assert line.startswith('1970-01-01T00:00:00Z INFO hi')
    assert fmt.converter is time.gmtime
