"""Complete config sections for the tests, in the 1.0.0 two-file shape.

Every science key is required by the package, so a test that wants to vary
one key starts from a complete section and overrides it. Values match
examples/retrieval.yaml unless a test says otherwise. Helpers for nested
sections take flat keyword names and place them (see `horizontal_wind`).
"""

import copy


def _apply(base, over, allowed):
    unknown = sorted(set(over) - set(allowed))
    if unknown:
        raise KeyError(f'unknown test config key(s) {unknown}; use {sorted(allowed)}')
    base.update(over)
    return base


def cloud_objects(**over):
    d = {'reflectivity_threshold_dbz': -20, 'min_area_km2': 7.0,
         'gate_quality': [], 'boundary_thickness_gates': 3,
         'exclude_lowest_rays': 3}
    return _apply(d, over, d)


def object_selection(**over):
    d = {'min_span_km': None, 'max_span_km': None, 'min_top_km': None,
         'min_depth_km': None}
    return _apply(d, over, d)


def attenuation(**over):
    d = {'correct': False, 'relation': None, 'require_band_match': True,
         'liquid_only': True, 'max_correction_db': 10.0}
    return _apply(d, over, d)


def corrections(attenuation_cfg=None):
    return {'attenuation': attenuation() if attenuation_cfg is None
            else attenuation_cfg}


def sedimentation(method='pid_giangrande_oklahoma2013'):
    return {'method': method}


_HW_PLACE = {
    'bin_size_m': (), 'fill_mirror': (), 'fill_interpolate_max_gap_bins': (),
    'min_bin_gates': (),
    'weak_echo_dbz': ('anchor',), 'min_gates': ('anchor',),
    'max_elevation_deg': ('anchor',), 'max_candidate_mps': ('anchor',),
    'robust': ('anchor',),
    'passes': ('spike_correction',), 'shear_per_km': ('spike_correction',),
    'k_mad': ('spike_correction',), 'max_fraction': ('spike_correction',),
}


def horizontal_wind(**over):
    """The horizontal_wind section; flat keyword names land in their subsection."""
    d = {
        'bin_size_m': 300, 'fill_mirror': True,
        'fill_interpolate_max_gap_bins': 2, 'min_bin_gates': 0,
        'anchor': {'weak_echo_dbz': [25, 30], 'min_gates': 15,
                   'max_elevation_deg': 65.0, 'max_candidate_mps': 40.0,
                   'robust': True},
        'spike_correction': {'passes': [3, 5], 'shear_per_km': 12.0,
                             'k_mad': 3.5, 'max_fraction': 0.25},
    }
    unknown = sorted(set(over) - set(_HW_PLACE))
    if unknown:
        raise KeyError(f'unknown horizontal_wind test key(s) {unknown}')
    for k, v in over.items():
        node = d
        for p in _HW_PLACE[k]:
            node = node[p]
        node[k] = v
    return d


def vv_compute(**over):
    d = {'max_horizontal_distance_km': 35.0, 'min_elevation_deg': 5.0}
    return _apply(d, over, d)


def vv_error(**over):
    d = {'dVr': 0.2, 'dVsed': 2.0, 'dVh2': 2.0, 'dVh1_table': 'chivo_tracer_2022'}
    return _apply(d, over, d)


def vv_usable(**over):
    d = {'min_elevation_deg': 30.0, 'max_horizontal_distance_km': None,
         'max_abs_w_mps': 60.0}
    return _apply(d, over, d)


def vertical_velocity(compute=None, error=None, usable=None):
    return {'compute': vv_compute() if compute is None else compute,
            'error': vv_error() if error is None else error,
            'usable': vv_usable() if usable is None else usable}


def co_threshold(**over):
    d = {'w_mps': 6.0, 'min_area_km2': 0.1}
    return _apply(d, over, d)


def co_persistence(**over):
    d = {'min_sigma': 2.0, 'floor_sigma': 1.0, 'min_area_km2': 0.1}
    return _apply(d, over, d)


def co_usable(**over):
    d = {'min_area_fraction': 0.7, 'core_peak_fraction': 0.8, 'core_min_gates': 3}
    return _apply(d, over, d)


def coherent_objects(threshold=None, persistence=None, usable=None):
    return {'threshold': co_threshold() if threshold is None else threshold,
            'persistence': co_persistence() if persistence is None else persistence,
            'usable': co_usable() if usable is None else usable}


def environment(**over):
    d = {'temperature_field': None, 'sounding_dir': None,
         'temperature_profile': None, 'melting_level_km': 4.89}
    return _apply(d, over, set(d) | {'max_sounding_age_hours',
                                     'max_sounding_distance_km'})


def run(**over):
    d = {'save_mode': 'full', 'overwrite_existing': True, 'plot_summary': True}
    return _apply(d, over, d)


def input_variables(refl_var='reflectivity', vr_var='vr', pid_var='PID'):
    return {'refl_var': refl_var, 'vr_var': vr_var, 'pid_var': pid_var}


def paths(**over):
    d = {'input_data_path': '/nonexistent/in', 'file_pattern': '*.nc',
         'output_dir': '/nonexistent/out', 'log_dir': '/nonexistent/log'}
    return _apply(d, over, d)


def provenance(**over):
    d = {'site_file': 'site_test.yaml', 'retrieval_file': 'retrieval_test.yaml',
         'retrieval_version': 'test', 'retrieval_overrides': {}}
    return _apply(d, over, d)


def retrieval():
    """A complete retrieval mapping, as a retrieval.yaml would load."""
    return copy.deepcopy({
        'config_version': 1,
        'retrieval_version': 'test',
        'corrections': corrections(),
        'cloud_objects': cloud_objects(),
        'object_selection': object_selection(),
        'sedimentation': sedimentation(),
        'horizontal_wind': horizontal_wind(),
        'vertical_velocity': vertical_velocity(),
        'coherent_objects': coherent_objects(),
    })


def site(**over):
    """A complete site mapping, as a site_<radar>.yaml would load."""
    d = {
        'config_version': 1,
        'retrieval': 'retrieval.yaml',
        'retrieval_overrides': {},
        'paths': paths(),
        'input_variables': input_variables(),
        'environment': environment(),
        'run': run(),
    }
    d.update(over)
    return copy.deepcopy(d)


def full(refl_var='reflectivity', vr_var='vr', pid_var='PID'):
    """Every section the pipeline reads, complete and resolved."""
    r = retrieval()
    del r['config_version'], r['retrieval_version']
    cfg = {'config_version': 1}
    cfg.update(r)
    cfg.update({
        'paths': paths(),
        'input_variables': input_variables(refl_var, vr_var, pid_var),
        'environment': environment(),
        'run': run(),
        'provenance': provenance(),
    })
    return copy.deepcopy(cfg)
