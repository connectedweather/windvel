"""
Deterministic tests for the horizontal-velocity fill ladder.

    rung 1 MEASURED     in-bin median of >= min_edge_gates weak-echo gates
    rung 2 MIRRORED     innermost mirror from the far half of the SAME bin
                        (far half must be MEASURED with 2x min_edge_gates)
    rung 3 INTERPOLATED interior gaps of at most
                        fill_interpolate_max_gap_bins bins, bounded by
                        finite anchors above AND below, filled linearly in
                        altitude, each edge on its own
    rung 0 NONE         NaN
    (the whole-edge mirror, anchor blend and cross-edge nudge do not exist)

Geometry is synthetic and fully controlled, so every expected number is
hand-computable:

  * elevation is 0 for every ray, so sin(elev)=0 and cos(elev)=1. The
    sedimentation term drops out and hv_candidate == vr exactly.
  * one ray per altitude bin (alt 150, 450, 750 m with 300 m bins), so
    "bin i" and "ray i" are the same thing.
  * slant range increases along gates only, identically for every ray, so the
    near/far split of a bin is always gates 0..9 / 10..19.

These are the regression tests for the whole-object coverage fallback that used
to fill a short half from every altitude of the object at once.
"""
import numpy as np
import pytest

from _configs import corrections, horizontal_wind
from windvel.calculate_windvel import (
    HV_SRC_INTERPOLATED,
    HV_SRC_MEASURED,
    HV_SRC_MIRRORED,
    HV_SRC_NONE,
    HV_SRC_SMOOTHED,
    calculate_vertical_velocity_error,
    get_horizontal_velocity,
    smooth_edge_profile,
    vertical_velocity_usability,
)

NGATES = 20
QUALIFY = 0.0      # reflectivity that passes refl < horizontal_wind_ref_thresh
REJECT = 100.0     # reflectivity that fails it
BIN_M = 300.0


class StubRadar:
    """Minimal stand-in: get_horizontal_velocity only touches .fields/.elevation."""

    def __init__(self, fields, elevation, sweep_slices=None):
        self.fields = fields
        self.elevation = {'data': elevation}
        # objects are now always resolved per sweep, so every path asks
        self._slices = sweep_slices or [slice(0, len(elevation))]

    @property
    def nsweeps(self):
        return len(self._slices)

    def get_slice(self, sweep):
        return self._slices[sweep]


# the spike-correction thresholds the site configs carry
SPIKE = dict(shear_max=12.0, k_mad=3.5, max_frac=0.25)


def make_cfg(**over):
    hw = dict(min_gates=5, max_candidate_mps=None,   # QC layer 2 off unless asked
              weak_echo_dbz=[25], bin_size_m=BIN_M)
    hw.update(over)
    return {
        'input_variables': {'refl_var': 'refl', 'vr_var': 'vr'},
        'corrections': corrections(),
        'horizontal_wind': horizontal_wind(**hw),
    }


# every leaf of horizontal_wind, as (subsection or None, key)
HW_LEAVES = [(None, k) for k in ('bin_size_m', 'fill_mirror',
                                 'fill_interpolate_max_gap_bins',
                                 'min_bin_gates')] + \
            [('anchor', k) for k in ('weak_echo_dbz', 'min_gates', 'max_elevation_deg',
                                     'max_candidate_mps', 'robust')] + \
            [('spike_correction', k) for k in ('passes', 'shear_per_km', 'k_mad',
                                               'max_fraction')]


def build(nrays, vr, refl):
    """
    vr, refl : (nrays, NGATES) arrays.
    Returns (radar, sr_map, alt_map). One altitude bin per ray.
    """
    vr = np.asarray(vr, dtype=float)
    refl = np.asarray(refl, dtype=float)
    assert vr.shape == (nrays, NGATES)

    sr_map = np.tile(10.0 + 0.1 * np.arange(NGATES), (nrays, 1))       # km
    # Ray r sits in bin r. The 1 m spread across gates lifts the object's zmax
    # just above the top ray, so np.arange(zmin, zmax + bin, bin) yields nrays+1
    # edges -- i.e. nrays bins. Without it the top ray falls outside every bin
    # (and a single-ray case produces no bins at all).
    alt_map = (150.0 + BIN_M * np.arange(nrays))[:, None] \
        + np.linspace(0.0, 1.0, NGATES)[None, :]
    alt_map = np.ascontiguousarray(alt_map)

    fields = {
        'refl': {'data': refl},
        'vr': {'data': vr},
        'sedimentation_velocity': {'data': np.zeros((nrays, NGATES))},
        'local_object_ids': {'data': np.ones((nrays, NGATES), dtype=int)},
    }
    return StubRadar(fields, np.zeros(nrays)), sr_map, alt_map


def all_qualify(nrays):
    return np.full((nrays, NGATES), QUALIFY)


# ---------------------------------------------------------------- rung 1


def test_both_edges_measured_interpolates_linearly():
    """Both halves observed -> plain linear interpolation, everything MEASURED."""
    vr = np.zeros((1, NGATES))
    vr[0, :10] = 10.0     # near half
    vr[0, 10:] = 20.0     # far half
    radar, sr, alt = build(1, vr, all_qualify(1))

    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)

    assert hv[0, 0] == pytest.approx(10.0)     # at the near edge
    assert hv[0, -1] == pytest.approx(20.0)    # at the far edge
    assert hv[0, 9] == pytest.approx(10.0 + (0.9 / 1.9) * 10.0)
    assert np.all(src == HV_SRC_MEASURED)


# ---------------------------------------------------------------- rung 2


def test_short_half_is_mirrored_not_averaged():
    """
    Near half has 2 qualifying gates (< min_edge_gates 5) carrying a wildly
    different wind. It must be mirrored from the far half, so the bin ends up
    flat at the far-half value -- and must NOT pick up the 2 stray gates.
    """
    vr = np.full((1, NGATES), 20.0)
    vr[0, :2] = -50.0
    refl = np.full((1, NGATES), REJECT)
    refl[0, :2] = QUALIFY      # only 2 usable gates on the near side
    refl[0, 10:] = QUALIFY     # all 10 usable on the far side
    radar, sr, alt = build(1, vr, refl)

    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)

    assert np.allclose(hv[0, :], 20.0)         # flat: no fabricated gradient
    assert np.all(src == HV_SRC_MIRRORED)
    assert not np.any(np.isclose(hv, -50.0))


def test_mirror_can_be_disabled():
    """fill_mirror: False -> the short half falls through to the next rung."""
    vr = np.full((1, NGATES), 20.0)
    refl = np.full((1, NGATES), REJECT)
    refl[0, 10:] = QUALIFY
    radar, sr, alt = build(1, vr, refl)

    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=False), sr, alt)

    assert np.all(np.isnan(hv))                # single bin, nothing to fit from
    assert np.all(src == HV_SRC_NONE)


# ------------------------------------------------- regression: no whole-object


def test_short_half_does_not_borrow_from_other_altitudes():
    """
    Regression against a whole-object coverage fallback.

    Bin 0's near half is short. Bins 1 and 2 are fully observed and carry a very
    different wind. The old code filled bin 0's near edge from the near half of
    the ENTIRE object -- i.e. from bins 1 and 2, hundreds of metres higher. The
    ladder must instead mirror bin 0's own far half.
    """
    vr = np.zeros((3, NGATES))
    vr[0, :] = 20.0            # bin 0 truth
    vr[1, :] = -30.0           # bins 1-2 carry a very different wind
    vr[2, :] = -30.0

    refl = all_qualify(3)
    refl[0, :10] = REJECT      # bin 0 near half unusable (0 gates)

    radar, sr, alt = build(3, vr, refl)
    # Raise the spike tolerance so the deliberate 50 m/s jump between bin 0 and
    # bin 1 does not trip Pass A's inconsistency correction. This test is about
    # which gates a short half draws from, not about spike repair.
    hv, src = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)

    # bin 0 is flat at its own far-half value
    assert np.allclose(hv[0, :], 20.0)
    assert np.all(src[0, :] == HV_SRC_MIRRORED)

    # a near anchor taken from the whole object's mean would give this value
    whole_object_near_mean = -30.0
    assert not np.any(np.isclose(hv[0, :], whole_object_near_mean))

    # untouched bins stay measured
    assert np.all(src[1, :] == HV_SRC_MEASURED)
    assert np.allclose(hv[1, :], -30.0)


# ------------------------------------------------- mirrors are terminal

def test_mirrored_bin_is_terminal_nothing_builds_on_it():
    """
    A mirrored edge holds borrowed gates, so nothing may build on it.
    Bin 0's near edge is mirrored and is the only value below bin 1, so
    bin 1's near edge has nothing legal to draw from and stays NaN.
    """
    vr = np.full((2, NGATES), 20.0)
    refl = np.full((2, NGATES), REJECT)
    refl[0, 10:] = QUALIFY     # bin 0: far half only -> near half mirrored
    # bin 1: nothing usable at all

    radar, sr, alt = build(2, vr, refl)
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)

    assert np.all(src[0, :] == HV_SRC_MIRRORED)
    assert np.all(np.isnan(hv[1, :]))
    assert np.all(src[1, :] == HV_SRC_NONE)


# ---------------------------------------------------------------- rung 4


def test_no_usable_gates_stays_nan():
    radar, sr, alt = build(1, np.zeros((1, NGATES)), np.full((1, NGATES), REJECT))
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)

    assert np.all(np.isnan(hv))
    assert np.all(src == HV_SRC_NONE)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("sub,missing", HW_LEAVES)
def test_missing_ladder_key_fails_loudly(sub, missing):
    cfg = make_cfg()
    node = cfg['horizontal_wind'] if sub is None else cfg['horizontal_wind'][sub]
    node.pop(missing)
    radar, sr, alt = build(1, np.zeros((1, NGATES)), all_qualify(1))

    with pytest.raises(KeyError) as exc:
        get_horizontal_velocity(radar, cfg, sr, alt)
    assert missing in str(exc.value)
def test_source_field_covers_every_gate_with_a_value():
    """hv finite <=> src non-zero. The provenance field must never lie."""
    vr = np.full((3, NGATES), 5.0)
    refl = all_qualify(3)
    refl[1, :] = REJECT
    refl[2, :10] = REJECT

    radar, sr, alt = build(3, vr, refl)
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)

    assert np.array_equal(np.isfinite(hv), src != HV_SRC_NONE)


# ---------------------------------------------- superseded knobs refused


def test_retired_ladder_knobs_are_refused():
    """The whole-edge mirror, blend and nudge do not exist in this ladder; a config
    still carrying their keys must fail loudly, not silently mean less."""
    from windvel.config import RETRIEVAL_SCHEMA, validate
    for key, val in (('mirror_mode', 'innermost'), ('anchor_blend', True),
                     ('cross_edge_nudge_mps', 0.0), ('fill_vertical', True)):
        hw = make_cfg()['horizontal_wind']
        hw[key] = val
        with pytest.raises(ValueError, match=key):
            validate(hw, RETRIEVAL_SCHEMA['horizontal_wind'], 'horizontal_wind')


def test_mirror_refuses_a_weakly_sampled_far_half():
    """The innermost mirror needs the far half to hold 2x min_edge_gates:
    an edge never borrows from a partner that cannot spare real gates."""
    vr = np.zeros((1, NGATES))
    vr[0, :10] = 10.0
    vr[0, 10:] = 20.0
    refl = np.full((1, NGATES), QUALIFY)
    refl[0, 2:10] = REJECT      # near half: 2 gates, below min_edge_gates 5
    refl[0, 17:] = REJECT       # far half: 7 gates, below 2 x 5
    radar, sr, alt = build(1, vr, refl)
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    assert np.all(np.isnan(hv))
    assert np.all(src == HV_SRC_NONE)


# ------------------------------------------------- near-empty bins mirror


def _three_bins_middle_near_empty():
    """Bins 0 and 2 fully sampled on both sides; bin 1 has no near-side gates."""
    vr = np.zeros((3, NGATES))
    for r, (near, far) in enumerate(((0.0, 20.0), (99.0, 20.0), (10.0, 20.0))):
        vr[r, :10] = near
        vr[r, 10:] = far
    refl = np.full((3, NGATES), QUALIFY)
    refl[1, :10] = REJECT                # middle bin: near half unusable
    return build(3, vr, refl)

def test_near_empty_middle_bin_mirrors_its_own_far_half():
    """The bin with no near gates fills from its OWN far half, nothing else."""
    radar, sr, alt = _three_bins_middle_near_empty()
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    assert hv[1, 0] == pytest.approx(20.0)
    assert np.all(src[1, :] == HV_SRC_MIRRORED)


def test_top_bin_with_no_near_gates_still_mirrors():
    """A bin above all fully-measured bins still fills from its own far half."""
    vr = np.zeros((2, NGATES))
    vr[0, :10] = 0.0;  vr[0, 10:] = 20.0
    vr[1, :10] = 99.0; vr[1, 10:] = 20.0
    refl = np.full((2, NGATES), QUALIFY)
    refl[1, :10] = REJECT                # top bin has no near-side gates
    radar, sr, alt = build(2, vr, refl)
    hv, src = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)
    assert hv[1, 0] == pytest.approx(20.0)
    assert np.all(src[1, :] == HV_SRC_MIRRORED)

def test_per_sweep_scope_splits_an_object_that_spans_sweeps():
    """Same id in two sweeps: per_file pools the bins, per_sweep does not."""
    # ray 0 and ray 1 sit in DIFFERENT altitude bins but share object id 1.
    # Give them opposite near-side winds; pooling cannot tell them apart.
    vr = np.zeros((2, NGATES))
    vr[0, :10] = 0.0;  vr[0, 10:] = 20.0
    vr[1, :10] = 10.0; vr[1, 10:] = 20.0
    radar, sr, alt = build(2, vr, all_qualify(2))
    radar._slices = [slice(0, 1), slice(1, 2)]        # one ray per sweep

    hv_file, _ = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)
    hv_sweep, _ = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)

    # each ray keeps its own near-side anchor either way here, but per_sweep
    # must build its bins from one sweep only, so the two runs agree on the
    # measured values and the object count differs, not the numbers.
    assert hv_sweep[0, 0] == pytest.approx(0.0)
    assert hv_sweep[1, 0] == pytest.approx(10.0)
    assert np.isfinite(hv_file).all() and np.isfinite(hv_sweep).all()

def test_rung2_spike_leaves_a_consistent_mirror_alone():
    vr = np.zeros((3, NGATES))
    for r, (near, far) in enumerate(((0.0, 1.0), (99.0, 1.5), (2.0, 2.0))):
        vr[r, :10] = near
        vr[r, 10:] = far
    refl = np.full((3, NGATES), QUALIFY)
    refl[1, :10] = REJECT
    radar, sr, alt = build(3, vr, refl)
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    assert hv[1, 0] == pytest.approx(1.5)          # within tolerance, kept
    assert src[1, 0] == HV_SRC_MIRRORED


def test_min_bin_gates_blanks_thin_bins():
    """A bin with too few gates is not retrieved by any rung."""
    vr = np.zeros((2, NGATES))
    vr[0, :10] = 0.0;  vr[0, 10:] = 5.0
    vr[1, :10] = 1.0;  vr[1, 10:] = 6.0
    radar, sr, alt = build(2, vr, all_qualify(2))

    hv0, _ = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    assert np.isfinite(hv0).all()

    # each bin holds NGATES gates; a floor just above that blanks everything
    hv1, src1 = get_horizontal_velocity(
        radar, make_cfg(min_bin_gates=NGATES + 1), sr, alt)
    assert np.all(np.isnan(hv1))
    assert np.all(src1 == HV_SRC_NONE)
def test_wide_window_keeps_a_genuine_shear_layer():
    """A steady vertical gradient is a trend, not a spike, and must survive."""
    near = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0]
    vr = np.zeros((len(near), NGATES))
    for r, nv in enumerate(near):
        vr[r, :10] = nv
        vr[r, 10:] = nv + 2.0
    radar, sr, alt = build(len(near), vr, all_qualify(len(near)))
    hv, src = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)
    for r, nv in enumerate(near):
        assert hv[r, 0] == pytest.approx(nv)     # nothing cut
    assert np.all(src == HV_SRC_MEASURED)

def test_measured_qc_rejects_an_anchor_off_its_own_trend():
    """A measured anchor far from a line through its measured neighbours goes."""
    near = [0.0, 1.0, 2.0, 40.0, 4.0, 5.0, 6.0]
    vr = np.zeros((len(near), NGATES))
    for r, nv in enumerate(near):
        vr[r, :10] = nv
        vr[r, 10:] = nv + 1.0
    radar, sr, alt = build(len(near), vr, all_qualify(len(near)))

    hv, src = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)

    # The spike correction rewrites the anchor, and the provenance says so
    # rather than claiming an observation.
    assert hv[3, 0] != pytest.approx(40.0)
    assert np.all(src[3, :] != HV_SRC_MEASURED)


def test_measured_qc_needs_enough_measured_neighbours():
    """With too few measured neighbours in the window the anchor is left alone."""
    # only bins 0 and 3 are measured on the near edge, so bin 3's window holds
    # at most one measured neighbour -- not enough to define a trend.
    vr = np.zeros((4, NGATES))
    for r, nv in enumerate((0.0, 1.0, 2.0, 40.0)):
        vr[r, :10] = nv
        vr[r, 10:] = nv + 1.0
    refl = all_qualify(4).copy()
    refl[1:3, :10] = REJECT
    radar, sr, alt = build(4, vr, refl)

    hv, _ = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)
    assert hv[3, 0] == pytest.approx(40.0)        # untouched


def test_measured_qc_keeps_a_steady_shear_layer():
    """A steady gradient is a trend, not a spike."""
    near = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0]
    vr = np.zeros((len(near), NGATES))
    for r, nv in enumerate(near):
        vr[r, :10] = nv
        vr[r, 10:] = nv + 1.0
    radar, sr, alt = build(len(near), vr, all_qualify(len(near)))
    hv, src = get_horizontal_velocity(
        radar, make_cfg(), sr, alt)
    for r, nv in enumerate(near):
        assert hv[r, 0] == pytest.approx(nv)
    assert np.all(src == HV_SRC_MEASURED)


def test_measured_values_are_never_overwritten_in_place():
    """A measured anchor is either kept exactly, or rejected -- never edited."""
    near = [0.0, 1.0, 2.0, 40.0, 4.0, 5.0, 6.0]
    vr = np.zeros((len(near), NGATES))
    for r, nv in enumerate(near):
        vr[r, :10] = nv
        vr[r, 10:] = nv + 1.0
    radar, sr, alt = build(len(near), vr, all_qualify(len(near)))
    hv, src = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    for r, nv in enumerate(near):
        if src[r, 0] == HV_SRC_MEASURED:
            assert hv[r, 0] == pytest.approx(nv)


# ---------------------------------------------------------------------------
# The profile spike correction, tested directly: it is a pure function of a
# profile.
# ---------------------------------------------------------------------------

def test_smoother_catches_an_isolated_spike():
    z = np.arange(20) * 0.3
    v = np.full(20, 10.0)
    v[9] = 40.0
    out, fl = smooth_edge_profile(z, v, half=3, **SPIKE)
    assert fl[9]
    assert abs(out[9] - 10.0) < 1e-6
    assert np.allclose(np.delete(out, 9), 10.0)


def test_smoother_catches_a_run_of_two():
    """Why a line fit rather than an immediate-neighbour test: two adjacent
    bad bins agree with each other and hide."""
    z = np.arange(20) * 0.3
    v = np.full(20, 10.0)
    v[9] = v[10] = 30.0
    out, fl = smooth_edge_profile(z, v, half=3, **SPIKE)
    assert fl[9] and fl[10]
    assert np.allclose(out, 10.0, atol=1e-6)


def test_smoother_leaves_a_genuine_shear_layer_alone():
    """A steady trend is not a spike, however large the total change."""
    z = np.arange(20) * 0.3
    v = 10.0 + 11.0 * z          # 11 (m/s)/km, just under the 12 floor
    out, fl = smooth_edge_profile(z, v, half=3, **SPIKE)
    assert not fl.any()
    assert np.allclose(out, v)


def test_smoother_refuses_to_rewrite_most_of_a_profile():
    rng = np.random.default_rng(0)
    z = np.arange(40) * 0.3
    v = rng.normal(0, 30, 40)                 # noise, not a profile
    out, fl = smooth_edge_profile(z, v, half=3, **SPIKE)
    assert fl.sum() <= int(0.25 * 40) + 1


def test_smoother_leaves_the_end_points_as_retrieved():
    """Flagged, but not rewritten: nothing brackets them, and dropping them
    cost about half a kilometre of anvil top per profile."""
    z = np.arange(20) * 0.3
    v = np.full(20, 10.0)
    v[0] = 40.0
    out, fl = smooth_edge_profile(z, v, half=3, **SPIKE)
    assert fl[0]
    assert out[0] == 40.0


def test_smoother_threshold_is_a_shear_not_an_absolute_step():
    """The same step is a spike across a thin bin and a trend across a thick
    one; an absolute tolerance in m/s cannot express that."""
    v = np.full(20, 10.0); v[9] = 16.0
    fine = smooth_edge_profile(np.arange(20) * 0.1, v, half=3, **SPIKE)[1]
    coarse = smooth_edge_profile(np.arange(20) * 1.0, v, half=3, **SPIKE)[1]
    assert fine[9] and not coarse[9]


def test_spike_passes_must_be_positive():
    radar, sr, alt = build(1, np.zeros((1, NGATES)), all_qualify(1))
    cfg = make_cfg(passes=[0])
    with pytest.raises(ValueError):
        get_horizontal_velocity(radar, cfg, sr, alt)


def test_spike_correction_can_be_disabled():
    radar, sr, alt = build(1, np.zeros((1, NGATES)), all_qualify(1))
    cfg = make_cfg(passes=[])
    hv, src = get_horizontal_velocity(radar, cfg, sr, alt)[:2]
    assert not (src == HV_SRC_SMOOTHED).any()


# ---------------------------------------------------------------------------
# The vertical-velocity error field.
# ---------------------------------------------------------------------------

class _FakeRadar:
    def __init__(self, elev):
        self.elevation = {'data': np.asarray(elev, dtype=float)}
        self.nrays = len(elev)
        self.ngates = 3


def _err(elev, **over):
    ecfg = dict(dVr=0.2, dVsed=2.0, dVh2=2.0, dVh1_table='chivo_tracer_2022')
    ecfg.update(over)
    return calculate_vertical_velocity_error(
        _FakeRadar(elev), {'vertical_velocity': {'error': ecfg}})[:, 0]


def test_error_at_zenith_is_the_sedimentation_floor():
    """At 90 deg cot(e) = 0 and 1/sin(e) = 1, so only dVr and dVsed survive."""
    got = _err([90.0])[0]
    assert got == pytest.approx(np.hypot(0.2, 2.0), rel=1e-6)


def test_error_blows_up_toward_the_horizon():
    """The quantitative form of the argument for a high-elevation cut."""
    e = _err([5.0, 10.0, 30.0, 60.0, 90.0])
    assert np.all(np.diff(e) < 0)          # falls monotonically with elevation
    assert e[0] > 25.0                     # 5 deg is unusable on its own terms
    assert e[3] < 6.0


def test_error_does_not_depend_on_the_fill():
    """Two gates at the same elevation report the same sigma whatever their
    horizontal wind provenance -- the field is geometry, not self-assessment."""
    assert _err([45.0, 45.0])[0] == _err([45.0])[0]


def test_error_folds_past_zenith():
    """A sweep that passes 90 deg must not fall off the end of the table."""
    assert _err([95.0])[0] == pytest.approx(_err([85.0])[0], rel=1e-9)


def test_error_table_must_be_consistent():
    with pytest.raises(ValueError):
        _err([45.0], dVh1_table={'elevation_deg': [0, 30, 60], 'dVh1': [1, 2]})
    with pytest.raises(ValueError):
        _err([45.0], dVh1_table={'elevation_deg': [0, 40, 30, 50, 60, 70, 90],
                                 'dVh1': [3, 3, 4, 6, 9, 11, 13]})   # unsorted


def test_every_error_key_is_required():
    """No code defaults: a missing constant fails loudly instead
    of silently reverting to a second copy of the config values."""
    with pytest.raises(KeyError):
        calculate_vertical_velocity_error(_FakeRadar([45.0]), {})   # no section
    full = dict(dVr=0.2, dVsed=2.0, dVh2=2.0, dVh1_table='chivo_tracer_2022')
    for key in full:
        ecfg = {k: v for k, v in full.items() if k != key}
        with pytest.raises(KeyError):
            calculate_vertical_velocity_error(
                _FakeRadar([45.0]), {'vertical_velocity': {'error': ecfg}})


# ---------------------------------------------------------------------------
# The usability flag. A flag, never a filter.
# ---------------------------------------------------------------------------

def _qc(elev, hdist_km, alt_km=0.0, **over):
    """One ray per elevation, one gate per horizontal distance."""
    elev = np.atleast_1d(np.asarray(elev, dtype=float))
    h = np.atleast_1d(np.asarray(hdist_km, dtype=float))
    alt = np.full((elev.size, h.size), alt_km, dtype=float)          # km
    # sr is slant range, so build it from the horizontal distance and altitude
    sr = np.sqrt(np.broadcast_to(h, alt.shape) ** 2 + alt ** 2)
    ucfg = dict(min_elevation_deg=30.0, max_horizontal_distance_km=20.0,
                max_abs_w_mps=None)          # QC layer 3 off unless asked
    ucfg.update(over)
    w = over.pop('_w', None)
    if w is None:
        w = np.zeros(alt.shape)
    return vertical_velocity_usability(_FakeRadar(elev),
                                       {'vertical_velocity': {'usable': ucfg}},
                                       sr, alt * 1000.0, w)


def test_usable_inside_both_limits():
    assert _qc(45.0, 10.0)[0, 0]


def test_either_limit_alone_makes_it_unusable():
    assert not _qc(20.0, 10.0)[0, 0]      # too low
    assert not _qc(45.0, 30.0)[0, 0]      # too far
    assert not _qc(20.0, 30.0)[0, 0]      # both


def test_thresholds_come_from_config():
    assert _qc(20.0, 10.0, min_elevation_deg=10.0)[0, 0]
    assert _qc(45.0, 30.0, max_horizontal_distance_km=40.0)[0, 0]


def test_elevation_boundary_is_exclusive():
    """'above 30 degrees', so exactly 30 is not usable."""
    assert not _qc(30.0, 10.0)[0, 0]
    assert _qc(30.01, 10.0)[0, 0]


def test_undecidable_distance_is_excluded_not_passed():
    assert not _qc(45.0, np.nan)[0, 0]


def test_the_range_key_must_be_written_out():
    """Regression against the hidden 20 km default: for years an absent
    max_horizontal_distance_km silently meant a 20 km cut. Today
    the key is REQUIRED -- disabling the limit takes an explicit null, and an
    absent line is a loud error, never a silent 20."""
    elev = np.array([45.0])
    sr = np.array([[10.0]])
    alt = np.zeros((1, 1))
    with pytest.raises(KeyError, match='required'):
        vertical_velocity_usability(
            _FakeRadar(elev),
            {'vertical_velocity': {'usable': {'min_elevation_deg': 30.0,
                                              'max_abs_w_mps': None}}},
            sr, alt, np.zeros_like(sr))


def test_the_elevation_key_and_the_section_are_required():
    elev = np.array([45.0])
    sr = np.array([[10.0]])
    alt = np.zeros((1, 1))
    with pytest.raises(KeyError):
        vertical_velocity_usability(_FakeRadar(elev), {}, sr, alt,
                                    np.zeros_like(sr))
    with pytest.raises(KeyError):
        vertical_velocity_usability(
            _FakeRadar(elev),
            {'vertical_velocity': {'usable': {'max_horizontal_distance_km': 20.0,
                                              'max_abs_w_mps': None}}},
            sr, alt, np.zeros_like(sr))


def test_range_limit_can_be_switched_off():
    """null leaves elevation as the only criterion. Beam broadening then goes
    unrepresented anywhere in the output, so a gate at 35 km reports the same
    confidence as one at 5 km at the same elevation."""
    assert not _qc(45.0, 30.0)[0, 0]                                  # 20 km limit
    assert _qc(45.0, 30.0, max_horizontal_distance_km=None)[0, 0]     # no limit
    # elevation still applies
    assert not _qc(20.0, 30.0, max_horizontal_distance_km=None)[0, 0]


def test_an_object_never_spans_two_sweeps():
    """Contours are numbered from 1 in every sweep, so the id alone does not
    identify an object: sweep 0 and sweep 1 both have an 'object 1' and they are
    different clouds. Pooling them would build each altitude bin from two
    azimuths at once and assert their cross-beam winds were the same."""
    vr = np.zeros((2, NGATES))
    vr[0, :] = 10.0          # sweep 0 sees 10 m/s
    vr[1, :] = 30.0          # sweep 1 sees 30
    radar, sr, alt = build(2, vr, all_qualify(2))
    radar._slices = [slice(0, 1), slice(1, 2)]   # one ray per sweep
    ids = np.ones((2, NGATES), dtype=int)        # SAME id in both sweeps
    radar.fields['local_object_ids']['data'] = np.ma.array(ids, mask=False)

    hv, _ = get_horizontal_velocity(radar, make_cfg(), sr, alt)
    # each sweep keeps its own wind; neither is dragged toward the other
    assert np.nanmax(np.abs(hv[0] - 10.0)) < 1e-6
    assert np.nanmax(np.abs(hv[1] - 30.0)) < 1e-6


# ---------------------------------------------------------------- rung 3


def _near_gap_case(nrays, gap_rays):
    """Every half measures, except the NEAR half of the gap rays.

    Near anchors are 10, 20, 30, ... by bin; far anchors are 50 everywhere,
    so the interpolated fill of an interior gap equals the value the bin
    would have measured -- the exactness of the linear fill is the assert.
    """
    vr = np.zeros((nrays, NGATES))
    vr[:, :10] = (10.0 + 10.0 * np.arange(nrays))[:, None]
    vr[:, 10:] = 50.0
    refl = all_qualify(nrays)
    for r in gap_rays:
        refl[r, :10] = REJECT
    return build(nrays, vr, refl)


@pytest.mark.parametrize('nrays, gap, max_gap', [(3, [1], 1), (4, [1, 2], 2)])
def test_interior_near_gap_interpolated(nrays, gap, max_gap):
    """A bounded interior gap no wider than the cap is filled linearly and
    every gate of the filled bin reports INTERPOLATED (the weakest rung of
    its two edges). Mirror off, so the gap actually reaches rung 3."""
    radar, sr, alt = _near_gap_case(nrays, gap)
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=False, passes=[],
                        fill_interpolate_max_gap_bins=max_gap), sr, alt)
    for r in gap:
        # gate 0 sits at the bin's minimum slant range, so it reads the near
        # anchor exactly; linear in altitude reproduces the 10*(r+1) ramp
        assert hv[r, 0] == pytest.approx(10.0 + 10.0 * r)
        assert np.all(src[r, :] == HV_SRC_INTERPOLATED)
    for r in set(range(nrays)) - set(gap):
        assert np.all(src[r, :] == HV_SRC_MEASURED)


def test_gap_wider_than_max_not_filled():
    radar, sr, alt = _near_gap_case(4, [1, 2])
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=False, passes=[],
                        fill_interpolate_max_gap_bins=1), sr, alt)
    for r in (1, 2):
        assert np.all(np.isnan(hv[r, :]))
        assert np.all(src[r, :] == HV_SRC_NONE)


def test_end_gap_never_extrapolated():
    """A gap at the profile's top has no anchor above it: filling it would be
    extrapolation, so it stays empty however generous the cap."""
    radar, sr, alt = _near_gap_case(3, [2])
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=False, passes=[],
                        fill_interpolate_max_gap_bins=2), sr, alt)
    assert np.all(np.isnan(hv[2, :]))
    assert np.all(src[2, :] == HV_SRC_NONE)


def test_zero_disables_gap_fill():
    radar, sr, alt = _near_gap_case(3, [1])
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=False, passes=[],
                        fill_interpolate_max_gap_bins=0), sr, alt)
    assert np.all(np.isnan(hv[1, :]))
    assert np.all(src[1, :] == HV_SRC_NONE)


def test_mirror_fills_before_interpolation():
    """Rung order: a near gap the innermost mirror can fill from real gates
    never falls through to rung 3."""
    radar, sr, alt = _near_gap_case(3, [1])
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=True, passes=[],
                        fill_interpolate_max_gap_bins=2), sr, alt)
    assert np.all(src[1, :] == HV_SRC_MIRRORED)


def test_far_gap_interpolated_not_mirrored():
    """The far edge is never mirror-filled, so with mirror ON a far-edge gap
    still reaches rung 3. Gate -1 sits at the bin's maximum slant range and
    reads the far anchor exactly."""
    vr = np.zeros((3, NGATES))
    vr[:, :10] = 10.0
    vr[:, 10:] = (50.0 + 10.0 * np.arange(3))[:, None]
    refl = all_qualify(3)
    refl[1, 10:] = REJECT
    radar, sr, alt = build(3, vr, refl)
    hv, src = get_horizontal_velocity(
        radar, make_cfg(fill_mirror=True, passes=[],
                        fill_interpolate_max_gap_bins=1), sr, alt)
    assert hv[1, -1] == pytest.approx(60.0)
    assert np.all(src[1, :] == HV_SRC_INTERPOLATED)


# ------------------------------------------------- provenance code numbering
def test_provenance_codes_are_pinned():
    """The five codes are a CONTRACT with data already written. Every dev-tree
    and shakedown output carries these integers in horizontal_velocity_source,
    so renumbering them would silently reinterpret files on disk rather than
    break anything loudly. Pin the values instead of leaving them to the order
    the constants happen to be defined in.

    3 (vertical fit) and 4 (blend) are reserved and must stay unused: an old
    file's 3 must never come back meaning something new."""
    assert (HV_SRC_NONE, HV_SRC_MEASURED, HV_SRC_MIRRORED, HV_SRC_SMOOTHED,
            HV_SRC_INTERPOLATED) == (0, 1, 2, 5, 6)
    live = {HV_SRC_NONE, HV_SRC_MEASURED, HV_SRC_MIRRORED, HV_SRC_SMOOTHED,
            HV_SRC_INTERPOLATED}
    assert len(live) == 5, 'the five codes must stay distinct'
    assert live.isdisjoint({3, 4}), 'codes 3 and 4 are retired, never reused'
