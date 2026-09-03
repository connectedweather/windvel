"""Quick-look figures: the field panel and the eight-panel summary.

Reads the fields the retrieval wrote and draws them; recomputes nothing.
Non-interactive (Agg), file output only.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from pyart.graph import RadarDisplay

from .config import require, section
from .utils import (
    altitude_bin_edges,
    as_float_nan,
    gate_dist_alt_km,
    make_sr_alt_maps,
    project_to_radial_component,
)

__all__ = ["plot_windvel_panel", "plot_windvel_summary"]

# Display limits for the quick-look panels (m/s). Drawing only; nothing here
# touches the saved fields.
VELOCITY_DISPLAY_LIMIT_MPS = 40.0
PANEL_MAX_ALT_KM = 15.0
SUMMARY_WIND_LIMIT_MPS = 20.0
SUMMARY_W_EXTENDED_LIMIT_MPS = 40.0
SUMMARY_DOPPLER_LIMIT_MPS = 30.0
SUMMARY_SIGMA_LIMIT = 4.0
SUMMARY_PROFILE_LIMIT_MPS = 30.0
SUMMARY_MIN_GATES_PER_BIN = 5

_UP, _DN, _BAD = '#b2182b', '#2166ac', '#999999'

# Labelling the object panels. An object smaller than this holds no legible
# number, and two labels closer than the spacing overprint each other; in both
# cases the object keeps its colour and outline and goes unlabelled, which is
# more use than a pile of unreadable digits.
_LABEL_MIN_GATES = 40
_LABEL_MIN_SEP_KM = 1.2


def plot_windvel_panel(outfile, radar, cfg, field_list, sweeps,
                       fig_dir=None, max_range_km=None):
    """One RHI panel per field, one figure per sweep.

    Parameters
    ----------
    outfile : str or Path
        The output netCDF; figures take its stem.
    radar : pyart.core.Radar
    cfg : dict
        The whole config; ``input_variables.vr_var`` is read.
    field_list : list of str
        Fields to draw, in order.
    sweeps : list of int
        The per-file RHI sweep list (`extract_rhi_sweep_indices`); an empty
        list draws nothing, never a silent sweep 0.
    fig_dir : str or Path, optional
        Write the PNGs here instead of ``<outfile>.parent/'figures'``.
    max_range_km : float, optional
        Cap the distance axis, e.g. to match the domain a set of statistics
        was computed over. None keeps the full sweep.

    Returns
    -------
    list of Path
        The figures written.
    """
    outfile = Path(outfile)
    fig_dir = Path(fig_dir) if fig_dir is not None else outfile.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    rhi_indices = list(sweeps)
    vr_var = require(section(cfg, 'input_variables'), 'input_variables', 'vr_var')

    velocity_bases = {
        vr_var, "horizontal_velocity", "vertical_velocity",
        "sedimentation_velocity"
    }

    def _is_velocity_field(fld):
        """True for a velocity field, so it gets clipped for display.

        Matches the plain names and any suffixed variant, e.g.
        horizontal_velocity_9dbz from a multi-threshold runner.
        """
        return any(fld == b or fld.startswith(b + "_") for b in velocity_bases)

    for sweep in rhi_indices:
        display = RadarDisplay(radar)
        n_fields = len(field_list)
        cols = 2
        rows = int(np.ceil(n_fields / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

        for ax, fld in zip(axes, field_list, strict=False):   # axes is padded to a full grid
            if _is_velocity_field(fld):
                data = radar.fields[fld]['data'].copy()
                data = np.ma.masked_outside(data, -VELOCITY_DISPLAY_LIMIT_MPS,
                                            VELOCITY_DISPLAY_LIMIT_MPS)
                # a temporary field, popped again below
                radar.add_field_like(fld, f"{fld}_clipped", data, replace_existing=True)
                display.plot_rhi(
                    f"{fld}_clipped",
                    sweep,
                    ax=ax,
                    title=fld,
                    cmap=None
                )
                radar.fields.pop(f"{fld}_clipped", None)
            else:
                display.plot_rhi(
                    fld,
                    sweep,
                    ax=ax,
                    title=fld,
                    cmap=None
                )

            ax.set_ylim(0, PANEL_MAX_ALT_KM)
            if max_range_km is not None:
                ax.set_xlim(0, max_range_km)

        for ax in axes[n_fields:]:
            ax.set_visible(False)

        fig.suptitle(f"RHI Sweep {sweep}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = fig_dir / f"{outfile.stem}_fields_sweep_{sweep}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    return [fig_dir / f"{outfile.stem}_fields_sweep_{s}.png" for s in rhi_indices]


# ===========================================================================
# Summary panel: what the retrieval produced, and what it found in it
# ===========================================================================
def _shade(ax, dk, ak, mask, colour, alpha=0.30, hatch=None):
    if not mask.any():
        return
    ax.contourf(dk, ak, mask.astype(float), levels=[0.5, 1.5],
                colors=[colour], alpha=alpha)
    if hatch:
        ax.contourf(dk, ak, mask.astype(float), levels=[0.5, 1.5],
                    colors='none', hatches=[hatch], alpha=0.0)


def _outline(ax, dk, ak, mask, colour, ls='-', lw=1.3):
    if mask.any():
        ax.contour(dk, ak, mask.astype(float), levels=[0.5], colors=[colour],
                   linewidths=lw, linestyles=[ls])


def _object_panel(ax, dk, ak, lab, flag, trunc, name):
    """One colour and one id per object, with no velocity underneath.

    The field is in the panels above; this one answers 'which object is that?'.
    Sign is in the id, a solid outline and bold id mean the object passed the
    usability test, and a hatch means it ran off the edge of the scan or up
    against a hole in the retrieval, so its area and depth are lower bounds.
    """
    ids = sorted({int(v) for v in np.unique(lab) if v != 0},
                 key=lambda i: -int((lab == i).sum()))   # big ones label first
    palette = plt.get_cmap('tab20')
    placed, n_ok, n_cut = [], 0, 0
    for n, i in enumerate(ids):
        k = lab == i
        ok = bool(flag[k].any())
        tr = bool((trunc[k] != 0).any())
        n_ok += ok
        n_cut += tr
        _shade(ax, dk, ak, k, palette(n % 20), alpha=0.75,
               hatch='///' if tr else None)
        _outline(ax, dk, ak, k, '#111111' if ok else '#888888',
                 ls='-' if ok else '--', lw=1.4 if ok else 0.9)
        if int(k.sum()) < _LABEL_MIN_GATES:
            continue
        yy, xx = np.where(k)
        cx, cy = dk[yy, xx].mean(), ak[yy, xx].mean()
        if any(abs(px - cx) < _LABEL_MIN_SEP_KM and abs(py - cy) < _LABEL_MIN_SEP_KM
               for px, py in placed):
            continue
        placed.append((cx, cy))
        ax.annotate(str(i), (cx, cy), ha='center', va='center', fontsize=7,
                    fontweight='bold' if ok else 'normal', color='#111111',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none',
                              alpha=0.8))
    ax.set_title(f'{name} objects: {len(ids)} found, {n_ok} usable, '
                 f'{n_cut} cut off\n(bold id = usable, hatched = incomplete)',
                 fontsize=9)


def _edge_profiles(hv, sr, alt, bin_m, ids):
    """The near and far anchors of every altitude bin, PER OBJECT.

    Read back off the assembled field rather than recomputed, so the panel
    shows what the retrieval actually used: within one object, the nearest
    and furthest gate of a bin are the two ends of the line it interpolated
    along.

    Per object, and binned from the OBJECT's own base by the same
    `altitude_bin_edges` the retrieval uses -- every anchor, bin and
    interpolation in `get_horizontal_velocity` belongs to a single object in
    a single sweep. Aggregating a bin across the sweep would pair the near
    edge of one cloud with the far edge of another tens of km away and draw
    the result as if it were one profile; where a cloud ended, the trace
    would step to a different cloud and look like wind shear.

    Returns {object_id: (z_km, near, far)}, largest object first.
    """
    ok = np.isfinite(hv) & np.isfinite(alt) & np.isfinite(sr) & (ids > 0)
    out = {}
    for oid in sorted({int(v) for v in np.unique(ids[ok])},
                      key=lambda o: -int((ids == o).sum())):
        om = ok & (ids == oid)
        if not om.any():
            continue
        zmin, zmax = float(np.nanmin(alt[om])), float(np.nanmax(alt[om]))
        z, near, far = [], [], []
        for base in altitude_bin_edges(zmin, zmax, bin_m):
            m = om & (alt >= base) & (alt < base + bin_m)
            if not m.any():
                continue
            d, v = sr[m], hv[m]
            z.append((base + bin_m / 2) / 1000.0)
            near.append(float(v[np.argmin(d)]))
            far.append(float(v[np.argmax(d)]))
        if z:
            out[oid] = (np.array(z), np.array(near), np.array(far))
    return out


def summary_w_limit(w_sweep):
    """Colorbar half-range for the vertical-velocity panel of one sweep.

    The standard wind limit, widened to `SUMMARY_W_EXTENDED_LIMIT_MPS`
    when any finite |w| in the sweep exceeds the standard limit -- a
    strong draft is then read on scale instead of saturating the panel.
    """
    finite = np.isfinite(w_sweep)
    if finite.any() and (float(np.max(np.abs(w_sweep[finite])))
                         > SUMMARY_WIND_LIMIT_MPS):
        return SUMMARY_W_EXTENDED_LIMIT_MPS
    return SUMMARY_WIND_LIMIT_MPS


def plot_windvel_summary(outfile, radar, cfg, sweeps, sweep_idx=None,
                         fig_dir=None, profile=None, max_range_km=40.0,
                         max_alt_km=20.0):
    """Eight panels per sweep: the retrieval, and the structures found in it.

    Parameters
    ----------
    outfile : str or Path
        The output netCDF; figures take its stem.
    radar : pyart.core.Radar
    cfg : dict
        The whole config; ``input_variables.vr_var`` and
        ``horizontal_wind.bin_size_m`` are read.
    sweeps : list of int
        The per-file RHI sweep list.
    sweep_idx : int, optional
        Draw this one sweep instead.
    fig_dir : str or Path, optional
        Write the PNGs here instead of ``<outfile>.parent/'figures'``.
    profile : tuple, optional
        (height_km, wind_ms, label) -- a sounding or a model column
        projected on the sweep azimuth -- drawn over the edge anchors.
        Omitted when absent, so nothing here depends on campaign paths.
    max_range_km, max_alt_km : float, optional
        Axis limits for the field panels.

    Returns
    -------
    list of Path
        The figures written; empty when the file has no vertical velocity.

    Notes
    -----
        top     Doppler - sedimentation | horizontal wind | vertical velocity
                | w / sigma
        bottom  per-bin spread | edge anchors | threshold objects
                | persistence objects

    The first top panel is the honest check on the last: Doppler with the
    fall speed removed is negative where the air is most likely descending
    and positive where it is rising, WITHOUT any horizontal-wind estimate
    entering, so a retrieved w that disagrees with it is telling on itself.

    Both object definitions are drawn side by side rather than in separate
    figures: five of the panels would be identical between them, and
    comparing the two is the point.
    """
    outfile = Path(outfile)
    fig_dir = Path(fig_dir) if fig_dir is not None else outfile.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    sweeps = [sweep_idx] if sweep_idx is not None else list(sweeps)
    bin_m = float(require(section(cfg, 'horizontal_wind'), 'horizontal_wind',
                          'bin_size_m'))
    vr_var = require(section(cfg, 'input_variables'), 'input_variables', 'vr_var')
    sr_map, alt_map = make_sr_alt_maps(radar)

    def _f(name, fill=np.nan):
        if name not in radar.fields:
            return None
        d = radar.fields[name]['data']
        return np.ma.filled(d, fill).astype(float)

    vr = _f(vr_var)
    sed_rad = as_float_nan(project_to_radial_component(
        radar, field_name='sedimentation_velocity', trig='sin'))
    dop_sed = vr + sed_rad if vr is not None else None
    hv, w = _f('horizontal_velocity'), _f('vertical_velocity')
    err = _f('vertical_velocity_error')
    if w is None:
        return []

    out = []
    for s in sweeps:
        sl = radar.get_slice(s)
        if not np.isfinite(w[sl]).any():
            continue
        dk, ak = gate_dist_alt_km(radar, s)
        az = float(np.median(radar.azimuth['data'][sl]))

        fig, ax = plt.subplots(2, 4, figsize=(25, 10.5), constrained_layout=True)
        wn = TwoSlopeNorm(vcenter=0, vmin=-SUMMARY_WIND_LIMIT_MPS,
                          vmax=SUMMARY_WIND_LIMIT_MPS)
        w_lim = summary_w_limit(w[sl])
        w_norm = TwoSlopeNorm(vcenter=0, vmin=-w_lim, vmax=w_lim)
        sig = (w[sl] / err[sl]) if err is not None else np.full_like(w[sl], np.nan)

        for a_, (F, norm, cmap, ttl, cb) in zip(
                ax[0],
                ((dop_sed[sl] if dop_sed is not None else np.full_like(w[sl], np.nan),
                  TwoSlopeNorm(vcenter=0, vmin=-SUMMARY_DOPPLER_LIMIT_MPS,
                               vmax=SUMMARY_DOPPLER_LIMIT_MPS), 'RdBu_r',
                  'Doppler - sedimentation\n(no wind estimate enters this)', 'm/s'),
                 (hv[sl] if hv is not None else np.full_like(w[sl], np.nan),
                  wn, 'RdBu_r', 'Horizontal wind', 'm/s'),
                 (w[sl], w_norm, 'RdBu_r', 'Vertical velocity', 'm/s'),
                 (sig, TwoSlopeNorm(vcenter=0, vmin=-SUMMARY_SIGMA_LIMIT,
                                    vmax=SUMMARY_SIGMA_LIMIT), 'PuOr_r',
                  'w / sigma\n(what the persistence detector works on)',
                  'w / sigma')), strict=True):
            im = a_.pcolormesh(dk, ak, np.ma.masked_invalid(F), cmap=cmap,
                               norm=norm, shading='auto')
            fig.colorbar(im, ax=a_, label=cb)
            a_.set_title(ttl, fontsize=9)

        # ---- per-bin spread, altitude on the vertical axis ----------------
        # Display binning only: floor-aligned across the whole sweep, unlike
        # the retrieval's per-object bins, because this panel pools objects.
        _cid = _f('local_object_ids', 0)
        prof = _edge_profiles(hv[sl] if hv is not None else w[sl],
                              sr_map[sl], alt_map[sl], bin_m,
                              (_cid[sl] if _cid is not None
                               else np.ones_like(w[sl])).astype(int))
        a_ = ax[1, 0]
        wb, hb, zb = [], [], []
        for base in np.arange(np.floor(np.nanmin(alt_map[sl]) / bin_m) * bin_m,
                              np.nanmax(alt_map[sl]) + bin_m, bin_m):
            m = (alt_map[sl] >= base) & (alt_map[sl] < base + bin_m)
            ww = w[sl][m & np.isfinite(w[sl])]
            if ww.size < SUMMARY_MIN_GATES_PER_BIN:
                continue
            zb.append((base + bin_m / 2) / 1000.0)
            wb.append(ww)
            hh = hv[sl][m & np.isfinite(hv[sl])] if hv is not None else np.array([])
            hb.append(hh if hh.size else np.array([np.nan]))
        lim = SUMMARY_PROFILE_LIMIT_MPS
        if zb:
            wid = bin_m / 1000.0 * 0.8
            a_.boxplot(wb, positions=zb, vert=False, widths=wid, showfliers=False,
                       manage_ticks=False,
                       boxprops=dict(color=_DN),
                       medianprops=dict(color=_DN),
                       whiskerprops=dict(color=_DN),
                       capprops=dict(color=_DN))
            a2 = a_.twiny()
            a2.boxplot(hb, positions=zb, vert=False, widths=wid, showfliers=False,
                       manage_ticks=False,
                       boxprops=dict(color=_UP),
                       medianprops=dict(color=_UP),
                       whiskerprops=dict(color=_UP),
                       capprops=dict(color=_UP))
            a2.set_xlim(-lim, lim); a2.set_xlabel('horizontal wind (m/s)',
                                                  color=_UP)
        a_.set_xlim(-lim, lim); a_.set_ylim(0, max_alt_km)
        a_.set_xlabel('vertical velocity (m/s)', color=_DN)
        a_.set_ylabel('altitude (km)')
        a_.set_title('Spread within each %g m bin (all objects pooled)' % bin_m,
                     fontsize=9)
        a_.grid(alpha=.3)

        # ---- edge anchors, ONE PROFILE PER OBJECT ---------------------------
        # never pooled across objects: each cloud is retrieved on its own bins
        a_ = ax[1, 1]
        for n, (oid, (zc, near, far)) in enumerate(prof.items()):
            first, alpha = (n == 0), (1.0 if n == 0 else 0.45)
            lw = 1.2 if first else 0.8
            a_.plot(near, zc, 'o-', ms=2.0, lw=lw, color=_DN,
                    alpha=alpha, label='near edge' if first else None)
            a_.plot(far, zc, 'o-', ms=2.0, lw=lw, color=_UP,
                    alpha=alpha, label='far edge' if first else None)
            if first and zc.size:
                a_.annotate(f'obj {oid}', (near[np.argmax(zc)], zc.max()),
                            fontsize=7, color=_DN,
                            xytext=(3, 3), textcoords='offset points')
        if profile is not None:
            pz, pv, plabel = profile
            a_.plot(pv, pz, '-', lw=1.2, color='#444444', label=plabel)
        a_.axvline(0, color='#111111', lw=0.8)
        a_.set_xlim(-lim, lim); a_.set_ylim(0, max_alt_km)
        a_.set_xlabel('horizontal wind (m/s)'); a_.set_ylabel('altitude (km)')
        a_.set_title(f'Edge anchors per object ({len(prof)} shown, '
                     'largest solid)', fontsize=9)
        a_.legend(fontsize=8); a_.grid(alpha=.3)

        # ---- the two object definitions, side by side ----------------------
        for a_, name in ((ax[1, 2], 'threshold'), (ax[1, 3], 'persistence')):
            lab = _f(f'coherent_objects_{name}', 0)
            flg = _f(f'coherent_objects_{name}_flag', 0)
            trc = _f(f'coherent_objects_{name}_truncated', 0)
            if lab is None:
                a_.set_visible(False)
                continue
            _object_panel(a_, dk, ak, lab[sl].astype(int),
                          flg[sl].astype(bool), trc[sl].astype(int), name)

        for a_ in ax.ravel():
            if not a_.get_visible():
                continue
            if a_ not in (ax[1, 0], ax[1, 1]):
                a_.set_xlabel('distance from radar (km)')
                a_.set_ylabel('altitude (km)')
                a_.set_xlim(0, max_range_km)
                a_.set_ylim(0, max_alt_km)
            a_.grid(alpha=.2)

        fig.suptitle(f'{outfile.stem}   sweep {s} (az {az:.1f} deg)', fontsize=13)
        path = fig_dir / f'{outfile.stem}_summary_sweep_{s}.png'
        fig.savefig(str(path), dpi=110)
        plt.close(fig)
        out.append(path)
    return out
