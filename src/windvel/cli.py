#!/usr/bin/env python3
"""The windvel command: read, retrieve, save, plot.

Usage:    windvel --config site_<radar>.yaml

The site file names the retrieval file (see `windvel.config`). This is the
only module that reads config files and prints; everything below it takes
values and logs.
"""

import argparse
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pyart
import yaml

from .calculate_windvel import OUTPUT_FIELDS, calculate_windvel
from .config import load_config, require, section
from .corrections import apply_corrections
from .plot_windvel import plot_windvel_panel, plot_windvel_summary
from .save_windvel import save_windvel_file, validate_save_mode
from .select_cloud_transects import select_cloud_transects
from .utils import extract_rhi_sweep_indices, ids_as_int

__all__ = [
    "has_retained_objects",
    "main",
    "parse_args",
    "prepare_file_list",
    "resolve_output_policy",
    "utc_log_formatter",
]


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with attribute `config` (path to the site YAML).
    """
    parser = argparse.ArgumentParser(
        description="Wind retrieval from RHI radar scans."
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to the site YAML (site_<radar>.yaml), which names the "
             "retrieval YAML."
    )
    return parser.parse_args()


def prepare_file_list(cfg: Dict[str, str | Path]) -> List[Path]:
    """Build the list of input files from the config.

    Parameters
    ----------
    cfg : dict
        The resolved config. REQUIRED under ``paths``:
        ``input_data_path`` (a file or a directory searched recursively)
        and ``file_pattern`` ('prefix*suffix', e.g.
        'houchivorhicelltracking*.nc'; prefix/suffix matching, not a glob).

    Returns
    -------
    list of Path
        All matching file paths.
    """
    paths = section(cfg, 'paths')
    inp = Path(require(paths, 'paths', 'input_data_path',
                       'a file or a directory to search'))
    if not inp.exists():
        raise FileNotFoundError(f"Input path '{inp}' does not exist.")

    pattern = str(require(paths, 'paths', 'file_pattern',
                          "'prefix*suffix' matched against file names"))
    prefix, _, suffix = pattern.partition('*')

    matches: List[Path] = []

    if inp.is_file():
        name = inp.name
        if name.startswith(prefix) and name.endswith(suffix):
            matches.append(inp)
        else:
            raise ValueError(
                f"File '{inp}' does not match pattern '{pattern}'.")

    elif inp.is_dir():
        for p in inp.rglob('*'):
            if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix):
                matches.append(p)

    else:
        raise FileNotFoundError(f"Path '{inp}' is neither file nor directory.")

    return matches


def resolve_output_policy(cfg):
    """The two output decisions, validated once at run start.

    Parameters
    ----------
    cfg : dict
        The resolved config. REQUIRED under ``run``:

        save_mode           what the output CONTAINS: 'full' writes every
                            field on the radar object, 'extract' only the
                            configured input variables plus the canonical
                            OUTPUT_FIELDS list.
        overwrite_existing  whether an existing output is recomputed from
                            the RAW scan and replaced (true), or the file
                            skipped (false).

    Returns
    -------
    save_mode : str
    overwrite_existing : bool
    """
    run = section(cfg, 'run')
    mode = validate_save_mode(require(run, 'run', 'save_mode',
                                      "'full' or 'extract'"))
    overwrite = require(run, 'run', 'overwrite_existing',
                        'true recomputes an existing output from the raw '
                        'scan and replaces it, false skips the file')
    return mode, bool(overwrite)


def has_retained_objects(radar):
    """True when any cloud object survived selection, over ALL sweeps.

    The write gate: a file with no surviving objects produced no retrieval
    and is not written. Masked gates count as no object.
    """
    fld = radar.fields.get('local_object_ids')
    if fld is None:
        return False
    return bool((ids_as_int(fld['data']) > 0).any())


def utc_log_formatter() -> logging.Formatter:
    """The run-log line format, with timestamps in UTC and marked as such."""
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = time.gmtime
    return fmt


def main():
    """Load the config, then read, retrieve, save and plot every scan."""
    args = parse_args()
    cfg = load_config(args.config)

    paths = section(cfg, 'paths')
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(require(paths, 'paths', 'log_dir')) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the RESOLVED config -- retrieval + overrides + site, with the
    # provenance section -- so the run is reproducible from this one file.
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger = logging.getLogger("windvel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(utc_log_formatter())
    logger.addHandler(fh)
    prov = section(cfg, 'provenance')
    logger.info("run_id=%s site=%s retrieval=%s (version %s)", run_id,
                prov['site_file'], prov['retrieval_file'],
                prov['retrieval_version'])
    for key, value in prov['retrieval_overrides'].items():
        logger.info("retrieval override %s = %r", key, value)

    # Every run-level decision, validated before any file is touched
    save_mode, overwrite_existing = resolve_output_policy(cfg)
    plot_summary = bool(require(section(cfg, 'run'), 'run',
                                'plot_summary', 'draw the summary panels'))
    output_dir = Path(require(paths, 'paths', 'output_dir'))
    ivars = section(cfg, 'input_variables')
    refl_var = require(ivars, 'input_variables', 'refl_var')
    vr_var = require(ivars, 'input_variables', 'vr_var')

    file_list = prepare_file_list(cfg)
    logger.info("file_count=%d", len(file_list))

    n_processed = n_skipped = n_errored = 0
    for file_path in file_list:
        try:
            outfile = output_dir / Path(file_path).parent.name / Path(file_path).name
            if outfile.exists() and not overwrite_existing:
                n_skipped += 1
                logger.info("skipped (output exists) %s", file_path)
                continue
            # ALWAYS the raw scan, never a previous output read back in.
            radar = pyart.io.read_cfradial(file_path)

            sweeps = extract_rhi_sweep_indices(radar)
            radar = apply_corrections(radar, cfg)
            radar = select_cloud_transects(radar, cfg, sweeps)
            radar = calculate_windvel(radar, cfg, sweeps)

            if not sweeps:
                n_skipped += 1
                logger.info("skipped (no RHI sweeps) %s", file_path)
                continue
            if not has_retained_objects(radar):
                n_skipped += 1
                logger.info("skipped (no cloud objects) %s", file_path)
                continue

            save_windvel_file(outfile, radar, cfg, mode=save_mode)
            # The quick-look: inputs plus every computed field present,
            # from the same canonical list the saver uses
            field_list = ([refl_var, vr_var]
                          + [f for f in OUTPUT_FIELDS if f in radar.fields])
            plot_windvel_panel(outfile, radar, cfg, field_list, sweeps)
            # The summary panel: the retrieval and the structures found in
            # it, on one sheet per sweep. A figure failing must not cost the
            # file that is already written.
            if plot_summary:
                try:
                    plot_windvel_summary(outfile, radar, cfg, sweeps)
                except Exception:
                    logger.exception('summary panel failed for %s', file_path)
            n_processed += 1
            logger.info("processed %s", file_path)

        except Exception as e:
            n_errored += 1
            logger.error("error %s: %s", file_path, e)
            logger.debug("traceback:\n%s", traceback.format_exc())
            print(f"Error processing {file_path}: {e}")

    logger.info(
        "summary processed=%d skipped=%d errored=%d",
        n_processed, n_skipped, n_errored,
    )


if __name__ == "__main__":
    main()
