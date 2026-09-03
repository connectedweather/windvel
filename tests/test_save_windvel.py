"""Regression tests for atomic saving.

Pinned bug: extract mode calling write_cfradial directly
on the target, so an interrupted write corrupted the file. Both modes now
stage to a sibling .tmp and os.replace -- an interrupted write must leave
the target either absent or its previous complete self, and no .tmp behind.
"""
import numpy as np
import pyart
import pytest

import windvel.save_windvel as sw
from _configs import provenance
from windvel.calculate_windvel import OUTPUT_FIELDS

_CFG = {'input_variables': {}, 'provenance': provenance()}


@pytest.fixture
def radar():
    pyart_testing = pytest.importorskip('pyart.testing')
    if not hasattr(pyart_testing, 'make_empty_rhi_radar'):
        pytest.skip('pyart.testing.make_empty_rhi_radar unavailable')
    return pyart_testing.make_empty_rhi_radar(10, 8, 1)


@pytest.mark.parametrize('mode', ['full', 'extract'])
def test_interrupted_write_preserves_target_and_leaves_no_tmp(
        tmp_path, monkeypatch, radar, mode):
    target = tmp_path / 'out.nc'
    target.write_bytes(b'PREVIOUS COMPLETE FILE')

    def boom(*a, **k):
        raise OSError('disk full')

    monkeypatch.setattr(pyart.io, 'write_cfradial', boom)
    with pytest.raises(OSError):
        sw.save_windvel_file(target, radar, _CFG, mode=mode)

    assert target.read_bytes() == b'PREVIOUS COMPLETE FILE'
    assert list(tmp_path.glob('*.tmp')) == []


@pytest.mark.parametrize('mode', ['full', 'extract'])
def test_successful_write_produces_target_and_no_tmp(tmp_path, radar, mode):
    target = tmp_path / 'out.nc'
    sw.save_windvel_file(target, radar, _CFG, mode=mode)
    assert target.exists() and target.stat().st_size > 0
    assert list(tmp_path.glob('*.tmp')) == []


def test_unknown_mode_raises(tmp_path, radar):
    with pytest.raises(ValueError):
        sw.save_windvel_file(tmp_path / 'out.nc', radar, _CFG, mode='banana')


def test_append_mode_is_refused_by_name(tmp_path, radar):
    """'append' never appended -- it wrote every field. Its name is 'full';
    the old name must die loudly, not silently keep working."""
    with pytest.raises(ValueError, match="renamed 'full'"):
        sw.save_windvel_file(tmp_path / 'out.nc', radar, _CFG, mode='append')


def test_extract_saves_every_computed_field(tmp_path, radar):
    """Regression for deferred item 1: the keep-list had drifted to 6 of the
    14 output fields, silently dropping w error/flag and all six coherent
    object fields from every extract-mode file."""
    shape = (radar.nrays, radar.ngates)
    for fld in OUTPUT_FIELDS:
        radar.fields[fld] = {'data': np.zeros(shape, dtype='float32'),
                             '_FillValue': -9999.0}
    target = tmp_path / 'out.nc'
    sw.save_windvel_file(target, radar, _CFG, mode='extract')
    back = pyart.io.read_cfradial(str(target))
    assert [f for f in OUTPUT_FIELDS if f not in back.fields] == []


def test_full_mode_writes_a_bool_field(tmp_path, radar):
    """Pinned by the shakedown: object_boundaries is a bool array,
    bool is not a netCDF primitive, and the int16 conversion ran only in
    extract mode -- so full mode died on the FIRST real file it ever saw."""
    shape = (radar.nrays, radar.ngates)
    radar.fields['object_boundaries'] = {'data': np.zeros(shape, dtype=bool)}
    target = tmp_path / 'out.nc'
    sw.save_windvel_file(target, radar, _CFG, mode='full')
    back = pyart.io.read_cfradial(str(target))
    assert back.fields['object_boundaries']['data'].dtype.kind == 'i'


def test_every_output_carries_the_code_version(tmp_path, radar):
    """The file records which windvel made it, so lineage against any earlier
    output tree is a metadata read, not archaeology."""
    import windvel
    target = tmp_path / 'out.nc'
    sw.save_windvel_file(target, radar, _CFG, mode='full')
    back = pyart.io.read_cfradial(str(target))
    assert back.metadata.get('windvel_version') == windvel.__version__


def test_the_keep_list_is_the_canonical_one():
    """ONE list of output fields, defined beside the code that creates them.
    The saver must consume that object, not carry a copy that can drift."""
    assert sw.OUTPUT_FIELDS is OUTPUT_FIELDS
    assert not hasattr(sw, '_ALWAYS_INCLUDE_FIELDS')
