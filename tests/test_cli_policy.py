"""Tests for the cli's output policy and write gate.

Pinned bugs: defaults.overwrite_existing silently choosing the output
FIELD SET ('append' when true, 'extract' when false); the write gate looked
only at the FIRST RHI sweep, so a file whose first sweep is clear air loses
the storms in its later sweeps; and a re-run read the previous OUTPUT back in
as its input. The policy is now two required keys resolved once at run start,
and the gate looks at every sweep.
"""
import numpy as np
import pytest

from windvel.cli import has_retained_objects, resolve_output_policy


def _cfg(**defaults):
    d = {'save_mode': 'full', 'overwrite_existing': True}
    d.update(defaults)
    return {'run': d}


# ---------------------------------------------------------------------------
# resolve_output_policy: two decisions, both explicit, validated up front.
# ---------------------------------------------------------------------------

def test_policy_round_trip():
    assert resolve_output_policy(_cfg()) == ('full', True)
    assert resolve_output_policy(
        _cfg(save_mode='extract', overwrite_existing=False)) == ('extract',
                                                                 False)


def test_every_policy_key_is_required():
    with pytest.raises(KeyError):
        resolve_output_policy({})                          # no defaults at all
    cfg = _cfg()
    del cfg['run']['save_mode']
    with pytest.raises(KeyError, match='save_mode'):
        resolve_output_policy(cfg)
    cfg = _cfg()
    del cfg['run']['overwrite_existing']
    with pytest.raises(KeyError, match='overwrite_existing'):
        resolve_output_policy(cfg)


def test_the_old_append_mode_is_refused():
    with pytest.raises(ValueError, match="renamed 'full'"):
        resolve_output_policy(_cfg(save_mode='append'))


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match='save_mode'):
        resolve_output_policy(_cfg(save_mode='banana'))


# ---------------------------------------------------------------------------
# has_retained_objects: the write gate, over ALL sweeps.
# ---------------------------------------------------------------------------

class _StubRadar:
    def __init__(self, fields):
        self.fields = fields


def _radar_with_ids(ids):
    return _StubRadar({'local_object_ids': {'data': ids}})


def test_no_field_means_no_objects():
    assert not has_retained_objects(_StubRadar({}))


def test_all_zero_ids_mean_no_objects():
    assert not has_retained_objects(_radar_with_ids(np.zeros((4, 6), int)))


def test_an_object_only_in_a_later_sweep_still_counts():
    """Regression against the first-sweep-only gate: rows 0-1 (sweep one) are
    clear air, the object lives in rows 2-3 (sweep two), and the file must
    still be written."""
    ids = np.zeros((4, 6), dtype=int)
    ids[2:, 3:] = 1
    assert has_retained_objects(_radar_with_ids(ids))


def test_masked_ids_do_not_count():
    ids = np.ma.array(np.ones((2, 3), dtype=int), mask=True)
    assert not has_retained_objects(_radar_with_ids(ids))
