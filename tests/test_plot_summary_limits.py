"""The vertical-velocity panel's adaptive colorbar limit."""
import numpy as np

from windvel.plot_windvel import (
    SUMMARY_W_EXTENDED_LIMIT_MPS,
    SUMMARY_WIND_LIMIT_MPS,
    summary_w_limit,
)


def test_standard_limit_when_within_range():
    w = np.array([[-19.9, 5.0, np.nan]])
    assert summary_w_limit(w) == SUMMARY_WIND_LIMIT_MPS


def test_extended_limit_when_any_gate_exceeds_the_standard():
    w = np.array([[3.0, -20.1], [np.nan, 10.0]])
    assert summary_w_limit(w) == SUMMARY_W_EXTENDED_LIMIT_MPS


def test_all_nan_sweep_keeps_the_standard_limit():
    assert summary_w_limit(np.full((2, 2), np.nan)) == SUMMARY_WIND_LIMIT_MPS
