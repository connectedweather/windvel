"""The package declares its surface, and the surface is real."""

import windvel


def test_version_is_a_string():
    assert isinstance(windvel.__version__, str)


def test_every_public_name_exists():
    for name in windvel.__all__:
        assert hasattr(windvel, name), name


def test_the_pipeline_stages_are_public():
    """What cli.py calls, in order, a user can call too."""
    for name in ("extract_rhi_sweep_indices", "select_cloud_transects",
                 "calculate_windvel", "save_windvel_file"):
        assert name in windvel.__all__


def test_errors_are_catchable_by_base_and_by_builtin():
    """ConfigError is caught as WindvelError and as ValueError, so old
    `except ValueError` clauses keep working during migration; likewise
    InputFieldError and KeyError."""
    assert issubclass(windvel.ConfigError, windvel.WindvelError)
    assert issubclass(windvel.ConfigError, ValueError)
    assert issubclass(windvel.InputFieldError, windvel.WindvelError)
    assert issubclass(windvel.InputFieldError, KeyError)
    assert str(windvel.InputFieldError("plain message")) == "plain message"
