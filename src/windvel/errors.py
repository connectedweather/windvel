"""Exceptions windvel raises.

A caller writes ``except WindvelError`` and catches everything this package
means, and nothing Python raised for its own reasons. Each subclass also
inherits the built-in it stands in for, so an existing ``except KeyError``
or ``except ValueError`` keeps working while callers migrate.
"""

__all__ = [
    "ConfigError",
    "InputDataError",
    "InputFieldError",
    "MissingConfigKeyError",
    "WindvelError",
]


class WindvelError(Exception):
    """Base for every error windvel raises."""


class ConfigError(WindvelError, ValueError):
    """A config key is not accepted any more, or holds an invalid value."""


class MissingConfigKeyError(ConfigError, KeyError):
    """A required config key is absent.

    Subclasses KeyError as well as ConfigError, so ``except KeyError`` still
    catches a missing key; ``str()`` gives the plain message rather than
    KeyError's quoted repr.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class InputFieldError(WindvelError, KeyError):
    """A field the config names is not in the radar file.

    Subclasses KeyError for compatibility; ``str()`` gives the plain message
    rather than KeyError's quoted repr.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class InputDataError(WindvelError, ValueError):
    """An input file holds something the retrieval cannot work with.

    A radar file with no readable start time or a non-uniform gate spacing;
    a sounding file with no profile in it. The file, not the config, is at
    fault.
    """
