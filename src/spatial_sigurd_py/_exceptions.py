# spatial_sigurd_py/_exceptions.py
"""Warning and exception types raised by spatial_sigurd_py"""

__all__ = [
    "FTUWarning",
    "SpatialSigurdWarning",
    "SpotFilterWarning",
]


class SpatialSigurdWarning(UserWarning):
    """Base class for all warnings raised by spatial_sigurd_py.

    This lets you handle every warning in one call.
    You can silence them with:
    warnings.filterwarnings("ignore", category=SpatialSigurdWarning)
    You can turn them into exceptions with:
    warnings.filterwarnings("error", category=SpatialSigurdWarning).
    """


class SpotFilterWarning(SpatialSigurdWarning):
    """A spot/cell-count filter is likely to remove all or nearly all features.

    This warning occurs when the minimum number of positive spots required is very high.
    This will most likely remove most if not all variants. Since this is still a
    possible, if unreasonable, requirement it only leads to a warning.
    The user can then easily lower the required number of spots.
    """


class FTUWarning(SpatialSigurdWarning):
    """FTU identification failed and was skipped; SVV detection results are unaffected."""
