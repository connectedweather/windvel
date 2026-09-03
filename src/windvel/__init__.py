"""windvel — horizontal and vertical wind retrieval from RHI radar scans,
and the coherent updraft/downdraft structures found in the result.

See docs/PIPELINE.md for how the retrieval works.
"""

# Defined before any submodule import: save_windvel stamps it into every
# output file and imports it from here, so it must exist first.
__version__ = "1.0.0"

from .calculate_windvel import (
    HV_SRC_INTERPOLATED,
    HV_SRC_MEASURED,
    HV_SRC_MIRRORED,
    HV_SRC_NONE,
    HV_SRC_SMOOTHED,
    OUTPUT_FIELDS,
    SED_METHODS,
    RepairSettings,
    calculate_windvel,
    resolve_repair_settings,
    smooth_edge_profile,
)
from .coherent_structures import (
    TRUNC_BOT,
    TRUNC_FAR,
    TRUNC_GAP,
    TRUNC_NEAR,
    TRUNC_TOP,
    cell_area_km2,
)
from .config import CONFIG_VERSION, load_config, resolve
from .errors import (
    ConfigError,
    InputDataError,
    InputFieldError,
    MissingConfigKeyError,
    WindvelError,
)
from .plot_windvel import plot_windvel_panel, plot_windvel_summary
from .save_windvel import save_windvel_file
from .select_cloud_transects import select_cloud_transects
from .tables import ATTENUATION_RELATIONS, VERTICAL_ERROR_TABLES
from .utils import (
    extract_rhi_sweep_indices,
    gate_dist_alt_km,
    make_sr_alt_maps,
    project_to_radial_component,
)

# The public surface. A name not listed here is internal and may change
# without notice; add to this list deliberately. The pipeline stages are in
# call order; the integer codes are the values written into output fields.
__all__ = [
    "ATTENUATION_RELATIONS",
    "CONFIG_VERSION",
    "HV_SRC_INTERPOLATED",
    "HV_SRC_MEASURED",
    "HV_SRC_MIRRORED",
    "HV_SRC_NONE",
    "HV_SRC_SMOOTHED",
    "OUTPUT_FIELDS",
    "SED_METHODS",
    "TRUNC_BOT",
    "TRUNC_FAR",
    "TRUNC_GAP",
    "TRUNC_NEAR",
    "TRUNC_TOP",
    "VERTICAL_ERROR_TABLES",
    "ConfigError",
    "InputDataError",
    "InputFieldError",
    "MissingConfigKeyError",
    "RepairSettings",
    "WindvelError",
    "__version__",
    "calculate_windvel",
    "cell_area_km2",
    "extract_rhi_sweep_indices",
    "gate_dist_alt_km",
    "load_config",
    "make_sr_alt_maps",
    "plot_windvel_panel",
    "plot_windvel_summary",
    "project_to_radial_component",
    "resolve",
    "resolve_repair_settings",
    "save_windvel_file",
    "select_cloud_transects",
    "smooth_edge_profile",
]
