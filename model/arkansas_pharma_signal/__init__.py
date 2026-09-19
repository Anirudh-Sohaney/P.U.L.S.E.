"""Arkansas pharmaceutical intelligence and forecasting package.

The package builds provenance-bearing external signals from real local or
public data.  Default statistical paths require only pandas and numpy; the
optional modular research model requires PyTorch.
"""

__version__ = "0.1.0"

from .canonical_graph import CanonicalGraph, build_canonical_graph
from .event_schema import EVENT_COLUMNS
from .universal_forecast import UNIVERSAL_OUTPUT_COLUMNS, build_universal_forecast_grid

__all__ = ["CanonicalGraph", "build_canonical_graph", "EVENT_COLUMNS",
           "UNIVERSAL_OUTPUT_COLUMNS", "build_universal_forecast_grid"]
