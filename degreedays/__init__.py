"""degreedays — heating and cooling degree days, normals, and weighted indices."""
from .degree_days import (anomaly_table, compute_degree_days, compute_rolling_climo,
                          fit_climo_and_trend, project_normal, project_rolling_normal)
from .index import build_index
from .weights import (REGIONS, aggregate_l48, aggregate_regions, load_county_weights,
                      population_weights, two_step_weights, weighted_series)

__all__ = [
    "anomaly_table", "compute_degree_days", "compute_rolling_climo", "fit_climo_and_trend",
    "project_normal", "project_rolling_normal", "build_index", "REGIONS", "aggregate_l48",
    "aggregate_regions", "load_county_weights", "population_weights", "two_step_weights",
    "weighted_series",
]
__version__ = "0.1.0"
