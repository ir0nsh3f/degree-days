"""Build a weighted degree-day index with normals, anomalies and spread from county temperatures."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .degree_days import (anomaly_table, compute_degree_days, fit_climo_and_trend,
                          project_normal, project_rolling_normal)
from .weights import population_weights, two_step_weights, weighted_series


def build_index(temps: pd.DataFrame, cf: pd.DataFrame, base: float = 65.0,
                projection_end: Optional[str] = None, rolling_window: int = 21,
                cdd_rolling_years: int = 10, hdd_by_month: bool = True,
                hdd_weighting: str = "gas", cdd_weighting: str = "population") -> pd.DataFrame:
    """County daily mean temperature (°F) -> one index table.

    Columns: ``hdd, hdd_normal, hdd_anom, hdd_std, cdd, cdd_normal, cdd_anom,
    cdd_std, day, year, week``. HDD uses the per-month trend normal and
    two-step gas weighting; CDD uses the trailing-10-year rolling normal and
    population weighting. Dates past the data through *projection_end* carry
    the normal as the value.
    """
    temps = temps.copy()
    temps.index = pd.to_datetime(temps.index)
    hdd, cdd = compute_degree_days(temps, base=base)
    dates = pd.date_range(temps.index.min(), projection_end or temps.index.max(), freq="D")

    climo, slope, intercept, t0 = fit_climo_and_trend(hdd, by_month=hdd_by_month)
    hdd_normal = project_normal(climo, slope, intercept, t0, dates, rolling_window=rolling_window)
    cdd_normal = project_rolling_normal(cdd, dates, n_years=cdd_rolling_years, rolling_window=rolling_window)

    w_h = two_step_weights(cf, available=hdd.columns) if hdd_weighting == "gas" else population_weights(cf, available=hdd.columns)
    w_c = two_step_weights(cf, available=cdd.columns) if cdd_weighting == "gas" else population_weights(cf, available=cdd.columns)

    h = anomaly_table(weighted_series(hdd, w_h), weighted_series(hdd_normal, w_h), rolling_window, name="hdd")
    c = anomaly_table(weighted_series(cdd, w_c), weighted_series(cdd_normal, w_c), rolling_window, name="cdd")
    out = h.join(c[["cdd", "cdd_normal", "cdd_anom", "cdd_std"]], how="inner")
    return out[["hdd", "hdd_normal", "hdd_anom", "hdd_std", "cdd", "cdd_normal", "cdd_anom", "cdd_std", "day", "year", "week"]]
