"""Degree days from daily temperature, and the normals they are measured against.

Everything here works on a DataFrame of daily values indexed by date with one
column per location (county FIPS codes in the examples). The functions are pure:
no I/O, no configuration, nothing hidden.

Method summary
--------------
* Degree days: ``HDD = max(base - T, 0)``, ``CDD = max(T - base, 0)`` with a
  65 °F base and ``T`` the daily mean temperature.
* Trend normal (used for HDD): a day-of-year climatology plus a linear trend
  fitted to the deseasonalised series. The trend is fitted **per calendar
  month**. Degree days are floored at zero, so a single all-season slope is a
  compromise between the season that has degree days to lose and the season
  that cannot lose any more: on a national gas-weighted HDD index the
  all-season slope understated the winter decline by roughly a third and left
  the forward winter normal ~80 HDD too cold. ``by_month=False`` reproduces
  the single-slope fit for comparison.
* Rolling normal (used for CDD): the trailing N-year mean for each day of
  year, which tracks structural change (air-conditioning adoption, population
  shifts) faster than a long linear trend.
* Both normals are smoothed with a centred 21-day window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_degree_days(temps: pd.DataFrame, base: float = 65.0):
    """Heating and cooling degree days from daily mean temperature (°F).

    Returns a ``(hdd, cdd)`` tuple of DataFrames with the same shape as *temps*.
    """
    hdd = (base - temps).clip(lower=0)
    cdd = (temps - base).clip(lower=0)
    return hdd, cdd


def _linear_fit(deseas: pd.DataFrame, t: np.ndarray):
    """Least-squares slope and intercept per column of *deseas* against *t* (days)."""
    t_mean = t.mean()
    t_var = (t ** 2).mean() - t_mean ** 2
    slope = (deseas.multiply(t, axis=0).mean() - deseas.mean() * t_mean) / t_var
    intercept = deseas.mean() - slope * t_mean
    return slope, intercept


def fit_climo_and_trend(df: pd.DataFrame, by_month: bool = True):
    """Fit a day-of-year climatology and a linear trend on the deseasonalised series.

    Parameters
    ----------
    df : DataFrame
        Daily values indexed by datetime, one column per location.
    by_month : bool
        Fit a separate slope and intercept for each calendar month (default).
        ``False`` fits one slope across all days of the year.

    Returns
    -------
    climatology : DataFrame
        Mean value per day of year (index 1..366).
    slope, intercept : DataFrame or Series
        Trend parameters in units per day. With ``by_month`` these are
        DataFrames indexed by month 1..12; otherwise Series per column.
    t0 : Timestamp
        Reference date for the trend (first date in *df*).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    climatology = df.groupby(df.index.dayofyear).mean()
    deseas = df - climatology.loc[df.index.dayofyear].values
    t = (df.index - df.index[0]).days.values

    if not by_month:
        slope, intercept = _linear_fit(deseas, t)
        return climatology, slope, intercept, df.index[0]

    months = df.index.month
    slopes, intercepts = {}, {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() < 2:
            raise ValueError(f"fit_climo_and_trend: no data for calendar month {m}")
        slopes[m], intercepts[m] = _linear_fit(deseas.loc[mask], t[mask])
    slope = pd.DataFrame(slopes).T.reindex(columns=df.columns)
    intercept = pd.DataFrame(intercepts).T.reindex(columns=df.columns)
    slope.index.name = intercept.index.name = "month"
    return climatology, slope, intercept, df.index[0]


def project_normal(climatology, slope, intercept, t0, dates, rolling_window: int = 21):
    """Trend-adjusted normals for *dates*: climatology plus the fitted trend, smoothed.

    Accepts the per-month (DataFrame) or single-slope (Series) parameters from
    :func:`fit_climo_and_trend`. The centred rolling window also smooths the
    small month-edge steps of the per-month trend. Values are clipped at zero.
    """
    dates = pd.to_datetime(dates)
    t = (dates - t0).days.values.reshape(-1, 1)
    climo_vals = climatology.loc[dates.dayofyear].values
    if isinstance(slope, pd.DataFrame):
        m = dates.month
        trend_vals = (intercept.reindex(columns=climatology.columns).loc[m].values
                      + slope.reindex(columns=climatology.columns).loc[m].values * t)
    else:
        trend_vals = intercept.values + slope.values * t
    normals = pd.DataFrame(climo_vals + trend_vals, index=dates, columns=climatology.columns)
    return normals.rolling(rolling_window, center=True, min_periods=1).mean().clip(lower=0)


def compute_rolling_climo(dd: pd.DataFrame, n_years: int = 10) -> pd.DataFrame:
    """Year-varying day-of-year climatology from the trailing *n_years*.

    For each date, the mean of the same day of year over the preceding
    *n_years* (the current year excluded). Early years with less history use
    what is available.
    """
    dd = dd.copy()
    dd.index = pd.to_datetime(dd.index)
    doy = dd.index.dayofyear
    result = pd.DataFrame(np.nan, index=dd.index, columns=dd.columns)
    for d in range(1, 367):
        mask = doy == d
        doy_slice = dd.loc[mask]
        if len(doy_slice) == 0:
            continue
        rolled = doy_slice.rolling(n_years, min_periods=1).mean().shift(1)
        rolled.iloc[0] = doy_slice.iloc[0].values
        result.loc[mask] = rolled.values
    return result


def project_rolling_normal(dd: pd.DataFrame, dates, n_years: int = 10, rolling_window: int = 21) -> pd.DataFrame:
    """Rolling-climatology normals for *dates*.

    Historical dates use the trailing *n_years* day-of-year mean; dates beyond
    the data repeat the most recent *n_years* average. Smoothed and clipped at zero.
    """
    dates = pd.to_datetime(dates)
    dd_index = pd.to_datetime(dd.index)
    hist = compute_rolling_climo(dd, n_years=n_years)

    cutoff = dd_index.max() - pd.DateOffset(years=n_years)
    recent = dd.loc[dd_index >= cutoff]
    latest_climo = recent.groupby(recent.index.dayofyear).mean()

    result = pd.DataFrame(np.nan, index=dates, columns=dd.columns)
    common = dates.intersection(hist.index)
    if len(common) > 0:
        result.loc[common] = hist.loc[common].values
    missing = result.index[result.iloc[:, 0].isna()]
    if len(missing) > 0:
        doys = missing.dayofyear
        valid = doys.isin(latest_climo.index)
        if valid.any():
            result.loc[missing[valid]] = latest_climo.loc[doys[valid]].values
    result = result.ffill().bfill()
    return result.rolling(rolling_window, center=True, min_periods=1).mean().clip(lower=0)


def anomaly_table(actual: pd.Series, normal: pd.Series, rolling_window: int = 21, name: str = "value") -> pd.DataFrame:
    """Actual, normal, anomaly and a smoothed day-of-year anomaly spread for one series.

    Dates with a normal but no actual (a projection period) carry the normal as
    the value, so the anomaly there is zero by construction.
    """
    df = pd.DataFrame({name: actual, f"{name}_normal": normal})
    df[name] = df[name].combine_first(df[f"{name}_normal"])
    df = df.dropna()
    df[f"{name}_anom"] = df[name] - df[f"{name}_normal"]
    df["day"] = df.index.dayofyear
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)
    std_by_doy = df.groupby(df.index.dayofyear)[f"{name}_anom"].std()
    df[f"{name}_std"] = (df["day"].map(std_by_doy.to_dict())
                         .rolling(rolling_window, center=True, min_periods=1).mean())
    return df
