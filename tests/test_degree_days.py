import numpy as np
import pandas as pd
import pytest

from degreedays import (compute_degree_days, fit_climo_and_trend, project_normal,
                        project_rolling_normal, anomaly_table)


def synthetic(years=40, winter_slope=-0.08, seed=0):
    """Daily 'HDD-like' series: seasonal cycle, a decline (units/day per year) in winter only, noise, zero floor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-01-01", periods=365 * years, freq="D")
    doy = dates.dayofyear.values
    t_years = np.arange(len(dates)) / 365.25
    season = 20 * np.cos(2 * np.pi * (doy - 15) / 365.25)          # +20 in January, -20 in July
    winter = np.isin(dates.month, [11, 12, 1, 2, 3])
    trend = np.where(winter, winter_slope * t_years, 0.0)
    raw = 15 + season + trend + rng.normal(0, 2, len(dates))
    return pd.DataFrame({"x": np.clip(raw, 0, None)}, index=dates)


def test_degree_day_identity():
    temps = pd.DataFrame({"a": [30.0, 65.0, 80.0]}, index=pd.date_range("2024-01-01", periods=3))
    hdd, cdd = compute_degree_days(temps, base=65.0)
    assert (hdd["a"].values == [35.0, 0.0, 0.0]).all()
    assert (cdd["a"].values == [0.0, 0.0, 15.0]).all()
    assert ((hdd - cdd)["a"].values == (65.0 - temps["a"]).values).all()
    assert (hdd >= 0).all().all() and (cdd >= 0).all().all()


def test_per_month_trend_recovers_winter_slope_where_single_slope_is_diluted():
    df = synthetic()
    true_winter_slope = -0.08                      # units/day lost per year, winter months only
    _, slope_m, _, _ = fit_climo_and_trend(df, by_month=True)
    _, slope_1, _, _ = fit_climo_and_trend(df, by_month=False)
    jan = float(slope_m.loc[1, "x"]) * 365.25       # per-day slopes -> per year
    single = float(slope_1["x"]) * 365.25
    assert abs(jan - true_winter_slope) < 0.3 * abs(true_winter_slope)    # per-month fit sees the winter decline
    assert abs(single) < 0.7 * abs(true_winter_slope)                     # one all-season slope understates it
    summer = float(slope_m.loc[7, "x"]) * 365.25
    assert abs(summer) < 0.3 * abs(true_winter_slope)                     # and summer, floored at zero, shows ~none


def test_projected_normal_is_smooth_and_nonnegative():
    df = synthetic()
    climo, slope, intercept, t0 = fit_climo_and_trend(df, by_month=True)
    dates = pd.date_range("2019-09-01", "2020-05-31")
    normal = project_normal(climo, slope, intercept, t0, dates, rolling_window=21)
    assert (normal["x"] >= 0).all()
    interior = normal.loc["2019-10-15":"2020-04-15", "x"]     # away from the window edges of the projection
    assert interior.diff().abs().max() < 0.6                   # seasonal cycle moves ~0.34/day; no month-edge jumps


def test_rolling_normal_of_constant_series_is_constant():
    dates = pd.date_range("2000-01-01", "2015-12-31")
    df = pd.DataFrame({"x": 7.0}, index=dates)
    normal = project_rolling_normal(df, pd.date_range("2016-01-01", "2016-12-31"), n_years=10)
    assert np.allclose(normal["x"].values, 7.0)


def test_anomaly_table_fills_projection_with_normal():
    idx = pd.date_range("2024-01-01", periods=10)
    actual = pd.Series(np.arange(7, dtype=float), index=idx[:7])
    normal = pd.Series(5.0, index=idx)
    t = anomaly_table(actual, normal, name="hdd")
    assert len(t) == 10
    assert (t.loc[idx[7:], "hdd_anom"] == 0).all()
    assert t.loc[idx[0], "hdd_anom"] == -5.0
