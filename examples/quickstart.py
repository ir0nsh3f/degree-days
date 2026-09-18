"""Build a gas-weighted HDD / population-weighted CDD index from the bundled sample.

The sample holds daily mean temperature for the five most populous counties in
each of five states, 1980-2025, so the two-step weighting (population within a
state, gas-heated households across states) has something to do. Run:

    python examples/quickstart.py
"""
from pathlib import Path

import pandas as pd

from degreedays import build_index, load_county_weights, fit_climo_and_trend, compute_degree_days

ROOT = Path(__file__).resolve().parents[1]
temps = pd.read_csv(ROOT / "data" / "sample_county_tavg_degF.csv", index_col=0, parse_dates=True)
cf = load_county_weights()

index = build_index(temps, cf, projection_end="2026-12-31")

winter = index[index.index.month.isin([11, 12, 1, 2, 3])].copy()
winter["season"] = winter.index.year.where(winter.index.month < 11, winter.index.year + 1)
totals = winter.groupby("season")[["hdd", "hdd_normal", "hdd_anom"]].sum().round(0)
print("Nov-Mar totals for the 25-county sample index (gas-weighted HDD):")
print(totals.tail(8).to_string())

# Why the trend is fitted per month: compare the two fits on the sample
hdd, _ = compute_degree_days(temps)
_, slope_month, _, _ = fit_climo_and_trend(hdd, by_month=True)
_, slope_single, _, _ = fit_climo_and_trend(hdd, by_month=False)
print("\nMean trend across the sample counties, HDD per day per year:")
print(f"  one all-season slope: {float(slope_single.mean()) * 365.25:+.3f}")
for m in (12, 1, 2, 7):
    print(f"  month {m:2d} slope:      {float(slope_month.loc[m].mean()) * 365.25:+.3f}")
