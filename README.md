# degreedays

Heating and cooling degree days from daily county temperatures, the normals they
are measured against, and the weighting that turns 3,108 counties into a national
gas-demand index. Pure pandas, no services, no configuration.

```
pip install -e ".[test]"
python examples/quickstart.py
pytest
```

## What it does

1. **Degree days.** `HDD = max(65 - T, 0)`, `CDD = max(T - 65, 0)` from the daily
   mean temperature in °F. The daily mean is `(min + max) / 2` of hourly ERA5
   2 m temperature at the county centroid over the US Central-time calendar day
   (`scripts/fetch_era5.py`, public archive, no account needed).

2. **Normals.** HDD normals are a day-of-year climatology plus a linear trend fitted
   to the deseasonalised series, **one slope per calendar month**, smoothed with a
   centred 21-day window. CDD normals are the trailing-10-year mean for each day of
   year, which follows air-conditioning adoption and population shifts faster than
   a long trend would.

3. **Weighting.** The national HDD index weights each county by its population
   within its state, then each state by its number of households that heat with
   utility gas (Census ACS "House Heating Fuel"). CDD and the regional series are
   population-weighted. Weights renormalise over whichever counties are present.

4. **Anomalies and spread.** Actual minus normal, plus a smoothed day-of-year
   standard deviation of the anomaly, for the usual "vs normal, in sigmas" reading.

## Why the trend is fitted per month

Degree days are floored at zero. In summer a county has no heating degree days to
lose, so a single all-season slope fitted to the whole year is a compromise between
the winter, which is warming, and the summer, which cannot show it. On a national
gas-weighted index the single slope declined about a third slower than the winter
totals themselves, which left the forward winter normal roughly 80 HDD (2 percent)
too cold. Fitting the slope month by month removes that; the 21-day smoothing hides
the small steps at month edges. `tests/test_degree_days.py` reproduces the effect on
a synthetic series, and `examples/quickstart.py` shows it on the sample data:
December and January slopes several times the all-season one, July near zero.

## Layout

```
degreedays/degree_days.py   compute_degree_days, fit_climo_and_trend, project_normal,
                            compute_rolling_climo, project_rolling_normal, anomaly_table
degreedays/weights.py       county weights, two-step gas weighting, regional aggregation
degreedays/index.py         build_index: temperatures -> index table in one call
scripts/fetch_era5.py       county daily temperature from the ARCO ERA5 archive
data/                       county weights (public Census data) and a 25-county sample
```

## Data sources and attribution

* ERA5 (Hersbach et al., 2020) via the Analysis-Ready, Cloud-Optimized ERA5 archive on
  Google Cloud. Contains modified Copernicus Climate Change Service information; neither
  the European Commission nor ECMWF is responsible for any use that may be made of it.
* U.S. Census Bureau, American Community Survey 5-year estimates (population, households,
  house heating fuel) and county population centroids. Public domain.

## License

MIT. See `LICENSE`.
