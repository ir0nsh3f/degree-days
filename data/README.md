# Data

**county_weights.csv** — one row per county in the contiguous United States (3,108 rows).
Columns: `FIPS`, `county`, `state`, `population`, `households`, `utility_gas_households`,
`lat`, `lon`. Population and households are from the U.S. Census Bureau's American
Community Survey 5-year estimates (households heating with utility gas: table B25040,
"House Heating Fuel"). Centroids are the Census Bureau's county population centroids.
All public domain.

**sample_county_tavg_degF.csv** — daily mean temperature in °F, 1980-01-01 to 2025-12-31,
for the five most populous counties in each of Colorado, Illinois, Massachusetts, New York
and Texas (25 columns, FIPS codes). Built with `scripts/fetch_era5.py`: ERA5 hourly 2 m
temperature at the county centroid, converted to °F, shifted to US Central time, daily
value = (min + max) / 2 over the Central-time calendar day.

Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020).
Neither the European Commission nor ECMWF is responsible for any use that may be made of it.
