"""Daily county mean temperature (°F) from the public ARCO ERA5 archive.

Reads hourly 2 m temperature at each county centroid from Google's
Analysis-Ready, Cloud-Optimized ERA5 store (anonymous access, no account),
converts to °F, shifts UTC to US Central time, and takes the daily mean as
(min + max) / 2 over the Central-time calendar day. That is the definition
behind the sample data and the index examples.

Requires the optional dependencies:  pip install "degreedays[era5]"
(xarray, zarr, gcsfs). Pulling a full year for all 3,108 counties is a few
GB of reads; start with a month and a handful of counties.

Usage:
    python scripts/fetch_era5.py --start 2024-01-01 --end 2024-01-31 --out county_tavg.csv
    python scripts/fetch_era5.py --start 2024-01-01 --end 2024-12-31 --fips 17031 48201 --out two.csv

ERA5 data: Hersbach et al. (2020), Copernicus Climate Change Service (C3S).
Contains modified Copernicus Climate Change Service information; neither the
European Commission nor ECMWF is responsible for any use of it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CT_OFFSET_HOURS = -6
WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "county_weights.csv"


def open_arco():
    import xarray as xr
    return xr.open_zarr(ARCO_URL, chunks=None, storage_options={"token": "anon"})


def extract_county_tavg(start: str, end: str, centroids: pd.DataFrame, ds=None) -> pd.DataFrame:
    """Daily (tmin + tmax) / 2 in °F on Central-time days, columns = FIPS."""
    import xarray as xr
    if ds is None:
        ds = open_arco()
    padded_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    padded_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lats = xr.DataArray(centroids["lat"].values, dims="county")
    lons = xr.DataArray((centroids["lon"].values % 360), dims="county")   # store uses 0..360 longitude
    t2m = ds["2m_temperature"].sel(time=slice(padded_start, padded_end))
    t2m = t2m.sel(latitude=lats, longitude=lons, method="nearest").load()
    hourly = t2m.to_pandas()
    hourly.columns = centroids["FIPS"].values
    hourly = (hourly - 273.15) * 9 / 5 + 32
    hourly.index = hourly.index + pd.Timedelta(hours=CT_OFFSET_HOURS)
    daily = (hourly.resample("D").min() + hourly.resample("D").max()) / 2
    daily = daily.loc[start:end]
    daily.index.name = "date"
    return daily


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fips", nargs="*", help="subset of county FIPS codes (default: all in data/county_weights.csv)")
    args = ap.parse_args()
    cf = pd.read_csv(WEIGHTS, dtype={"FIPS": str})
    cf["FIPS"] = cf["FIPS"].str.zfill(5)
    if args.fips:
        cf = cf[cf["FIPS"].isin([f.zfill(5) for f in args.fips])]
    ds = open_arco()
    frames = []
    for year in range(pd.Timestamp(args.start).year, pd.Timestamp(args.end).year + 1):   # one year at a time
        s = max(pd.Timestamp(args.start), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        e = min(pd.Timestamp(args.end), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        print(f"  {s} .. {e}: {len(cf)} counties")
        frames.append(extract_county_tavg(s, e, cf[["FIPS", "lat", "lon"]], ds=ds))
    out = pd.concat(frames).round(2)
    out.to_csv(args.out)
    print(f"wrote {args.out}: {out.shape[0]} days x {out.shape[1]} counties")


if __name__ == "__main__":
    main()
