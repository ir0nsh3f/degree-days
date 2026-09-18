"""Weighting county degree days into regional and national indices.

Two weighting schemes:

* **Population** — each county weighted by population. Used for cooling
  degree days and for the regional series.
* **Two-step gas** — population weight within each state, then each state
  weighted by its number of households heating with utility gas (Census ACS
  table B25040, "House Heating Fuel"). Used for the national heating degree
  day index: it puts the weight where gas heating demand actually is, rather
  than where people live.

Weights are renormalised over the counties actually present in the data, so a
missing county scales its neighbours up rather than silently lowering the
index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Weather regions used for the regional series (state names as in the weights table).
REGIONS = {
    "MI": ["Michigan"],
    "NW": ["Oregon", "Washington", "Idaho", "Montana", "Wyoming", "Utah", "Colorado"],
    "CAL": ["California"],
    "SW": ["Arizona", "New Mexico", "Nevada"],
    "TEX": ["Texas"],
    "CENT": ["North Dakota", "South Dakota", "Nebraska", "Kansas", "Oklahoma"],
    "MIDW": ["Minnesota", "Iowa", "Missouri", "Arkansas", "Louisiana", "Wisconsin", "Michigan", "Indiana"],
    "MIDA": ["Illinois", "Ohio", "Pennsylvania", "Kentucky", "West Virginia", "Virginia", "Maryland",
             "Delaware", "New Jersey", "District of Columbia"],
    "TEN": ["Tennessee"],
    "CAR": ["North Carolina", "South Carolina"],
    "SE": ["Mississippi", "Alabama", "Georgia"],
    "FLA": ["Florida"],
    "NY": ["New York"],
    "NE": ["Connecticut", "Massachusetts", "Vermont", "New Hampshire", "Maine", "Rhode Island"],
}

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "county_weights.csv"


def load_county_weights(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the county weights table (FIPS zero-padded to five digits)."""
    cf = pd.read_csv(path or DEFAULT_WEIGHTS, dtype={"FIPS": str})
    cf["FIPS"] = cf["FIPS"].str.zfill(5)
    return cf


def _restrict(cf: pd.DataFrame, available: Optional[Iterable[str]]) -> pd.DataFrame:
    if available is None:
        return cf
    return cf[cf["FIPS"].isin(set(available))]


def population_weights(cf: pd.DataFrame, available: Optional[Iterable[str]] = None,
                       column: str = "population") -> pd.Series:
    """Population share per county, over the counties in *available* (all if None)."""
    sub = _restrict(cf, available)
    w = sub[column] / sub[column].sum()
    return pd.Series(w.values, index=sub["FIPS"].values, name="weight")


def two_step_weights(cf: pd.DataFrame, available: Optional[Iterable[str]] = None,
                     within: str = "population", across: str = "utility_gas_households") -> pd.Series:
    """Population weight within each state, times the state's share of gas-heated households.

    ``w_county = pop_county / pop_state  *  gas_state / gas_total`` — sums to one.

    A state's gas share is a fact about the state, so it is taken from the
    full table even when some of its counties are missing from the data; the
    missing counties' population weight is redistributed within the state.
    States with no county present are dropped and the shares renormalised.
    """
    sub = _restrict(cf, available)
    w_within = sub[within] / sub.groupby("state")[within].transform("sum")
    state_share = cf.groupby("state")[across].sum().reindex(sub["state"].unique())
    state_share = state_share / state_share.sum()
    w = w_within * sub["state"].map(state_share)
    return pd.Series(w.values, index=sub["FIPS"].values, name="weight")


def weighted_series(dd: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted average across columns of *dd*, renormalised over the columns present."""
    common = weights.index.intersection(dd.columns)
    if len(common) == 0:
        raise ValueError("no overlap between weights and data columns")
    w = weights[common]
    w = w / w.sum()
    return (dd[common] * w).sum(axis=1)


def aggregate_l48(dd: pd.DataFrame, cf: pd.DataFrame, method: str = "gas") -> pd.Series:
    """National index: ``"gas"`` for two-step gas weighting, ``"population"`` otherwise."""
    if method.lower() == "gas":
        weights = two_step_weights(cf, available=dd.columns)
    else:
        weights = population_weights(cf, available=dd.columns)
    return weighted_series(dd, weights)


def aggregate_regions(dd: pd.DataFrame, cf: pd.DataFrame, regions: dict = REGIONS,
                      method: str = "population") -> pd.DataFrame:
    """One population-weighted (or two-step) series per region, regions with no data skipped."""
    out = {}
    for name, states in regions.items():
        sub = cf[cf["state"].isin(states)]
        present = sub["FIPS"][sub["FIPS"].isin(dd.columns)]
        if present.empty:
            continue
        weights = two_step_weights(sub, available=present) if method.lower() == "gas" \
            else population_weights(sub, available=present)
        out[name] = weighted_series(dd, weights)
    return pd.DataFrame(out)
