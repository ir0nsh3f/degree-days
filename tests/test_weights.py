import numpy as np
import pandas as pd

from degreedays import (aggregate_l48, aggregate_regions, load_county_weights,
                        population_weights, two_step_weights, weighted_series, build_index)


def toy_cf():
    return pd.DataFrame({
        "FIPS": ["01001", "01002", "02001", "02002"],
        "county": ["a", "b", "c", "d"],
        "state": ["Texas", "Texas", "Illinois", "Illinois"],
        "population": [300, 100, 200, 200],
        "households": [100, 40, 80, 80],
        "utility_gas_households": [10, 10, 60, 20],    # Texas 20, Illinois 80 -> 20% / 80%
        "lat": [30.0, 30.5, 41.0, 41.5],
        "lon": [-97.0, -97.5, -88.0, -88.5],
    })


def test_two_step_weights_match_hand_calculation():
    w = two_step_weights(toy_cf())
    assert abs(w.sum() - 1) < 1e-12
    assert np.isclose(w["01001"], 0.75 * 0.20)     # 300/400 within Texas, Texas 20% of gas households
    assert np.isclose(w["02001"], 0.50 * 0.80)


def test_weights_renormalise_over_available_counties():
    w = two_step_weights(toy_cf(), available=["01001", "02001", "02002"])   # 01002 missing
    assert abs(w.sum() - 1) < 1e-12
    assert np.isclose(w["01001"], 1.0 * 0.20)


def test_weighted_series_of_identical_columns_is_unchanged():
    idx = pd.date_range("2024-01-01", periods=5)
    dd = pd.DataFrame({f: np.arange(5.0) for f in ["01001", "01002", "02001", "02002"]}, index=idx)
    out = weighted_series(dd, two_step_weights(toy_cf()))
    assert np.allclose(out.values, np.arange(5.0))
    pop = aggregate_l48(dd, toy_cf(), method="population")
    assert np.allclose(pop.values, np.arange(5.0))


def test_regions_use_only_their_states():
    idx = pd.date_range("2024-01-01", periods=3)
    dd = pd.DataFrame({"01001": 1.0, "01002": 1.0, "02001": 9.0, "02002": 9.0}, index=idx)
    r = aggregate_regions(dd, toy_cf(), regions={"TEX": ["Texas"], "MIDA": ["Illinois"], "EMPTY": ["Ohio"]})
    assert list(r.columns) == ["TEX", "MIDA"]
    assert (r["TEX"] == 1.0).all() and (r["MIDA"] == 9.0).all()


def test_bundled_weights_and_sample_build_an_index():
    cf = load_county_weights()
    assert len(cf) == 3108 and cf["FIPS"].str.len().eq(5).all()
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    temps = pd.read_csv(root / "data" / "sample_county_tavg_degF.csv", index_col=0, parse_dates=True)
    temps = temps.loc["2010-01-01":"2015-12-31"]
    out = build_index(temps, cf, projection_end="2016-03-31")
    assert out.index.max() == pd.Timestamp("2016-03-31")
    assert out.loc["2016-01-15", "hdd_anom"] == 0            # projection carries the normal
    assert out.loc["2010-01-01":"2015-12-31", "hdd"].mean() > 0
    assert (out["hdd_normal"] >= 0).all() and (out["cdd_normal"] >= 0).all()
