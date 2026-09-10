"""Unit tests for spatial calculations, filtering and dataset generation.

Run from the repository root::

    pytest -v

Assertions focus on mathematical correctness (distance metrics), invariants and
filtering behaviour, so they stay robust across environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    DC_FAST,
    LEVEL_2,
    DataConfig,
    filter_stations,
    load_demand_points,
    load_stations,
)
from src.spatial_analysis import (
    compute_summary,
    deployment_trend,
    detect_charging_deserts,
    haversine_km,
    nearest_station_distances,
)


@pytest.fixture(scope="module")
def config():
    return DataConfig(n_stations=600, n_demand_points=200, random_seed=11)


@pytest.fixture(scope="module")
def stations(config):
    return load_stations(config)


@pytest.fixture(scope="module")
def demand(config):
    return load_demand_points(config)


# --------------------------------------------------------------------------- #
# Haversine distance metric
# --------------------------------------------------------------------------- #
def test_haversine_zero_distance():
    assert haversine_km(40.0, -75.0, 40.0, -75.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_longitude_at_equator():
    # 1 degree of longitude at the equator is ~111.19 km.
    d = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert d == pytest.approx(111.19, abs=0.5)


def test_haversine_known_city_pair():
    # New York City -> Los Angeles is ~3936 km.
    d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert d == pytest.approx(3936, rel=0.02)


def test_haversine_is_symmetric():
    a = haversine_km(47.6, -122.3, 25.8, -80.2)
    b = haversine_km(25.8, -80.2, 47.6, -122.3)
    assert a == pytest.approx(b)


def test_haversine_vectorized():
    lat1 = np.array([0.0, 0.0])
    lon1 = np.array([0.0, 0.0])
    d = haversine_km(lat1, lon1, np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    assert d.shape == (2,)
    assert d[1] == pytest.approx(2 * d[0], rel=1e-6)


# --------------------------------------------------------------------------- #
# Nearest-station distances
# --------------------------------------------------------------------------- #
def test_nearest_station_picks_closest():
    stations = pd.DataFrame(
        {"latitude": [40.0, 41.0], "longitude": [-75.0, -76.0]}
    )
    points = pd.DataFrame({"latitude": [40.01], "longitude": [-75.01]})
    nn = nearest_station_distances(points, stations)
    # Nearest is the (40.0, -75.0) station: well under 2 km.
    assert nn[0] < 2.0


def test_nearest_station_no_stations_is_inf():
    points = pd.DataFrame({"latitude": [40.0], "longitude": [-75.0]})
    nn = nearest_station_distances(points, pd.DataFrame(columns=["latitude", "longitude"]))
    assert np.isinf(nn[0])


def test_nearest_station_empty_points():
    stations = pd.DataFrame({"latitude": [40.0], "longitude": [-75.0]})
    nn = nearest_station_distances(pd.DataFrame(columns=["latitude", "longitude"]), stations)
    assert nn.shape == (0,)


# --------------------------------------------------------------------------- #
# Charging-desert detection
# --------------------------------------------------------------------------- #
def test_desert_detection_flags_far_points():
    stations = pd.DataFrame({"latitude": [40.0], "longitude": [-75.0]})
    points = pd.DataFrame(
        {
            "latitude": [40.001, 44.0],   # ~0.1 km away, ~445 km away
            "longitude": [-75.001, -75.0],
        }
    )
    result = detect_charging_deserts(points, stations, threshold_km=10.0)
    assert bool(result.loc[0, "is_desert"]) is False
    assert bool(result.loc[1, "is_desert"]) is True


def test_desert_threshold_is_monotonic(stations, demand):
    """A larger coverage radius can only reduce (never increase) desert count."""
    few = detect_charging_deserts(demand, stations, threshold_km=5.0)["is_desert"].sum()
    many = detect_charging_deserts(demand, stations, threshold_km=25.0)["is_desert"].sum()
    assert many <= few


def test_desert_all_when_no_stations(demand):
    empty = pd.DataFrame(columns=["latitude", "longitude"])
    result = detect_charging_deserts(demand, empty, threshold_km=10.0)
    assert result["is_desert"].all()


def test_desert_invalid_threshold():
    with pytest.raises(ValueError):
        detect_charging_deserts(pd.DataFrame(), pd.DataFrame(), threshold_km=0)


def test_desert_adds_expected_columns(stations, demand):
    result = detect_charging_deserts(demand, stations, threshold_km=10.0)
    assert {"nearest_station_km", "is_desert"}.issubset(result.columns)
    assert len(result) == len(demand)


# --------------------------------------------------------------------------- #
# Filtering behaviour
# --------------------------------------------------------------------------- #
def test_filter_by_state(stations):
    filtered = filter_stations(stations, states=["CA"])
    assert (filtered["state"] == "CA").all()
    assert len(filtered) > 0


def test_filter_by_network(stations):
    filtered = filter_stations(stations, networks=["Tesla"])
    assert (filtered["ev_network"] == "Tesla").all()


def test_filter_by_dc_fast_level(stations):
    filtered = filter_stations(stations, levels=[DC_FAST])
    assert (filtered["ev_dc_fast_num"] > 0).all()


def test_filter_by_level2(stations):
    filtered = filter_stations(stations, levels=[LEVEL_2])
    assert (filtered["ev_level2_evse_num"] > 0).all()


def test_filter_no_criteria_returns_all(stations):
    assert len(filter_stations(stations)) == len(stations)


def test_filter_combined(stations):
    filtered = filter_stations(stations, states=["CA", "TX"], networks=["Tesla"])
    assert set(filtered["state"]).issubset({"CA", "TX"})
    assert (filtered["ev_network"] == "Tesla").all()


# --------------------------------------------------------------------------- #
# Dataset generation & schema
# --------------------------------------------------------------------------- #
def test_station_schema(stations):
    expected = {
        "station_name", "city", "state", "latitude", "longitude",
        "ev_network", "ev_level2_evse_num", "ev_dc_fast_num",
        "ev_connector_types", "open_date", "access_code", "charging_level",
    }
    assert expected.issubset(set(stations.columns))


def test_generation_is_deterministic():
    cfg = DataConfig(n_stations=100, random_seed=7)
    a = load_stations(cfg)
    b = load_stations(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_coordinates_within_conus_bounds(stations):
    assert stations["latitude"].between(24.0, 50.0).all()
    assert stations["longitude"].between(-125.0, -66.0).all()


def test_charging_level_consistent(stations):
    dc = stations[stations["charging_level"] == DC_FAST]
    assert (dc["ev_dc_fast_num"] > 0).all()


# --------------------------------------------------------------------------- #
# Summary & temporal trend
# --------------------------------------------------------------------------- #
def test_summary_ratios_in_range(stations, demand):
    deserts = detect_charging_deserts(demand, stations, threshold_km=10.0)
    summary = compute_summary(stations, deserts)
    assert 0.0 <= summary["dc_fast_station_ratio"] <= 1.0
    assert 0.0 <= summary["desert_ratio"] <= 1.0
    assert summary["total_stations"] == len(stations)


def test_deployment_trend_is_cumulative(stations):
    trend = deployment_trend(stations)
    assert (trend["cumulative_stations"].diff().dropna() >= 0).all()
    assert trend["cumulative_stations"].iloc[-1] == len(stations)
