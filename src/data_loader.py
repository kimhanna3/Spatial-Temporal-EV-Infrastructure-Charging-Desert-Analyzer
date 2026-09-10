"""Data ingestion, synthetic generation and preprocessing.

The application runs fully offline: :func:`load_stations` returns a synthetic
dataset whose schema mimics NREL's Alternative Fuel Data Center (AFDC) station
export, so the analytics and UI layers behave exactly as they would against the
real feed.

All spatial coordinates use WGS84 (EPSG:4326, decimal degrees).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

#: Charging-level categories used throughout the app.
LEVEL_2 = "Level 2"
DC_FAST = "DC Fast"
CHARGING_LEVELS: tuple[str, ...] = (LEVEL_2, DC_FAST)

#: EV networks (providers), roughly weighted by real-world prevalence.
NETWORKS: dict[str, float] = {
    "Tesla": 0.24,
    "ChargePoint": 0.22,
    "Electrify America": 0.12,
    "EVgo": 0.10,
    "Blink": 0.08,
    "Rivian Adventure Network": 0.04,
    "Volta": 0.04,
    "Non-Networked": 0.16,
}

#: Approximate CONUS state bounding boxes ``(lat_min, lat_max, lon_min, lon_max)``
#: with a population weight (millions) used to distribute stations/demand.
STATE_BOXES: dict[str, tuple[float, float, float, float, float]] = {
    "CA": (32.5, 42.0, -124.4, -114.1, 39.0),
    "TX": (25.8, 36.5, -106.6, -93.5, 30.0),
    "FL": (24.5, 31.0, -87.6, -80.0, 22.0),
    "NY": (40.5, 45.0, -79.8, -71.9, 19.0),
    "IL": (37.0, 42.5, -91.5, -87.5, 12.0),
    "NC": (33.8, 36.6, -84.3, -75.5, 10.5),
    "GA": (30.4, 35.0, -85.6, -80.8, 11.0),
    "WA": (45.5, 49.0, -124.8, -117.0, 7.8),
    "AZ": (31.3, 37.0, -114.8, -109.0, 7.4),
    "MA": (41.2, 42.9, -73.5, -69.9, 7.0),
    "CO": (37.0, 41.0, -109.0, -102.0, 5.9),
    "OR": (42.0, 46.3, -124.6, -116.5, 4.2),
}

_L2_CONNECTORS = ["J1772"]
_DCFAST_CONNECTORS = ["J1772COMBO", "CHADEMO", "TESLA"]

# Geography model: both stations and demand concentrate around a shared set of
# per-state "metro" centers, mirroring how real EV infrastructure and demand
# cluster in urban areas. Charging deserts then emerge naturally at the urban
# periphery and in the fraction of demand placed in underserved (rural) areas.
_METROS_PER_STATE = 4
_STATION_SPREAD_DEG = 0.15   # ~16 km std-dev around a metro center
_DEMAND_SPREAD_DEG = 0.30    # demand spreads wider than the charger network
_STATION_RURAL_FRACTION = 0.15   # share of stations placed uniformly (highways)
_DEMAND_RURAL_FRACTION = 0.20    # share of demand in underserved areas


@dataclass
class DataConfig:
    """Parameters controlling synthetic dataset generation."""

    n_stations: int = 1200
    n_demand_points: int = 400
    dc_fast_fraction: float = 0.35  # share of stations that are DC fast
    start_year: int = 2015
    end_year: int = 2025
    random_seed: Optional[int] = 42
    states: tuple[str, ...] = field(default_factory=lambda: tuple(STATE_BOXES))


# --------------------------------------------------------------------------- #
# Synthetic generation
# --------------------------------------------------------------------------- #
def _weighted_choice(rng: np.random.Generator, options, weights, size):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(options, size=size, p=weights)


def _metro_centers(config: "DataConfig") -> dict[str, np.ndarray]:
    """Deterministic per-state metro centers shared by stations and demand.

    Seeded independently of the station/demand seed offsets so that both
    datasets reference the *same* urban centers for a given config.
    """
    seed = None if config.random_seed is None else config.random_seed + 500
    rng = np.random.default_rng(seed)
    centers: dict[str, np.ndarray] = {}
    for st in config.states:
        if st not in STATE_BOXES:
            continue
        lat_min, lat_max, lon_min, lon_max, _ = STATE_BOXES[st]
        lats = rng.uniform(lat_min, lat_max, _METROS_PER_STATE)
        lons = rng.uniform(lon_min, lon_max, _METROS_PER_STATE)
        centers[st] = np.column_stack([lats, lons])
    return centers


def _place_points(
    rng: np.random.Generator,
    chosen_states: np.ndarray,
    metros: dict[str, np.ndarray],
    spread_deg: float,
    rural_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place points around metro centers, with a fraction scattered rurally."""
    n = len(chosen_states)
    lat = np.empty(n)
    lon = np.empty(n)
    for i, st in enumerate(chosen_states):
        lat_min, lat_max, lon_min, lon_max, _ = STATE_BOXES[st]
        if rng.random() < rural_fraction:
            lat[i] = rng.uniform(lat_min, lat_max)
            lon[i] = rng.uniform(lon_min, lon_max)
        else:
            center = metros[st][rng.integers(len(metros[st]))]
            lat[i] = np.clip(rng.normal(center[0], spread_deg), lat_min, lat_max)
            lon[i] = np.clip(rng.normal(center[1], spread_deg), lon_min, lon_max)
    return lat, lon


def generate_synthetic_stations(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """Generate a synthetic station table matching the AFDC schema.

    Returns:
        DataFrame with columns: ``station_name``, ``city``, ``state``,
        ``latitude``, ``longitude``, ``ev_network``, ``ev_level2_evse_num``,
        ``ev_dc_fast_num``, ``ev_connector_types``, ``open_date``,
        ``access_code``, and a derived ``charging_level``.
    """
    config = config or DataConfig()
    rng = np.random.default_rng(config.random_seed)
    n = config.n_stations

    states = [s for s in config.states if s in STATE_BOXES]
    if not states:
        raise ValueError("No valid states configured.")
    state_weights = np.array([STATE_BOXES[s][4] for s in states])
    chosen_states = _weighted_choice(rng, states, state_weights, n)

    lat, lon = _place_points(
        rng, chosen_states, _metro_centers(config),
        _STATION_SPREAD_DEG, _STATION_RURAL_FRACTION,
    )

    networks = _weighted_choice(rng, list(NETWORKS), list(NETWORKS.values()), n)
    is_dc_fast = rng.random(n) < config.dc_fast_fraction

    level2_ports = np.where(is_dc_fast, rng.integers(0, 3, n), rng.integers(1, 9, n))
    dcfast_ports = np.where(is_dc_fast, rng.integers(1, 7, n), 0)

    # Deployment dates spread across the configured window.
    start = np.datetime64(f"{config.start_year}-01-01")
    end = np.datetime64(f"{config.end_year}-06-30")
    span_days = (end - start).astype(int)
    open_dates = start + rng.integers(0, span_days, n).astype("timedelta64[D]")

    connectors = [
        ";".join(
            [_DCFAST_CONNECTORS[0] if net != "Tesla" else "TESLA"]
            if dc
            else _L2_CONNECTORS
        )
        for net, dc in zip(networks, is_dc_fast)
    ]

    df = pd.DataFrame(
        {
            "station_name": [f"{net} Station {i:04d}" for i, net in enumerate(networks)],
            "city": [f"{st} Metro {i % 50:02d}" for i, st in enumerate(chosen_states)],
            "state": chosen_states,
            "latitude": np.round(lat, 6),
            "longitude": np.round(lon, 6),
            "ev_network": networks,
            "ev_level2_evse_num": level2_ports.astype(int),
            "ev_dc_fast_num": dcfast_ports.astype(int),
            "ev_connector_types": connectors,
            "open_date": pd.to_datetime(open_dates),
            "access_code": rng.choice(
                ["public", "private"], size=n, p=[0.88, 0.12]
            ),
        }
    )
    df["charging_level"] = np.where(df["ev_dc_fast_num"] > 0, DC_FAST, LEVEL_2)
    return df


def generate_demand_points(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """Generate synthetic EV-demand hotspots (proxy for density/traffic).

    Points cluster around random urban centers within each state, weighted by
    population, each carrying a ``demand_weight``. These are the reference
    points against which charging deserts are evaluated.
    """
    config = config or DataConfig()
    # Offset the seed so demand points are not co-located with stations.
    seed = None if config.random_seed is None else config.random_seed + 1000
    rng = np.random.default_rng(seed)
    m = config.n_demand_points

    states = [s for s in config.states if s in STATE_BOXES]
    state_weights = np.array([STATE_BOXES[s][4] for s in states])
    chosen_states = _weighted_choice(rng, states, state_weights, m)

    # Demand clusters around the same metro centers as stations, but with a
    # wider spread and a larger rural share so genuine deserts appear.
    lat, lon = _place_points(
        rng, chosen_states, _metro_centers(config),
        _DEMAND_SPREAD_DEG, _DEMAND_RURAL_FRACTION,
    )

    return pd.DataFrame(
        {
            "point_id": np.arange(m),
            "state": chosen_states,
            "latitude": np.round(lat, 6),
            "longitude": np.round(lon, 6),
            "demand_weight": np.round(rng.lognormal(mean=1.0, sigma=0.5, size=m), 3),
        }
    )


def load_stations(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """Public entry point: load the (synthetic) station dataset."""
    return generate_synthetic_stations(config)


def load_demand_points(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """Public entry point: load the (synthetic) demand hotspots."""
    return generate_demand_points(config)


# --------------------------------------------------------------------------- #
# Preprocessing / filtering
# --------------------------------------------------------------------------- #
def filter_stations(
    stations: pd.DataFrame,
    states: Optional[Iterable[str]] = None,
    levels: Optional[Iterable[str]] = None,
    networks: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter stations by state, charging level and/or network.

    A station matches a ``levels`` selection if it has at least one port of the
    requested type (a mixed station can satisfy both ``Level 2`` and ``DC Fast``).
    ``None``/empty for any facet means "no filter on that facet".
    """
    df = stations
    if states:
        df = df[df["state"].isin(set(states))]
    if networks:
        df = df[df["ev_network"].isin(set(networks))]
    if levels:
        levels = set(levels)
        mask = pd.Series(False, index=df.index)
        if LEVEL_2 in levels:
            mask |= df["ev_level2_evse_num"] > 0
        if DC_FAST in levels:
            mask |= df["ev_dc_fast_num"] > 0
        df = df[mask]
    return df.reset_index(drop=True)


def to_geodataframe(stations: pd.DataFrame):
    """Convert a station/point DataFrame to a GeoDataFrame (EPSG:4326).

    Imported lazily so the core data path does not hard-require GeoPandas.
    """
    import geopandas as gpd

    return gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    )
