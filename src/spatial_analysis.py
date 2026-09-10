"""Spatial proximity, clustering and charging-desert detection.

This module holds the mathematical core of the application and is deliberately
free of any UI or plotting concern. All distances are great-circle (haversine)
distances in **kilometres**.

Methodology (charging deserts)
------------------------------
For every demand point *d* we find the distance to its nearest charging
station, ``nn_km(d) = min over stations s of haversine(d, s)``. A demand point
is classified as lying in a **charging desert** when that nearest-station
distance exceeds a coverage threshold ``T``::

    is_desert(d) = nn_km(d) > T

This is equivalent to a buffer-coverage test: draw a circle of radius ``T``
around every station; any demand point falling outside the union of those
circles is uncovered. We compute it via a :class:`~sklearn.neighbors.BallTree`
with the haversine metric for O(m log n) nearest-neighbour queries.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

#: Mean Earth radius (km), IUGG value.
EARTH_RADIUS_KM: float = 6371.0088


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in km between two (arrays of) lat/lon points.

    Inputs are in decimal degrees and broadcast against one another.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _latlon_radians(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))


def nearest_station_distances(
    points: pd.DataFrame, stations: pd.DataFrame
) -> np.ndarray:
    """Distance (km) from each point to its nearest station.

    Returns an array of length ``len(points)``; entries are ``inf`` when there
    are no stations to compare against.
    """
    if len(points) == 0:
        return np.array([], dtype=float)
    if len(stations) == 0:
        return np.full(len(points), np.inf)

    tree = BallTree(_latlon_radians(stations), metric="haversine")
    dist_rad, _ = tree.query(_latlon_radians(points), k=1)
    return dist_rad[:, 0] * EARTH_RADIUS_KM


def detect_charging_deserts(
    demand_points: pd.DataFrame,
    stations: pd.DataFrame,
    threshold_km: float = 10.0,
) -> pd.DataFrame:
    """Label demand points as in/out of a charging desert.

    Args:
        demand_points: Points with ``latitude``/``longitude`` (and optional
            ``demand_weight``).
        stations: Charging stations with ``latitude``/``longitude``.
        threshold_km: Coverage radius; points farther than this from any
            station are deserts.

    Returns:
        A copy of ``demand_points`` with added ``nearest_station_km`` and
        ``is_desert`` columns.
    """
    if threshold_km <= 0:
        raise ValueError("threshold_km must be positive.")

    out = demand_points.copy()
    nn = nearest_station_distances(demand_points, stations)
    out["nearest_station_km"] = np.round(nn, 3)
    out["is_desert"] = nn > threshold_km
    return out


def cluster_stations(
    stations: pd.DataFrame, eps_km: float = 25.0, min_samples: int = 5
) -> pd.DataFrame:
    """Cluster stations by geographic density using haversine DBSCAN.

    Returns a copy of ``stations`` with a ``cluster`` label column
    (``-1`` denotes noise / sparse stations that form no cluster).
    """
    out = stations.copy()
    if len(stations) == 0:
        out["cluster"] = pd.Series(dtype=int)
        return out
    if eps_km <= 0:
        raise ValueError("eps_km must be positive.")

    labels = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
    ).fit_predict(_latlon_radians(stations))
    out["cluster"] = labels
    return out


class CoverageSummary(TypedDict):
    """Headline metrics for a station selection and its desert analysis."""

    total_stations: int
    total_level2_ports: int
    total_dcfast_ports: int
    dc_fast_station_ratio: float
    fast_charger_port_ratio: float
    n_networks: int
    n_demand_points: int
    n_desert_points: int
    desert_ratio: float
    mean_nearest_km: float


def compute_summary(
    stations: pd.DataFrame, deserts: Optional[pd.DataFrame] = None
) -> CoverageSummary:
    """Compute headline KPIs for a station set and (optional) desert analysis."""
    total = int(len(stations))
    l2_ports = int(stations["ev_level2_evse_num"].sum()) if total else 0
    dc_ports = int(stations["ev_dc_fast_num"].sum()) if total else 0
    dc_stations = int((stations["ev_dc_fast_num"] > 0).sum()) if total else 0
    total_ports = l2_ports + dc_ports

    n_demand = int(len(deserts)) if deserts is not None else 0
    n_desert = int(deserts["is_desert"].sum()) if deserts is not None and n_demand else 0
    mean_nn = (
        float(deserts["nearest_station_km"].replace(np.inf, np.nan).mean())
        if deserts is not None and n_demand
        else 0.0
    )

    return CoverageSummary(
        total_stations=total,
        total_level2_ports=l2_ports,
        total_dcfast_ports=dc_ports,
        dc_fast_station_ratio=(dc_stations / total) if total else 0.0,
        fast_charger_port_ratio=(dc_ports / total_ports) if total_ports else 0.0,
        n_networks=int(stations["ev_network"].nunique()) if total else 0,
        n_demand_points=n_demand,
        n_desert_points=n_desert,
        desert_ratio=(n_desert / n_demand) if n_demand else 0.0,
        mean_nearest_km=round(mean_nn, 3) if not np.isnan(mean_nn) else 0.0,
    )


def deployment_trend(stations: pd.DataFrame, freq: str = "YS") -> pd.DataFrame:
    """Cumulative station deployment over time.

    Returns a DataFrame with columns ``period``, ``new_stations`` and
    ``cumulative_stations``, resampled at ``freq`` (default: year-start).
    """
    columns = ["period", "new_stations", "cumulative_stations"]
    if len(stations) == 0 or "open_date" not in stations:
        return pd.DataFrame(columns=columns)

    s = pd.to_datetime(stations["open_date"])
    counts = (
        s.dt.to_period(_freq_to_period(freq))
        .value_counts()
        .sort_index()
    )
    trend = counts.rename("new_stations").to_frame()
    trend["cumulative_stations"] = trend["new_stations"].cumsum()
    trend = trend.reset_index(names="period")
    trend["period"] = trend["period"].astype(str)
    return trend[columns]


def _freq_to_period(freq: str) -> str:
    """Map a resample freq alias to a Period alias (YS->Y, MS->M)."""
    return {"YS": "Y", "MS": "M", "Y": "Y", "M": "M"}.get(freq, "Y")
