"""Spatial-Temporal EV Infrastructure & Charging Desert Analyzer — core package.

Public surface:
    - Data:      :mod:`src.data_loader` (synthetic AFDC-schema generator, filters)
    - Analytics: :mod:`src.spatial_analysis` (haversine, deserts, clustering, KPIs)
    - Visuals:   :mod:`src.visualization` (Folium maps, Plotly charts)
"""

from .data_loader import (
    CHARGING_LEVELS,
    DC_FAST,
    LEVEL_2,
    NETWORKS,
    STATE_BOXES,
    DataConfig,
    filter_stations,
    load_demand_points,
    load_stations,
    to_geodataframe,
)
from .spatial_analysis import (
    EARTH_RADIUS_KM,
    cluster_stations,
    compute_summary,
    deployment_trend,
    detect_charging_deserts,
    haversine_km,
    nearest_station_distances,
)

__all__ = [
    "CHARGING_LEVELS",
    "DC_FAST",
    "LEVEL_2",
    "NETWORKS",
    "STATE_BOXES",
    "DataConfig",
    "filter_stations",
    "load_demand_points",
    "load_stations",
    "to_geodataframe",
    "EARTH_RADIUS_KM",
    "cluster_stations",
    "compute_summary",
    "deployment_trend",
    "detect_charging_deserts",
    "haversine_km",
    "nearest_station_distances",
]

__version__ = "1.0.0"
