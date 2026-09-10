"""Map rendering (Folium) and chart components (Plotly).

Kept separate from analytics so that visual concerns never leak into the
spatial computations. Functions return ready-to-render objects
(:class:`folium.Map`, :class:`plotly.graph_objects.Figure`) which the Streamlit
layer displays.
"""

from __future__ import annotations

from typing import Optional

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap, MarkerCluster

from .data_loader import DC_FAST

# CONUS centroid & default zoom for the base map.
_CONUS_CENTER = (39.5, -98.35)
_DEFAULT_ZOOM = 4

_LEVEL_COLOR = {"DC Fast": "#E45756", "Level 2": "#4C78A8"}


def _base_map() -> folium.Map:
    return folium.Map(
        location=list(_CONUS_CENTER),
        zoom_start=_DEFAULT_ZOOM,
        tiles="OpenStreetMap",  # free, no API key required
        control_scale=True,
    )


def build_station_map(
    stations: pd.DataFrame,
    deserts: Optional[pd.DataFrame] = None,
    show_heatmap: bool = False,
    max_markers: int = 1500,
) -> folium.Map:
    """Render stations (and optional desert points) on an interactive map.

    Args:
        stations: Filtered station set to plot.
        deserts: Optional demand points with an ``is_desert`` flag; desert
            points are drawn as red markers.
        show_heatmap: If True, overlay a station-density heatmap instead of
            individual clustered markers.
        max_markers: Cap on individually-drawn markers (protects the browser).
    """
    fmap = _base_map()

    if show_heatmap and len(stations) > 0:
        HeatMap(
            stations[["latitude", "longitude"]].to_numpy().tolist(),
            radius=12,
            blur=18,
            name="Station density",
        ).add_to(fmap)
    elif len(stations) > 0:
        cluster = MarkerCluster(name="Charging stations").add_to(fmap)
        plotted = stations if len(stations) <= max_markers else stations.sample(
            max_markers, random_state=0
        )
        for row in plotted.itertuples(index=False):
            color = _LEVEL_COLOR.get(row.charging_level, "#4C78A8")
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{row.station_name}</b><br>"
                    f"{row.ev_network}<br>"
                    f"{row.charging_level}<br>"
                    f"L2 ports: {row.ev_level2_evse_num} · "
                    f"DCFC ports: {row.ev_dc_fast_num}",
                    max_width=250,
                ),
            ).add_to(cluster)

    if deserts is not None and "is_desert" in deserts:
        desert_pts = deserts[deserts["is_desert"]]
        if len(desert_pts) > 0:
            gap_layer = folium.FeatureGroup(name="Charging deserts").add_to(fmap)
            for row in desert_pts.itertuples(index=False):
                folium.CircleMarker(
                    location=(row.latitude, row.longitude),
                    radius=5,
                    color="#8B0000",
                    fill=True,
                    fill_color="#FF3333",
                    fill_opacity=0.6,
                    popup=f"Charging desert — nearest station "
                    f"{getattr(row, 'nearest_station_km', float('nan')):.1f} km",
                ).add_to(gap_layer)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def deployment_trend_chart(trend: pd.DataFrame) -> go.Figure:
    """Line+bar chart of new and cumulative station deployment over time."""
    fig = go.Figure()
    if not trend.empty:
        fig.add_bar(
            x=trend["period"], y=trend["new_stations"],
            name="New stations", marker_color="#B279A2",
        )
        fig.add_trace(
            go.Scatter(
                x=trend["period"], y=trend["cumulative_stations"],
                name="Cumulative", mode="lines+markers",
                line=dict(color="#4C78A8", width=3), yaxis="y2",
            )
        )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Period", yaxis=dict(title="New"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def network_breakdown_chart(stations: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of station counts per network provider."""
    if stations.empty:
        return go.Figure()
    counts = stations["ev_network"].value_counts().sort_values()
    fig = px.bar(
        x=counts.to_numpy(), y=counts.index.tolist(), orientation="h",
        labels={"x": "Stations", "y": "Network"},
        color=counts.to_numpy(), color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_showscale=False,
    )
    return fig


def level_split_chart(stations: pd.DataFrame) -> go.Figure:
    """Donut chart of Level 2 vs DC Fast stations."""
    if stations.empty:
        return go.Figure()
    counts = stations["charging_level"].value_counts()
    fig = px.pie(
        names=counts.index.tolist(), values=counts.to_numpy(), hole=0.55,
        color=counts.index.tolist(), color_discrete_map=_LEVEL_COLOR,
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig
