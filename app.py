"""Streamlit entry point — Spatial-Temporal EV Infrastructure Analyzer.

Run with::

    streamlit run app.py

Thin presentation layer: all data, spatial analytics and figure construction
live in :mod:`src`. This module wires sidebar controls to those functions and
lays out the map, metric cards and charts.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.data_loader import (
    CHARGING_LEVELS,
    NETWORKS,
    STATE_BOXES,
    DataConfig,
    filter_stations,
    load_demand_points,
    load_stations,
)
from src.spatial_analysis import (
    compute_summary,
    deployment_trend,
    detect_charging_deserts,
)
from src.visualization import (
    build_station_map,
    deployment_trend_chart,
    level_split_chart,
    network_breakdown_chart,
)

st.set_page_config(
    page_title="EV Charging Desert Analyzer",
    page_icon="🔌",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_data(seed: int, n_stations: int, n_demand: int):
    """Load and cache the synthetic station + demand datasets."""
    cfg = DataConfig(n_stations=n_stations, n_demand_points=n_demand, random_seed=seed)
    return load_stations(cfg), load_demand_points(cfg)


def sidebar_controls():
    """Render sidebar filters; return the user's selections."""
    st.sidebar.header("🔎 Filters")

    states = st.sidebar.multiselect(
        "State / region", options=sorted(STATE_BOXES), default=[],
        help="Leave empty to include all states.",
    )
    levels = st.sidebar.multiselect(
        "Charging level", options=list(CHARGING_LEVELS), default=[],
        help="Leave empty to include all levels.",
    )
    networks = st.sidebar.multiselect(
        "Network provider", options=list(NETWORKS), default=[],
        help="Leave empty to include all networks.",
    )

    st.sidebar.header("🗺️ Map & Analysis")
    show_heatmap = st.sidebar.toggle(
        "Show density heatmap", value=False,
        help="Toggle between clustered markers and a station-density heatmap.",
    )
    show_deserts = st.sidebar.toggle("Highlight charging deserts", value=True)
    threshold_km = st.sidebar.slider(
        "Desert threshold (km)", 2.0, 50.0, 10.0, step=1.0,
        help="A demand point farther than this from any station is a 'desert'.",
    )

    st.sidebar.header("🧪 Synthetic Dataset")
    n_stations = st.sidebar.slider("Number of stations", 200, 3000, 1200, step=100)
    n_demand = st.sidebar.slider("Number of demand points", 100, 1000, 400, step=50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    return {
        "states": states,
        "levels": levels,
        "networks": networks,
        "show_heatmap": show_heatmap,
        "show_deserts": show_deserts,
        "threshold_km": threshold_km,
        "n_stations": n_stations,
        "n_demand": n_demand,
        "seed": int(seed),
    }


def render_metrics(summary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total stations", f"{summary['total_stations']:,}")
    c2.metric("DC-fast station ratio", f"{summary['dc_fast_station_ratio']:.0%}")
    c3.metric("Charging deserts", f"{summary['n_desert_points']:,}")
    c4.metric("Desert share", f"{summary['desert_ratio']:.0%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Level 2 ports", f"{summary['total_level2_ports']:,}")
    c6.metric("DC-fast ports", f"{summary['total_dcfast_ports']:,}")
    c7.metric("Networks", f"{summary['n_networks']}")
    c8.metric("Mean nearest station", f"{summary['mean_nearest_km']:.1f} km")


def main() -> None:
    st.title("🔌 Spatial-Temporal EV Infrastructure & Charging Desert Analyzer")
    st.caption(
        "Explore EV charging-station coverage across the U.S., track deployment "
        "over time, and detect **charging deserts** — demand areas left "
        "underserved by nearby infrastructure."
    )

    controls = sidebar_controls()
    stations_all, demand = _load_data(
        controls["seed"], controls["n_stations"], controls["n_demand"]
    )

    stations = filter_stations(
        stations_all,
        states=controls["states"] or None,
        levels=controls["levels"] or None,
        networks=controls["networks"] or None,
    )

    deserts = detect_charging_deserts(demand, stations, controls["threshold_km"])
    summary = compute_summary(stations, deserts)

    render_metrics(summary)
    st.divider()

    st.subheader("🗺️ Coverage Map")
    if controls["show_heatmap"]:
        st.caption("Heatmap of station density. Red points mark charging deserts.")
    else:
        st.caption("Clustered stations (blue = Level 2, red = DC Fast). Dark-red points mark charging deserts.")
    fmap = build_station_map(
        stations,
        deserts=deserts if controls["show_deserts"] else None,
        show_heatmap=controls["show_heatmap"],
    )
    st_folium(fmap, use_container_width=True, height=520, returned_objects=[])
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("📈 Deployment Over Time")
        st.plotly_chart(
            deployment_trend_chart(deployment_trend(stations)),
            use_container_width=True,
        )
    with right:
        st.subheader("🏷️ Stations by Network")
        st.plotly_chart(network_breakdown_chart(stations), use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("⚡ Level 2 vs DC Fast")
        st.plotly_chart(level_split_chart(stations), use_container_width=True)
    with right2:
        st.subheader("🏜️ Deserts by State")
        if summary["n_desert_points"] > 0:
            by_state = (
                deserts[deserts["is_desert"]]
                .groupby("state")
                .size()
                .sort_values(ascending=False)
                .rename("desert_points")
                .reset_index()
            )
            st.dataframe(by_state, use_container_width=True, hide_index=True)
        else:
            st.success("No charging deserts detected for the current selection.")

    with st.expander("📄 Filtered station table"):
        st.dataframe(stations, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
