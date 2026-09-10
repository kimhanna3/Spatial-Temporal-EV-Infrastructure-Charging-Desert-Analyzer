# 🔌 Spatial-Temporal EV Infrastructure & Charging Desert Analyzer

An interactive **geospatial analytics** application that maps electric-vehicle
(EV) charging infrastructure across the U.S., tracks how it has been deployed
over time, and **automatically detects "charging deserts"** — demand areas left
underserved by nearby stations.

Filter by state, charging level and network; toggle between a station-density
heatmap and coverage-gap view; and read off headline metrics — all from a live
Streamlit dashboard that runs **fully offline** on a synthetic dataset modelled
on NREL's Alternative Fuel Data Center (AFDC) schema.

> **Stack:** Python 3.10+ · Streamlit · GeoPandas · Folium · Streamlit-Folium ·
> Plotly · Scikit-learn · Shapely · Pandas · Pytest

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Methodology: How Charging Deserts Are Calculated](#-methodology-how-charging-deserts-are-calculated)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Running the Tests](#-running-the-tests)
- [Screenshots](#-screenshots)
- [Configuration Reference](#-configuration-reference)

---

## 🔍 Overview

EV adoption is racing ahead of charging infrastructure, and coverage is deeply
uneven. This tool makes that gap measurable and visible. It:

- visualises the **spatial distribution** of charging stations,
- reconstructs **historical deployment trends** from station open-dates, and
- runs a **spatial proximity algorithm** to flag demand areas that are too far
  from any charger — the *charging deserts*.

Because it ships with a synthetic data generator that mirrors the AFDC export
schema, the app runs out-of-the-box with **no API keys, downloads or data
files**.

---

## ✨ Features

- 🗺️ **Interactive Folium map** with clustered station markers (colour-coded by
  charging level) and a togglable **density heatmap**.
- 🏜️ **Charging-desert overlay** highlighting underserved demand points in red.
- 🎛️ **Filters** for state/region, charging level (Level 2 / DC Fast) and
  network provider.
- 📊 **Metric cards**: total stations, DC-fast ratio, identified gap zones,
  ports by level, mean distance to nearest station.
- 📈 **Temporal analytics**: cumulative + per-period deployment trend.
- 🏷️ **Provider & level breakdowns** (Plotly bar / donut charts).
- 🧮 **Vectorised, tested spatial core** (haversine + `BallTree`), decoupled
  from the UI.

---

## 🧮 Methodology: How Charging Deserts Are Calculated

The analysis reduces coverage to a single, well-defined geometric question:
**how far is each unit of EV demand from its nearest charger?**

**1. Great-circle distance.** All distances use the **haversine** formula on
WGS84 coordinates, which gives the distance along the Earth's surface between
two points `(φ₁, λ₁)` and `(φ₂, λ₂)`:

```
a = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
d = 2R · arcsin(√a)          where R = 6371.0088 km (mean Earth radius)
```

**2. Nearest-station distance.** Demand hotspots (a synthetic proxy for EV
density / traffic) are the reference points. For each demand point *d* we
compute the distance to its nearest station:

```
nn_km(d) = min over all stations s of  haversine(d, s)
```

Nearest neighbours are found with a **`sklearn.neighbors.BallTree` using the
haversine metric** (coordinates in radians, distances × R), giving
`O(m · log n)` queries instead of a brute-force `O(m · n)` scan.

**3. Desert classification.** Given a coverage radius **T** (the user-set
threshold, default **10 km**), a demand point is a charging desert when its
nearest station lies beyond that radius:

```
is_desert(d) = nn_km(d) > T
```

This is mathematically equivalent to a **buffer-coverage test**: draw a circle
of radius *T* around every station; any demand point outside the union of those
circles is uncovered. As *T* grows, the desert set can only shrink — a
monotonicity property that is explicitly asserted in the test suite.

**4. Density clustering (optional).** Station agglomerations are found with
**haversine DBSCAN** (`eps` expressed in km ÷ R), useful for characterising
well-served corridors versus sparse regions.

---

## 🏗 Architecture

Spatial computation is cleanly separated from presentation:

```
┌───────────────────────────────────────────────────────────────┐
│                         app.py (Streamlit)                      │
│   sidebar filters · metric cards · Folium map · Plotly charts   │
└───────────┬───────────────────────┬───────────────────┬────────┘
            │ DataConfig / filters   │ analytics          │ figures
            ▼                        ▼                     ▼
┌────────────────────┐  ┌───────────────────────┐  ┌────────────────────┐
│  src/data_loader.py│  │ src/spatial_analysis.py│  │ src/visualization.py│
│  synthetic AFDC gen│  │ haversine · BallTree   │  │ Folium maps +       │
│  + preprocessing/  │  │ desert detection ·     │  │ HeatMap · Plotly    │
│  filtering         │  │ DBSCAN · KPIs · trends │  │ charts              │
└────────────────────┘  └───────────────────────┘  └────────────────────┘
```

- **`data_loader.py`** — synthetic AFDC-schema station & demand generation
  (seeded, reproducible), plus state/level/network filtering and a
  GeoPandas conversion helper.
- **`spatial_analysis.py`** — the pure geospatial core: haversine distances,
  nearest-neighbour queries, desert detection, DBSCAN clustering, KPI and
  deployment-trend computation.
- **`visualization.py`** — Folium map rendering (markers, clusters, heatmap,
  desert overlay) and Plotly charts.
- **`app.py`** — thin Streamlit layer wiring controls to the modules above.

---

## 📁 Project Structure

```
Spatial-Temporal-EV-Infrastructure-Charging-Desert-Analyzer/
├── app.py                  # Streamlit app entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data ingestion, synthetic AFDC generator, filters
│   ├── spatial_analysis.py # Proximity, clustering & desert-detection algorithms
│   └── visualization.py    # Folium map rendering & Plotly components
├── tests/
│   ├── __init__.py
│   └── test_spatial.py     # pytest unit tests for spatial calculations
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Requires **Python 3.10+**. GeoPandas/Shapely wheels install cleanly on modern
pip; no system GDAL is required for this project (no file-based I/O is used).

```bash
# 1. Clone
git clone https://github.com/kimhanna3/Spatial-Temporal-EV-Infrastructure-Charging-Desert-Analyzer.git
cd Spatial-Temporal-EV-Infrastructure-Charging-Desert-Analyzer

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the App

From the repository root:

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Adjust filters and the desert
threshold in the sidebar; the map, metrics and charts update live.

---

## 🧪 Running the Tests

```bash
pytest -v
```

The suite verifies the haversine metric against known distances (equator
degree, NYC↔LA, symmetry, vectorisation), nearest-neighbour correctness, desert
classification and its monotonicity in the threshold, all filtering paths, the
generated schema/bounds, determinism, and KPI/trend calculations.

---

## 🖼 Screenshots

> _Placeholders — add exported images after your first run._

| Coverage map + deserts | Deployment trend | Metric cards |
|---|---|---|
| `docs/screenshot_map.png` | `docs/screenshot_trend.png` | `docs/screenshot_metrics.png` |

---

## 📖 Configuration Reference

Dataset generation is controlled by `DataConfig` (`src/data_loader.py`), exposed
via sidebar controls:

| Parameter | Default | Description |
|---|---|---|
| `n_stations` | 1200 | Number of synthetic charging stations. |
| `n_demand_points` | 400 | Number of EV-demand hotspots. |
| `dc_fast_fraction` | 0.35 | Share of stations that are DC fast. |
| `start_year` / `end_year` | 2015 / 2025 | Deployment-date window. |
| `random_seed` | 42 | Seed for reproducibility. |
| `states` | all 12 | CONUS states included in generation. |

Analysis parameters (set in the UI): **desert threshold** (km), **heatmap
toggle**, and **desert overlay toggle**.

---

## 📝 License

Released under the MIT License — free to use, modify and distribute.

_Synthetic data only: this project ships no real NREL data. The schema mirrors
the public AFDC station export so it can be swapped for the live feed._
