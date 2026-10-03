from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from engine import (
    ALIASES,
    ResearchConfig,
    harmonize_frames,
    infer_column,
    infer_schema,
    make_demo_dataset,
    read_any_file,
    run_paper_pipeline,
    zip_results,
)

st.set_page_config(page_title="MOAPO–CatBoost HVAC Research Engine", page_icon="🧪", layout="wide")

st.title("MOAPO–CatBoost HVAC Research Engine")
st.caption("Surrogate-assisted 4-objective HVAC optimization: Energy · Thermal discomfort · Degradation · Economic cost")

with st.expander("Scientific scope and claim boundary", expanded=False):
    st.markdown(
        """
This software is intended for **research simulation and manuscript analysis**. It trains CatBoost response surrogates from uploaded DesignBuilder/EnergyPlus/strategy datasets and then compares **MOAPO-HVAC, NSGA-II, and MOPSO** in a common four-objective search space.

**Important:** the MOAPO-HVAC implementation retains APO behavior classes (autotrophic/heterotrophic foraging, dormancy, reproduction) and adds a Pareto archive and diversity-based leader selection. It is a documented Pareto adaptation for this HVAC study; it is **not claimed to be bit-for-bit identical to a previously published MOAPO variant**.

The program cannot make uploaded aggregate S0–S3 summaries into a valid control surrogate unless the data also contain varying **control inputs** (normally zone setpoint and airflow fraction) and corresponding response targets. Final Pareto controls should be re-evaluated with the original ROM/EnergyPlus/DesignBuilder model before publication.
        """
    )

# -----------------------------
# Data
# -----------------------------
st.sidebar.header("1 · Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV/XLSX files (multiple allowed)",
    type=["csv", "txt", "xlsx", "xls"],
    accept_multiple_files=True,
)
use_demo = st.sidebar.checkbox("Use synthetic demonstration data", value=not bool(uploaded))

if uploaded:
    frames = []
    errors = []
    for f in uploaded:
        try:
            frames.extend(read_any_file(f, f.name))
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    if errors:
        st.error("\n".join(errors))
    df = harmonize_frames(frames)
elif use_demo:
    df = make_demo_dataset()
else:
    st.info("Upload your DesignBuilder/strategy data or enable demonstration data.")
    st.stop()

if df.empty:
    st.error("No rows were loaded from the supplied files.")
    st.stop()

schema = infer_schema(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Strategies", df[schema["strategy"]].nunique() if schema.get("strategy") else "—")
c4.metric("Severities", df[schema["severity"]].nunique() if schema.get("severity") else "—")

with st.expander("Loaded data preview", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Automatically inferred schema:", schema)

# -----------------------------
# Column mapping
# -----------------------------
st.sidebar.header("2 · Column mapping")
cols = list(df.columns)

def idx_or_zero(val):
    try:
        return cols.index(val) if val in cols else 0
    except Exception:
        return 0

energy_default = schema.get("energy") or cols[0]
comfort_default = schema.get("comfort") or cols[0]
deg_default = schema.get("degradation") or cols[0]

energy_col = st.sidebar.selectbox("Energy target", cols, index=idx_or_zero(energy_default))
comfort_col = st.sidebar.selectbox("Comfort/discomfort target", cols, index=idx_or_zero(comfort_default))
degradation_col = st.sidebar.selectbox("Degradation target", cols, index=idx_or_zero(deg_default))

cost_options = ["<computed from energy + maintenance>"] + cols
cost_guess = schema.get("cost")
cost_index = cost_options.index(cost_guess) if cost_guess in cost_options else 0
cost_pick = st.sidebar.selectbox("Observed cost target (optional)", cost_options, index=cost_index)
cost_col = None if cost_pick.startswith("<") else cost_pick

control_candidates = [c for c in [schema.get("setpoint"), schema.get("airflow"), schema.get("chw"), schema.get("pump")] if c]
control_default = list(dict.fromkeys(control_candidates))[:2]
control_cols = st.sidebar.multiselect("Decision/control variables", cols, default=control_default)

if len(control_cols) < 2:
    st.warning("Select at least two decision variables, normally zone setpoint and airflow fraction.")

strategy_col = schema.get("strategy")
severity_col = schema.get("severity")
timestamp_col = schema.get("timestamp")
group_col = schema.get("group")

# -----------------------------
# Research configuration
# -----------------------------
st.sidebar.header("3 · Optimization")
population = st.sidebar.number_input("Population", 12, 150, 30, step=2)
iterations = st.sidebar.number_input("Iterations", 5, 250, 25, step=5)
archive_size = st.sidebar.number_input("Pareto archive size", 30, 500, 100, step=10)
repetitions = st.sidebar.number_input("Independent runs / optimizer", 1, 30, 3, step=1, help="Use 30 for final publication statistics; use 1–3 for a quick diagnostic run.")
seed = st.sidebar.number_input("Random seed", 0, 999999, 42)

st.sidebar.header("4 · Economics / maintenance")
tariff = st.sidebar.number_input("Electricity tariff [currency/kWh]", min_value=0.0, value=0.12, step=0.01)
maint_threshold = st.sidebar.number_input("Maintenance trigger δ_on", min_value=0.0, max_value=1.5, value=0.55, step=0.01)
maint_cost = st.sidebar.number_input("Maintenance cost per triggered interval", min_value=0.0, value=50.0, step=5.0)
downtime_cost = st.sidebar.number_input("Downtime cost per triggered interval", min_value=0.0, value=0.0, step=5.0)

st.sidebar.header("5 · CatBoost")
cb_iter = st.sidebar.number_input("Trees", 100, 2000, 350, step=50)
cb_depth = st.sidebar.slider("Depth", 4, 10, 7)
cb_lr = st.sidebar.number_input("Learning rate", min_value=0.005, max_value=0.30, value=0.04, step=0.005, format="%.3f")

# Control bounds
st.subheader("Control bounds")
bounds = {}
if control_cols:
    bcols = st.columns(min(4, len(control_cols)))
    for i, c in enumerate(control_cols):
        v = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)
        lo0 = float(v.quantile(0.01)) if v.notna().sum() and v.max() > v.min() else (22.0 if "setpoint" in c.lower() else 0.5)
        hi0 = float(v.quantile(0.99)) if v.notna().sum() and v.max() > v.min() else (26.0 if "setpoint" in c.lower() else 1.0)
        with bcols[i % len(bcols)]:
            st.markdown(f"**{c}**")
            lo = st.number_input(f"Lower · {c}", value=lo0, key=f"lo_{c}")
            hi = st.number_input(f"Upper · {c}", value=hi0, key=f"hi_{c}")
            bounds[c] = (float(lo), float(hi))

severity_choice = None
if severity_col and severity_col in df.columns:
    vals = [str(x) for x in df[severity_col].dropna().unique()]
    severity_choice = st.selectbox("Representative optimization context", ["All severities (median context)"] + vals)
    severity_choice = None if severity_choice.startswith("All") else severity_choice

cfg = ResearchConfig(
    energy_col=energy_col,
    comfort_col=comfort_col,
    degradation_col=degradation_col,
    cost_col=cost_col,
    control_cols=tuple(control_cols),
    strategy_col=strategy_col,
    severity_col=severity_col,
    timestamp_col=timestamp_col,
    group_col=group_col,
    electricity_tariff=float(tariff),
    maintenance_threshold=float(maint_threshold),
    maintenance_cost=float(maint_cost),
    downtime_cost=float(downtime_cost),
    population=int(population),
    iterations=int(iterations),
    archive_size=int(archive_size),
    repetitions=int(repetitions),
    seed=int(seed),
    catboost_iterations=int(cb_iter),
    catboost_depth=int(cb_depth),
    catboost_learning_rate=float(cb_lr),
    bounds=bounds or None,
)

st.subheader("Run paper pipeline")
st.write(
    "One run trains the three CatBoost surrogates, creates the four objectives, generates the Pareto archive, "
    "compares MOAPO / NSGA-II / MOPSO, and exports paper-ready CSV tables plus 300-dpi figures."
)

run = st.button("▶ Run MOAPO–CatBoost paper simulation", type="primary", use_container_width=True, disabled=len(control_cols) < 2)

if run:
    with st.spinner("Training surrogates and running three multi-objective optimizers…"):
        try:
            tmp = Path(tempfile.mkdtemp(prefix="moapo_hvac_"))
            results = run_paper_pipeline(df, cfg, tmp, context_severity=severity_choice)
            st.session_state["paper_results"] = results
            st.session_state["paper_dir"] = str(tmp)
            st.session_state["paper_zip"] = zip_results(tmp)
        except Exception as e:
            st.exception(e)

if "paper_results" in st.session_state:
    results = st.session_state["paper_results"]
    outdir = Path(st.session_state["paper_dir"])

    st.success("Simulation completed. Review the diagnostics before using any values in the manuscript.")

    st.subheader("A · CatBoost validation")
    st.dataframe(results["surrogate_metrics"], use_container_width=True)

    st.subheader("B · Optimizer benchmark")
    st.dataframe(results["benchmark"], use_container_width=True)

    if not results["optimizer_statistics"].empty:
        st.subheader("C · Repeated-run statistical tests")
        st.dataframe(results["optimizer_statistics"], use_container_width=True)

    st.subheader("D · Representative Pareto solutions")
    st.dataframe(results["representatives"], use_container_width=True)

    st.subheader("E · Pareto archive")
    st.dataframe(results["pareto"], use_container_width=True, height=380)

    if not results["strategy_summary"].empty:
        st.subheader("F · Uploaded S0–S3 / severity summary")
        st.dataframe(results["strategy_summary"], use_container_width=True)

    st.subheader("Generated manuscript figures")
    figs = sorted(outdir.glob("*.png"))
    for p in figs:
        st.image(str(p), caption=p.name, use_container_width=True)

    st.download_button(
        "⬇ Download complete paper-results package (.zip)",
        data=st.session_state["paper_zip"],
        file_name="MOAPO_CatBoost_HVAC_Paper_Results.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()
st.caption("Recommended publication workflow: surrogate screening → Pareto optimization → select representative/knee points → physics-model re-evaluation → report regret/robustness → manuscript claims.")
