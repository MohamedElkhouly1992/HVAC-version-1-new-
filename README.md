# MOAPO–CatBoost HVAC Research Engine v1.0

A deployable Streamlit research application for the next HVAC manuscript. It accepts control-aware DesignBuilder/EnergyPlus/strategy datasets, trains CatBoost surrogate models, defines four objectives, and compares MOAPO-HVAC, NSGA-II, and MOPSO.

## Four minimization objectives

1. HVAC energy consumption
2. Occupant thermal discomfort (PPD or your mapped discomfort metric)
3. Degradation state / next-step composite degradation index
4. Economic cost = energy cost + maintenance trigger cost + optional downtime cost (or an observed uploaded cost target)

## What is automatically exported

- `Table_surrogate_metrics.csv`
- `surrogate_test_predictions.csv`
- `Table_all_Pareto_solutions.csv`
- `Table_optimizer_benchmark.csv`
- `Table_optimizer_runs.csv` and non-parametric optimizer statistics when repetitions ≥ 3
- `Table_representative_Pareto_solutions.csv`
- `Table_strategy_severity_summary.csv` when S0–S3 data are available
- optimizer convergence history
- selected optimization context and control bounds
- CatBoost parity figures at 300 dpi
- energy-vs-discomfort Pareto figure
- four-objective parallel-coordinate figure
- optimizer hypervolume convergence figure
- strategy × severity energy figure when available
- reproducibility manifest
- automatically generated manuscript-results summary

## Required data structure for a valid control-optimization study

The software can read CSV and Excel files and can concatenate multiple S0/S1/S2/S3 files. However, a valid surrogate-based controller needs **control-aware rows**, not only annual/daily totals. Each training observation should include current state/context, candidate control values, and the resulting target response.

Recommended minimum columns:

```text
context_id
timestamp
strategy
severity
outdoor_drybulb_C
outdoor_rh_pct
zone_people_count
zone_air_temp_C
fouling_resistance_m2K_W
filter_dp_Pa
delta_current
candidate_zone_setpoint_C
candidate_airflow_fraction
target_hvac_energy_kWh
target_ppd_pct
target_delta_next
```

Optional additional controls:

```text
candidate_chw_setpoint_C
candidate_pump_speed_fraction
```

The UI lets you map different column names, so the exact names above are recommended rather than mandatory.

## Local deployment

Windows: double-click `run_local.bat`.

Or:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud deployment

1. Put the files in a GitHub repository.
2. Create a Streamlit app from the repository.
3. Set the main file to `streamlit_app.py`.
4. No secrets are required.

The included `.streamlit/config.toml` raises the upload limit to 2 GB.

## Command-line test / reproducible run

The core engine can run without the UI:

```bash
python engine.py --demo --out paper_results --population 30 --iterations 25 --repetitions 3
```

Or with your data:

```bash
python engine.py --input control_surrogate_dataset.csv --out paper_results --population 40 --iterations 50 --repetitions 30
```

If your columns differ from the default names, use the Streamlit UI for mapping.

## MOAPO methodology note

`run_moapo()` is a documented **Pareto extension for this HVAC research software**. It retains the APO behavioral classes—autotrophic/heterotrophic foraging, dormancy, and reproduction—and replaces a scalar global-best mechanism with an external non-dominated archive plus diversity-biased leader selection.

It is intentionally labelled **MOAPO-HVAC**. Do not write in the manuscript that it is a bit-for-bit implementation of a specific published MOAPO variant. In the methodology, state the exact operators implemented in `engine.py` and cite the original APO paper:

Wang X. et al., *Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization*, Knowledge-Based Systems, 295 (2024), 111737. DOI: 10.1016/j.knosys.2024.111737.

## Publication safeguards

- Do not call synthetic demonstration results publishable evidence.
- Do not treat annual S0–S3 summary data as a control-aware surrogate dataset unless control inputs vary within the data.
- Final representative Pareto controls should be replayed through the original physics/ROM/EnergyPlus/DesignBuilder model.
- Report surrogate prediction metrics and objective-wise optimization regret.
- Use repeated independent optimizer runs for final statistics. The app supports 1–30 independent seeded runs per optimizer. Use 1–3 for debugging and 30 for the final publication benchmark when computationally feasible.
