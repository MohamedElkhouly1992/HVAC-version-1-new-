# Deployment checklist

## Fast local deployment

### Windows
Run `run_local.bat`.

### macOS/Linux
```bash
chmod +x run_local.sh
./run_local.sh
```

## Streamlit Community Cloud

- Main file: `streamlit_app.py`
- Python requirements: `requirements.txt`
- Secrets: none
- Upload cap: configured to 2048 MB in `.streamlit/config.toml`

## First real-data run

Start with:
- population = 18–30
- iterations = 10–25
- independent runs = 1–3

After verifying the schema, control bounds, CatBoost validation, and Pareto plots, use a publication run such as:
- population = 40–60
- iterations = 50–100
- independent runs = 30
- archive = 100–200

The exact final settings must be reported in the manuscript and checked for convergence sensitivity.

## Minimum validity checks before using the generated values in Paper 1

1. CatBoost test metrics are acceptable for all three response targets.
2. The training data contain varying control variables and not only strategy totals.
3. Control bounds reflect actual HVAC operating limits.
4. Pareto results are stable across repeated seeds.
5. Representative Pareto/knee solutions are replayed in the original ROM/EnergyPlus/DesignBuilder model.
6. Surrogate-vs-physics objective errors and optimization regret are reported.
7. Maintenance costs and trigger assumptions are documented, not silently treated as measured facts.
