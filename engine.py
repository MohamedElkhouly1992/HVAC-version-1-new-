from __future__ import annotations

import io
import json
import math
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class ResearchConfig:
    energy_col: str = "target_hvac_energy_kWh"
    comfort_col: str = "target_ppd_pct"
    degradation_col: str = "target_delta_next"
    cost_col: Optional[str] = None

    control_cols: Tuple[str, ...] = ("candidate_zone_setpoint_C", "candidate_airflow_fraction")
    strategy_col: Optional[str] = "strategy"
    severity_col: Optional[str] = "severity"
    timestamp_col: Optional[str] = "timestamp"
    group_col: Optional[str] = "context_id"

    electricity_tariff: float = 0.12
    maintenance_threshold: float = 0.55
    maintenance_cost: float = 50.0
    downtime_cost: float = 0.0
    maintenance_hysteresis_off: float = 0.45

    population: int = 36
    iterations: int = 35
    archive_size: int = 100
    repetitions: int = 3
    seed: int = 42
    test_fraction: float = 0.20
    catboost_iterations: int = 350
    catboost_depth: int = 7
    catboost_learning_rate: float = 0.04

    # If no user bounds are supplied, infer from data.
    bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # Objective labels in fixed order used throughout the software.
    objective_names: Tuple[str, ...] = ("Energy", "Discomfort", "Degradation", "Cost")


ALIASES = {
    "energy": [
        "target_hvac_energy_kWh", "hvac_total_energy_kWh", "hvac_energy_kwh", "energy_kwh",
        "energy", "total_energy", "total hvac energy", "hvac energy", "Energy (MWh)",
    ],
    "comfort": [
        "target_ppd_pct", "ppd_pct", "ppd", "comfort", "comfort_dev_C", "discomfort_days",
        "Discomfort Days", "Mean Comfort Deviation C", "occupied_discomfort_h",
    ],
    "degradation": [
        "target_delta_next", "delta_composite", "delta", "Mean Degradation Index",
        "composite_degradation_index", "degradation", "severity_index",
    ],
    "cost": ["total_operating_cost", "total_cost", "Total Cost (USD)", "cost", "operating_cost"],
    "setpoint": [
        "candidate_zone_setpoint_C", "zone_setpoint_C", "zone_cooling_setpoint_C", "Tsp", "tset",
        "setpoint", "setpoint_C",
    ],
    "airflow": [
        "candidate_airflow_fraction", "airflow_fraction", "alpha_flow", "air_cap", "flow_fraction",
    ],
    "chw": ["candidate_chw_setpoint_C", "chw_supply_setpoint_C", "CHWS_C", "chw_setpoint_C"],
    "pump": ["candidate_pump_speed_fraction", "pump_speed_fraction", "water_cap", "pump_fraction"],
    "strategy": ["strategy", "Strategy"],
    "severity": ["severity", "Severity"],
    "timestamp": ["timestamp", "datetime", "date_time", "Date/Time"],
    "group": ["context_id", "scenario_id", "simulation_id", "case", "scenario_key"],
}


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def infer_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    nmap = {_norm(c): c for c in columns}
    for a in aliases:
        if _norm(a) in nmap:
            return nmap[_norm(a)]
    # soft containment match only after exact normalized matching
    for a in aliases:
        na = _norm(a)
        if len(na) < 5:
            continue
        for c in columns:
            nc = _norm(c)
            if na in nc or nc in na:
                return c
    return None


def infer_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {k: infer_column(df.columns, v) for k, v in ALIASES.items()}


# -----------------------------
# Data loading / harmonization
# -----------------------------
def read_any_file(file_obj, name: str) -> List[pd.DataFrame]:
    """Read CSV or Excel. For Excel, return all non-empty sheets."""
    suffix = Path(name).suffix.lower()
    frames: List[pd.DataFrame] = []
    if suffix in {".csv", ".txt"}:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        try:
            df = pd.read_csv(file_obj)
        except UnicodeDecodeError:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="latin-1")
        df["__source_file"] = Path(name).name
        frames.append(df)
    elif suffix in {".xlsx", ".xls"}:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        xl = pd.ExcelFile(file_obj)
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            if len(df) == 0:
                continue
            df["__source_file"] = Path(name).name
            df["__sheet"] = sheet
            frames.append(df)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return frames


def harmonize_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    cleaned = []
    for i, f in enumerate(frames):
        x = f.copy()
        x = x.loc[:, [c for c in x.columns if not str(c).lower().startswith("unnamed")]]
        # derive strategy from common file names when absent
        if "strategy" not in x.columns and "__source_file" in x.columns:
            src = str(x["__source_file"].iloc[0]) if len(x) else ""
            for s in ["S0", "S1", "S2", "S3"]:
                if s.lower() in src.lower():
                    x["strategy"] = s
                    break
        x["__frame_id"] = i
        cleaned.append(x)
    if not cleaned:
        return pd.DataFrame()
    return pd.concat(cleaned, ignore_index=True, sort=False)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            # Avoid destroying obvious categoricals.
            converted = pd.to_numeric(out[c], errors="coerce")
            if converted.notna().mean() >= 0.85:
                out[c] = converted
    return out


# -----------------------------
# Surrogate models
# -----------------------------
class SurrogateBundle:
    def __init__(self, models: Dict[str, CatBoostRegressor], feature_cols: List[str],
                 categorical_cols: List[str], medians: Dict[str, float], categories: Dict[str, str]):
        self.models = models
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.medians = medians
        self.categories = categories

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        for c in self.feature_cols:
            if c in df.columns:
                x[c] = df[c]
            elif c in self.categorical_cols:
                x[c] = self.categories.get(c, "UNKNOWN")
            else:
                x[c] = self.medians.get(c, 0.0)
        for c in self.categorical_cols:
            x[c] = x[c].fillna(self.categories.get(c, "UNKNOWN")).astype(str)
        for c in [cc for cc in self.feature_cols if cc not in self.categorical_cols]:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(self.medians.get(c, 0.0))
        return x

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        x = self._prepare(df)
        return {name: model.predict(x) for name, model in self.models.items()}


def choose_features(df: pd.DataFrame, cfg: ResearchConfig) -> Tuple[List[str], List[str]]:
    target_cols = {cfg.energy_col, cfg.comfort_col, cfg.degradation_col}
    if cfg.cost_col:
        target_cols.add(cfg.cost_col)
    exclude = target_cols | {"__source_file", "__sheet"}
    candidates: List[str] = []
    cats: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if cfg.timestamp_col and c == cfg.timestamp_col:
            continue
        # Keep controls even when nearly constant, as long as they are numeric.
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() >= 10 and (df[c].nunique(dropna=True) > 1 or c in cfg.control_cols):
                candidates.append(c)
        elif c in {cfg.strategy_col, cfg.severity_col}:
            candidates.append(c)
            cats.append(c)
    # Ensure controls exist in features.
    for c in cfg.control_cols:
        if c in df.columns and c not in candidates:
            candidates.append(c)
    return candidates, cats


def _split_indices(df: pd.DataFrame, cfg: ResearchConfig, valid_mask: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.flatnonzero(valid_mask.to_numpy())
    work = df.loc[valid_mask].reset_index(drop=False)
    # Prefer scenario/context grouping to avoid leakage.
    if cfg.group_col and cfg.group_col in work.columns and work[cfg.group_col].nunique() >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seed)
        tr, te = next(gss.split(work, groups=work[cfg.group_col].astype(str)))
        return ix[tr], ix[te]
    # Prefer chronological block split when a timestamp exists.
    if cfg.timestamp_col and cfg.timestamp_col in work.columns:
        ts = pd.to_datetime(work[cfg.timestamp_col], errors="coerce")
        if ts.notna().mean() > 0.8:
            order = np.argsort(ts.fillna(pd.Timestamp.min).to_numpy())
            cut = max(1, int((1 - cfg.test_fraction) * len(order)))
            return ix[order[:cut]], ix[order[cut:]]
    tr, te = train_test_split(ix, test_size=cfg.test_fraction, random_state=cfg.seed)
    return np.asarray(tr), np.asarray(te)


def train_surrogates(df: pd.DataFrame, cfg: ResearchConfig) -> Tuple[SurrogateBundle, pd.DataFrame, pd.DataFrame]:
    features, cats = choose_features(df, cfg)
    if not features:
        raise ValueError("No usable surrogate features were found. Include state variables and control columns.")

    targets = {
        "energy": cfg.energy_col,
        "comfort": cfg.comfort_col,
        "degradation": cfg.degradation_col,
    }
    if cfg.cost_col and cfg.cost_col in df.columns:
        targets["cost_observed"] = cfg.cost_col

    valid = pd.Series(True, index=df.index)
    for c in targets.values():
        if c not in df.columns:
            raise ValueError(f"Required target column not found: {c}")
        valid &= pd.to_numeric(df[c], errors="coerce").notna()
    if valid.sum() < 80:
        raise ValueError(f"Only {int(valid.sum())} complete target rows are available; at least 80 are recommended.")

    tr_idx, te_idx = _split_indices(df, cfg, valid)
    medians: Dict[str, float] = {}
    categories: Dict[str, str] = {}
    X = df[features].copy()
    for c in features:
        if c in cats:
            mode = X.loc[tr_idx, c].dropna().astype(str).mode()
            categories[c] = mode.iloc[0] if len(mode) else "UNKNOWN"
            X[c] = X[c].fillna(categories[c]).astype(str)
        else:
            vals = pd.to_numeric(X.loc[tr_idx, c], errors="coerce")
            medians[c] = float(vals.median()) if vals.notna().any() else 0.0
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])

    cat_indices = [features.index(c) for c in cats]
    models: Dict[str, CatBoostRegressor] = {}
    metrics = []
    parity = []
    for name, col in targets.items():
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        model = CatBoostRegressor(
            iterations=cfg.catboost_iterations,
            depth=cfg.catboost_depth,
            learning_rate=cfg.catboost_learning_rate,
            loss_function="RMSE",
            random_seed=cfg.seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_indices if cat_indices else None)
        pred = model.predict(X.iloc[te_idx])
        yt = y[te_idx]
        rmse = float(np.sqrt(mean_squared_error(yt, pred)))
        mae = float(mean_absolute_error(yt, pred))
        r2 = float(r2_score(yt, pred)) if len(np.unique(yt)) > 1 else np.nan
        denom = np.maximum(np.abs(yt), 1e-9)
        mape = float(np.mean(np.abs((yt - pred) / denom)) * 100)
        metrics.append({"target": name, "column": col, "n_train": len(tr_idx), "n_test": len(te_idx),
                        "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape})
        parity.extend({"target": name, "observed": float(o), "predicted": float(p)} for o, p in zip(yt, pred))
        models[name] = model

    bundle = SurrogateBundle(models, features, cats, medians, categories)
    return bundle, pd.DataFrame(metrics), pd.DataFrame(parity)


# -----------------------------
# Multi-objective utilities
# -----------------------------
def dominates(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.all(a <= b + eps) and np.any(a < b - eps))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(F[j], F[i]):
                keep[i] = False
                break
    return keep


def fast_nondominated_sort(F: np.ndarray) -> List[List[int]]:
    n = len(F)
    S = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        d[order[0]] = d[order[-1]] = np.inf
        lo, hi = F[order[0], j], F[order[-1], j]
        scale = hi - lo
        if scale <= 1e-15:
            continue
        for k in range(1, n - 1):
            d[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / scale
    return d


def update_archive(Xa: np.ndarray, Fa: np.ndarray, Xn: np.ndarray, Fn: np.ndarray, max_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if Xa.size == 0:
        X = Xn.copy(); F = Fn.copy()
    else:
        X = np.vstack([Xa, Xn]); F = np.vstack([Fa, Fn])
    # remove exact/near duplicates in decision space first
    rounded = np.round(X, 10)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    X, F = X[idx], F[idx]
    keep = nondominated_mask(F)
    X, F = X[keep], F[keep]
    if len(X) > max_size:
        cd = crowding_distance(F)
        order = np.argsort(-np.nan_to_num(cd, nan=np.inf))
        X, F = X[order[:max_size]], F[order[:max_size]]
    return X, F


def normalize_objectives(F: np.ndarray, ref_min: Optional[np.ndarray] = None, ref_max: Optional[np.ndarray] = None):
    lo = np.nanmin(F, axis=0) if ref_min is None else ref_min
    hi = np.nanmax(F, axis=0) if ref_max is None else ref_max
    return (F - lo) / np.maximum(hi - lo, 1e-12), lo, hi


def spacing_metric(F: np.ndarray) -> float:
    if len(F) < 2:
        return np.nan
    Z, _, _ = normalize_objectives(F)
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    D[D == 0] = np.inf
    nearest = np.min(D, axis=1)
    return float(np.std(nearest, ddof=1)) if len(nearest) > 1 else 0.0


def igd_metric(F: np.ndarray, Fref: np.ndarray) -> float:
    if len(F) == 0 or len(Fref) == 0:
        return np.nan
    union = np.vstack([F, Fref])
    Z, lo, hi = normalize_objectives(union)
    zf = (F - lo) / np.maximum(hi - lo, 1e-12)
    zr = (Fref - lo) / np.maximum(hi - lo, 1e-12)
    dist = np.linalg.norm(zr[:, None, :] - zf[None, :, :], axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def approximate_hypervolume(F: np.ndarray, samples: int = 6000, seed: int = 12345) -> float:
    """Normalized Monte-Carlo HV in [0,1] using the current set's ideal/nadir and a 1.1 reference box."""
    if len(F) == 0:
        return 0.0
    Z, _, _ = normalize_objectives(F)
    Z = np.clip(Z, 0, 1.0)
    ref = np.full(Z.shape[1], 1.10)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, ref, size=(samples, Z.shape[1]))
    dominated = np.zeros(samples, dtype=bool)
    for f in Z:
        dominated |= np.all(pts >= f, axis=1)
    # Divide by the reference-box volume so the reported indicator is in [0, 1].
    return float(dominated.mean())


def select_knee(F: np.ndarray, weights: Optional[np.ndarray] = None) -> int:
    Z, _, _ = normalize_objectives(F)
    if weights is None:
        weights = np.ones(F.shape[1]) / F.shape[1]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    d = np.sqrt(np.sum(weights * Z**2, axis=1))
    return int(np.argmin(d))


# -----------------------------
# Objective evaluator
# -----------------------------
class HVACObjective:
    def __init__(self, bundle: SurrogateBundle, context: pd.Series, cfg: ResearchConfig,
                 control_cols: Sequence[str], bounds: np.ndarray):
        self.bundle = bundle
        self.context = context.copy()
        self.cfg = cfg
        self.control_cols = list(control_cols)
        self.bounds = np.asarray(bounds, dtype=float)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        rows = pd.DataFrame([self.context.to_dict()] * len(X))
        for j, c in enumerate(self.control_cols):
            rows[c] = X[:, j]
        pred = self.bundle.predict(rows)
        energy = np.maximum(np.asarray(pred["energy"], dtype=float), 0.0)
        comfort = np.maximum(np.asarray(pred["comfort"], dtype=float), 0.0)
        degradation = np.clip(np.asarray(pred["degradation"], dtype=float), 0.0, 1.5)
        # The economic objective intentionally differs from energy by adding maintenance/downtime costs.
        if "cost_observed" in pred:
            base_cost = np.maximum(np.asarray(pred["cost_observed"], dtype=float), 0.0)
        else:
            base_cost = energy * self.cfg.electricity_tariff
        maintenance_flag = degradation >= self.cfg.maintenance_threshold
        cost = base_cost + maintenance_flag.astype(float) * (self.cfg.maintenance_cost + self.cfg.downtime_cost)
        return np.column_stack([energy, comfort, degradation, cost])


# -----------------------------
# Optimizers
# -----------------------------
def random_population(rng: np.random.Generator, n: int, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + rng.random((n, len(bounds))) * (hi - lo)


def repair(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(X, bounds[:, 0]), bounds[:, 1])


def _rank_and_crowding(F: np.ndarray):
    fronts = fast_nondominated_sort(F)
    rank = np.full(len(F), 10**9, dtype=int)
    crowd = np.zeros(len(F))
    for r, front in enumerate(fronts):
        rank[front] = r
        cd = crowding_distance(F[front])
        crowd[front] = cd
    return rank, crowd, fronts


def _environmental_selection(X: np.ndarray, F: np.ndarray, n: int):
    fronts = fast_nondominated_sort(F)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
        else:
            cd = crowding_distance(F[front])
            order = np.argsort(-np.nan_to_num(cd, nan=np.inf))
            selected.extend([front[i] for i in order[: n - len(selected)]])
            break
    idx = np.asarray(selected, dtype=int)
    return X[idx], F[idx]


def _tournament(rng, rank, crowd):
    a, b = rng.integers(0, len(rank), 2)
    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b
    if crowd[a] > crowd[b]:
        return a
    if crowd[b] > crowd[a]:
        return b
    return a if rng.random() < 0.5 else b


def _sbx(rng, p1, p2, bounds, eta=15.0):
    c1, c2 = p1.copy(), p2.copy()
    for j in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[j] - p2[j]) < 1e-14:
            continue
        x1, x2 = sorted([p1[j], p2[j]])
        yl, yu = bounds[j]
        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - yl) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1))
        betaq = (rand * alpha) ** (1.0 / (eta + 1)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        beta = 1.0 + 2.0 * (yu - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1))
        betaq = (rand * alpha) ** (1.0 / (eta + 1)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        if rng.random() < 0.5:
            c1[j], c2[j] = child2, child1
        else:
            c1[j], c2[j] = child1, child2
    return repair(c1[None, :], bounds)[0], repair(c2[None, :], bounds)[0]


def _poly_mutation(rng, x, bounds, eta=20.0, pm=None):
    y = x.copy()
    d = len(y)
    if pm is None:
        pm = 1.0 / d
    for j in range(d):
        if rng.random() >= pm:
            continue
        yl, yu = bounds[j]
        if yu <= yl:
            continue
        delta1 = (y[j] - yl) / (yu - yl)
        delta2 = (yu - y[j]) / (yu - yl)
        rnd = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rnd <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mut_pow
        y[j] += deltaq * (yu - yl)
    return repair(y[None, :], bounds)[0]


def run_nsga2(obj: HVACObjective, cfg: ResearchConfig, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n, K, bounds = cfg.population, cfg.iterations, obj.bounds
    X = random_population(rng, n, bounds)
    F = obj.evaluate(X)
    Xa, Fa = update_archive(np.empty((0, X.shape[1])), np.empty((0, 4)), X, F, cfg.archive_size)
    hist = []
    for k in range(K):
        rank, crowd, _ = _rank_and_crowding(F)
        offspring = []
        while len(offspring) < n:
            p1 = X[_tournament(rng, rank, crowd)]
            p2 = X[_tournament(rng, rank, crowd)]
            c1, c2 = _sbx(rng, p1, p2, bounds)
            offspring.extend([_poly_mutation(rng, c1, bounds), _poly_mutation(rng, c2, bounds)])
        Xo = np.asarray(offspring[:n])
        Fo = obj.evaluate(Xo)
        X, F = _environmental_selection(np.vstack([X, Xo]), np.vstack([F, Fo]), n)
        Xa, Fa = update_archive(Xa, Fa, X, F, cfg.archive_size)
        if k == 0 or (k + 1) % max(1, K // 12) == 0 or k == K - 1:
            hist.append({"iteration": k + 1, "hypervolume": approximate_hypervolume(Fa, samples=2500, seed=seed + k)})
    return {"X": Xa, "F": Fa, "history": hist}


def run_mopso(obj: HVACObjective, cfg: ResearchConfig, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n, K, bounds = cfg.population, cfg.iterations, obj.bounds
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    X = random_population(rng, n, bounds)
    V = rng.uniform(-0.1, 0.1, size=X.shape) * span
    F = obj.evaluate(X)
    P, FP = X.copy(), F.copy()
    Xa, Fa = update_archive(np.empty((0, X.shape[1])), np.empty((0, 4)), X, F, cfg.archive_size)
    hist = []
    for k in range(K):
        cd = crowding_distance(Fa)
        finite = np.where(np.isfinite(cd), np.maximum(cd, 1e-9), np.nanmax(cd[np.isfinite(cd)]) * 2 if np.any(np.isfinite(cd)) else 1.0)
        probs = finite / finite.sum() if finite.sum() > 0 else np.ones(len(Fa)) / len(Fa)
        leaders = Xa[rng.choice(len(Xa), size=n, p=probs)]
        w = 0.9 - 0.5 * (k / max(K - 1, 1))
        c1, c2 = 1.7, 1.7
        V = w * V + c1 * rng.random(X.shape) * (P - X) + c2 * rng.random(X.shape) * (leaders - X)
        V = np.clip(V, -0.25 * span, 0.25 * span)
        X = repair(X + V, bounds)
        F = obj.evaluate(X)
        for i in range(n):
            if dominates(F[i], FP[i]) or (not dominates(FP[i], F[i]) and rng.random() < 0.5):
                P[i], FP[i] = X[i].copy(), F[i].copy()
        Xa, Fa = update_archive(Xa, Fa, X, F, cfg.archive_size)
        if k == 0 or (k + 1) % max(1, K // 12) == 0 or k == K - 1:
            hist.append({"iteration": k + 1, "hypervolume": approximate_hypervolume(Fa, samples=2500, seed=seed + k)})
    return {"X": Xa, "F": Fa, "history": hist}


def run_moapo(obj: HVACObjective, cfg: ResearchConfig, seed: int) -> Dict:
    """
    Pareto extension of APO for HVAC.

    The implementation retains the APO behavioral classes (autotrophic/heterotrophic
    foraging, dormancy, reproduction) but replaces scalar-best guidance with an external
    non-dominated archive and diversity-biased leader selection. This is an explicit
    MOAPO-HVAC adaptation, not a claim of bit-for-bit equivalence with any published MOAPO variant.
    """
    rng = np.random.default_rng(seed)
    n, K, bounds = cfg.population, cfg.iterations, obj.bounds
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    X = random_population(rng, n, bounds)
    F = obj.evaluate(X)
    Xa, Fa = update_archive(np.empty((0, X.shape[1])), np.empty((0, 4)), X, F, cfg.archive_size)
    hist = []

    for k in range(K):
        progress = k / max(K - 1, 1)
        p_ah = 0.5 * (1.0 + math.cos(math.pi * progress))  # exploration -> exploitation
        f_scale = 0.5 * (1.0 - progress) + 0.05
        pf = 0.10 + 0.20 * (1.0 - progress)  # fraction undergoing dormancy/reproduction

        rank, crowd, _ = _rank_and_crowding(F)
        order = np.lexsort((-np.nan_to_num(crowd, nan=1e9), rank))
        n_dr = max(1, int(math.ceil(n * pf)))
        poor = set(order[-n_dr:].tolist())
        good = set(order[:n_dr].tolist())

        cd_a = crowding_distance(Fa)
        finite = np.where(np.isfinite(cd_a), np.maximum(cd_a, 1e-9), np.nanmax(cd_a[np.isfinite(cd_a)]) * 2 if np.any(np.isfinite(cd_a)) else 1.0)
        p_lead = finite / finite.sum() if finite.sum() > 0 else np.ones(len(Fa)) / len(Fa)

        Xnew = X.copy()
        for i in range(n):
            leader = Xa[rng.choice(len(Xa), p=p_lead)]
            # Dormancy: weak protozoa are regenerated to restore diversity.
            if i in poor and rng.random() < 0.55:
                candidate = lo + rng.random(len(lo)) * span
            # Reproduction: strong protozoa create a perturbed offspring near an archive leader.
            elif i in good and rng.random() < 0.50:
                mask = rng.random(len(lo)) < max(1.0 / len(lo), 0.5 * (1.0 - progress))
                if not np.any(mask):
                    mask[rng.integers(0, len(lo))] = True
                candidate = X[i].copy()
                noise = rng.normal(0, 1, len(lo)) * span * (0.08 * (1.0 - progress) + 0.01)
                candidate[mask] = 0.5 * X[i, mask] + 0.5 * leader[mask] + noise[mask]
            else:
                a, b = rng.choice(n, size=2, replace=False)
                # Autotrophic foraging: exploratory movement using population differences.
                if rng.random() < p_ah:
                    candidate = X[i] + f_scale * (X[a] - X[b]) + rng.uniform(-0.25, 0.25, len(lo)) * span * (1.0 - progress)
                # Heterotrophic foraging: exploit nutrient-rich (Pareto leader) regions.
                else:
                    candidate = X[i] + (0.8 + 0.6 * rng.random()) * f_scale * (leader - X[i]) + 0.35 * f_scale * (X[a] - X[b])
            Xnew[i] = repair(candidate[None, :], bounds)[0]

        Fnew = obj.evaluate(Xnew)
        # Greedy Pareto selection with diversity tie-break.
        for i in range(n):
            if dominates(Fnew[i], F[i]):
                X[i], F[i] = Xnew[i], Fnew[i]
            elif not dominates(F[i], Fnew[i]):
                # Keep nondominated newcomer with a probability that is higher early in search.
                if rng.random() < 0.45 + 0.25 * (1.0 - progress):
                    X[i], F[i] = Xnew[i], Fnew[i]
        Xa, Fa = update_archive(Xa, Fa, X, F, cfg.archive_size)
        if k == 0 or (k + 1) % max(1, K // 12) == 0 or k == K - 1:
            hist.append({"iteration": k + 1, "hypervolume": approximate_hypervolume(Fa, samples=2500, seed=seed + k)})
    return {"X": Xa, "F": Fa, "history": hist}


OPTIMIZERS = {"MOAPO": run_moapo, "NSGA-II": run_nsga2, "MOPSO": run_mopso}


# -----------------------------
# Paper pipeline
# -----------------------------
def infer_bounds(df: pd.DataFrame, control_cols: Sequence[str], user_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    rows = []
    defaults = {
        "candidate_zone_setpoint_C": (22.0, 26.0),
        "zone_setpoint_C": (22.0, 26.0),
        "candidate_airflow_fraction": (0.5, 1.0),
        "airflow_fraction": (0.5, 1.0),
        "candidate_chw_setpoint_C": (5.0, 9.0),
        "chw_supply_setpoint_C": (5.0, 9.0),
        "candidate_pump_speed_fraction": (0.5, 1.0),
        "pump_speed_fraction": (0.5, 1.0),
    }
    for c in control_cols:
        if user_bounds and c in user_bounds:
            lo, hi = user_bounds[c]
        elif c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and v.max() > v.min():
                lo, hi = float(v.quantile(0.01)), float(v.quantile(0.99))
            elif c in defaults:
                lo, hi = defaults[c]
            else:
                base = float(v.iloc[0]) if len(v) else 0.0
                lo, hi = base - 1.0, base + 1.0
        elif c in defaults:
            lo, hi = defaults[c]
        else:
            raise ValueError(f"Cannot infer bounds for missing control column {c}")
        if hi <= lo:
            hi = lo + 1e-6
        rows.append((lo, hi))
    return np.asarray(rows, dtype=float)


def choose_context(df: pd.DataFrame, cfg: ResearchConfig, mode: str = "median", row_index: Optional[int] = None,
                   severity: Optional[str] = None) -> pd.Series:
    work = df.copy()
    if severity and cfg.severity_col and cfg.severity_col in work.columns:
        sub = work[work[cfg.severity_col].astype(str) == str(severity)]
        if len(sub):
            work = sub
    if row_index is not None and row_index in work.index:
        return work.loc[row_index].copy()
    if mode == "first":
        return work.iloc[0].copy()
    # Select the actual row closest to numeric medians, preserving categorical consistency.
    num = work.select_dtypes(include=[np.number]).columns
    num = [c for c in num if c not in {cfg.energy_col, cfg.comfort_col, cfg.degradation_col, cfg.cost_col}]
    if not num:
        return work.iloc[len(work) // 2].copy()
    z = work[num].apply(pd.to_numeric, errors="coerce")
    med = z.median()
    scale = z.std().replace(0, 1).fillna(1)
    dist = (((z - med) / scale) ** 2).sum(axis=1)
    return work.loc[dist.idxmin()].copy()


def _pareto_dataframe(name: str, result: Dict, control_cols: Sequence[str], cfg: ResearchConfig) -> pd.DataFrame:
    X, F = result["X"], result["F"]
    rows = []
    for i in range(len(X)):
        r = {"optimizer": name, "solution_id": i + 1}
        r.update({c: float(X[i, j]) for j, c in enumerate(control_cols)})
        r.update({cfg.objective_names[j]: float(F[i, j]) for j in range(4)})
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        knee = select_knee(F)
        out["is_knee"] = False
        out.loc[knee, "is_knee"] = True
    return out


def _representatives(pareto: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    if pareto.empty:
        return pareto
    F = pareto[list(cfg.objective_names)].to_numpy(float)
    labels = {}
    labels["Energy-priority"] = int(np.argmin(F[:, 0]))
    labels["Comfort-priority"] = int(np.argmin(F[:, 1]))
    labels["Degradation-priority"] = int(np.argmin(F[:, 2]))
    labels["Cost-priority"] = int(np.argmin(F[:, 3]))
    labels["Balanced knee"] = select_knee(F)
    rows = []
    for label, idx in labels.items():
        rr = pareto.iloc[idx].copy().to_dict()
        rr["representative"] = label
        rows.append(rr)
    return pd.DataFrame(rows).drop_duplicates(subset=["optimizer", "solution_id", "representative"])


def _plot_parity(parity: pd.DataFrame, outdir: Path):
    for target in parity["target"].unique():
        sub = parity[parity["target"] == target]
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(sub["observed"], sub["predicted"], s=18, alpha=0.65)
        lo = min(sub["observed"].min(), sub["predicted"].min())
        hi = max(sub["observed"].max(), sub["predicted"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(f"CatBoost parity: {target}")
        fig.tight_layout()
        fig.savefig(outdir / f"Fig_parity_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_pareto(all_pareto: pd.DataFrame, outdir: Path):
    if all_pareto.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name, sub in all_pareto.groupby("optimizer"):
        ax.scatter(sub["Energy"], sub["Discomfort"], s=32, alpha=0.75, label=name)
    ax.set_xlabel("Energy objective")
    ax.set_ylabel("Thermal discomfort objective")
    ax.set_title("Pareto front: energy vs discomfort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Pareto_energy_discomfort.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Parallel-coordinate-like normalized plot for the four objectives.
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    cols = ["Energy", "Discomfort", "Degradation", "Cost"]
    Z = all_pareto[cols].copy()
    for c in cols:
        den = max(Z[c].max() - Z[c].min(), 1e-12)
        Z[c] = (Z[c] - Z[c].min()) / den
    x = np.arange(len(cols))
    for idx in range(len(all_pareto)):
        ax.plot(x, Z.iloc[idx].values, alpha=0.18, linewidth=0.8)
    ax.set_xticks(x, cols)
    ax.set_ylabel("Normalized objective (lower is better)")
    ax.set_title("Four-objective Pareto trade-offs")
    fig.tight_layout()
    fig.savefig(outdir / "Fig_Pareto_parallel_coordinates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(hist_df: pd.DataFrame, outdir: Path):
    if hist_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, sub in hist_df.groupby("optimizer"):
        ax.plot(sub["iteration"], sub["hypervolume"], marker="o", markersize=3, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Approx. normalized hypervolume")
    ax.set_title("Optimizer convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Fig_optimizer_hypervolume_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_summary(df: pd.DataFrame, cfg: ResearchConfig, outdir: Path) -> Optional[pd.DataFrame]:
    if not cfg.strategy_col or cfg.strategy_col not in df.columns:
        return None
    metrics = [c for c in [cfg.energy_col, cfg.comfort_col, cfg.degradation_col, cfg.cost_col] if c and c in df.columns]
    if not metrics:
        return None
    group = [cfg.strategy_col]
    if cfg.severity_col and cfg.severity_col in df.columns:
        group.append(cfg.severity_col)
    summary = df.groupby(group, dropna=False)[metrics].mean(numeric_only=True).reset_index()
    if cfg.energy_col in summary.columns:
        pivot = summary.pivot(index=group[0], columns=group[1] if len(group) > 1 else None, values=cfg.energy_col) if len(group) > 1 else summary.set_index(group[0])[[cfg.energy_col]]
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        if len(group) > 1:
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(np.arange(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=25)
            ax.set_yticks(np.arange(len(pivot.index)), [str(x) for x in pivot.index])
            fig.colorbar(im, ax=ax, label=cfg.energy_col)
            ax.set_xlabel(group[1]); ax.set_ylabel(group[0])
        else:
            ax.bar(pivot.index.astype(str), pivot[cfg.energy_col])
        ax.set_title("Strategy / severity energy summary")
        fig.tight_layout()
        fig.savefig(outdir / "Fig_strategy_severity_energy.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return summary


def run_paper_pipeline(df: pd.DataFrame, cfg: ResearchConfig, outdir: Path,
                       context_mode: str = "median", context_severity: Optional[str] = None,
                       algorithms: Sequence[str] = ("MOAPO", "NSGA-II", "MOPSO")) -> Dict[str, pd.DataFrame]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = coerce_numeric(df)

    bundle, surrogate_metrics, parity = train_surrogates(df, cfg)
    surrogate_metrics.to_csv(outdir / "Table_surrogate_metrics.csv", index=False)
    parity.to_csv(outdir / "surrogate_test_predictions.csv", index=False)
    _plot_parity(parity, outdir)

    control_cols = [c for c in cfg.control_cols if c in df.columns]
    if len(control_cols) < 2:
        schema = infer_schema(df)
        inferred = [schema.get("setpoint"), schema.get("airflow"), schema.get("chw"), schema.get("pump")]
        control_cols = [c for c in inferred if c and c in df.columns]
    control_cols = list(dict.fromkeys(control_cols))
    if len(control_cols) < 2:
        raise ValueError("At least two numeric control columns are required (normally zone setpoint and airflow fraction).")

    bounds = infer_bounds(df, control_cols, cfg.bounds)
    context = choose_context(df, cfg, mode=context_mode, severity=context_severity)
    objective = HVACObjective(bundle, context, cfg, control_cols, bounds)

    results = {}
    raw_pareto_frames = []
    raw_hist_frames = []
    run_rows = []
    repetitions = max(1, int(cfg.repetitions))

    # Repeated seeded trials: publication work should normally use 30.
    for i, name in enumerate(algorithms):
        if name not in OPTIMIZERS:
            continue
        algorithm_runs = []
        for rep in range(repetitions):
            run_seed = int(cfg.seed + i * 100000 + rep * 7919)
            t0 = time.perf_counter()
            res = OPTIMIZERS[name](objective, cfg, run_seed)
            runtime = time.perf_counter() - t0
            algorithm_runs.append(res)
            pdf = _pareto_dataframe(name, res, control_cols, cfg)
            pdf["repeat"] = rep + 1
            pdf["seed"] = run_seed
            raw_pareto_frames.append(pdf)
            hf = pd.DataFrame(res["history"])
            if len(hf):
                hf["optimizer"] = name
                hf["repeat"] = rep + 1
                hf["seed"] = run_seed
                raw_hist_frames.append(hf)
            run_rows.append({
                "optimizer": name, "repeat": rep + 1, "seed": run_seed,
                "runtime_s": runtime, "n_nondominated": len(res["F"]),
                "hypervolume": approximate_hypervolume(res["F"], samples=12000, seed=run_seed),
                "spacing": spacing_metric(res["F"]),
            })

        # Union non-dominated archive across repeated runs for this optimizer.
        Xall = np.vstack([r["X"] for r in algorithm_runs])
        Fall = np.vstack([r["F"] for r in algorithm_runs])
        Xu, Fu = update_archive(np.empty((0, Xall.shape[1])), np.empty((0, 4)), Xall, Fall, cfg.archive_size)
        results[name] = {"X": Xu, "F": Fu, "runs": algorithm_runs}

    raw_pareto = pd.concat(raw_pareto_frames, ignore_index=True) if raw_pareto_frames else pd.DataFrame()
    hist_raw = pd.concat(raw_hist_frames, ignore_index=True) if raw_hist_frames else pd.DataFrame()
    raw_pareto.to_csv(outdir / "Table_all_Pareto_solutions_all_runs.csv", index=False)
    hist_raw.to_csv(outdir / "optimizer_convergence_all_runs.csv", index=False)

    # Final per-optimizer non-dominated union.
    pareto_frames = [_pareto_dataframe(name, res, control_cols, cfg) for name, res in results.items()]
    all_pareto = pd.concat(pareto_frames, ignore_index=True) if pareto_frames else pd.DataFrame()
    all_pareto.to_csv(outdir / "Table_all_Pareto_solutions.csv", index=False)

    # Mean convergence trace across repetitions for figures.
    hist_df = (hist_raw.groupby(["optimizer", "iteration"], as_index=False)["hypervolume"].agg(["mean", "std"]).reset_index()
               if len(hist_raw) else pd.DataFrame())
    if len(hist_df):
        hist_df = hist_df.rename(columns={"mean": "hypervolume", "std": "hypervolume_sd"})
    hist_df.to_csv(outdir / "optimizer_convergence.csv", index=False)

    # Unified reference front from the non-dominated union of final optimizer archives.
    F_union = all_pareto[list(cfg.objective_names)].to_numpy(float)
    ref_mask = nondominated_mask(F_union) if len(F_union) else np.array([], dtype=bool)
    Fref = F_union[ref_mask] if len(F_union) else np.empty((0, 4))

    # Add IGD for each individual run against the common union reference.
    rr = pd.DataFrame(run_rows)
    for idx, row in rr.iterrows():
        name = row["optimizer"]; rep = int(row["repeat"]) - 1
        rr.loc[idx, "IGD_to_union_reference"] = igd_metric(results[name]["runs"][rep]["F"], Fref)
    rr.to_csv(outdir / "Table_optimizer_runs.csv", index=False)

    benchmark_rows = []
    for name, sub in rr.groupby("optimizer"):
        q1, q3 = sub["hypervolume"].quantile([0.25, 0.75])
        benchmark_rows.append({
            "optimizer": name,
            "runs": len(sub),
            "hypervolume_median": sub["hypervolume"].median(),
            "hypervolume_IQR": q3 - q1,
            "hypervolume_mean": sub["hypervolume"].mean(),
            "hypervolume_sd": sub["hypervolume"].std(ddof=1) if len(sub) > 1 else 0.0,
            "IGD_mean": sub["IGD_to_union_reference"].mean(),
            "IGD_sd": sub["IGD_to_union_reference"].std(ddof=1) if len(sub) > 1 else 0.0,
            "spacing_mean": sub["spacing"].mean(),
            "runtime_s_mean": sub["runtime_s"].mean(),
            "n_nondominated_mean": sub["n_nondominated"].mean(),
        })
    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(["IGD_mean", "hypervolume_median"], ascending=[True, False])
    benchmark_df.to_csv(outdir / "Table_optimizer_benchmark.csv", index=False)

    # Non-parametric statistics when enough repeated runs are available.
    stats_rows = []
    if repetitions >= 3 and len(algorithms) >= 2:
        try:
            from scipy.stats import friedmanchisquare, wilcoxon
            available = [a for a in algorithms if a in rr["optimizer"].unique()]
            if len(available) >= 3:
                arrays = [rr[rr["optimizer"] == a].sort_values("repeat")["hypervolume"].to_numpy() for a in available]
                stat, pval = friedmanchisquare(*arrays)
                stats_rows.append({"test": "Friedman", "metric": "hypervolume", "comparison": " vs ".join(available), "statistic": stat, "p_value": pval})
            for a_i in range(len(available)):
                for b_i in range(a_i + 1, len(available)):
                    a, b = available[a_i], available[b_i]
                    va = rr[rr["optimizer"] == a].sort_values("repeat")["hypervolume"].to_numpy()
                    vb = rr[rr["optimizer"] == b].sort_values("repeat")["hypervolume"].to_numpy()
                    try:
                        stat, pval = wilcoxon(va, vb, zero_method="wilcox")
                    except ValueError:
                        stat, pval = np.nan, 1.0
                    stats_rows.append({"test": "Wilcoxon signed-rank", "metric": "hypervolume", "comparison": f"{a} vs {b}", "statistic": stat, "p_value": pval})
        except Exception as exc:
            stats_rows.append({"test": "statistics unavailable", "metric": "hypervolume", "comparison": str(exc), "statistic": np.nan, "p_value": np.nan})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(outdir / "Table_optimizer_statistics.csv", index=False)

    reps = []
    for name, sub in all_pareto.groupby("optimizer"):
        reps.append(_representatives(sub.reset_index(drop=True), cfg))
    representatives = pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()
    representatives.to_csv(outdir / "Table_representative_Pareto_solutions.csv", index=False)

    strategy_summary = _plot_strategy_summary(df, cfg, outdir)
    if strategy_summary is not None:
        strategy_summary.to_csv(outdir / "Table_strategy_severity_summary.csv", index=False)

    _plot_pareto(all_pareto, outdir)
    _plot_convergence(hist_df, outdir)

    context_df = pd.DataFrame([context])
    context_df.to_csv(outdir / "selected_optimization_context.csv", index=False)
    pd.DataFrame({"control": control_cols, "lower_bound": bounds[:, 0], "upper_bound": bounds[:, 1]}).to_csv(outdir / "control_bounds.csv", index=False)

    manifest = {
        "software": "MOAPO-CatBoost HVAC Research Engine",
        "version": "1.0.0",
        "config": asdict(cfg),
        "control_cols": control_cols,
        "bounds": {control_cols[i]: [float(bounds[i, 0]), float(bounds[i, 1])] for i in range(len(control_cols))},
        "n_rows": int(len(df)),
        "n_features": len(bundle.feature_cols),
        "feature_cols": bundle.feature_cols,
        "categorical_cols": bundle.categorical_cols,
        "algorithm_note": "MOAPO-HVAC retains APO behavior classes and adds external Pareto archive/diversity guidance; it is not claimed to reproduce a specific published MOAPO variant bit-for-bit.",
        "hypervolume_note": "Hypervolume is a deterministic Monte-Carlo approximation in internally normalized objective space.",
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Ready-to-edit manuscript summary, numbers drawn from generated results only.
    best_name = benchmark_df.iloc[0]["optimizer"] if len(benchmark_df) else "N/A"
    knee = representatives[(representatives["optimizer"] == best_name) & (representatives["representative"] == "Balanced knee")] if len(representatives) else pd.DataFrame()
    lines = [
        "# Automatically generated results summary",
        "",
        f"The surrogate models were trained on {len(df):,} observations using a leakage-aware split when grouping/time information was available.",
        f"The optimization compared {', '.join(algorithms)} using population={cfg.population}, iterations={cfg.iterations}, archive size={cfg.archive_size}, and {cfg.repetitions} repeated run(s) per optimizer.",
        f"Based on the generated reference-front indicators, the first-ranked optimizer in this run was **{best_name}**.",
    ]
    if len(knee):
        r = knee.iloc[0]
        lines.append(
            "The balanced Pareto solution for this optimizer produced "
            f"Energy={r['Energy']:.4g}, Discomfort={r['Discomfort']:.4g}, "
            f"Degradation={r['Degradation']:.4g}, and Cost={r['Cost']:.4g}."
        )
    lines += [
        "",
        "## Claim boundary",
        "These are surrogate-assisted optimization results for the uploaded/synthetic operating envelope. They are not field-performance claims unless the selected controls are independently validated against measured or physics-model data.",
        "",
        "## Required manuscript validation",
        "Re-evaluate representative Pareto controls with the original ROM/EnergyPlus/DesignBuilder model and report objective-wise error and optimization regret before treating them as final physics-verified results.",
    ]
    (outdir / "manuscript_results_summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "surrogate_metrics": surrogate_metrics,
        "parity": parity,
        "pareto": all_pareto,
        "benchmark": benchmark_df,
        "optimizer_runs": rr,
        "optimizer_statistics": stats_df,
        "representatives": representatives,
        "history": hist_df,
        "strategy_summary": strategy_summary if strategy_summary is not None else pd.DataFrame(),
    }


def zip_results(outdir: Path) -> bytes:
    outdir = Path(outdir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(outdir))
    return buf.getvalue()


# -----------------------------
# Demo dataset (clearly synthetic)
# -----------------------------
def make_demo_dataset(n: int = 3500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strategy = rng.choice(["S0", "S1", "S2", "S3"], n)
    severity = rng.choice(["Mild", "Moderate", "Severe", "High"], n, p=[0.25, 0.30, 0.27, 0.18])
    sev_base = pd.Series(severity).map({"Mild": 0.25, "Moderate": 0.45, "Severe": 0.65, "High": 0.85}).to_numpy()
    tout = rng.normal(31, 5, n)
    rh = np.clip(rng.normal(55, 12, n), 20, 95)
    occ = rng.integers(0, 130, n)
    solar = np.clip(rng.normal(350, 240, n), 0, 950)
    tset = rng.uniform(22, 26, n)
    airflow = rng.uniform(0.5, 1.0, n)
    fouling = np.clip(sev_base * 2e-4 + rng.normal(0, 1.2e-5, n), 0, 2.2e-4)
    filter_dp = np.clip(150 + sev_base * 300 + rng.normal(0, 15, n), 150, 470)
    strat_effect = pd.Series(strategy).map({"S0": 0.08, "S1": 0.03, "S2": -0.02, "S3": -0.08}).to_numpy()
    delta = np.clip(0.52 * (fouling / 2e-4) + 0.38 * ((filter_dp - 150) / 300) + 0.10 * sev_base + strat_effect, 0, 1)
    load = 55 + 3.7 * np.maximum(tout - 22, 0) + 0.13 * occ + 0.025 * solar
    fan = 18 * airflow**3
    chiller = load * (1 + 0.35 * delta) * (1 + 0.018 * (24 - tset))
    energy = np.maximum(20, chiller + fan + rng.normal(0, 4, n))
    comfort = np.clip(5 + 6.5 * np.abs(tset - 24.1) + 8 * (1 - airflow) + 0.035 * occ + 4 * delta + rng.normal(0, 1.4, n), 4, 60)
    next_delta = np.clip(delta + 0.006 * sev_base - 0.005 * (airflow - 0.5) + rng.normal(0, 0.015, n), 0, 1)
    timestamps = pd.date_range("2026-05-01", periods=n, freq="h")
    context_id = np.arange(n) // 12
    return pd.DataFrame({
        "context_id": context_id,
        "timestamp": timestamps,
        "strategy": strategy,
        "severity": severity,
        "outdoor_drybulb_C": tout,
        "outdoor_rh_pct": rh,
        "solar_direct_W_m2": solar,
        "zone_people_count": occ,
        "zone_air_temp_C": tset + rng.normal(0.3, 0.7, n),
        "fouling_resistance_m2K_W": fouling,
        "filter_dp_Pa": filter_dp,
        "delta_current": delta,
        "candidate_zone_setpoint_C": tset,
        "candidate_airflow_fraction": airflow,
        "target_hvac_energy_kWh": energy,
        "target_ppd_pct": comfort,
        "target_delta_next": next_delta,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOAPO-CatBoost HVAC Research Engine")
    parser.add_argument("--demo", action="store_true", help="Run using the built-in synthetic demonstration dataset")
    parser.add_argument("--input", nargs="*", help="CSV/XLSX input files")
    parser.add_argument("--out", default="paper_results")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    if args.demo or not args.input:
        df = make_demo_dataset()
    else:
        frames = []
        for fn in args.input:
            with open(fn, "rb") as f:
                frames += read_any_file(f, fn)
        df = harmonize_frames(frames)
    cfg = ResearchConfig(population=args.population, iterations=args.iterations, repetitions=args.repetitions)
    out = Path(args.out)
    run_paper_pipeline(df, cfg, out)
    print(f"Finished. Results written to: {out.resolve()}")
