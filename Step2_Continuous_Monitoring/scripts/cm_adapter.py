"""Adapter between the frozen MIMIC-IV baseline and the upstream online-simulation code.

Evaluation only. Nothing in this module fits, trains, tunes or rewrites a baseline
artifact: Baseline scripts and the pinned upstream repository are imported read-only,
and every fitted object is checked before and after use.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # Keep frozen Step 1 and upstream trees free of bytecode writes.

import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from project_paths import repo_path  # noqa: E402

LOGGER = logging.getLogger("continuous_monitoring")

MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]
ML_MODELS = ("logistic_regression", "xgboost")
HORIZONS = [0, 48, 72]
UPSTREAM_MODEL_NAMES = {
    "ite_transformer": "ITE Transformer",
    "lstm_attention": "LSTM-Attention",
    "masked_cnn": "Masked CNN",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}
DATASET_KEY = "mimic-iv_test"
DATASET_DISPLAY = "MIMIC-IV (adapted held-out test)"
# Irregular token times are float32 hours since ICU intime; one minute is 0.0167 h.
TIME_TOLERANCE_HOURS = 1e-3


class MonitoringAbort(RuntimeError):
    """Raised when a frozen-evaluation or leakage guarantee does not hold."""


# --------------------------------------------------------------------------- setup
def load_config() -> tuple[dict[str, Path], dict]:
    raw_paths = yaml.safe_load((CM_ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    monitoring = yaml.safe_load((CM_ROOT / "configs" / "monitoring.yaml").read_text(encoding="utf-8"))
    monitoring["artifact_manifest"] = repo_path(monitoring["artifact_manifest"])
    return {key: repo_path(value) for key, value in raw_paths.items()}, monitoring


def enforce_frozen_policy(monitoring: dict) -> None:
    required = {
        "reuse_subject_level_split": True,
        "held_out_split": "test",
        "allow_split_generation": False,
        "allow_model_training": False,
        "allow_preprocessing_fit": False,
        "allow_test_set_tuning": False,
        "require_existing_checkpoints": True,
        "require_existing_preprocessing_artifacts": True,
        "artifact_scope": "full",
        "threshold_source": "baseline_config_validation_threshold",
    }
    for key, expected in required.items():
        if monitoring.get(key) != expected:
            raise MonitoringAbort(f"monitoring.yaml policy violation: {key}={monitoring.get(key)!r}, expected {expected!r}")
    if sorted(monitoring.get("prediction_horizons", [])) != HORIZONS:
        raise MonitoringAbort(f"prediction_horizons must be {HORIZONS}")
    if monitoring.get("time_points_hours") != [-72, -60, -48, -36, -24, -12, 0]:
        raise MonitoringAbort("time_points_hours must be [-72, -60, -48, -36, -24, -12, 0]")
    if sorted(monitoring.get("models", [])) != sorted(MODELS):
        raise MonitoringAbort(f"models must be {MODELS}")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def process_memory_gb() -> float:
    """Private (commit) memory of this process in GB; NaN if unavailable. Windows API via ctypes."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # default c_int truncates the 64-bit handle
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PagefileUsage / 1024**3
    except Exception:
        return float("nan")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_sha256(frame: pd.DataFrame) -> str:
    """Canonical cohort hash used by the preflight audit and the artifact manifest."""
    ids = frame[["subject_id", "hadm_id", "stay_id"]].astype("int64")
    ids = ids.sort_values(["subject_id", "hadm_id", "stay_id"], kind="stable")
    return hashlib.sha256(ids.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def import_baseline(baseline_root: Path) -> SimpleNamespace:
    """Import the phase-one encoder and inference helpers without modifying them."""
    build_inputs = _load_module("baseline_build_model_inputs", baseline_root / "scripts" / "build_model_inputs.py")
    run_baseline = _load_module("baseline_run_baseline", baseline_root / "scripts" / "run_baseline.py")
    config = sys.modules["src.utils.config"]
    if not Path(config.__file__).resolve().is_relative_to(baseline_root.resolve()):
        raise MonitoringAbort(f"src.utils.config resolved outside Baseline: {config.__file__}")
    return SimpleNamespace(
        build_one=build_inputs.build_one,
        run=run_baseline,
        config=config,
        processing=config.load_baseline_config()["processing"],
    )


def import_upstream(upstream_root: Path, results_dir: Path, figures_dir: Path, modules: list[str]) -> SimpleNamespace:
    """Import upstream analysis modules with their output roots redirected into ContinuousMonitoring.

    Upstream modules create their output folders at import time relative to
    utils.config.RESULTS_DIR / FIGURES_DIR, so those are overridden before import.
    """
    import matplotlib

    matplotlib.use("Agg")
    saved_path = list(sys.path)
    sys.path.insert(0, str(upstream_root))
    try:
        upstream_config = importlib.import_module("utils.config")
        if Path(upstream_config.PROJECT_ROOT).resolve() != upstream_root.resolve():
            raise MonitoringAbort(f"utils.config resolved to {upstream_config.PROJECT_ROOT}, not {upstream_root}")
        upstream_config.RESULTS_DIR = results_dir
        upstream_config.FIGURES_DIR = figures_dir
        loaded = {"config": upstream_config}
        for module in modules:
            loaded[module.rsplit(".", 1)[-1]] = importlib.import_module(module)
    finally:
        sys.path[:] = saved_path
    return SimpleNamespace(**loaded)


# ----------------------------------------------------------------------- artifacts
PATH_COLUMNS = [
    "checkpoint_path", "scaler_path", "imputer_path", "normalization_table_path",
    "feature_list_path", "config_path", "test_prediction_path",
]


def select_artifacts(monitoring: dict, models: list[str], horizons: list[int]) -> tuple[list[dict], pd.DataFrame]:
    """Select checkpoints strictly through the manifest; smoke artifacts can never be returned."""
    manifest = pd.read_csv(monitoring["artifact_manifest"], dtype=str, keep_default_na=False)
    for column in PATH_COLUMNS:
        manifest[column] = manifest[column].map(lambda value: str(repo_path(value)) if value else "")
    manifest["prediction_window_hours"] = manifest.prediction_window_hours.astype(int)
    final_flag = manifest.used_for_final_evaluation.str.lower().eq("true")
    formal = manifest[manifest.scope.eq("full") & final_flag]
    excluded = manifest[~(manifest.scope.eq("full") & final_flag)]

    expected_pairs = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    actual_pairs = set(zip(formal.model, formal.prediction_window_hours))
    if len(formal) != monitoring["expected_formal_artifacts"] or actual_pairs != expected_pairs:
        raise MonitoringAbort(f"manifest must contain exactly one formal artifact per model/horizon; found {len(formal)}")
    if formal.duplicated(["model", "prediction_window_hours"]).any():
        raise MonitoringAbort("duplicate formal artifacts in manifest")
    if not excluded.scope.eq("smoke").all() or excluded.used_for_final_evaluation.str.lower().eq("true").any():
        raise MonitoringAbort("unexpected non-formal manifest rows")
    formal_paths = {value for col in PATH_COLUMNS for value in formal[col] if value}
    smoke_paths = {value for col in PATH_COLUMNS for value in excluded[col] if value}
    if any("/smoke/" in value.replace("\\", "/") for value in formal_paths) or formal_paths & smoke_paths:
        raise MonitoringAbort("a formal artifact path points at a smoke artifact")

    selected = formal[formal.model.isin(models) & formal.prediction_window_hours.isin(horizons)]
    selected = selected.sort_values(["prediction_window_hours", "model"], key=lambda s: s.map(MODELS.index) if s.name == "model" else s)
    records = []
    for row in selected.to_dict("records"):
        for col in PATH_COLUMNS:
            if row[col] and not Path(row[col]).exists():
                raise MonitoringAbort(f"missing artifact path {row[col]}")
        observed = sha256_file(Path(row["checkpoint_path"]))
        if observed != row["checkpoint_sha256"]:
            raise MonitoringAbort(f"checkpoint SHA-256 mismatch for {row['artifact_id']}")
        records.append(row)
    table = pd.concat([
        selected.assign(selection="selected"),
        formal.drop(selected.index).assign(selection="formal_not_in_this_stage"),
        excluded.assign(selection="excluded_smoke"),
    ])[["artifact_id", "model", "prediction_window_hours", "scope", "used_for_final_evaluation", "checkpoint_path", "checkpoint_sha256", "selection"]]
    return records, table.reset_index(drop=True)


# -------------------------------------------------------------------------- cohort
def load_and_verify_cohort(paths: dict[str, Path], monitoring: dict, artifacts: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    master = pd.read_parquet(paths["baseline_cohort_root"] / "master_include.parquet")
    master["sex_num"] = master.sex.map({"M": 0.0, "F": 1.0})  # same mapping as build_model_inputs.py
    test = master[master.split.eq(monitoring["held_out_split"])].reset_index(drop=True)
    checks = []
    digest = cohort_sha256(test)
    checks.append({"check": "test_cohort_sha256", "passed": digest == monitoring["expected_test_cohort_sha256"], "detail": digest})
    checks.append({"check": "test_cohort_size", "passed": len(test) == monitoring["expected_test_stays"] and int(test.label.sum()) == monitoring["expected_test_aki"], "detail": f"stays={len(test)}, aki={int(test.label.sum())}"})
    subjects = pd.read_csv(paths["test_subjects"]).subject_id
    checks.append({"check": "test_subject_file_matches", "passed": set(subjects) == set(test.subject_id), "detail": f"subjects={subjects.nunique()}"})
    leakage = master.groupby("subject_id").split.nunique().max() > 1
    checks.append({"check": "no_subject_in_multiple_splits", "passed": not leakage, "detail": "frozen seed-42 subject split"})
    for artifact in artifacts:
        predictions = pd.read_parquet(artifact["test_prediction_path"])
        same = artifact["test_cohort_sha256"] == digest and cohort_sha256(predictions) == digest and len(predictions) == len(test)
        checks.append({"check": f"{artifact['artifact_id']}_prediction_cohort", "passed": same, "detail": artifact["test_prediction_path"]})
    for horizon in sorted({a["prediction_window_hours"] for a in artifacts}):
        processed = pd.read_parquet(paths["baseline_processed_data_root"] / "mimiciv" / "full" / f"horizon_{horizon}" / "master.parquet")
        order_ok = processed.loc[processed.split.eq("test"), "stay_id"].tolist() == test.stay_id.tolist()
        checks.append({"check": f"h{horizon}_processed_test_order", "passed": order_ok, "detail": "processed master test rows in master_include order"})
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise MonitoringAbort(f"cohort verification failed: {failed}")
    return master, test, checks


def compute_reference_times(master: pd.DataFrame, test: pd.DataFrame, mode: str, step1, monitoring: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """AKI reference = KDIGO onset. Non-AKI reference = upstream matched pseudo-onset or ICU outtime."""
    frame = test[["subject_id", "hadm_id", "stay_id", "visit_id", "label", "intime", "outtime", "aki_dt", "los", "loo"]].copy()
    case = frame.label.eq(1)
    matching = None
    if mode == "matched":
        settings = monitoring["reference_time"]
        matching = step1.match_reference_times(
            master, "mimic-iv", seed=settings["matching_seed"], min_obs_hours=settings["matching_min_obs_hours"],
        )
        pseudo = frame.visit_id.map(matching.set_index("visit_id").pseudo_onset_hours)
        if pseudo[~case].isna().any():
            raise MonitoringAbort("matched reference time missing for a non-AKI test stay")
        pseudo_dt = frame.intime + pd.to_timedelta(pseudo, unit="h")
        # Upstream clips pseudo-onset to [min_obs_hours, los]; float hours -> ns can overshoot outtime by
        # ~1 ns when the clip hits los. Snap only such round-off back to outtime; larger gaps still abort below.
        roundoff = (pseudo_dt > frame.outtime) & ((pseudo_dt - frame.outtime) <= pd.Timedelta(milliseconds=1))
        pseudo_dt = pseudo_dt.mask(roundoff, frame.outtime)
        LOGGER.info("matched reference: %d non-AKI test stays snapped from <=1 ms past outtime (clip at los)", int((roundoff & ~case).sum()))
        frame["reference_dt"] = frame.aki_dt.where(case, pseudo_dt)
        frame["sampled_ratio"] = frame.visit_id.map(matching.set_index("visit_id").sampled_ratio)
    elif mode == "discharge":
        frame["reference_dt"] = frame.aki_dt.where(case, frame.outtime)
    else:
        raise MonitoringAbort(f"unknown reference mode {mode}")
    frame["reference_dt"] = pd.to_datetime(frame.reference_dt)
    frame["reference_hours_since_intime"] = (frame.reference_dt - frame.intime).dt.total_seconds() / 3600
    if (frame.reference_dt < frame.intime).any() or (frame.reference_dt > frame.outtime).any():
        raise MonitoringAbort(f"{mode} reference time outside the ICU stay")
    frame["reference_mode"] = mode
    return frame, matching


# -------------------------------------------------------------------------- events
# Verbatim from Step 1 build_model_inputs.py, except that the upper bound is the
# ICU outtime (a superset). Every time slice is cut from this superset with charttime <= cutoff.
EVENT_SQL = """
    WITH raw_events AS (
      SELECT c.stay_id, c.charttime, c.feature,
        CASE WHEN c.itemid=223761 THEN (c.valuenum-32)*5/9 ELSE c.valuenum END AS valuenum
      FROM read_parquet('{chart_path}') c JOIN selected_stays s USING(stay_id)
      WHERE c.feature IS NOT NULL AND c.charttime BETWEEN s.intime AND s.upper_dt
      UNION ALL
      SELECT s.stay_id, l.charttime, l.feature, l.valuenum
      FROM read_parquet('{labs_path}') l JOIN selected_stays s USING(hadm_id)
      WHERE l.charttime BETWEEN s.intime AND s.upper_dt
    )
    SELECT stay_id, charttime, feature, avg(valuenum) AS valuenum FROM raw_events
    WHERE valuenum IS NOT NULL AND isfinite(valuenum)
      AND (feature!='sbp' OR valuenum BETWEEN 40 AND 300)
      AND (feature!='dbp' OR valuenum BETWEEN 20 AND 200)
      AND (feature!='pulse' OR valuenum BETWEEN 20 AND 300)
      AND (feature!='resp' OR valuenum BETWEEN 1 AND 80)
      AND (feature!='spo2' OR valuenum BETWEEN 1 AND 100)
      AND (feature!='temp' OR valuenum BETWEEN 25 AND 45)
    GROUP BY stay_id, charttime, feature ORDER BY stay_id, charttime, feature
"""


def extract_stay_events(interim_root: Path, test: pd.DataFrame) -> pd.DataFrame:
    events_dir = interim_root / "mimiciv" / "events"
    quote = lambda path: str(path).replace("\\", "/").replace("'", "''")
    stays = test[["stay_id", "hadm_id", "intime"]].assign(upper_dt=test.outtime)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")
    con.register("selected_stays", stays)
    events = con.execute(EVENT_SQL.format(chart_path=quote(events_dir / "chartevents.parquet"), labs_path=quote(events_dir / "labevents.parquet"))).df()
    con.close()
    events["charttime"] = pd.to_datetime(events.charttime)
    return events


# ---------------------------------------------------------------------- time slices
def audit_patient_slice(slice_events: pd.DataFrame, intime, cutoff, irregular: np.ndarray, cnn: np.ndarray, cnn_hours: int) -> dict:
    """Independent re-check that no encoded input is later than the cutoff."""
    cutoff_h = (cutoff - intime).total_seconds() / 3600
    slice_future = int((slice_events.charttime > cutoff).sum())
    token_hours = irregular[:, 1].astype(float) if len(irregular) else np.empty(0)
    token_future = int((token_hours > cutoff_h + TIME_TOLERANCE_HOURS).sum())
    if len(slice_events):
        last = slice_events.charttime.max()
        last_h = (last - intime).total_seconds() / 3600
        token_consistent = bool(abs(token_hours.max() - last_h) <= TIME_TOLERANCE_HOURS)
        bins = np.floor((slice_events.charttime - cutoff).dt.total_seconds().to_numpy() / 3600).astype(int) + cnn_hours - 1
        cnn_future = int((bins > cnn_hours - 1).sum())
        inside = (bins >= 0) & (bins <= cnn_hours - 1)
        expected_cells = int(pd.DataFrame({"bin": bins[inside], "feature": slice_events.feature.to_numpy()[inside]}).drop_duplicates().shape[0])
    else:
        last, last_h, token_consistent, cnn_future, expected_cells = pd.NaT, np.nan, True, 0, 0
    observed_cells = int(np.isfinite(cnn).sum())
    return {
        "n_events_in_slice": int(len(slice_events)),
        "last_event_dt": last,
        "last_event_hours_since_intime": last_h,
        "slice_events_after_cutoff": slice_future,
        "irregular_tokens": int(len(irregular)),
        "irregular_tokens_after_cutoff": token_future,
        "irregular_last_token_consistent": token_consistent,
        "cnn_bins_after_cutoff": cnn_future,
        "cnn_cells_consistent": observed_cells == expected_cells,
        "future_events_detected": slice_future + token_future + cnn_future,
    }


def build_timepoint_inputs(
    frame: pd.DataFrame,
    cutoffs: pd.Series,
    events_by_stay: dict[int, pd.DataFrame],
    empty_events: pd.DataFrame,
    scale: pd.DataFrame,
    baseline: SimpleNamespace,
    apply_eligibility: bool,
    min_observations: int,
    reference_dt: pd.Series | None = None,
    next_cutoffs: pd.Series | None = None,
) -> tuple[dict[int, dict], pd.DataFrame]:
    cnn_hours = int(baseline.processing["cnn_hours"])
    max_events = int(baseline.processing["irregular_max_events"])
    work = frame.assign(cutoff_dt=pd.to_datetime(cutoffs).to_numpy())
    reference = pd.to_datetime(reference_dt).to_numpy() if reference_dt is not None else [pd.NaT] * len(work)
    upcoming = pd.to_datetime(next_cutoffs).to_numpy() if next_cutoffs is not None else [pd.NaT] * len(work)
    inputs: dict[int, dict] = {}
    audits = []
    for row, ref_dt, next_dt in zip(work.itertuples(index=False), reference, upcoming):
        stay_id = int(row.stay_id)
        source = events_by_stay.get(stay_id, empty_events)
        in_window = source[(source.charttime >= row.intime) & (source.charttime <= row.cutoff_dt)]
        record = {
            "stay_id": stay_id,
            "cutoff_dt": row.cutoff_dt,
            "cutoff_hours_since_intime": (row.cutoff_dt - row.intime).total_seconds() / 3600,
            "source_events_after_cutoff": int((source.charttime > row.cutoff_dt).sum()),
        }
        reason = None
        if row.cutoff_dt < row.intime:
            reason = "cutoff_before_icu_admission"
        elif len(in_window) < min_observations:
            reason = "no_timeseries_observation_before_cutoff"
        if apply_eligibility and reason is not None:
            record.update(eligible=False, ineligible_reason=reason, n_events_in_slice=0, future_events_detected=0)
            audits.append(record)
            continue
        irregular, cnn, demo, flat = baseline.build_one(row, in_window, scale, cnn_hours, max_events)
        record.update(audit_patient_slice(in_window, row.intime, row.cutoff_dt, irregular, cnn, cnn_hours))
        record.update(eligible=True, ineligible_reason=None)
        if not pd.isna(ref_dt) and len(in_window):
            record["last_event_offset_from_reference_h"] = (record["last_event_dt"] - pd.Timestamp(ref_dt)).total_seconds() / 3600
        if not pd.isna(next_dt):
            record["events_after_next_timepoint"] = int((in_window.charttime > pd.Timestamp(next_dt)).sum())
        inputs[stay_id] = {
            "irregular": irregular,
            "cnn": cnn,
            "demo": demo,
            # Same column construction as build_model_inputs.py flat_features.parquet
            "flat": {**flat, "age": row.age, "sex": row.sex_num, "bmi": row.bmi},
        }
        audits.append(record)
    return inputs, pd.DataFrame(audits)


def validator_self_test(baseline: SimpleNamespace, scale: pd.DataFrame) -> dict:
    """Negative control: the audit must flag an input that contains a post-cutoff event."""
    cnn_hours = int(baseline.processing["cnn_hours"])
    max_events = int(baseline.processing["irregular_max_events"])
    intime = pd.Timestamp("2150-01-01 00:00")
    cutoff = intime + pd.Timedelta(hours=48)
    events = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "charttime": [intime + pd.Timedelta(hours=1), cutoff, cutoff + pd.Timedelta(minutes=1)],
        "feature": ["sbp", "creatinine", "sbp"],
        "valuenum": [120.0, 1.1, 118.0],
    })
    row = SimpleNamespace(age=60, sex_num=0.0, bmi=25.0, intime=intime, cutoff_dt=cutoff)
    clean = events[events.charttime <= cutoff]
    irregular, cnn, _, _ = baseline.build_one(row, clean, scale, cnn_hours, max_events)
    clean_audit = audit_patient_slice(clean, intime, cutoff, irregular, cnn, cnn_hours)
    irregular_leak, cnn_leak, _, _ = baseline.build_one(row, events, scale, cnn_hours, max_events)
    leak_audit = audit_patient_slice(events, intime, cutoff, irregular_leak, cnn_leak, cnn_hours)
    passed = (
        clean_audit["future_events_detected"] == 0
        and leak_audit["slice_events_after_cutoff"] == 1
        and leak_audit["irregular_tokens_after_cutoff"] == 1
        and leak_audit["future_events_detected"] >= 2
    )
    return {"passed": bool(passed), "clean": clean_audit["future_events_detected"], "leaked": leak_audit["future_events_detected"]}


def compare_with_baseline_inputs(stay_ids: list[int], inputs: dict[int, dict], processed_dir: Path, features: list[str]) -> dict:
    """Check that re-encoding at the baseline cutoff reproduces the saved phase-one model inputs."""
    flat = pd.read_parquet(processed_dir / "flat_features.parquet").set_index("stay_id")
    mismatches = {"irregular": 0, "cnn": 0, "demo": 0, "flat": 0}
    for stay_id in stay_ids:
        mine = inputs[stay_id]
        with np.load(processed_dir / "sequences" / f"{stay_id}.npz") as saved:
            mismatches["irregular"] += not np.array_equal(saved["irregular"], mine["irregular"])
            mismatches["cnn"] += not np.array_equal(saved["cnn"], mine["cnn"], equal_nan=True)
            mismatches["demo"] += not np.array_equal(saved["demo"], mine["demo"])
        expected = flat.loc[stay_id, features].to_numpy(float)
        observed = np.array([mine["flat"][f] for f in features], dtype=float)
        mismatches["flat"] += not np.allclose(expected, observed, rtol=0, atol=1e-9, equal_nan=True)
    return {"n_stays": len(stay_ids), **{f"{key}_mismatches": int(value) for key, value in mismatches.items()}}


# -------------------------------------------------------------------------- models
def _forbid_fitting(obj, label: str) -> None:
    def blocked(*_args, **_kwargs):
        raise MonitoringAbort(f"fitting {label} is forbidden during continuous monitoring")

    for name in ("fit", "fit_transform", "partial_fit"):
        if hasattr(obj, name):
            setattr(obj, name, blocked)


class InMemorySequenceDataset:
    """Item contract of Baseline scripts/run_baseline.py::SequenceDataset, reading arrays from memory."""

    def __init__(self, stay_ids: list[int], inputs: dict[int, dict], kind: str, max_events: int | None):
        import torch

        self.stay_ids, self.inputs, self.kind, self.max_events, self.torch = stay_ids, inputs, kind, max_events, torch

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        data = self.inputs[self.stay_ids[idx]]
        label = self.torch.tensor(0, dtype=self.torch.long)  # labels are never used for inference
        if self.kind == "cnn":
            return (
                self.torch.tensor(data["demo"], dtype=self.torch.float32),
                self.torch.tensor(np.nan_to_num(data["cnn"], nan=0.0), dtype=self.torch.float32),
                label,
            )
        irregular = data["irregular"]
        if self.max_events is not None and len(irregular) > self.max_events:
            demo_count = min(3, len(irregular))
            irregular = np.concatenate([irregular[:demo_count], irregular[-(self.max_events - demo_count):]], axis=0)
        return self.torch.tensor(irregular, dtype=self.torch.float32), label


class FrozenModel:
    """One manifest artifact, loaded read-only with its authoritative preprocessing."""

    KIND = {"masked_cnn": "cnn", "lstm_attention": "lstm", "ite_transformer": "transformer"}

    def __init__(self, artifact: dict, baseline: SimpleNamespace, paths: dict[str, Path], device: str):
        self.artifact = artifact
        self.model_name = artifact["model"]
        self.horizon = int(artifact["prediction_window_hours"])
        self.config = json.loads(Path(artifact["config_path"]).read_text(encoding="utf-8"))
        if (self.config["scope"], self.config["model"], int(self.config["prediction_horizon_hours"])) != ("full", self.model_name, self.horizon):
            raise MonitoringAbort(f"config.json does not describe {artifact['artifact_id']}")
        metric_path = paths["baseline_results_root"] / "metrics" / "full" / f"horizon_{self.horizon}" / f"{self.model_name}.json"
        saved_threshold = json.loads(metric_path.read_text(encoding="utf-8"))["threshold"]
        self.threshold = float(self.config["metrics"]["threshold"])
        if not np.isfinite(self.threshold) or self.threshold != float(saved_threshold):
            raise MonitoringAbort(f"frozen validation threshold mismatch for {artifact['artifact_id']}")
        self.features = list(baseline.config.ALL_FEATURES)
        self.scale_path = Path(artifact["normalization_table_path"])
        self.scale = pd.read_parquet(self.scale_path)
        self.tracked_files = {name: Path(artifact[name]) for name in ("checkpoint_path", "scaler_path", "normalization_table_path", "config_path")}
        self.file_hashes_before = {name: sha256_file(path) for name, path in self.tracked_files.items()}
        if self.file_hashes_before["checkpoint_path"] != artifact["checkpoint_sha256"]:
            raise MonitoringAbort(f"checkpoint SHA-256 mismatch for {artifact['artifact_id']}")

        if self.model_name in ML_MODELS:
            self.estimator = joblib.load(artifact["checkpoint_path"])
            if artifact["scaler_path"] != artifact["imputer_path"] or not artifact["scaler_path"].endswith("preprocessing.joblib"):
                raise MonitoringAbort("ML preprocessing must come from preprocessing.joblib")
            bundle = joblib.load(artifact["scaler_path"])
            self.imputer, self.scaler = bundle["imputer"], bundle["scaler"]
            # preprocessing.joblib["features"] is the authoritative ML column order (not feature_list.json).
            self.columns = list(bundle["features"])
            if sorted(self.columns) != sorted(self.features) or len(self.columns) != 51:
                raise MonitoringAbort("preprocessing.joblib feature vocabulary differs from the 51-feature schema")
            json_order = json.loads(Path(artifact["feature_list_path"]).read_text(encoding="utf-8"))
            self.feature_order_note = "bundle order differs from feature_list.json (bundle used)" if json_order != self.columns else "bundle order equals feature_list.json"
            for obj, label in ((self.estimator, "model"), (self.imputer, "imputer"), (self.scaler, "scaler")):
                _forbid_fitting(obj, f"{artifact['artifact_id']} {label}")
        else:
            import torch

            self.torch = torch
            self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(artifact["checkpoint_path"], map_location=self.device, weights_only=False)
            if checkpoint["features"] != self.features or checkpoint["model_config"].get("type_dict", baseline.config.FEATURE_TO_ID) != baseline.config.FEATURE_TO_ID:
                raise MonitoringAbort("checkpoint feature encoding differs from the baseline schema")
            upstream = paths["paper_repo_root"]
            model_config = checkpoint["model_config"]
            if self.model_name == "masked_cnn":
                model_class = baseline.run.load_symbol(upstream / "train_cnn" / "models" / "mask_rms_1d_cnn.py", "Timeseries_CNN_Model")
                self.net = model_class(**model_config)
                self.collate = baseline.run.cnn_collate
            elif self.model_name == "lstm_attention":
                model_class = baseline.run.load_symbol(upstream / "train_lstm_attention" / "src" / "models" / "lstm_attention.py", "LSTMAttentionModel")
                self.net = model_class(model_config)
                self.collate = baseline.run.irregular_collate
            else:
                model_class = baseline.run.load_symbol(upstream / "train_transformer" / "src" / "models" / "ite_transformer.py", "ITETransformer")
                self.net = model_class(model_config)
                self.collate = baseline.run.irregular_collate
            self.net.load_state_dict(checkpoint["state_dict"], strict=True)
            self.net.to(self.device).eval()
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)
            self.kind = self.KIND[self.model_name]
            self.max_events = self.config["hyperparameters"].get("adapter_max_events")
            self.batch_size = int(self.config["hyperparameters"]["batch_size"])
            self.best_epoch = checkpoint.get("epoch")
        self.fingerprint_before = self.fingerprint()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        if self.model_name in ML_MODELS:
            if self.model_name == "xgboost":
                digest.update(bytes(self.estimator.get_booster().save_raw(raw_format="ubj")))
            else:
                digest.update(np.ascontiguousarray(self.estimator.coef_).tobytes())
                digest.update(np.ascontiguousarray(self.estimator.intercept_).tobytes())
            digest.update(np.ascontiguousarray(self.imputer.statistics_).tobytes())
            digest.update(np.ascontiguousarray(self.scaler.center_).tobytes())
            digest.update(np.ascontiguousarray(self.scaler.scale_).tobytes())
        else:
            for key, tensor in self.net.state_dict().items():
                digest.update(key.encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(self.scale, index=True).to_numpy().tobytes())
        return digest.hexdigest()

    def predict(self, stay_ids: list[int], inputs: dict[int, dict]) -> np.ndarray:
        if not stay_ids:
            return np.empty(0, dtype=float)
        if self.model_name in ML_MODELS:
            matrix = pd.DataFrame([inputs[s]["flat"] for s in stay_ids])[self.columns].to_numpy(float)
            return self.estimator.predict_proba(self.scaler.transform(self.imputer.transform(matrix)))[:, 1]
        torch = self.torch
        from torch.utils.data import DataLoader

        loader = DataLoader(InMemorySequenceDataset(stay_ids, inputs, self.kind, self.max_events), batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=0)
        if self.net.training:
            raise MonitoringAbort("model left eval mode")
        values = []
        with torch.inference_mode():
            for batch in loader:
                if self.kind == "cnn":
                    demo, sequence, _ = batch
                    logits = self.net(demo.to(self.device), sequence.to(self.device))
                else:
                    sequence, mask, _ = batch
                    logits = self.net(sequence.to(self.device), mask.to(self.device))
                values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(values)

    def integrity_after(self) -> dict:
        hashes_after = {name: sha256_file(path) for name, path in self.tracked_files.items()}
        fingerprint_after = self.fingerprint()
        return {
            "artifact_id": self.artifact["artifact_id"],
            "fingerprint_before": self.fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "parameters_unchanged": fingerprint_after == self.fingerprint_before,
            "artifact_files_unchanged": hashes_after == self.file_hashes_before,
            **{f"{name}_sha256": value for name, value in hashes_after.items()},
        }


# ------------------------------------------------------------------------- metrics
def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Discrimination and frozen-threshold metrics; undefined values are NaN with a stated reason."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n, n_pos = len(y_true), int(y_true.sum())
    n_neg = n - n_pos
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    ratio = lambda num, den: num / den if den > 0 else np.nan
    notes = []
    if n == 0:
        notes.append("no_eligible_patients")
    if n_pos == 0 or n_neg == 0:
        auroc = auprc = np.nan
        notes.append("single_class:auroc_auprc_undefined")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    ppv = ratio(tp, tp + fp)
    if tp + fp == 0:
        notes.append("no_alerts:ppv_nne_undefined")
    elif tp == 0:
        notes.append("ppv_zero:nne_undefined")
    for name, den in (("sensitivity", tp + fn), ("specificity", tn + fp), ("npv", tn + fn)):
        if den == 0 and n:
            notes.append(f"{name}_undefined")
    return {
        "N": n, "N_positive": n_pos, "N_negative": n_neg, "prevalence": ratio(n_pos, n),
        "auroc": auroc, "auprc": auprc,
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
        "ppv": ppv, "npv": ratio(tn, tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn), "accuracy": ratio(tp + tn, n),
        "nne": 1.0 / ppv if (not np.isnan(ppv) and ppv > 0) else np.nan,
        "alert_rate": ratio(tp + fp, n), "false_positive_rate": ratio(fp, fp + tn),
        "brier_score": float(brier_score_loss(y_true, y_prob)) if n else np.nan,
        "metric_notes": ";".join(notes),
    }
