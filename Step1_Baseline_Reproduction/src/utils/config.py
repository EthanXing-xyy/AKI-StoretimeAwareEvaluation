"""Load repository-relative configuration and define feature mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from project_paths import MIMIC_IV_ZIP, repo_path  # noqa: E402


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_paths() -> dict[str, Path]:
    raw = load_yaml(PROJECT_ROOT / "configs" / "paths.yaml")
    paths = {key: repo_path(value) for key, value in raw.items()}
    paths["mimic_zip"] = MIMIC_IV_ZIP
    return paths


def load_baseline_config() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "baseline.yaml")


DEMO_FEATURES = ["age", "sex", "bmi"]
VITAL_FEATURES = ["sbp", "dbp", "pulse", "resp", "spo2", "temp"]
LAB_FEATURES = [
    "albumin", "alp", "alt", "aptt", "ast", "bilirubin", "bnp", "bun",
    "calcium", "chloride", "cholesterol", "ck", "creatinine", "crp", "esr",
    "ggt", "glucose", "hba1c", "hco3", "hdl", "hematocrit", "hemoglobin",
    "lactate", "ldh", "magnesium", "pco2", "pdw", "ph", "phosphorus",
    "platelet", "po2", "potassium", "procalcitonin", "protein", "pt_inr",
    "rbc", "rdw", "sodium", "tco2", "triglyceride", "uric_acid", "wbc",
]
TS_FEATURES = VITAL_FEATURES + LAB_FEATURES
ALL_FEATURES = DEMO_FEATURES + TS_FEATURES
FEATURE_TO_ID = {feature: idx for idx, feature in enumerate(ALL_FEATURES)}


# Mappings are explicit so every extraction is reproducible. The newer 52/53-series
# ids are included for MIMIC-IV 3.1 where they represent the same blood analyte.
LAB_ITEMIDS: dict[str, list[int]] = {
    "albumin": [50862, 53085],
    "alp": [50863, 53086],
    "alt": [50861, 53084],
    "aptt": [51275, 52923],
    "ast": [50878],
    "bilirubin": [50885, 53089],
    "bnp": [50963],
    "bun": [51006, 52647],
    "calcium": [50893],
    "chloride": [50806, 50902, 52434, 52535],
    "cholesterol": [],  # paper: not routinely collected in MIMIC-IV
    "ck": [50910],
    "creatinine": [50912, 52024, 52546],
    "crp": [50889],
    "esr": [51288],
    "ggt": [50927, 53093],
    "glucose": [50809, 50931, 52027, 52569],
    "hba1c": [50852],
    "hco3": [50803, 50882],
    "hdl": [50904],
    "hematocrit": [50810, 51221, 51638, 51639, 52028],
    "hemoglobin": [50811, 51222, 51640],
    "lactate": [50813, 52442, 53154],
    "ldh": [50954],
    "magnesium": [50960],
    "pco2": [50818, 52040],
    "pdw": [],  # paper: unavailable in MIMIC-IV
    "ph": [50820],
    "phosphorus": [50970],
    "platelet": [51265, 53189],
    "po2": [50821],
    "potassium": [50822, 50971, 52452, 52610],
    "procalcitonin": [],  # paper: unavailable in MIMIC-IV
    "protein": [50976],
    "pt_inr": [51237, 51675],
    "rbc": [51279],
    "rdw": [51277, 52172, 53165, 53166],
    "sodium": [50824, 50983, 52455, 52623],
    "tco2": [50804, 51739],
    "triglyceride": [51000],
    "uric_acid": [51007],
    "wbc": [51300, 51755, 51756],
}

CHART_ITEMIDS: dict[str, list[int]] = {
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "pulse": [220045],
    "resp": [220210, 224690],
    "spo2": [220277],
    "temp": [223761, 223762],
}
WEIGHT_ITEMIDS = [224639, 226512, 226531]
HEIGHT_ITEMIDS = [226707, 226730]
URINE_ITEMIDS = [
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227488, 227489,
]
CREATININE_LABEL_ITEMID = 50912  # MIT-LCP KDIGO concept


def reverse_item_mapping(mapping: dict[str, list[int]]) -> dict[int, str]:
    return {itemid: feature for feature, itemids in mapping.items() for itemid in itemids}
