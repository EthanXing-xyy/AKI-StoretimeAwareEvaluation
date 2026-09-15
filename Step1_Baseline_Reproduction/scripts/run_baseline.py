"""Train and evaluate one baseline model and prediction horizon."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_metrics, youden_threshold
from src.utils.config import ALL_FEATURES, DEMO_FEATURES, FEATURE_TO_ID, TS_FEATURES, load_baseline_config, load_paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_outputs(model_name: str, horizon: int, scope: str, master: pd.DataFrame, y_prob: np.ndarray, threshold: float, metrics: dict, paths: dict, record: dict) -> None:
    test = master.loc[master.split == "test", ["subject_id", "hadm_id", "stay_id", "label"]].reset_index(drop=True)
    test = test.rename(columns={"label": "y_true"})
    test["y_prob"] = y_prob
    test["y_pred"] = (y_prob >= threshold).astype(int)
    pred_dir = paths["results_root"] / "predictions" / scope / f"horizon_{horizon}"
    metric_dir = paths["results_root"] / "metrics" / scope / f"horizon_{horizon}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    test.to_parquet(pred_dir / f"{model_name}.parquet", index=False)
    test.to_csv(pred_dir / f"{model_name}.csv", index=False)
    (metric_dir / f"{model_name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "confusion_matrix"}]).to_csv(metric_dir / f"{model_name}.csv", index=False)
    log_dir = paths["logs_root"] / scope
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{record['experiment_id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def train_ml(model_name: str, master: pd.DataFrame, data_dir: Path, model_dir: Path, cfg: dict, device: str):
    flat = pd.read_parquet(data_dir / "flat_features.parquet").set_index("stay_id")
    columns = TS_FEATURES + DEMO_FEATURES
    sets = {}
    for split in ["train", "val", "test"]:
        meta = master[master.split == split]
        sets[split] = (flat.loc[meta.stay_id, columns].to_numpy(float), meta.label.to_numpy(int))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(sets["train"][0]))
    x_val = scaler.transform(imputer.transform(sets["val"][0]))
    x_test = scaler.transform(imputer.transform(sets["test"][0]))
    y_train, y_val, y_test = sets["train"][1], sets["val"][1], sets["test"][1]
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="liblinear", random_state=cfg["seed"])
        hyperparameters = model.get_params()
        model.fit(x_train, y_train)
    else:
        scale_pos = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
        hyperparameters = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, eval_metric="logloss", early_stopping_rounds=20, random_state=cfg["seed"], n_jobs=-1, tree_method="hist", device="cuda" if device == "cuda" else "cpu")
        model = XGBClassifier(**hyperparameters)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model_best.joblib")
    joblib.dump({"imputer": imputer, "scaler": scaler, "features": columns}, model_dir / "preprocessing.joblib")
    if model_name == "logistic_regression":
        history = [{"iteration": int(np.max(model.n_iter_)), "converged": bool(np.max(model.n_iter_) < model.max_iter)}]
    else:
        eval_history = model.evals_result().get("validation_0", {}).get("logloss", [])
        history = [{"iteration": i + 1, "val_logloss": value} for i, value in enumerate(eval_history)]
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    return test_prob, youden_threshold(y_val, val_prob), hyperparameters, None


def load_symbol(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"upstream_{path.stem}_{symbol}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, symbol)


class SequenceDataset:
    def __init__(self, meta, sequence_dir, kind, max_events=None):
        import torch
        self.meta = meta.reset_index(drop=True)
        self.sequence_dir = sequence_dir
        self.kind = kind
        self.max_events = max_events
        self.torch = torch

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        with np.load(self.sequence_dir / f"{int(row.stay_id)}.npz") as data:
            if self.kind == "cnn":
                return (
                    self.torch.tensor(data["demo"], dtype=self.torch.float32),
                    self.torch.tensor(np.nan_to_num(data["cnn"], nan=0.0), dtype=self.torch.float32),
                    self.torch.tensor(int(row.label), dtype=self.torch.long),
                )
            irregular = data["irregular"]
            if self.max_events is not None and len(irregular) > self.max_events:
                demo_count = min(3, len(irregular))
                irregular = np.concatenate([irregular[:demo_count], irregular[-(self.max_events-demo_count):]], axis=0)
            return (
                self.torch.tensor(irregular, dtype=self.torch.float32),
                self.torch.tensor(int(row.label), dtype=self.torch.long),
            )


def irregular_collate(batch):
    import torch
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences])
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.stack(labels)


def cnn_collate(batch):
    import torch
    demo, sequence, label = zip(*batch)
    return torch.stack(demo), torch.stack(sequence), torch.stack(label)


def train_deep(model_name: str, master: pd.DataFrame, data_dir: Path, model_dir: Path, cfg: dict, device_name: str, smoke: bool):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    upstream = load_paths()["paper_repo_root"]
    if model_name == "masked_cnn":
        model_class = load_symbol(upstream / "train_cnn" / "models" / "mask_rms_1d_cnn.py", "Timeseries_CNN_Model")
        model_config = {"num_demo_features": 3, "num_timeseries_features": 48, "num_classes": 2, "seq_len": cfg["processing"]["cnn_hours"]}
        model = model_class(**model_config)
        kind, collate, max_events = "cnn", cnn_collate, None
    elif model_name == "lstm_attention":
        model_class = load_symbol(upstream / "train_lstm_attention" / "src" / "models" / "lstm_attention.py", "LSTMAttentionModel")
        model_config = {"d_model": 64, "num_layers": 2, "num_lstm_layers": 2, "num_heads": 4, "dropout": 0.1, "num_classes": 2, "bidirectional": True, "type_dict": FEATURE_TO_ID, "max_seq_len": cfg["processing"]["irregular_max_events"], "gradient_checkpointing": False, "ffn_expansion": 2}
        model = model_class(model_config)
        kind, collate, max_events = "lstm", irregular_collate, 512
    else:
        model_class = load_symbol(upstream / "train_transformer" / "src" / "models" / "ite_transformer.py", "ITETransformer")
        model_config = {"d_model": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1, "num_classes": 2, "type_dict": FEATURE_TO_ID}
        model = model_class(model_config)
        kind, collate, max_events = "transformer", irregular_collate, 1024

    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    seq_dir = data_dir / "sequences"
    datasets = {split: SequenceDataset(master[master.split == split], seq_dir, kind, max_events=max_events) for split in ["train", "val", "test"]}
    train_labels = datasets["train"].meta.label.to_numpy(int)
    counts = np.bincount(train_labels, minlength=2)
    weights = np.array([1 / max(counts[label], 1) for label in train_labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    batch_size = cfg["training"]["batch_size"] if kind == "cnn" else (16 if kind == "lstm" else 8)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, collate_fn=collate, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    max_epochs = cfg["training"]["smoke_max_epochs"] if smoke else cfg["training"]["max_epochs"]
    patience = min(cfg["training"]["early_stopping_patience"], max_epochs)
    history, best_loss, stale = [], float("inf"), 0
    model_dir.mkdir(parents=True, exist_ok=True)

    def forward(batch):
        if kind == "cnn":
            demo, sequence, label = batch
            return model(demo.to(device), sequence.to(device)), label.to(device)
        sequence, mask, label = batch
        return model(sequence.to(device), mask.to(device)), label.to(device)

    def loss_fn(logits, labels):
        one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
        cosine = torch.nn.functional.cosine_embedding_loss(logits, one_hot, torch.ones(len(labels), device=device))
        return cosine + 0.1 * torch.nn.functional.cross_entropy(logits, labels)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["val"]:
                logits, labels = forward(batch)
                val_losses.append(float(loss_fn(logits, labels).cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.5f} val_loss={row['val_loss']:.5f}", flush=True)
        if row["val_loss"] < best_loss:
            best_loss, stale = row["val_loss"], 0
            torch.save({"state_dict": model.state_dict(), "model_config": model_config, "features": ALL_FEATURES, "seed": cfg["seed"], "epoch": epoch, "val_loss": best_loss}, model_dir / "model_best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    checkpoint = torch.load(model_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def probabilities(loader):
        values = []
        with torch.no_grad():
            for batch in loader:
                logits, _ = forward(batch)
                values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(values)
    val_prob = probabilities(loaders["val"])
    test_prob = probabilities(loaders["test"])
    threshold = youden_threshold(datasets["val"].meta.label.to_numpy(int), val_prob)
    hyperparameters = {**model_config, "optimizer": "AdamW", "learning_rate": cfg["training"]["learning_rate"], "batch_size": batch_size, "adapter_max_events": max_events, "max_epochs": max_epochs, "loss": "upstream Cosine_Loss formula"}
    return test_prob, threshold, hyperparameters, checkpoint["epoch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one paper baseline on adapted MIMIC-IV")
    parser.add_argument("--model", choices=["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"], required=True)
    parser.add_argument("--horizon", type=int, choices=[0, 48, 72], required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    paths, cfg = load_paths(), load_baseline_config()
    set_seed(cfg["seed"])
    data_dir = paths["processed_data_root"] / "mimiciv" / args.scope / f"horizon_{args.horizon}"
    master = pd.read_parquet(data_dir / "master.parquet")
    model_dir = paths["model_root"] / args.model / f"horizon_{args.horizon}" / args.scope
    started = time.time()
    experiment_id = f"{args.scope}_{args.model}_h{args.horizon}_seed{cfg['seed']}"
    if args.model in ["logistic_regression", "xgboost"]:
        test_prob, threshold, hyperparameters, best_epoch = train_ml(args.model, master, data_dir, model_dir, cfg, args.device)
    else:
        test_prob, threshold, hyperparameters, best_epoch = train_deep(args.model, master, data_dir, model_dir, cfg, args.device, args.scope == "smoke")
    for artifact_name in ["scaling_info.parquet", "feature_list.json"]:
        source = data_dir / artifact_name
        if source.exists():
            shutil.copy2(source, model_dir / artifact_name)
    test_y = master.loc[master.split == "test", "label"].to_numpy(int)
    metrics = binary_metrics(test_y, test_prob, threshold)
    metrics.update({"model": args.model, "prediction_window": args.horizon, "scope": args.scope, "n_train": int((master.split == "train").sum()), "n_val": int((master.split == "val").sum()), "seed": cfg["seed"]})
    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prediction_horizon_hours": args.horizon,
        "scope": args.scope,
        "dataset_size": len(master),
        "positive_prevalence": float(master.label.mean()),
        "split_counts": master.split.value_counts().to_dict(),
        "features": ALL_FEATURES,
        "hyperparameters": hyperparameters,
        "seed": cfg["seed"],
        "training_seconds": time.time() - started,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "checkpoint_path": str(model_dir),
    }
    (model_dir / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    save_outputs(args.model, args.horizon, args.scope, master, test_prob, threshold, metrics, paths, record)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
