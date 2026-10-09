"""Controlled frozen-feature TorNet adaptations with a sealed final evaluation.

The only fit inputs are adaptation-training rows. Validation chooses checkpoints,
configuration and thresholds. ``test_once`` applies that saved decision and saves
predictions before making figures or resampling them. No unselected adaptation is
evaluated on test. Paths can be supplied explicitly; defaults are local to this
study so the previously completed MRMS project stays untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


CONFIGURATIONS = ("baseline", "history", "hard_negatives", "nonlinear_head")
REFERENCES = ("majority", "original_cnn", "radar_statistics", "baseline")
METRICS = ("average_precision", "precision", "recall", "f1", "false_positive_rate",
           "warning_negative_false_positive_rate", "accuracy")
STATISTIC_NAMES = ("DBZ_p95", "DBZ_fraction_ge40", "VEL_p95_minus_p05", "RHOHV_p05",
                   "ZDR_median", "KDP_p95", "WIDTH_p95", "radar_range_midpoint_m")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Creating a hard link commits a fully written file atomically and fails
        # if another process has already committed the same exclusive lock.
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_write(path, frame, *, exclusive=False, before_commit=None):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            # Preserve float32 scores exactly when reloaded as float64; otherwise
            # a rounded score at a locked threshold could change classification.
            frame.to_csv(handle, index=False, float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        if before_commit is not None:
            before_commit(sha256(temporary))
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    return value


def append_event(results_dir, stage, status, **details):
    event = {"time_utc": utc_now(), "stage": stage, "status": status, **details}
    with (results_dir / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(finite_json(event), allow_nan=False) + "\n")


def resolve_paths(study_dir=None, data_dir=None):
    study = Path(study_dir or Path(__file__).parent).resolve()
    data = Path(data_dir).resolve() if data_dir else study / "data"
    results = study / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "models").mkdir(exist_ok=True)
    (results / "figures").mkdir(exist_ok=True)
    return study, data, results


def load_data(data):
    """Require an explicit cache alignment receipt, not merely matching row counts."""
    manifest_path = data / "runtime_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"filename": str, "storm_group": str})
    if "split" not in manifest and "study_split" in manifest:
        manifest["split"] = manifest["study_split"]
    required = {"filename", "split", "label", "category", "storm_group"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    if manifest.filename.isna().any() or not manifest.filename.is_unique:
        raise ValueError("Manifest filenames must be nonempty and unique")
    if manifest.storm_group.isna().any():
        raise ValueError("Manifest storm_group values must be nonempty")
    manifest["split"] = manifest.split.replace({"val": "validation"})
    if set(manifest.split) != {"train", "validation", "test"}:
        raise ValueError("Manifest must have exactly train/validation/test split names")
    if not set(manifest.label.unique()).issubset({0, 1}):
        raise ValueError("Terminal-frame labels must be binary")
    manifest["label"] = manifest.label.astype(int)
    if not set(manifest.category).issubset({"TOR", "WRN", "NUL"}):
        raise ValueError("Unrecognized catalog category")
    for column in ("storm_group", "episode_id", "event_id"):
        if column in manifest:
            overlap = manifest.dropna(subset=[column]).groupby(column).split.nunique()
            if overlap.max() > 1:
                raise ValueError(f"{column} occurs in multiple study splits")
    arrays = {}
    for name in ("features_current", "features_previous", "original_cnn_scores", "radar_statistics"):
        arrays[name] = np.load(data / f"{name}.npy")
        if len(arrays[name]) != len(manifest):
            raise ValueError(f"{name} row count does not match manifest")
    for name in ("features_current", "features_previous"):
        if arrays[name].shape != (len(manifest), 384) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid {name} shape or nonfinite values")
    original = np.asarray(arrays["original_cnn_scores"]).reshape(-1)
    if len(original) != len(manifest) or not np.isfinite(original).all() or ((original < 0) | (original > 1)).any():
        raise ValueError("Original CNN scores must be finite sigmoid scores in [0,1]")
    arrays["original_cnn_scores"] = original
    if arrays["radar_statistics"].shape != (len(manifest), 8) or np.isinf(arrays["radar_statistics"]).any():
        raise ValueError("Radar statistics must contain eight finite-or-NaN physical-unit columns")
    receipt_path = data / "feature_provenance.json"
    if not receipt_path.exists():
        receipt_path = data / "feature_cache.json"
    if not receipt_path.exists():
        raise ValueError("Missing feature_provenance.json alignment receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = sha256(manifest_path)
    if receipt.get("manifest_sha256") != expected:
        raise ValueError("Feature cache receipt does not match runtime manifest SHA-256")
    if receipt.get("feature_rows", receipt.get("samples")) != len(manifest):
        raise ValueError("Feature cache receipt must record exact feature_rows")
    if "radar_statistics_columns" in receipt and tuple(receipt["radar_statistics_columns"]) != STATISTIC_NAMES:
        raise ValueError("Radar-statistics feature names/order do not match the declared baseline")
    sample_ids_path = data / "features_sample_ids.json"
    if not sample_ids_path.exists():
        raise ValueError("Missing ordered features_sample_ids.json alignment proof")
    sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if sample_ids != manifest.filename.tolist():
        raise ValueError("Feature sample IDs differ from runtime manifest order")
    for name in arrays:
        filename = f"{name}.npy"
        cache_file = receipt.get("files", {}).get(filename, receipt.get("files", {}).get(name, {}))
        if cache_file.get("sha256") and cache_file["sha256"] != sha256(data / filename):
            raise ValueError(f"Feature provenance hash mismatch: {filename}")
    return manifest, arrays


def input_fingerprint(study, data):
    receipt_path = data / "feature_provenance.json"
    if not receipt_path.exists():
        receipt_path = data / "feature_cache.json"
    paths = [study / "planned_config.json", data / "runtime_manifest.csv", receipt_path,
             data / "features_sample_ids.json"]
    paths += [data / f"{name}.npy" for name in (
        "features_current", "features_previous", "original_cnn_scores", "radar_statistics")]
    checkpoint = study / "probe" / "tornado_detector_baseline.keras"
    if checkpoint.exists():
        paths.append(checkpoint)
    payload = {path.relative_to(study).as_posix() if path.is_relative_to(study) else path.as_posix(): sha256(path)
               for path in paths}
    overall = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"sha256": overall, "files": payload}


def assert_audit(manifest):
    audit = {}
    for split in ("train", "validation", "test"):
        subset = manifest.loc[manifest.split == split]
        positives = subset.loc[subset.label == 1]
        warning_negatives = subset.loc[(subset.category == "WRN") & (subset.label == 0)]
        info = {"clips": len(subset), "positive_clips": len(positives),
                "storm_groups": subset.storm_group.nunique(),
                "positive_storm_groups": positives.storm_group.nunique(),
                "warning_negative_clips": len(warning_negatives),
                "warning_negative_storm_groups": warning_negatives.storm_group.nunique()}
        audit[split] = {key: int(value) for key, value in info.items()}
        if subset.label.nunique() != 2:
            raise ValueError(f"{split} must contain both actual target classes")
        if split != "train" and (len(positives) < 40 or positives.storm_group.nunique() < 8
                                  or warning_negatives.storm_group.nunique() < 3):
            raise ValueError(f"Predeclared evaluation label/group audit failed: {split}: {info}")
    return audit


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_head(configuration):
    if configuration == "nonlinear_head":
        return nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    return nn.Linear(384, 1)


def predict_head(head, x):
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).reshape(-1)).numpy()


def representation(configuration, current, previous):
    return np.asarray(0.5 * (current + previous) if configuration == "history" else current, dtype=np.float32)


def sample_weights(y, category, warning_multiplier=1):
    counts = np.bincount(y.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training loss requires both classes")
    weights = len(y) / (2.0 * counts[y.astype(int)])
    weights[(category == "WRN") & (y == 0)] *= warning_multiplier
    weights /= weights.mean()
    return weights.astype(np.float32)


def recall_threshold(y, scores, target=0.8):
    """Largest observed score whose >= rule retains the required positive count."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid aligned scores for threshold selection")
    positives = np.sort(scores[y == 1])[::-1]
    if not len(positives):
        raise ValueError("Cannot select a recall threshold without positives")
    rank = math.ceil(target * len(positives)) - 1
    threshold = float(positives[rank])
    if np.mean(scores[y == 1] >= threshold) + 1e-12 < target:
        raise AssertionError("Threshold failed its declared recall target")
    return threshold


def metrics(y, scores, threshold, category, *, constant_prediction=None):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = np.full(len(y), constant_prediction, dtype=int) if constant_prediction is not None else (scores >= threshold).astype(int)
    positive, negative = y == 1, y == 0
    tp = int(np.sum(positive & (predicted == 1)))
    fn = int(np.sum(positive & (predicted == 0)))
    fp = int(np.sum(negative & (predicted == 1)))
    tn = int(np.sum(negative & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    warning = (np.asarray(category) == "WRN") & negative
    return {"clips": len(y), "positives": int(positive.sum()), "prevalence": float(positive.mean()),
            "average_precision": float(average_precision_score(y, scores)) if positive.any() else float("nan"),
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
            "warning_negative_false_positive_rate": float(predicted[warning].mean()) if warning.any() else float("nan"),
            "warning_negative_clips": int(warning.sum()), "accuracy": float((predicted == y).mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "threshold": None if threshold is None else float(threshold)}


def train_one(configuration, seed, x_train, x_val, y_train, y_val, train_categories, config, results, fingerprint):
    destination = results / "models" / f"{configuration}_seed{seed}.pt"
    summary_path = results / f"{configuration}_seed{seed}.json"
    curve_path = results / f"{configuration}_seed{seed}_curve.csv"
    if summary_path.exists() and destination.exists() and curve_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["input_fingerprint"] != fingerprint["sha256"]:
            raise ValueError("Existing training run came from different data/configuration")
        if summary.get("checkpoint_sha256") != sha256(destination) or summary.get("curve_sha256") != sha256(curve_path):
            raise ValueError("Existing training checkpoint or curve changed after run completion")
        checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
        if (checkpoint["configuration"] != configuration or checkpoint["seed"] != seed
                or checkpoint["best_epoch"] != summary["best_epoch"]):
            raise ValueError("Existing training checkpoint metadata does not match run summary")
        head = make_head(configuration)
        head.load_state_dict(checkpoint["state_dict"])
        return summary, pd.read_csv(curve_path).to_dict("records"), predict_head(head, x_val)
    start = time.perf_counter()
    append_event(results, "head_fit", "started", configuration=configuration, seed=seed)
    seed_everything(seed)
    head = make_head(configuration)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    multiplier = 2 if configuration == "hard_negatives" else 1
    weights = torch.from_numpy(sample_weights(y_train, train_categories, multiplier))
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))
    xv = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32))
    yv = torch.from_numpy(y_val.astype(np.float32))
    order_generator = np.random.default_rng(seed)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_ap, best_epoch, best_state = -float("inf"), None, None
    curves = []
    try:
        for epoch in range(1, config["epochs"] + 1):
            head.train()
            order = order_generator.permutation(len(y_train))
            step_loss_sum = 0.0
            for offset in range(0, len(order), config["batch_size"]):
                ix = torch.as_tensor(order[offset:offset + config["batch_size"]], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = head(xt[ix]).reshape(-1)
                loss = (criterion(logits, yt[ix]) * weights[ix]).mean()
                loss.backward()
                optimizer.step()
                step_loss_sum += float(loss.detach()) * len(ix)
            head.eval()
            with torch.no_grad():
                train_logits = head(xt).reshape(-1)
                val_logits = head(xv).reshape(-1)
                train_scores = torch.sigmoid(train_logits).numpy()
                val_scores = torch.sigmoid(val_logits).numpy()
                row = {"configuration": configuration, "seed": seed, "epoch": epoch,
                       "optimizer_step_weighted_train_bce": step_loss_sum / len(y_train),
                       "weighted_train_bce": float((criterion(train_logits, yt) * weights).mean()),
                       "unweighted_validation_bce": float(criterion(val_logits, yv).mean()),
                       "train_average_precision": float(average_precision_score(y_train, train_scores)),
                       "validation_average_precision": float(average_precision_score(y_val, val_scores))}
            curves.append(row)
            if row["validation_average_precision"] > best_ap:
                best_ap, best_epoch = row["validation_average_precision"], epoch
                best_state = copy.deepcopy(head.state_dict())
        head.load_state_dict(best_state)
        val_scores = predict_head(head, x_val)
        selected_curve = curves[best_epoch - 1]
        summary = {"configuration": configuration, "seed": seed, "status": "completed",
                   "best_epoch": best_epoch, "validation_average_precision": best_ap,
                   "train_average_precision_at_selected_epoch": selected_curve["train_average_precision"],
                   "weighted_train_bce_at_selected_epoch": selected_curve["weighted_train_bce"],
                   "unweighted_validation_bce_at_selected_epoch": selected_curve["unweighted_validation_bce"],
                   "final_epoch_validation_average_precision": curves[-1]["validation_average_precision"],
                   "final_epoch_train_average_precision": curves[-1]["train_average_precision"],
                   "parameters": sum(p.numel() for p in head.parameters()),
                   "runtime_seconds": time.perf_counter() - start,
                   "checkpoint": destination.relative_to(results).as_posix(),
                   "input_fingerprint": fingerprint["sha256"]}
        temporary_checkpoint = destination.with_suffix(".partial")
        torch.save({"state_dict": best_state, "configuration": configuration, "seed": seed,
                    "best_epoch": best_epoch}, temporary_checkpoint)
        os.replace(temporary_checkpoint, destination)
        csv_write(curve_path, pd.DataFrame(curves))
        summary["checkpoint_sha256"] = sha256(destination)
        summary["curve_sha256"] = sha256(curve_path)
        json_write(summary_path, summary)
        append_event(results, "head_fit", "completed", **{key: value for key, value in summary.items() if key != "status"})
        return summary, curves, val_scores
    except Exception as exc:
        append_event(results, "head_fit", "failed", configuration=configuration, seed=seed,
                     runtime_seconds=time.perf_counter() - start, error=repr(exc))
        raise


def environment():
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "keras", "netCDF4", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "training_device": "CPU", "torch_threads": torch.get_num_threads()}


def update_live_experiment_log(study, config, summary, y_val, scores, categories):
    """Fill the assignment's planned log immediately, before the next head fit."""
    path = study / "design" / "experiment_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    hypotheses = {
        "baseline": "Establish a trained frozen-feature adaptation reference",
        "history": "Persistence across two scans may suppress brief lookalikes",
        "hard_negatives": "Extra attention to non-tornadic warning cases may reduce false alarms",
        "nonlinear_head": "Nonlinear feature interactions may improve discrimination, with overfitting risk",
    }
    if path.exists():
        log = pd.read_csv(path).astype(object)
    else:
        log = pd.DataFrame([{"configuration": item["name"], "seed": seed, "status": "planned",
                             "what_changed": item["change"], "why_tried": hypotheses[item["name"]]}
                            for item in config["experiments"] for seed in config["training_seeds"]]).astype(object)
    match = (log.configuration == summary["configuration"]) & (log.seed.astype(int) == summary["seed"])
    if int(match.sum()) != 1:
        raise ValueError("Planned experiment log must contain exactly one row per configuration/seed")
    threshold = recall_threshold(y_val, scores)
    measured = metrics(y_val, scores, threshold, categories)
    values = {
        "status": "completed", "best_epoch": summary["best_epoch"],
        "train_loss": summary["weighted_train_bce_at_selected_epoch"],
        "val_loss": summary["unweighted_validation_bce_at_selected_epoch"],
        "val_AP": summary["validation_average_precision"], "val_precision": measured["precision"],
        "val_recall": measured["recall"], "val_WRN_false_positive_rate": measured["warning_negative_false_positive_rate"],
        "runtime_seconds": summary["runtime_seconds"], "validation_threshold": threshold,
        "result_interpretation": f"Observed seed validation AP {summary['validation_average_precision']:.6f}; "
                                  f"saved earliest best epoch {summary['best_epoch']}. Final selection awaits all runs.",
        "next_decision": "Continue the remaining fixed comparisons; do not inspect final test scores",
        "completed_at_utc": utc_now(), "input_fingerprint": summary["input_fingerprint"],
    }
    for name, value in values.items():
        if name not in log:
            log[name] = None
        log.loc[match, name] = value
    csv_write(path, log)


def finalize_experiment_log(study, comparison, winner):
    path = study / "design" / "experiment_log.csv"
    log = pd.read_csv(path).astype(object)
    baseline_ap = float(comparison.loc[comparison.configuration == "baseline", "mean_validation_ap"].iloc[0])
    for row in comparison.to_dict("records"):
        match = log.configuration == row["configuration"]
        delta = row["mean_validation_ap"] - baseline_ap
        log.loc[match, "result_interpretation"] = (
            f"Three-seed mean validation AP {row['mean_validation_ap']:.6f} ± {row['std_validation_ap']:.6f}; "
            f"difference from baseline {delta:+.6f}. No test result informed this comparison.")
        log.loc[match, "next_decision"] = (
            "Selected by the predeclared validation rule; freeze ensemble and threshold before testing"
            if row["configuration"] == winner else
            "Retain as a completed validation comparison; do not combine this change with other experiments")
    csv_write(path, log)


def plot_validation(results, curves, summaries, comparison, validation, validation_scores, thresholds):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    colors = {"baseline": "#2166ac", "history": "#1b9e77", "hard_negatives": "#d95f02", "nonlinear_head": "#7570b3"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    for ax, configuration in zip(axes.ravel(), CONFIGURATIONS):
        for seed in sorted(curves.seed.unique()):
            sub = curves[(curves.configuration == configuration) & (curves.seed == seed)]
            ax.plot(sub.epoch, sub.train_average_precision, color=colors[configuration], alpha=.45, linestyle="--")
            ax.plot(sub.epoch, sub.validation_average_precision, color=colors[configuration], alpha=.8)
            best = summaries[(summaries.configuration == configuration) & (summaries.seed == seed)].iloc[0]
            ax.scatter(best.best_epoch, best.validation_average_precision, s=22, color=colors[configuration], zorder=3)
        ax.set_title(configuration.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average precision")
        ax.set_ylim(0, 1)
    axes[0, 0].plot([], [], "--", color="#444444", label="Training")
    axes[0, 0].plot([], [], "-", color="#444444", label="Validation")
    axes[0, 0].scatter([], [], color="#444444", label="Saved epoch")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Frozen CNN adaptations · three seeds per configuration")
    fig.tight_layout()
    fig.savefig(results / "figures" / "training_validation_ap.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    for ax, configuration in zip(axes.ravel(), CONFIGURATIONS):
        for seed in sorted(curves.seed.unique()):
            sub = curves[(curves.configuration == configuration) & (curves.seed == seed)]
            ax.plot(sub.epoch, sub.weighted_train_bce, color=colors[configuration], alpha=.5, linestyle="--")
            ax.plot(sub.epoch, sub.unweighted_validation_bce, color=colors[configuration], alpha=.8)
        ax.set_title(configuration.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross entropy")
    axes[0, 0].plot([], [], "--", color="#444444", label="Weighted training BCE")
    axes[0, 0].plot([], [], "-", color="#444444", label="Unweighted validation BCE")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Loss curves · training and validation losses use different weighting")
    fig.tight_layout()
    fig.savefig(results / "figures" / "training_validation_loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.7, 4.2))
    positions = np.arange(len(comparison))
    ax.errorbar(positions, comparison.mean_validation_ap, yerr=comparison.std_validation_ap,
                fmt="o", markersize=8, capsize=5, color="#2166ac", label="Mean ± sample SD across seeds")
    ax.scatter(positions, comparison.ensemble_validation_ap, marker="D", color="#d95f02", label="Three-head ensemble")
    ax.set_xticks(positions, [v.replace("_", " ") for v in comparison.configuration])
    ax.set_ylabel("Validation average precision")
    ax.set_title("Controlled comparisons · final test not used for selection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figures" / "validation_comparison.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    y = validation.label.to_numpy()
    for name, scores in validation_scores.items():
        if name == "majority":
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, label=f"{name.replace('_', ' ')} · AP {ap:.3f}", alpha=.9)
    ax.axhline(y.mean(), linestyle=":", color="gray", label=f"Positive prevalence {y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Validation precision–recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figures" / "validation_precision_recall.png")
    plt.close(fig)


def train(study_dir=None, data_dir=None):
    study, data, results = resolve_paths(study_dir, data_dir)
    config = json.loads((study / "planned_config.json").read_text(encoding="utf-8"))
    fingerprint = input_fingerprint(study, data)
    decision_path = results / "final_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision["input_fingerprint"] != fingerprint:
            raise ValueError("Final decision is already locked against different inputs")
        verify_locked_artifacts(study, data, results, decision)
        return decision
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    manifest, arrays = load_data(data)
    audit = assert_audit(manifest)
    training = manifest.split.to_numpy() == "train"
    validating = manifest.split.to_numpy() == "validation"
    y_train = manifest.loc[training, "label"].to_numpy()
    y_val = manifest.loc[validating, "label"].to_numpy()
    categories = manifest.loc[training, "category"].to_numpy()
    scaler = StandardScaler().fit(np.asarray(arrays["features_current"][training]))
    np.savez(results / "models" / "feature_scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_, n_samples=scaler.n_samples_seen_)
    current_train = scaler.transform(arrays["features_current"][training]).astype(np.float32)
    previous_train = scaler.transform(arrays["features_previous"][training]).astype(np.float32)
    current_val = scaler.transform(arrays["features_current"][validating]).astype(np.float32)
    previous_val = scaler.transform(arrays["features_previous"][validating]).astype(np.float32)
    summaries, curves, ensembles = [], [], {}
    start = time.perf_counter()
    append_event(results, "adaptation_training", "started", input_fingerprint=fingerprint["sha256"])
    for configuration in CONFIGURATIONS:
        seed_scores = []
        for seed in config["training_seeds"]:
            summary, run_curve, scores = train_one(configuration, seed,
                representation(configuration, current_train, previous_train),
                representation(configuration, current_val, previous_val),
                y_train, y_val, categories, config["training"], results, fingerprint)
            summaries.append(summary)
            curves.extend(run_curve)
            seed_scores.append(scores)
            update_live_experiment_log(study, config, summary, y_val, scores,
                                       manifest.loc[validating, "category"].to_numpy())
            pd.DataFrame(summaries).to_csv(results / "run_summary.csv", index=False)
            pd.DataFrame(curves).to_csv(results / "learning_curves.csv", index=False)
            print(f"{configuration} seed {seed}: AP {summary['validation_average_precision']:.6f}, "
                  f"epoch {summary['best_epoch']}, {summary['runtime_seconds']:.2f} s", flush=True)
            if len(summaries) == 1:
                pilot = {"configuration": configuration, "seed": seed,
                         "runtime_seconds": summary["runtime_seconds"], "limit_seconds": 600,
                         "passed": summary["runtime_seconds"] < 600, "reused_as_actual_run": True}
                json_write(results / "pilot_timing.json", pilot)
                if not pilot["passed"]:
                    append_event(results, "pilot", "failed", **pilot)
                    raise RuntimeError("First head fit exceeded the ten-minute pilot limit")
        ensembles[configuration] = np.mean(seed_scores, axis=0)
    summary_frame = pd.DataFrame(summaries)
    comparisons = []
    for order, configuration in enumerate(CONFIGURATIONS):
        subset = summary_frame[summary_frame.configuration == configuration]
        threshold = recall_threshold(y_val, ensembles[configuration])
        measured = metrics(y_val, ensembles[configuration], threshold, manifest.loc[validating, "category"].to_numpy())
        comparisons.append({"configuration": configuration, "order": order,
                            "parameters": int(subset.parameters.iloc[0]), "uses_previous_frame": configuration == "history",
                            "mean_validation_ap": float(subset.validation_average_precision.mean()),
                            "std_validation_ap": float(subset.validation_average_precision.std(ddof=1)),
                            "ensemble_validation_ap": measured["average_precision"],
                            "validation_threshold": threshold, "validation_precision": measured["precision"],
                            "validation_recall": measured["recall"], "validation_f1": measured["f1"],
                            "validation_warning_negative_fpr": measured["warning_negative_false_positive_rate"],
                            "total_head_training_seconds": float(subset.runtime_seconds.sum()),
                            "best_epochs": ",".join(str(v) for v in subset.best_epoch)})
    comparison = pd.DataFrame(comparisons)
    best = comparison.mean_validation_ap.max()
    eligible = comparison[comparison.mean_validation_ap >= best - 0.01 - 1e-12]
    winner = eligible.sort_values(["parameters", "uses_previous_frame", "order"]).iloc[0].configuration
    comparison["selected"] = comparison.configuration == winner
    comparison.to_csv(results / "experiment_comparison.csv", index=False)
    finalize_experiment_log(study, comparison, winner)
    majority_label = int(np.bincount(y_train, minlength=2).argmax())
    logistic = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42))])
    logistic.fit(np.asarray(arrays["radar_statistics"][training]), y_train)
    if np.max(logistic.named_steps["classifier"].n_iter_) >= 1000:
        append_event(results, "radar_statistics_fit", "warning", message="Reached maximum iterations; inspect convergence")
    joblib.dump(logistic, results / "models" / "radar_statistics.joblib")
    validation_scores = dict(ensembles)
    validation_scores["original_cnn"] = arrays["original_cnn_scores"][validating]
    validation_scores["radar_statistics"] = logistic.predict_proba(arrays["radar_statistics"][validating])[:, 1]
    validation_scores["majority"] = np.full(len(y_val), majority_label, dtype=float)
    model_names = list(dict.fromkeys([winner, *REFERENCES]))
    thresholds = {name: (None if name == "majority" else recall_threshold(y_val, validation_scores[name])) for name in model_names}
    validation_metrics = {name: metrics(y_val, validation_scores[name], thresholds[name], manifest.loc[validating, "category"].to_numpy(),
                                       constant_prediction=majority_label if name == "majority" else None) for name in model_names}
    score_frame = manifest.loc[validating, ["filename", "split", "label", "previous_label", "category", "storm_group"]].copy()
    for name, scores in validation_scores.items():
        score_frame[f"score_{name}"] = scores
    csv_write(results / "validation_scores.csv", score_frame)
    json_write(results / "validation_metrics.json", finite_json(validation_metrics))
    log = summary_frame.copy()
    descriptions = {entry["name"]: entry["change"] for entry in config["experiments"]}
    log["what_changed"] = log.configuration.map(descriptions)
    log["next_decision"] = np.where(log.configuration == winner, "Selected by predeclared validation rule", "Retained as validation comparison")
    log.to_csv(results / "experiment_log.csv", index=False)
    plot_validation(results, pd.DataFrame(curves), summary_frame, comparison, manifest.loc[validating], validation_scores, thresholds)
    model_artifacts = [results / "models" / "feature_scaler.npz", results / "models" / "radar_statistics.joblib"]
    for name in set((winner, "baseline")):
        model_artifacts += [results / "models" / f"{name}_seed{seed}.pt" for seed in config["training_seeds"]]
    decision = {"locked_at_utc": utc_now(), "status": "locked_before_test", "input_fingerprint": fingerprint,
                "selected_configuration": winner, "selection_metric": "mean seed validation average precision",
                "selection_rule": config["evaluation"]["selection"], "training_seeds": config["training_seeds"],
                "chosen_best_epochs": {str(row["seed"]): row["best_epoch"] for row in summaries if row["configuration"] == winner},
                "evaluated_test_models": model_names, "thresholds": thresholds, "majority_prediction": majority_label,
                "validation_metrics": finite_json(validation_metrics), "data_audit": audit,
                "feature_scaler_fit": "Current adaptation-training rows only; same scaler reused for both frames",
                "radar_statistics_names": list(STATISTIC_NAMES), "environment": environment(),
                "training_wall_seconds": time.perf_counter() - start,
                "implementation_sha256": sha256(Path(__file__)),
                "model_artifact_hashes": {path.relative_to(results).as_posix(): sha256(path) for path in model_artifacts},
                "test_score_inspection_before_lock": False}
    json_write(decision_path, decision, exclusive=True)
    append_event(results, "final_decision", "locked", selected_configuration=winner, decision_sha256=sha256(decision_path))
    return decision


def verify_locked_artifacts(study, data, results, decision):
    if decision["input_fingerprint"] != input_fingerprint(study, data):
        raise ValueError("Data, feature cache, checkpoint or configuration changed after decision lock")
    if decision["implementation_sha256"] != sha256(Path(__file__)):
        raise ValueError("Experiment implementation changed after decision lock")
    for relative, expected in decision["model_artifact_hashes"].items():
        if sha256(results / relative) != expected:
            raise ValueError(f"Locked model artifact changed: {relative}")


def grouped_bootstrap(frame, model_names, thresholds, majority_label, winner, draws=1000, seed=42):
    """Resample whole groups jointly for all models; duplicate groups keep all rows."""
    groups = frame.storm_group.unique()
    group_rows = [np.flatnonzero(frame.storm_group.to_numpy() == group) for group in groups]
    generator = np.random.default_rng(seed)
    category = frame.category.to_numpy()
    y = frame.label.to_numpy()
    score_arrays = {name: frame[f"score_{name}"].to_numpy() for name in model_names}
    samples = {(name, metric): [] for name in model_names for metric in METRICS}
    references = [name for name in ("baseline", "radar_statistics", "original_cnn") if name != winner and name in model_names]
    differences = {(name, metric): [] for name in references for metric in METRICS}
    for _ in range(draws):
        chosen = generator.integers(0, len(groups), size=len(groups))
        ix = np.concatenate([group_rows[group] for group in chosen])
        draw_metrics = {}
        for name in model_names:
            values = metrics(y[ix], score_arrays[name][ix], thresholds[name], category[ix],
                             constant_prediction=majority_label if name == "majority" else None)
            draw_metrics[name] = values
            for metric in METRICS:
                samples[name, metric].append(values[metric])
        for name in references:
            for metric in METRICS:
                differences[name, metric].append(draw_metrics[winner][metric] - draw_metrics[name][metric])
    rows = []
    def append_interval(kind, name, metric, values):
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]
        low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else [np.nan, np.nan]
        rows.append({"kind": kind, "model": name, "metric": metric, "draws": draws, "valid_draws": len(valid),
                     "ci_lower": low, "ci_upper": high, "seed": seed, "resampling_unit": "whole storm group"})
    for (name, metric), values in samples.items():
        append_interval("model_metric", name, metric, values)
    for (name, metric), values in differences.items():
        append_interval(f"{winner}_minus_reference", name, metric, values)
    return pd.DataFrame(rows)


def plot_final(results, frame, final, winner):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    values = final["metrics"][winner]
    confusion = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    im = ax.imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=17,
                    color="white" if confusion[row, column] > confusion.max() * .55 else "#222222")
    ax.set_xticks([0, 1], ["No tornado", "Tornado"])
    ax.set_yticks([0, 1], ["No tornado", "Tornado"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Final test · {winner.replace('_', ' ')}")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(results / "figures" / "final_confusion_matrix.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    y = frame.label.to_numpy()
    for name in final["evaluated_models"]:
        if name == "majority":
            continue
        precision, recall, _ = precision_recall_curve(y, frame[f"score_{name}"])
        ap = final["metrics"][name]["average_precision"]
        ax.plot(recall, precision, linewidth=2 if name == winner else 1.4,
                label=f"{name.replace('_', ' ')} · AP {ap:.3f}")
    ax.axhline(y.mean(), color="gray", linestyle=":", label=f"Prevalence {y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Final test · fixed configurations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figures" / "final_precision_recall.png")
    plt.close(fig)
    group = pd.read_csv(results / "per_group_metrics.csv", dtype={"storm_group": str})
    chosen = group[group.model == winner].sort_values("clips")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(chosen))
    ax.barh(positions, chosen.average_precision.fillna(0), color="#2166ac")
    ax.set_yticks(positions, [f"{str(g)[-16:]} · n={n}, pos={p}" for g, n, p in zip(chosen.storm_group, chosen.clips, chosen.positives)])
    ax.set(xlabel="Average precision", xlim=(0, 1), title="Final model by storm group · AP undefined without positives")
    for position, value in enumerate(chosen.average_precision):
        if pd.isna(value):
            ax.text(.02, position, "undefined", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figures" / "final_per_group_ap.png")
    plt.close(fig)


def test_once(study_dir=None, data_dir=None):
    study, data, results = resolve_paths(study_dir, data_dir)
    decision_path = results / "final_decision.json"
    if not decision_path.exists():
        raise ValueError("Train/validation decision must be locked before final testing")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    verify_locked_artifacts(study, data, results, decision)
    final_path = results / "final_test.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final["decision_sha256"] != sha256(decision_path):
            raise ValueError("Existing sealed test result belongs to another decision")
        if final["predictions_sha256"] != sha256(results / "final_test_predictions.csv"):
            raise ValueError("Saved final predictions changed after sealing")
        return final
    manifest, arrays = load_data(data)
    test_mask = manifest.split.to_numpy() == "test"
    frame = manifest.loc[test_mask].copy().reset_index(drop=True)
    model_names = decision["evaluated_test_models"]
    winner = decision["selected_configuration"]
    thresholds = decision["thresholds"]
    majority_label = decision["majority_prediction"]
    predictions_path = results / "final_test_predictions.csv"
    receipt_path = results / "test_evaluation_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["decision_sha256"] != sha256(decision_path):
            raise ValueError("A test evaluation has already started for another decision")
    else:
        receipt = {"started_at_utc": utc_now(), "decision_sha256": sha256(decision_path),
                   "models": model_names, "policy": "Apply locked decision; save predictions once"}
        json_write(receipt_path, receipt, exclusive=True)
        append_event(results, "final_test", "started", decision_sha256=sha256(decision_path))
    if predictions_path.exists():
        if receipt.get("predictions_sha256") != sha256(predictions_path):
            raise ValueError("Cached final predictions do not match the committed evaluation receipt")
        saved = pd.read_csv(predictions_path, dtype={"filename": str, "storm_group": str}, float_precision="round_trip")
        for column in ("filename", "label", "category", "storm_group"):
            if not saved[column].equals(frame[column]):
                raise ValueError(f"Cached final predictions do not match locked test {column}")
        frame = saved
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        scaler = np.load(results / "models" / "feature_scaler.npz")
        current = ((arrays["features_current"][test_mask] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
        previous = ((arrays["features_previous"][test_mask] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
        for name in model_names:
            if name in CONFIGURATIONS:
                scores = []
                for seed in decision["training_seeds"]:
                    checkpoint = torch.load(results / "models" / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=True)
                    head = make_head(name)
                    head.load_state_dict(checkpoint["state_dict"])
                    scores.append(predict_head(head, representation(name, current, previous)))
                score = np.mean(scores, axis=0)
            elif name == "original_cnn":
                score = arrays["original_cnn_scores"][test_mask]
            elif name == "radar_statistics":
                logistic = joblib.load(results / "models" / "radar_statistics.joblib")
                score = logistic.predict_proba(arrays["radar_statistics"][test_mask])[:, 1]
            elif name == "majority":
                score = np.full(len(frame), majority_label, dtype=float)
            else:
                raise ValueError(f"Unknown locked model: {name}")
            frame[f"score_{name}"] = score
            frame[f"prediction_{name}"] = majority_label if name == "majority" else (score >= thresholds[name]).astype(int)
        def commit_receipt(prediction_hash):
            if receipt.get("predictions_sha256") not in (None, prediction_hash):
                raise ValueError("Resumed prediction values differ from the started test evaluation")
            receipt["predictions_sha256"] = prediction_hash
            receipt["predictions_committed_at_utc"] = utc_now()
            json_write(receipt_path, receipt)
        csv_write(predictions_path, frame, exclusive=True, before_commit=commit_receipt)
    measured = {}
    per_group = []
    for name in model_names:
        measured[name] = metrics(frame.label.to_numpy(), frame[f"score_{name}"].to_numpy(), thresholds[name], frame.category.to_numpy(),
                                 constant_prediction=majority_label if name == "majority" else None)
        for group_id, subset in frame.groupby("storm_group", sort=True):
            values = metrics(subset.label.to_numpy(), subset[f"score_{name}"].to_numpy(), thresholds[name], subset.category.to_numpy(),
                             constant_prediction=majority_label if name == "majority" else None)
            per_group.append({"storm_group": group_id, "model": name, **values})
    pd.DataFrame(per_group).to_csv(results / "per_group_metrics.csv", index=False)
    intervals = grouped_bootstrap(frame, model_names, thresholds, majority_label, winner)
    intervals.to_csv(results / "bootstrap_intervals.csv", index=False)
    final = {"sealed_at_utc": utc_now(), "status": "sealed_final_test",
             "decision_sha256": sha256(decision_path), "predictions_sha256": sha256(predictions_path),
             "selected_configuration": winner, "evaluated_models": model_names,
             "test_clips": len(frame), "test_storm_groups": int(frame.storm_group.nunique()),
             "metrics": finite_json(measured), "bootstrap": {"draws": 1000, "seed": 42,
                 "unit": "whole storm group", "paired_between_models": True,
                 "intervals": "bootstrap_intervals.csv", "interpretation": "Descriptive with only ten independent groups"},
             "policy": "No unselected adaptation evaluated on test; no settings changed after validation lock"}
    plot_final(results, frame, final, winner)
    json_write(final_path, final, exclusive=True)
    append_event(results, "final_test", "sealed", final_test_sha256=sha256(final_path), selected_configuration=winner)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--study-dir")
    parser.add_argument("--data-dir")
    options = parser.parse_args()
    result = train(options.study_dir, options.data_dir) if options.action == "train" else test_once(options.study_dir, options.data_dir)
    print(json.dumps({key: result[key] for key in ("status", "selected_configuration")}, indent=2))
