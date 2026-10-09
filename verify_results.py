"""Independently verify sealed real TorNet results without refitting any model.

Recompute metrics from committed predictions, check validation-only selection and
thresholds, verify all twelve run fingerprints, and prove cached test replay does
not invoke prediction or alter the sealed artifacts. This script requires real
data and feature QA receipts; it does not manufacture or substitute any results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

import experiments

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same_number(actual, expected, name):
    if expected is None or (isinstance(expected, (float, np.floating)) and np.isnan(expected)):
        require(actual is None or pd.isna(actual), f"{name}: expected undefined, got {actual}")
    else:
        require(actual is not None and np.isclose(float(actual), float(expected), atol=5e-10, rtol=5e-10),
                f"{name}: got {actual}, independently recomputed {expected}")


def independently_measure(frame, model, threshold, majority_label):
    y = frame.label.to_numpy(dtype=int)
    scores = frame[f"score_{model}"].to_numpy(dtype=float)
    predicted = np.full(len(frame), majority_label, dtype=int) if model == "majority" else (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    warning = (frame.category.to_numpy() == "WRN") & (y == 0)
    return {"clips": len(frame), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "average_precision": average_precision_score(y, scores) if y.sum() else None,
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0) if y.sum() else None,
            "f1": f1_score(y, predicted, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "warning_negative_false_positive_rate": float(predicted[warning].mean()) if warning.any() else None,
            "warning_negative_clips": int(warning.sum()), "accuracy": accuracy_score(y, predicted),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": threshold}, predicted


def verify(study_dir=None, *, write_receipt=True):
    study = Path(study_dir or ROOT).resolve()
    data, results = study / "data", study / "results"
    decision_path, final_path = results / "final_decision.json", results / "final_test.json"
    require(decision_path.exists() and final_path.exists(), "Real train/validation decision and sealed final test must exist first")
    config = read_json(study / "planned_config.json")
    decision, final = read_json(decision_path), read_json(final_path)
    audit = read_json(results / "data_audit.json")
    features = read_json(data / "feature_provenance.json")
    require(audit.get("status") == "passed", "Physical-frame data QA did not pass")
    require(features.get("status") == "completed", "Real frozen feature extraction did not complete")
    require(features.get("checkpoint_revision") == config["model"]["revision"], "Feature extraction checkpoint revision differs from protocol")
    require(features.get("backbone_trainable_parameters") == 0, "Feature extraction did not freeze the pretrained backbone")
    require(decision["status"] == "locked_before_test" and final["status"] == "sealed_final_test", "Final holdout was not sealed")
    require(decision["test_score_inspection_before_lock"] is False, "Decision reports test inspection before lock")
    require(final["decision_sha256"] == experiments.sha256(decision_path), "Sealed test decision hash mismatch")
    manifest_path = data / "runtime_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"filename": str, "storm_group": str}, float_precision="round_trip")
    if "split" not in manifest:
        manifest["split"] = manifest.study_split
    manifest["split"] = manifest.split.replace({"val": "validation"})
    require(audit["runtime_manifest_sha256"] == experiments.sha256(manifest_path), "Physical audit manifest hash mismatch")
    require(audit["planned_manifest_sha256"] == experiments.sha256(study / "design/planned_manifest.csv"), "Data audit planned-manifest hash mismatch")
    require(audit["planned_config_sha256"] == experiments.sha256(study / "planned_config.json"), "Data audit configuration hash mismatch")
    require(audit["valid_samples"] == len(manifest), "Physical data audit count mismatch")
    require(features["manifest_sha256"] == experiments.sha256(manifest_path), "Feature manifest hash mismatch")
    require(features["feature_rows"] == len(manifest), "Feature row count mismatch")
    require(features["features_script_sha256"] == experiments.sha256(study / "features.py"), "Feature extraction script changed after caching")
    for relative, expected in features["source_sha256"].items():
        require(experiments.sha256(study / "upstream" / relative) == expected, f"Pretrained model/preprocessing source changed: {relative}")
    if "type" in manifest:
        require(set(manifest.type) == {"test"}, "Study includes samples outside predeclared source-test pool")
    samples = read_json(data / "features_sample_ids.json")
    require(samples == manifest.filename.tolist(), "Feature rows and runtime filenames are not exactly aligned")
    experiments.verify_locked_artifacts(study, data, results, decision)
    summaries = pd.read_csv(results / "run_summary.csv", float_precision="round_trip")
    expected_runs = {(name, int(seed)) for name in experiments.CONFIGURATIONS for seed in config["training_seeds"]}
    actual_runs = set(zip(summaries.configuration, summaries.seed.astype(int)))
    require(len(summaries) == 12 and actual_runs == expected_runs, "Expected exactly four configurations × three seeds")
    require(set(summaries.status) == {"completed"}, "A declared real run did not complete")
    require(set(summaries.input_fingerprint) == {decision["input_fingerprint"]["sha256"]}, "The twelve runs do not share the locked input fingerprint")
    curves = pd.read_csv(results / "learning_curves.csv", float_precision="round_trip")
    for row in summaries.to_dict("records"):
        name, seed = row["configuration"], int(row["seed"])
        checkpoint = results / row["checkpoint"]
        curve_path = results / f"{name}_seed{seed}_curve.csv"
        require(experiments.sha256(checkpoint) == row["checkpoint_sha256"], f"{name}/{seed}: checkpoint changed")
        require(experiments.sha256(curve_path) == row["curve_sha256"], f"{name}/{seed}: learning curve changed")
        group = curves[(curves.configuration == name) & (curves.seed == seed)].sort_values("epoch")
        require(group.epoch.tolist() == list(range(1, config["training"]["epochs"] + 1)), f"{name}/{seed}: epoch budget mismatch")
        best_value = group.validation_average_precision.max()
        earliest = int(group.loc[group.validation_average_precision == best_value, "epoch"].min())
        require(int(row["best_epoch"]) == earliest, f"{name}/{seed}: checkpoint is not earliest maximum validation AP")
        same_number(row["validation_average_precision"], best_value, f"{name}/{seed} selected validation AP")
        require(int(row["parameters"]) == (24705 if name == "nonlinear_head" else 385), f"{name}/{seed}: trainable-head parameter count mismatch")
    comparison = pd.read_csv(results / "experiment_comparison.csv", float_precision="round_trip")
    validation = pd.read_csv(results / "validation_scores.csv", dtype={"filename": str, "storm_group": str}, float_precision="round_trip")
    expected_validation = manifest[manifest.split == "validation"].reset_index(drop=True)
    require(validation.filename.tolist() == expected_validation.filename.tolist(), "Validation scores do not match validation rows")
    require(np.array_equal(validation.label, expected_validation.label), "Validation labels differ from runtime labels")
    for name in experiments.CONFIGURATIONS:
        summary = summaries[summaries.configuration == name]
        row = comparison[comparison.configuration == name].iloc[0]
        same_number(row.mean_validation_ap, summary.validation_average_precision.mean(), f"{name} mean seed AP")
        same_number(row.std_validation_ap, summary.validation_average_precision.std(ddof=1), f"{name} seed AP standard deviation")
        same_number(row.ensemble_validation_ap, average_precision_score(validation.label, validation[f"score_{name}"]), f"{name} ensemble AP")
    eligible = comparison[comparison.mean_validation_ap >= comparison.mean_validation_ap.max() - .01 - 1e-12]
    chosen = eligible.sort_values(["parameters", "uses_previous_frame", "order"]).iloc[0].configuration
    require(chosen == decision["selected_configuration"] == final["selected_configuration"], "Final configuration does not follow the predeclared validation rule")
    require(comparison.loc[comparison.selected, "configuration"].tolist() == [chosen], "Comparison table selection flag mismatch")
    expected_models = list(dict.fromkeys([chosen, *experiments.REFERENCES]))
    require(decision["evaluated_test_models"] == final["evaluated_models"] == expected_models, "An unselected adaptation was tested, or a reference is missing")
    train_labels = manifest.loc[manifest.split == "train", "label"].to_numpy(dtype=int)
    majority = int(np.bincount(train_labels, minlength=2).argmax())
    require(majority == decision["majority_prediction"], "Majority reference is not derived solely from training labels")
    threshold_checks = {}
    for name in expected_models:
        threshold = decision["thresholds"][name]
        if name == "majority":
            require(threshold is None, "Majority reference should not have a recall-tuned threshold")
            require(np.all(validation[f"score_{name}"] == majority), "Validation majority scores changed")
        else:
            scores = validation[f"score_{name}"].to_numpy(dtype=float)
            positives = validation.label.to_numpy(dtype=int) == 1
            observed = np.unique(scores)
            require(threshold in observed, f"{name}: threshold is not an observed validation score")
            recall = float(np.mean(scores[positives] >= threshold))
            require(recall >= .8 - 1e-12, f"{name}: locked threshold misses validation recall target")
            larger = observed[observed > threshold]
            if len(larger):
                require(np.mean(scores[positives] >= larger.min()) < .8, f"{name}: a larger observed threshold still meets recall")
            cutoff_rank = math.ceil(.8 * int(positives.sum())) - 1
            same_number(threshold, np.sort(scores[positives])[::-1][cutoff_rank], f"{name} retained-ties cutoff")
            threshold_checks[name] = {"threshold": threshold, "validation_recall": recall}
        calculated, _ = independently_measure(validation, name, threshold, majority)
        for metric, value in calculated.items():
            same_number(decision["validation_metrics"][name][metric], value, f"Validation {name}/{metric}")
    predictions_path = results / "final_test_predictions.csv"
    require(final["predictions_sha256"] == experiments.sha256(predictions_path), "Sealed prediction hash mismatch")
    receipt = read_json(results / "test_evaluation_receipt.json")
    require(receipt["predictions_sha256"] == experiments.sha256(predictions_path), "Test execution receipt hash mismatch")
    predictions = pd.read_csv(predictions_path, dtype={"filename": str, "storm_group": str}, float_precision="round_trip")
    expected_test = manifest[manifest.split == "test"].reset_index(drop=True)
    for column in ("filename", "label", "category", "storm_group"):
        require(predictions[column].equals(expected_test[column]), f"Final predictions {column} differs from locked test rows")
    require(final["test_clips"] == len(predictions), "Final test clip count mismatch")
    require(final["test_storm_groups"] == predictions.storm_group.nunique(), "Final independent group count mismatch")
    recomputed = {}
    for name in expected_models:
        measured, predicted = independently_measure(predictions, name, decision["thresholds"][name], majority)
        require(np.array_equal(predicted, predictions[f"prediction_{name}"]), f"{name}: saved classes differ from locked threshold")
        for metric, value in measured.items():
            same_number(final["metrics"][name][metric], value, f"Final {name}/{metric}")
        recomputed[name] = measured
    per_group = pd.read_csv(results / "per_group_metrics.csv", dtype={"storm_group": str}, float_precision="round_trip")
    require(len(per_group) == len(expected_models) * predictions.storm_group.nunique(), "Per-group table row count mismatch")
    for (name, group), rows in per_group.groupby(["model", "storm_group"]):
        require(len(rows) == 1, "Per-group metric table contains duplicate model/group rows")
        subset = predictions[predictions.storm_group == group]
        measured, _ = independently_measure(subset, name, decision["thresholds"][name], majority)
        for metric, value in measured.items():
            same_number(rows.iloc[0][metric], value, f"Group {group}/{name}/{metric}")
    intervals = pd.read_csv(results / "bootstrap_intervals.csv")
    require(set(intervals.draws) == {1000} and set(intervals.seed) == {42}, "Bootstrap draw count or seed differs from protocol")
    require((intervals.valid_draws <= intervals.draws).all(), "Bootstrap valid draw counts are invalid")
    require(set(intervals.resampling_unit) == {"whole storm group"}, "Bootstrap does not use independent storm groups")
    stable = [decision_path, final_path, predictions_path, results / "test_evaluation_receipt.json", results / "run_events.jsonl"]
    stable += [results / path for path in decision["model_artifact_hashes"]]
    before = {str(path): (path.stat().st_mtime_ns, experiments.sha256(path)) for path in stable}
    with patch.object(experiments, "predict_head", side_effect=AssertionError("Cached final replay must never make new predictions")), \
         patch.object(experiments, "load_data", side_effect=AssertionError("Cached final replay must not reopen data for prediction")):
        replay = experiments.test_once(study)
    require(replay == final, "Cached final replay returned a different result")
    after = {str(path): (path.stat().st_mtime_ns, experiments.sha256(path)) for path in stable}
    require(before == after, "Cached final replay modified sealed artifacts or appended a prediction event")
    report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "real_runs": len(summaries), "shared_input_fingerprint": decision["input_fingerprint"]["sha256"],
              "selected_configuration": chosen, "checked_test_models": expected_models,
              "test_clips": len(predictions), "test_storm_groups": int(predictions.storm_group.nunique()),
              "validation_threshold_checks": threshold_checks,
              "checks": ["Passed real physical-data and feature QA receipts", "Exact twelve-run shared input fingerprint",
                         "All checkpoint and learning-curve hashes", "Thirty-epoch budgets and earliest validation maxima",
                         "Validation-only configuration selection and threshold maximality with ties",
                         "Independent sklearn final/per-group metric recomputation from saved predictions",
                         "Final test contains only predeclared winner and references", "Cached final replay produces no predictions or changes"],
              "decision_sha256": experiments.sha256(decision_path), "predictions_sha256": experiments.sha256(predictions_path),
              "verification_script_sha256": experiments.sha256(Path(__file__))}
    if write_receipt:
        experiments.json_write(results / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--no-write-receipt", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.study_dir, write_receipt=not arguments.no_write_receipt), indent=2))
