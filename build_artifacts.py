"""Build classroom artifacts strictly from the completed, sealed real study."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
NAMES = {"baseline": "Current scan / linear", "history": "Two-scan mean / linear",
         "hard_negatives": "Warning negatives x2", "nonlinear_head": "Current scan / MLP",
         "majority": "Training majority", "original_cnn": "Unchanged TorNet CNN",
         "radar_statistics": "Radar-statistics logistic"}
SOURCES = [
    ("A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data (2025)", "https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml"),
    ("Authors' accessible preprint", "https://arxiv.org/html/2401.16437v1"),
    ("TorNet code and distribution notice", "https://github.com/mit-ll/tornet"),
    ("Public pretrained checkpoint", "https://huggingface.co/tornet-ml/tornado_detector_baseline_v1"),
    ("2014 v1.1 data / CC BY 4.0", "https://zenodo.org/records/12637032"),
    ("2015 v1.1 data / CC BY 4.0", "https://zenodo.org/records/12655151"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value, digits=3):
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def metadata_correction_note(audit):
    correction = audit.get("catalog_id_corrections", {})
    if not correction.get("count"):
        return ""
    return (f"A genuine pre-fit QA failure exposed {correction['count']:,} WRN files with legacy embedded event/episode IDs. "
            "The authors document corrected warning IDs in v1.1. Filenames match the corrected catalog IDs, and actual terminal timestamps "
            "match the catalog exactly. We use catalog IDs for grouping, retain embedded IDs as diagnostics, and keep strict ID equality "
            "for TOR/NUL. Frame labels and the predeclared split are unchanged; the failed audit is preserved.")


def operating_point_note(study):
    chosen = study["decision"]["selected_configuration"]
    adapted = study["final"]["metrics"][chosen]
    original = study["final"]["metrics"]["original_cnn"]
    return (f"At the locked validation-derived thresholds, the adaptation detected {adapted['tp']} positive clips "
            f"versus {original['tp']} for the unchanged CNN, but produced {adapted['fp']} false positives versus {original['fp']}. "
            "AP describes overall ranking, while these counts describe one operating point. "
            "We retain the validation-selected configuration as this experiment's final adaptation; the evidence does not support warning deployment.")


def load_completed():
    audit = read_json(RESULTS / "data_audit.json")
    feature = read_json(ROOT / "data/feature_provenance.json")
    decision = read_json(RESULTS / "final_decision.json")
    final = read_json(RESULTS / "final_test.json")
    if audit["status"] != "passed" or feature["status"] != "completed" or final["status"] != "sealed_final_test":
        raise RuntimeError("Artifacts require passed real-data QA, complete features, and sealed final evaluation.")
    if feature.get("raw_data_hashes_verified") != audit["valid_samples"]:
        raise RuntimeError("Every real feature row must come from its audited physical-file hash.")
    if final["decision_sha256"] != digest(RESULTS / "final_decision.json"):
        raise RuntimeError("Final decision changed after testing.")
    if final["predictions_sha256"] != digest(RESULTS / "final_test_predictions.csv"):
        raise RuntimeError("Saved final predictions changed.")
    runs = pd.read_csv(RESULTS / "run_summary.csv")
    comparison = pd.read_csv(RESULTS / "experiment_comparison.csv", float_precision="round_trip")
    if len(runs) != 12 or not runs.status.eq("completed").all():
        raise RuntimeError("All twelve real fits must be complete.")
    return {"audit": audit, "feature": feature, "decision": decision, "final": final,
            "runs": runs, "comparison": comparison,
            "intervals": pd.read_csv(RESULTS / "bootstrap_intervals.csv"),
            "predictions": pd.read_csv(RESULTS / "final_test_predictions.csv", float_precision="round_trip")}


def unsuccessful_attempt(study):
    c = study["comparison"].set_index("configuration")
    base = c.loc["baseline", "mean_validation_ap"]
    alternatives = c.drop(index="baseline")
    worst = alternatives.mean_validation_ap.idxmin()
    delta = float(c.loc[worst, "mean_validation_ap"] - base)
    hypotheses = {
        "history": ("A persistent signal across two scans might suppress brief lookalikes.",
                    "Averaging can dilute a newly emerging signal, and these frozen features may not encode useful temporal changes."),
        "hard_negatives": ("Giving non-tornadic warning examples more weight might reduce false alarms.",
                           "Warning labels are uncertain; extra negative pressure can also suppress real tornado cues. The authors instead downweighted warning cases."),
        "nonlinear_head": ("A nonlinear head might learn useful interactions that a linear head misses.",
                           "Its larger capacity may overfit the limited independent storm groups; the curves can support this explanation but do not prove it."),
    }
    if delta < 0:
        tried, possible = hypotheses[worst]
        text = (f"{NAMES[worst]} was a genuine less-successful adaptation. {tried} "
                f"Mean validation AP was {number(c.loc[worst, 'mean_validation_ap'])}, versus {number(base)} for the baseline "
                f"(difference {delta:+.3f}). {possible} This is a plausible explanation, not an established mechanism. "
                "We kept the predeclared comparison and selected the final configuration using all three seeds, without adding the unsuccessful change to other runs.")
        if worst == "hard_negatives":
            text += (f" At each model's validation-recall threshold, warning-negative FPR was "
                     f"{number(c.loc[worst, 'validation_warning_negative_fpr'])} versus "
                     f"{number(c.loc['baseline', 'validation_warning_negative_fpr'])} for baseline; "
                     "this secondary measure can trade off with overall ranking quality. "
                     "The AP difference is small relative to seed variation. Rechoosing a threshold to preserve 80% recall can offset "
                     "a loss-weight change's effect on raw scores, so extra negative weight does not guarantee fewer false alarms.")
        elif worst == "history" and "validation_diagnostics" in study:
            diagnostic = study["validation_diagnostics"]
            onset = diagnostic[diagnostic.subgroup == "onset_0_to_1"].set_index("configuration")
            if "history" in onset.index and onset.loc["history", "clips"] > 0:
                text += (f" On {int(onset.loc['history', 'clips'])} validation onset clips, history recall was "
                         f"{number(onset.loc['history', 'recall'])}, versus {number(onset.loc['baseline', 'recall'])} "
                         "for baseline, using each model's globally chosen threshold. This diagnosis does not prove a cause.")
        return text
    r = study["runs"].copy()
    r["drop"] = r.validation_average_precision - r.final_epoch_validation_average_precision
    row = r.sort_values("drop", ascending=False).iloc[0]
    if row["drop"] > .001:
        return (f"Continuing {NAMES[row.configuration]} (seed {int(row.seed)}) to the full 30-epoch budget was less successful "
                f"than its saved epoch {int(row.best_epoch)}: validation AP fell from {number(row.validation_average_precision)} "
                f"to {number(row.final_epoch_validation_average_precision)}. More optimization seemed reasonable, but its "
                "validation benefit did not persist. We retained the earliest best-validation checkpoint. A decline alone "
                "does not establish overfitting; compare it with the training curve and variation across seeds.")
    return ("The initial genuine data-split attempt used only 2014 with grouped stratification. It produced 599 training "
            "clips, 1,269 validation clips and 678 test clips, with only 17 TOR-category test clips. Large storm groups "
            "made that split unsuitable for dependable comparisons. Before reading model scores, we expanded to 2014-2015 "
            "and used the saved metadata-only group assignment. The repaired split passed the actual-label audit. "
            "This was a development failure rather than a failed training result.")


def make_examples(study):
    from features import read_physical_frames, resolve_sample, verify_sample_hash
    frame = study["predictions"]
    prior_path = RESULTS / "artifact_receipt.json"
    if prior_path.exists():
        prior = read_json(prior_path)
        figure, cases = FIGURES / "example_predictions.png", RESULTS / "example_cases.csv"
        if (prior.get("predictions_sha256") == study["final"]["predictions_sha256"]
                and figure.exists() and cases.exists()
                and prior.get("example_figure_sha256") == digest(figure)
                and prior.get("example_cases_sha256") == digest(cases)):
            return pd.read_csv(cases).to_dict("records")
    chosen = study["decision"]["selected_configuration"]
    score, prediction = "score_" + chosen, "prediction_" + chosen
    selections = []
    for title, label, predicted, ascending in [("False positive", 0, 1, False), ("False negative", 1, 0, True)]:
        candidates = frame[(frame.label == label) & (frame[prediction] == predicted)]
        if len(candidates):
            selections.append((title, candidates.sort_values([score, "filename"], ascending=[ascending, True]).iloc[0]))
    candidates = frame[(frame.label == 1) & (frame[prediction] == 1)]
    if len(candidates):
        selections.append(("Correct positive", candidates.sort_values("filename").sample(1, random_state=42).iloc[0]))
    if not selections:
        selections = [("Example", frame.sort_values("filename").iloc[0])]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selections), 2, figsize=(8.3, 2.05 * len(selections)), squeeze=False)
    examples = []
    for i, (title, row) in enumerate(selections):
        sample_path = resolve_sample(row, ROOT / "data/netcdf")
        verify_sample_hash(row, sample_path)
        current, _ = read_physical_frames(sample_path)
        for j, (variable, low, high, cmap, unit) in enumerate([
            ("DBZ", 0, 70, "turbo", "dBZ"), ("VEL", -70, 70, "RdBu_r", "m/s")]):
            values = current[variable][:, :, 0].copy()
            if variable == "VEL":
                values[current["range_folded_mask"][:, :, 0] != 0] = np.nan
            im = axes[i, j].imshow(values, origin="lower", aspect="auto", vmin=low, vmax=high, cmap=cmap)
            axes[i, j].set_title(f"{title}: {variable} / truth {int(row.label)}, prediction {int(row[prediction])}, score {row[score]:.3f}", fontsize=8)
            axes[i, j].set_xlabel("Range-bin index", fontsize=7)
            axes[i, j].set_ylabel("Azimuth-bin index", fontsize=7)
            axes[i, j].tick_params(labelsize=7)
            fig.colorbar(im, ax=axes[i, j], fraction=.03, pad=.02).set_label(unit, fontsize=7)
        examples.append({"case": title, "filename": row.filename, "label": int(row.label),
                         "prediction": int(row[prediction]), "score": float(row[score]),
                         "category": row.category, "storm_group": row.storm_group})
    fig.tight_layout()
    fig.savefig(FIGURES / "example_predictions.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(examples).to_csv(RESULTS / "example_cases.csv", index=False)
    return examples


def make_report_curves():
    """A legible report panel; notebook retains each individual seed curve."""
    curves = pd.read_csv(RESULTS / "learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.7), sharey=True)
    colors = {"baseline": "#175b85", "history": "#23918d", "hard_negatives": "#be7636", "nonlinear_head": "#9b4b80"}
    for ax, metric, title in zip(axes, ["train_average_precision", "validation_average_precision"], ["Training", "Validation"]):
        for name, color in colors.items():
            grouped = curves[curves.configuration == name].groupby("epoch")[metric]
            mean, sd = grouped.mean(), grouped.std(ddof=1)
            ax.plot(mean.index, mean, color=color, label=NAMES[name], linewidth=1.6)
            ax.fill_between(mean.index, mean-sd, mean+sd, color=color, alpha=.12)
        ax.set(title=title, xlabel="Epoch", ylim=(0, 1))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Average precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .2, 1, 1))
    fig.savefig(FIGURES / "report_learning_curves.png", dpi=220)
    plt.close(fig)


def make_report_examples(study, examples):
    """Reuse the fixed cases, with readable labels at printed report size."""
    from features import read_physical_frames, resolve_sample, verify_sample_hash
    path = FIGURES / "report_example_predictions.png"
    prior_path = RESULTS / "artifact_receipt.json"
    if prior_path.exists() and path.exists():
        prior = read_json(prior_path)
        if (prior.get("predictions_sha256") == study["final"]["predictions_sha256"] and
                prior.get("report_example_figure_sha256") == digest(path)):
            return
    fig, axes = plt.subplots(len(examples), 2, figsize=(8.3, 1.83 * len(examples)), squeeze=False)
    manifest = study["predictions"].set_index("filename")
    for i, case in enumerate(examples):
        row = manifest.loc[case["filename"]].copy()
        row["filename"] = case["filename"]
        sample_path = resolve_sample(row, ROOT / "data/netcdf")
        verify_sample_hash(row, sample_path)
        current, _ = read_physical_frames(sample_path)
        for j, (variable, low, high, cmap, unit) in enumerate([
                ("DBZ", 0, 70, "turbo", "dBZ"), ("VEL", -70, 70, "RdBu_r", "m/s")]):
            values = current[variable][:, :, 0].copy()
            if variable == "VEL":
                values[current["range_folded_mask"][:, :, 0] != 0] = np.nan
            im = axes[i, j].imshow(values, origin="lower", aspect="auto", vmin=low, vmax=high, cmap=cmap)
            axes[i, j].set_title(f"{case['case']}: score {case['score']:.3f}\n"
                                f"{variable} / truth {case['label']}, prediction {case['prediction']}", fontsize=10)
            axes[i, j].set_xlabel("Range-bin index", fontsize=9)
            axes[i, j].set_ylabel("Azimuth-bin index", fontsize=9)
            axes[i, j].tick_params(labelsize=9)
            bar = fig.colorbar(im, ax=axes[i, j], fraction=.03, pad=.02)
            bar.set_label(unit, fontsize=9)
            bar.ax.tick_params(labelsize=9)
    fig.tight_layout(pad=.65, h_pad=.9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_validation_diagnostics(study):
    """Interpret history with saved validation predictions; no additional fits."""
    from experiments import metrics
    validation = pd.read_csv(RESULTS / "validation_scores.csv", float_precision="round_trip")
    if "previous_label" not in validation:
        raise RuntimeError("Temporal diagnosis needs the audited previous-frame labels.")
    masks = {
        "quiet_0_to_0": (validation.previous_label == 0) & (validation.label == 0),
        "onset_0_to_1": (validation.previous_label == 0) & (validation.label == 1),
        "ending_1_to_0": (validation.previous_label == 1) & (validation.label == 0),
        "persisting_1_to_1": (validation.previous_label == 1) & (validation.label == 1),
        "warning_negative": (validation.category == "WRN") & (validation.label == 0),
    }
    rows = []
    for _, configuration in study["comparison"].iterrows():
        for name, mask in masks.items():
            sub = validation[mask]
            if not len(sub):
                rows.append({"configuration": configuration.configuration, "subgroup": name, "clips": 0,
                             "storm_groups": 0, "recall": None, "false_positive_rate": None})
                continue
            measured = metrics(sub.label, sub["score_" + configuration.configuration],
                               configuration.validation_threshold, sub.category)
            rows.append({"configuration": configuration.configuration, "subgroup": name,
                         "storm_groups": int(sub.storm_group.nunique()), **measured})
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "validation_subgroup_metrics.csv", index=False)
    study["validation_diagnostics"] = output
    return output


def write_results(study, examples):
    d, f, a = study["decision"], study["final"], study["audit"]
    weather = a["source_training_weather_date_overlap"]
    chosen = d["selected_configuration"]
    lines = ["# Executed TorNet transfer-learning study", "",
             f"All four configurations were fitted with seeds 42, 43 and 44: **12 completed real-data fits**. "
             f"Validation selected **{NAMES[chosen]}**. Final testing used the locked decision and saved predictions; the holdout was not used for configuration selection.", "",
             f"Actual-label QA passed for {a['valid_samples']:,} clips; {a['excluded_samples']} planned clips were excluded identically for all runs. "
             "Targets are the stored terminal-frame labels, not the TOR/WRN/NUL sampling categories.", "",
             "| Split | Clips | Positives | Storm groups | Positive groups | WRN negatives |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, counts in a["split_census"].items():
        lines.append(f"| {split} | {counts['samples']} | {counts['positive_samples']} | {counts['storm_groups']} | {counts['positive_storm_groups']} | {counts['warning_negative_samples']} |")
    lines += ["", metadata_correction_note(a), "",
              "The backbone was frozen and used only as a feature extractor. Each run trained a new head. "
              "Embedded checkpoint normalization and missing-value handling were preserved; the public loader and repeated-inference checks passed.", "",
              "| Configuration | Mean validation AP ± SD | Ensemble validation AP | Head parameters | Head fit seconds (3 seeds) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for _, r in study["comparison"].iterrows():
        lines.append(f"| {NAMES[r.configuration]} | {r.mean_validation_ap:.3f} ± {r.std_validation_ap:.3f} | {r.ensemble_validation_ap:.3f} | {int(r.parameters):,} | {r.total_head_training_seconds:.1f} |")
    lines += ["", "Configurations within 0.01 AP of the highest mean favor fewer parameters, then one scan, then declared order. "
              "Each experiment changed one major factor relative to the baseline; changes were not combined. "
              "Each scored model received its own validation threshold targeting at least 80% recall. This does not guarantee 80% test recall.", "",
              unsuccessful_attempt(study), "", "| Final-test model | AP | Precision | Recall | F1 | Accuracy | WRN-negative FPR |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in f["evaluated_models"]:
        m = f["metrics"][model]
        lines.append("| " + NAMES[model] + " | " + " | ".join(number(m[key]) for key in ["average_precision", "precision", "recall", "f1", "accuracy", "warning_negative_false_positive_rate"]) + " |")
    original = f["metrics"]["original_cnn"]["average_precision"]
    adapted = f["metrics"][chosen]["average_precision"]
    lines += ["", f"Validation prevalence is {d['validation_metrics'][chosen]['prevalence']:.3f}; test prevalence is "
              f"{f['metrics'][chosen]['prevalence']:.3f}. Precision and AP depend on prevalence, so compare configurations "
              "within the same fixed split rather than treating the validation-to-test AP change as a pure measure of overfitting."]
    lines += ["", operating_point_note(study)]
    difference = study["intervals"][(study["intervals"].kind == f"{chosen}_minus_reference") &
                                    (study["intervals"].model == "original_cnn") &
                                    (study["intervals"].metric == "average_precision")]
    if len(difference):
        interval = difference.iloc[0]
        lines += ["", f"Adaptation minus unchanged-CNN test AP: {adapted-original:+.3f}; descriptive 95% paired storm-group "
                  f"bootstrap interval [{number(interval.ci_lower)}, {number(interval.ci_upper)}]. "
                  "An interval containing zero does not establish a robust ranking improvement."]
    if adapted <= original:
        lines += ["", "The selected adaptation did not beat the unchanged CNN's final-test AP. This study therefore does not establish "
                  "a benefit over the pretrained detector. The chosen configuration is the best adaptation under the predeclared validation rule. "
                  "Global feature pooling may discard spatial relationships preserved by the original head; that explanation needs a separate controlled ablation."]
    lines += ["", f"The final set contains {f['test_clips']:,} clips in only {f['test_storm_groups']} storm groups. "
              "Paired storm-group bootstrap intervals and per-group metrics are in results/. Adjacent clips are correlated; "
              "these are descriptive uncertainty estimates rather than evidence of operational reliability.", "",
              f"Measured feature extraction: {study['feature']['total_elapsed_seconds']:.1f} seconds. "
              f"Longest individual head fit: {study['runs'].runtime_seconds.max():.1f} seconds. "
              "These timings exclude downloading the two full archives (32.46 GB decimal) and include no claim about a production system.", "",
              "The example figure uses the highest-scoring false positive, lowest-scoring false negative, and one seeded correct positive "
              "when those cases exist. Selection occurred after the final result was sealed; examples did not drive training decisions.", "",
              "Limitations: current tornado detection, not advance forecasting; retrospective event-centered crops; two years and few independent "
              "storm groups; selected-case prevalence; uncertain warning labels; source-training exclusion relies on documented authors' practice "
              "rather than checkpoint sample metadata; other episodes on the same weather days may have been in pretraining. "
              "Class-weighted sigmoid scores are not calibrated tornado probabilities.", "",
              f"In the predeclared pool, {weather['selected_clips_on_source_train_dates']:,}/{weather['selected_clips']:,} clips "
              f"share a terminal calendar date with authors' source-training records ({weather['shared_terminal_dates']}/{weather['selected_terminal_dates']} dates). "
              "This is date overlap, not evidence of duplicate samples or necessarily the same weather system; it limits claims about backbone exposure.", "",
              "The committed pre-run configuration remains unchanged, including its historical planning status. "
              "Completion is documented by data_audit.json, feature_provenance.json, all twelve run rows, final_decision.json and final_test.json.", "",
              "This is a technical study draft for team review. It does not supply signatures, personal contribution claims or peer evaluations. "
              "Both students must review the code and reasoning before submitting the report.", "",
              "Sources and attribution:", ""]
    lines += [f"- [{title}]({url})" for title, url in SOURCES]
    lines += ["", "MIT Lincoln Laboratory TorNet data: CC BY 4.0 as recorded by the official archive metadata; retain its release notice. "
              "TorNet software: MIT license. Checkpoint revision and upstream source commit are recorded in provenance."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_report(study):
    from report_layout import Report
    a, d, f, p = [study[k] for k in ("audit", "decision", "final", "feature")]
    chosen = d["selected_configuration"]
    weather = a["source_training_weather_date_overlap"]
    report = Report(ROOT / "output/pdf/tornet_report.pdf")
    y = report.new_page("1 / Question, data and transfer", "Can a small adaptation help?")
    y = report.paragraph("<b>Question.</b> Can a small trained head on frozen TorNet radar features improve current tornado detection? "
                         "We compare the current scan with five-minute history, stronger warning-negative weighting and a nonlinear head. "
                         "Warning researchers benefit from knowing which changes improve discrimination and which increase missed cases.", y)
    y = report.paragraph("<b>Contribution.</b> We reuse a public detector's learned features, train new prediction heads and run controlled comparisons. "
                         "This is an adaptation study, not a new weather foundation model. It classifies tornado presence at the current scan; it does not predict future tornadoes.", y)
    y = report.heading("Public data and defensible holdouts", y)
    y = report.paragraph(f"MIT TorNet v1.1 contains 203,133 catalog clips in TOR, WRN and NUL sampling categories. We used 2014-2015 authors' "
                         f"source-test cases. Removing shared source-training episodes/events and nearby source-training records left 6,414 planned clips. "
                         f"Actual NetCDF QA retained {a['valid_samples']:,}; {a['excluded_samples']} exclusions applied to every configuration. "
                         "The target is the supplied binary terminal-frame label, never the sampling category.", y)
    rows = [["Actual-label split", "Clips", "Positive", "Groups", "Positive groups"]]
    for split, c in a["split_census"].items():
        rows.append([split, c["samples"], c["positive_samples"], c["storm_groups"], c["positive_storm_groups"]])
    y = report.table(rows, [164, 80, 90, 85, 105], y)
    y = report.paragraph("Complete episodes sharing any input-window UTC date stay together. The seed-42 metadata-only split was fixed before loading labels or scores. "
                         "Actual time, label, shape, episode/event/date isolation and source-archive MD5 checks passed. Training fits all scalers and coefficients; "
                         "validation selects epochs, configuration and thresholds; final testing happens after a hashed decision lock.", y)
    if metadata_correction_note(a):
        y = report.paragraph(html.escape(metadata_correction_note(a)), y, style="small")
    y = report.heading("What was pretrained, and what did we train?", y)
    y = report.paragraph(f"The public TorNet CNN has {p['backbone_parameters']:,} parameters. We froze every backbone weight and disabled dropout, "
                         "then spatially max-pooled dropout_11 into 384 features per frame. Inputs retain six physical radar fields, two tilts, range-fold mask "
                         "and the original range/inverse-range coordinates. Normalization and NaN filling are already inside the checkpoint. "
                         "Our inputs and original logits matched the authors' loader within the verified tolerances; repeated features were identical.", y)
    y = report.paragraph("Model revision b4d103aa...; upstream code 92d9d97b...; complete hashes and versions are in the provenance receipt. "
                         "Data archives declare CC BY 4.0; TorNet software is MIT licensed. We retain MIT Lincoln Laboratory attribution and its release notice. "
                         f"The checkpoint lacks a sample list. {weather['selected_clips_on_source_train_dates']:,}/{weather['selected_clips']:,} planned clips "
                         "share a terminal date with source-training records; date overlap does not establish duplicate samples, but entirely unseen weather is not guaranteed.", y, style="small")

    y = report.new_page("2 / Controlled experiments", "One change, three repeated seeds")
    y = report.paragraph("Each experiment starts from the trained current-scan linear baseline. History averages current and previous standardized features; "
                         "hard negatives double only WRN label-zero loss weights; the MLP adds 64 ReLU units and dropout 0.2. No experimental changes are combined.", y)
    rows = [["Configuration", "Head params", "Mean val AP ± SD", "Ensemble AP"]]
    highlight = None
    for i, (_, r) in enumerate(study["comparison"].iterrows(), 1):
        rows.append([NAMES[r.configuration], f"{int(r.parameters):,}", f"{r.mean_validation_ap:.3f} ± {r.std_validation_ap:.3f}", f"{r.ensemble_validation_ap:.3f}"])
        if r.configuration == chosen:
            highlight = i
    y = report.table(rows, [204, 85, 139, 96], y, highlight=highlight)
    y = report.paragraph("Seeds 42/43/44; AdamW lr 0.001, weight decay 0.01; batch 64; 30 epochs. Train-only class-balanced BCE weights are normalized to mean one. "
                         "All linear heads use the same initialization and batch order within each seed. Keep the earliest maximum-validation-AP epoch. "
                         "AP summarizes precision across recall thresholds and is more informative than accuracy with uncommon positives.", y)
    y = report.image(FIGURES / "report_learning_curves.png", y, max_height=195)
    y = report.paragraph("Training and validation AP: mean over three seeds, shading ±1 standard deviation. The best seed is not selected. "
                         "Weighted training BCE and unweighted validation BCE are separately plotted in the notebook and figure files.", y, style="caption")
    y = report.heading("A real less-successful attempt", y)
    y = report.paragraph(html.escape(unsuccessful_attempt(study)), y, style="small")
    y = report.paragraph(f"Frozen-feature extraction took {p['total_elapsed_seconds']:.1f} s; the longest head fit took "
                         f"{study['runs'].runtime_seconds.max():.1f} s. The two full archives required 32.46 GB of transfer, a substantial data cost "
                         "despite inexpensive head training. Timings are observed on this machine, not guarantees for other hardware.", y, style="small")

    y = report.new_page("3 / Decision and final evidence", "A holdout used after selection")
    y = report.paragraph(f"<b>Selected adaptation: {NAMES[chosen]}.</b> We ranked mean validation AP across three seeds. Candidates within 0.01 AP of the best "
                         "favor fewer parameters, then one frame, then declared order. The selected model averages three saved heads' sigmoid scores. "
                         f"Its locked threshold is {d['thresholds'][chosen]:.4f}. Each scored reference gets its own highest observed validation threshold "
                         "achieving at least 80% recall, with ties retained.", y)
    rows = [["Final-test model", "AP", "Precision", "Recall", "F1", "WRN FPR"]]
    for name in f["evaluated_models"]:
        m = f["metrics"][name]
        rows.append([NAMES[name]] + [number(m[key]) for key in ("average_precision", "precision", "recall", "f1", "warning_negative_false_positive_rate")])
    y = report.table(rows, [199, 55, 73, 62, 55, 80], y, highlight=1)
    m = f["metrics"][chosen]
    majority = f["metrics"]["majority"]
    y = report.paragraph(f"Validation prevalence {d['validation_metrics'][chosen]['prevalence']:.3f}; test prevalence {m['prevalence']:.3f}; "
                         f"majority accuracy {majority['accuracy']:.3f}; selected accuracy {m['accuracy']:.3f}. AP depends on prevalence, so compare models within each split. "
                         "The physical-statistics logistic reference uses eight fixed radar summaries and train-only imputation/scaling. "
                         "The unchanged CNN reference retains its original spatial prediction head. Scores are not calibrated operational probabilities.", y, style="small")
    y = report.image(FIGURES / "final_confusion_matrix.png", y, width=275, max_height=205, x=44)
    y = report.paragraph(f"Selected-model confusion counts: TN {m['tn']}, FP {m['fp']}, FN {m['fn']}, TP {m['tp']}. "
                         "Validation sensitivity targets need not carry over to test storms. WRN FPR uses only warning-category cases whose stored target is zero.", y, style="caption")
    intervals = study["intervals"]
    ci = intervals[(intervals.kind == "model_metric") & (intervals.model == chosen) & (intervals.metric == "average_precision")]
    if len(ci):
        r = ci.iloc[0]
        y = report.paragraph(f"Selected AP's descriptive 95% storm-group bootstrap interval: [{number(r.ci_lower)}, {number(r.ci_upper)}], "
                             f"{int(r.valid_draws)}/{int(r.draws)} valid paired draws (seed 42). Only {f['test_storm_groups']} test groups are available, "
                             "and one planned group contains 703 clips. Per-group metrics and paired differences accompany the notebook; individual clips are not independent trials.", y, style="small")
    delta_ci = intervals[(intervals.kind == f"{chosen}_minus_reference") & (intervals.model == "original_cnn") & (intervals.metric == "average_precision")]
    if len(delta_ci):
        r = delta_ci.iloc[0]
        delta = m["average_precision"] - f["metrics"]["original_cnn"]["average_precision"]
        y = report.paragraph(f"Paired test AP difference, adaptation minus unchanged CNN: {delta:+.3f}; "
                             f"95% group-bootstrap interval [{number(r.ci_lower)}, {number(r.ci_upper)}]. "
                             "This comparison uses the same resampled storm groups for both models.", y, style="small")
    if f["metrics"][chosen]["average_precision"] <= f["metrics"]["original_cnn"]["average_precision"]:
        y = report.paragraph("<b>Interpretation:</b> the selected adaptation did not exceed the unchanged CNN's test AP. "
                             "The evidence does not establish an improvement over TorNet. Validation still selects this configuration among the adaptations. "
                             "A later study could test whether spatial pooling loses co-located cues preserved by the original head; this remains a hypothesis.", y, style="small")
    else:
        uncertainty = ("The paired interval includes zero. " if len(delta_ci) and
                       delta_ci.iloc[0].ci_lower <= 0 <= delta_ci.iloc[0].ci_upper else "")
        y = report.paragraph("The selected adaptation exceeded the unchanged CNN's test AP in this particular grouped holdout. "
                             + uncertainty + operating_point_note(study), y, style="small")

    y = report.new_page("4 / Errors, limits and reproduction", "What the evidence can support")
    y = report.image(FIGURES / "report_example_predictions.png", y, max_height=325)
    y = report.paragraph("Lowest tilt in native polar array coordinates. Examples are the highest-score false positive, lowest-score false negative and a seeded correct positive "
                         "when present. DBZ shows reflectivity; VEL shows radial velocity. These are illustrations of saved predictions, not proof of why a decision was wrong.", y, style="caption")
    y = report.paragraph("<b>Limits.</b> Retrospective event-centered chips differ from searching live radar. This is current detection, not advance forecasting. "
                         "Only two years and few independent storms were studied; curated prevalence differs from real warning operations. Warning targets can be uncertain. "
                         "Source-training exclusion follows the authors' protocol, while other episodes on the same weather days may have been seen during pretraining. "
                         "Temporal averaging tests smoothing, not directional storm evolution.", y, style="small")
    packages = p["packages"]
    y = report.paragraph("<b>Reproduce.</b> run_study.py orchestrates archive verification, audit, feature extraction, twelve fits, decision locking and final evaluation. "
                         "project.ipynb was executed with visible results. The pre-run JSON remains unchanged; completion is documented by the run receipts. "
                         "Identical reruns validate and reuse sealed artifacts; fresh experiments require a separate study copy. "
                         f"Python {p['python'].split()[0]}; torch {packages['torch']}; Keras {packages['keras']}; NumPy {packages['numpy']}; "
                         f"scikit-learn {packages['scikit-learn']}; hardware {html.escape(str(p.get('gpu')))}. Full settings, hashes, versions and run logs are included.", y, style="small")
    y = report.heading("References and attribution", y)
    for title, url in SOURCES:
        y = report.paragraph(f'<a href="{url}" color="#247e83">{html.escape(title)}</a>', y, style="small")
    y = report.paragraph("Technical draft for team review. Both members must understand the complete project; personal statements, signatures and private peer evaluations "
                         "must reflect their own participation. No teamwork evidence is inferred from the automated experiment.", y, style="small")
    report.save()


def notebook_source():
    import nbformat as nbf
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md("# TorNet: an executed controlled transfer-learning study\n\nWe adapt a frozen public radar CNN by training new heads. The target is **current tornado presence**, not future development. "
       "All twelve actual fits and the final decision are stored with hashes. This notebook executes the pipeline's verified reuse paths when results already exist, "
       "then displays those measured outputs. It does not retune the final holdout. The frozen plan retains its historical status string; "
       "actual completion is documented by the receipts and run rows below. Source code is in the accompanying Python modules.")
    code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd().resolve()\nassert (ROOT / 'planned_config.json').exists(), 'Run from study_tornado'\nsys.path.insert(0, str(ROOT))\nfrom run_study import ensure_data, ensure_features\nfrom experiments import train, test_once\nconfig = json.loads((ROOT / 'planned_config.json').read_text())\ndisplay(config)")
    md("## Verify data before modeling\n\nThe official archives pass their advertised MD5 hashes. The metadata-only assignment keeps episodes and connected UTC dates together. "
       "NetCDF terminal labels, native shapes, time order and actual split isolation are checked before fitting. Categories are sampling groups, not target labels.")
    code("audit = ensure_data()\nassert audit['status'] == 'passed'\ndisplay(pd.DataFrame(audit['split_census']).T)\ndisplay({k:audit[k] for k in ['valid_samples','excluded_samples','terminal_ef_mismatches','gap_seconds_min','gap_seconds_max','split_isolation','data_licenses']})\ndisplay(audit.get('catalog_id_corrections', {}))")
    md("## Frozen pretrained features\n\nSix physical radar fields, two tilts, mask and coordinates enter the unchanged frozen backbone. The model already contains normalization and NaN filling. "
       "Spatial maximum pooling creates 384 features per frame. Current and previous arrays retain manifest row alignment. Feature extraction verifies hashes on reruns.")
    code("feature = ensure_features()\ndisplay({k:feature[k] for k in ['samples','backbone_parameters','backbone_trainable_parameters','feature_dimension','gpu','total_elapsed_seconds','packages','qa','development_attempts']})")
    md("## Baseline plus three independent changes\n\nBaseline: current features with a linear head. Experiment 1: mean current/previous features, same head. "
       "Experiment 2: warning-negative loss weights ×2, baseline head/input. Experiment 3: 64-unit nonlinear head, baseline input. "
       "All use the same data, train-only scaler, epochs, optimizer settings and three seeds. Each comparison is against baseline. "
       "The earliest maximum-validation-AP checkpoint is retained. The function below fits all twelve runs in a fresh study and verifies/reuses the committed fits here.")
    code("decision = train()\nruns = pd.read_csv(ROOT / 'results/run_summary.csv')\ncomparison = pd.read_csv(ROOT / 'results/experiment_comparison.csv')\nassert len(runs)==12 and runs.status.eq('completed').all()\ndisplay(runs[['configuration','seed','best_epoch','validation_average_precision','final_epoch_validation_average_precision','parameters','runtime_seconds']])\ndisplay(comparison)\ndisplay({'selected_configuration':decision['selected_configuration'],'thresholds':decision['thresholds'],'chosen_best_epochs':decision['chosen_best_epochs'],'test_score_inspection_before_lock':decision['test_score_inspection_before_lock']})")
    code("for name in ['training_validation_ap','training_validation_loss','validation_comparison','validation_precision_recall']:\n    display(Image(filename=str(ROOT / 'results/figures' / (name+'.png')), width=850))\ndisplay(pd.read_csv(ROOT / 'results/validation_subgroup_metrics.csv')[['configuration','subgroup','clips','storm_groups','recall','false_positive_rate']])")
    md("## Seal and evaluate the final test\n\nThe selected three-head ensemble and predeclared references are evaluated after the decision file is locked. "
       "The trained baseline remains a predeclared reference; no unselected experimental variant is tested. "
       "Each scored model uses its own validation threshold targeting ≥80% recall. "
       "The majority classifier predicts the training majority class. Here test_once validates and reuses saved final predictions; it makes no new tuning decisions.")
    code("final = test_once()\nassert final['status']=='sealed_final_test'\ndisplay(pd.DataFrame(final['metrics']).T)\ndisplay({'test_clips':final['test_clips'],'test_storm_groups':final['test_storm_groups'],'decision_sha256':final['decision_sha256'],'bootstrap':final['bootstrap']})")
    code("for name in ['final_confusion_matrix','final_precision_recall','final_per_group_ap','example_predictions']:\n    display(Image(filename=str(ROOT / 'results/figures' / (name+'.png')), width=850))\ndisplay(pd.read_csv(ROOT / 'results/example_cases.csv'))\ndisplay(pd.read_csv(ROOT / 'results/bootstrap_intervals.csv'))\ndisplay(pd.read_csv(ROOT / 'results/per_group_metrics.csv'))")
    md("## Reasoning, unsuccessful attempt and limitations\n\nThe following account is generated from actual results. A poorer configuration or development attempt is discussed without inventing an unsuccessful run. "
       "Storm-group intervals describe uncertainty from this small, correlated holdout. Scores are not calibrated tornado probabilities.")
    code("display(Markdown((ROOT / 'RESULTS.md').read_text(encoding='utf-8')))\nfrom verify_results import verify\ndisplay(verify())")
    md("## Team submission\n\nThis notebook and the four-page PDF are technical artifacts for Jeffrey and Jake to review together. "
       "The team charter, individual contribution statements and private peer evaluations must describe real participation; this notebook does not fabricate them. "
       "Both teammates should be able to explain the label policy, grouped split, frozen backbone, four comparisons, validation selection and errors.")
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "TorNet study", "language": "python", "name": "tornet-study"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    return nb


def execute_notebook():
    import nbformat
    from nbclient import NotebookClient
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(ROOT / ".venv"),
                    "--name", "tornet-study", "--display-name", "TorNet study"], check=True, capture_output=True, text=True)
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    nb = notebook_source()
    NotebookClient(nb, timeout=1800, kernel_name="tornet-study", resources={"metadata": {"path": str(ROOT)}}).execute()
    if any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])):
        raise RuntimeError("Notebook contains an execution error.")
    nbformat.write(nb, ROOT / "project.ipynb")
    return {"code_cells": sum(c.cell_type == "code" for c in nb.cells),
            "all_code_cells_executed": all(c.execution_count is not None for c in nb.cells if c.cell_type == "code"),
            "sha256": digest(ROOT / "project.ipynb")}


def update_readme(study):
    path = ROOT / "README.md"
    existing = path.read_text(encoding="utf-8")
    # Preserve the registered design as historical context and keep its config immutable.
    protocol_path = ROOT / "PROTOCOL.md"
    if not protocol_path.exists():
        protocol_path.write_text(existing, encoding="utf-8")
    chosen = study["decision"]["selected_configuration"]
    path.write_text("# Executed TorNet transfer-learning study\n\n"
                    f"Completed all twelve real-data fits, validation selection and one sealed final-test evaluation. "
                    f"The selected adaptation is **{NAMES[chosen]}**. See [RESULTS.md](RESULTS.md) for measured comparisons, "
                    "errors and limitations, [project.ipynb](project.ipynb) for executed outputs, and [the report](output/pdf/tornet_report.pdf).\n\n"
                    "Run from this directory with the isolated study Python:\n\n```powershell\n"
                    ".\\.venv\\Scripts\\python.exe run_study.py\n```\n\n"
                    "The runner verifies and reuses complete data, feature, training and sealed test artifacts. "
                    "It will reject changed locked inputs. Fresh experiments belong in a separate study copy; keep this final holdout intact. "
                    "The unchanged pre-run design is in [PROTOCOL.md](PROTOCOL.md), [planned_config.json](planned_config.json), "
                    "and design/planned_manifest.csv. Its historical planning status is intentionally preserved for auditability.\n\n"
                    "Official archive downloads total 32.46 GB. The shareable bundle excludes archives, raw radar files and virtual environments; "
                    "it includes features, trained heads, manifests, provenance and the public checkpoint. Cached result inspection works without downloading "
                    "radar archives. Re-extracting features or rebuilding raw-radar example figures requires the verified archives and selected files.\n\n"
                    "Install the recorded dependencies in Python 3.11 and use KERAS_BACKEND=torch. For this RTX 5060 Ti, PyTorch 2.8.0 uses the "
                    "CUDA 12.8 wheel index https://download.pytorch.org/whl/cu128. The bundle includes the pinned authors' upstream/tornet source package; "
                    "its commit is 92d9d97bbd174be2e5cea4896c62208c5be7900d. Full requirements and checkpoint revision are recorded.\n\n"
                    "For an external Jupyter installation, register the kernel from your own study environment before opening the notebook:\n\n"
                    "```powershell\n.\\.venv\\Scripts\\python.exe -m ipykernel install --user --name tornet-study --display-name \"TorNet study\"\n```\n\n"
                    "This is current detection on retrospectively selected storm chips. It is not an operational warning system or advance tornado forecast. "
                    "The technical report is a draft for both students to review. Team signatures, contributions and peer ratings must be supplied truthfully by the students.\n",
                    encoding="utf-8")


def make_bundle():
    output = ROOT / "output/tornet_study_bundle.zip"
    output.parent.mkdir(exist_ok=True)
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.json")) + list(ROOT.glob("*.md")) + list(ROOT.glob("*.txt"))
    files += [ROOT / "project.ipynb", ROOT / "probe/tornado_detector_baseline.keras", ROOT / "output/pdf/tornet_report.pdf"]
    for folder in ["design", "results"]:
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in (ROOT / "data").glob("*") if p.is_file() and p.suffix in {".json", ".csv", ".npy"}]
    files += [p for p in (ROOT / "data/archives").glob("*.json")]
    files += [p for p in (ROOT / "upstream").glob("*") if p.is_file() and p.name.startswith(("LICENSE", "README"))]
    files += [p for p in (ROOT / "upstream/tornet").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files = sorted(set(p for p in files if p.exists()))
    inventory = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in files}
    (ROOT / "output/bundle_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [ROOT / "output/bundle_inventory.json"]:
            z.write(p, "study_tornado/" + str(p.relative_to(ROOT)).replace("\\", "/"))
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest(output), "files": len(files) + 1}


def build_all(execute_notebook=True):
    study = load_completed()
    examples = make_examples(study)
    make_report_curves()
    make_report_examples(study, examples)
    make_validation_diagnostics(study)
    write_results(study, examples)
    if execute_notebook:
        notebook = globals()["execute_notebook"]()
    else:
        previous_path = RESULTS / "artifact_receipt.json"
        if not previous_path.exists():
            raise RuntimeError("Skipping notebook execution requires a previous verified complete notebook.")
        previous = read_json(previous_path)
        notebook = previous.get("notebook", {})
        if (not notebook.get("all_code_cells_executed") or not (ROOT / "project.ipynb").exists()
                or notebook.get("sha256") != digest(ROOT / "project.ipynb")
                or previous.get("decision_sha256") != study["final"]["decision_sha256"]):
            raise RuntimeError("Existing notebook does not match the completed locked study.")
    create_report(study)
    update_readme(study)
    receipt = {"status": "complete", "notebook": notebook,
               "report": str(ROOT / "output/pdf/tornet_report.pdf"), "report_sha256": digest(ROOT / "output/pdf/tornet_report.pdf"),
               "decision_sha256": study["final"]["decision_sha256"], "predictions_sha256": study["final"]["predictions_sha256"],
               "example_figure_sha256": digest(FIGURES / "example_predictions.png"),
               "report_example_figure_sha256": digest(FIGURES / "report_example_predictions.png"),
               "example_cases_sha256": digest(RESULTS / "example_cases.csv")}
    (RESULTS / "artifact_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["bundle"] = make_bundle()
    (ROOT / "output/delivery_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute-notebook", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_all(not args.no_execute_notebook), indent=2))
