"""Small algorithm/holdout checks; synthetic fixtures are never study results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments as experiment


class ExperimentChecks(unittest.TestCase):
    def test_exclusive_atomic_json_preserves_an_existing_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decision.json"
            experiment.json_write(path, {"fixed": 1}, exclusive=True)
            original_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                experiment.json_write(path, {"fixed": 2}, exclusive=True)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_threshold_retains_ties_and_is_largest_allowed_observed_value(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0])
        scores = np.array([.95, .8, .8, .4, .1, .8, .3])
        selected = experiment.recall_threshold(labels, scores)
        self.assertEqual(selected, .4)
        self.assertEqual(np.mean(scores[labels == 1] >= selected), .8)
        for larger in np.unique(scores[scores > selected]):
            self.assertLess(np.mean(scores[labels == 1] >= larger), .8)

    def test_paired_group_bootstrap_keeps_identical_models_identical(self):
        frame = pd.DataFrame({"storm_group": ["a", "a", "b", "b", "b", "c"],
                              "label": [1, 0, 1, 0, 0, 1], "category": ["TOR", "WRN", "TOR", "WRN", "NUL", "TOR"],
                              "score_history": [.9, .7, .8, .4, .2, .6],
                              "score_baseline": [.9, .7, .8, .4, .2, .6]})
        result = experiment.grouped_bootstrap(frame, ["history", "baseline"], {"history": .6, "baseline": .6}, 0,
                                             "history", draws=100, seed=42)
        paired = result[result.kind == "history_minus_reference"]
        self.assertTrue((paired.ci_lower == 0).all())
        self.assertTrue((paired.ci_upper == 0).all())

    def test_split_overlap_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            pd.DataFrame({"filename": ["a", "b", "c"], "study_split": ["train", "val", "test"],
                          "label": [0, 1, 0], "category": ["WRN", "TOR", "NUL"],
                          "storm_group": ["same", "same", "other"]}).to_csv(data / "runtime_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "storm_group occurs in multiple"):
                experiment.load_data(data)

    def test_end_to_end_seals_choice_and_reuses_saved_test_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            data = study / "data"
            data.mkdir()
            config = json.loads((Path(__file__).resolve().parents[1] / "planned_config.json").read_text(encoding="utf-8"))
            # Two epochs suffice to exercise the protocol; this temporary fixture
            # is explicitly separate from all actual 30-epoch experiment outputs.
            config["training"]["epochs"] = 2
            (study / "planned_config.json").write_text(json.dumps(config), encoding="utf-8")
            rows = []
            for split, groups in (("train", 28), ("val", 10), ("test", 10)):
                for group in range(groups):
                    for scan in range(10):
                        positive = int(scan < 5)
                        rows.append({"filename": f"{split}_{group}_{scan}", "study_split": split,
                                     "label": positive, "previous_label": positive,
                                     "category": "TOR" if positive else ("WRN" if scan < 8 else "NUL"),
                                     "storm_group": f"{split}_{group}"})
            frame = pd.DataFrame(rows)
            frame.to_csv(data / "runtime_manifest.csv", index=False)
            rng = np.random.default_rng(19)
            current = rng.normal(size=(len(frame), 384)).astype(np.float32)
            current[:, 0] += frame.label.to_numpy() * 2
            statistics = current[:, :8].copy()
            statistics[::17, 3] = np.nan
            arrays = {"features_current": current, "features_previous": current + rng.normal(0, .2, current.shape).astype(np.float32),
                      "original_cnn_scores": 1 / (1 + np.exp(-current[:, 0])), "radar_statistics": statistics}
            files = {}
            for name, values in arrays.items():
                path = data / f"{name}.npy"
                np.save(path, values)
                files[path.name] = {"sha256": experiment.sha256(path)}
            (data / "features_sample_ids.json").write_text(json.dumps(frame.filename.tolist()), encoding="utf-8")
            (data / "feature_provenance.json").write_text(json.dumps({"manifest_sha256": experiment.sha256(data / "runtime_manifest.csv"),
                "feature_rows": len(frame), "files": files}), encoding="utf-8")
            decision = experiment.train(study)
            self.assertFalse(decision["test_score_inspection_before_lock"])
            self.assertFalse((study / "results" / "final_test.json").exists())
            summaries = pd.read_csv(study / "results" / "run_summary.csv")
            self.assertEqual(len(summaries), 12)
            with patch.object(experiment, "plot_final", side_effect=RuntimeError("synthetic report interruption")):
                with self.assertRaisesRegex(RuntimeError, "synthetic report interruption"):
                    experiment.test_once(study)
            self.assertTrue((study / "results" / "final_test_predictions.csv").exists())
            self.assertFalse((study / "results" / "final_test.json").exists())
            with patch.object(experiment, "predict_head", side_effect=AssertionError("must reuse committed predictions after interruption")):
                final = experiment.test_once(study)
            self.assertEqual(final["test_clips"], 100)
            allowed = {decision["selected_configuration"], *experiment.REFERENCES}
            self.assertEqual(set(final["evaluated_models"]), allowed)
            with patch.object(experiment, "predict_head", side_effect=AssertionError("must not recompute test predictions")):
                self.assertEqual(experiment.test_once(study), final)
            tampered = arrays["features_previous"].copy()
            tampered[0, 0] += 1
            np.save(data / "features_previous.npy", tampered)
            with self.assertRaisesRegex(ValueError, "changed after decision lock"):
                experiment.test_once(study)


if __name__ == "__main__":
    unittest.main()
