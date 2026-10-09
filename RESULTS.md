# Executed TorNet transfer-learning study

All four configurations were fitted with seeds 42, 43 and 44: **12 completed real-data fits**. Validation selected **Two-scan mean / linear**. Final testing used the locked decision and saved predictions; the holdout was not used for configuration selection.

Actual-label QA passed for 6,414 clips; 0 planned clips were excluded identically for all runs. Targets are the stored terminal-frame labels, not the TOR/WRN/NUL sampling categories.

| Split | Clips | Positives | Storm groups | Positive groups | WRN negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3667 | 438 | 28 | 12 | 1333 |
| val | 1325 | 177 | 10 | 8 | 519 |
| test | 1422 | 89 | 10 | 8 | 473 |

A genuine pre-fit QA failure exposed 2,325 WRN files with legacy embedded event/episode IDs. The authors document corrected warning IDs in v1.1. Filenames match the corrected catalog IDs, and actual terminal timestamps match the catalog exactly. We use catalog IDs for grouping, retain embedded IDs as diagnostics, and keep strict ID equality for TOR/NUL. Frame labels and the predeclared split are unchanged; the failed audit is preserved.

The backbone was frozen and used only as a feature extractor. Each run trained a new head. Embedded checkpoint normalization and missing-value handling were preserved; the public loader and repeated-inference checks passed.

| Configuration | Mean validation AP ± SD | Ensemble validation AP | Head parameters | Head fit seconds (3 seeds) |
| --- | ---: | ---: | ---: | ---: |
| Current scan / linear | 0.658 ± 0.005 | 0.660 | 385 | 3.2 |
| Two-scan mean / linear | 0.675 ± 0.001 | 0.676 | 385 | 2.4 |
| Warning negatives x2 | 0.656 ± 0.005 | 0.656 | 385 | 2.4 |
| Current scan / MLP | 0.670 ± 0.007 | 0.674 | 24,705 | 3.6 |

Configurations within 0.01 AP of the highest mean favor fewer parameters, then one scan, then declared order. Each experiment changed one major factor relative to the baseline; changes were not combined. Each scored model received its own validation threshold targeting at least 80% recall. This does not guarantee 80% test recall.

Warning negatives x2 was a genuine less-successful adaptation. Giving non-tornadic warning examples more weight might reduce false alarms. Mean validation AP was 0.656, versus 0.658 for the baseline (difference -0.003). Warning labels are uncertain; extra negative pressure can also suppress real tornado cues. The authors instead downweighted warning cases. This is a plausible explanation, not an established mechanism. We kept the predeclared comparison and selected the final configuration using all three seeds, without adding the unsuccessful change to other runs. At each model's validation-recall threshold, warning-negative FPR was 0.530 versus 0.507 for baseline; this secondary measure can trade off with overall ranking quality. The AP difference is small relative to seed variation. Rechoosing a threshold to preserve 80% recall can offset a loss-weight change's effect on raw scores, so extra negative weight does not guarantee fewer false alarms.

| Final-test model | AP | Precision | Recall | F1 | Accuracy | WRN-negative FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-scan mean / linear | 0.548 | 0.109 | 0.775 | 0.191 | 0.589 | 0.550 |
| Training majority | 0.063 | 0.000 | 0.000 | 0.000 | 0.937 | 0.000 |
| Unchanged TorNet CNN | 0.517 | 0.198 | 0.764 | 0.314 | 0.791 | 0.387 |
| Radar-statistics logistic | 0.142 | 0.086 | 0.876 | 0.156 | 0.408 | 0.858 |
| Current scan / linear | 0.508 | 0.109 | 0.787 | 0.192 | 0.585 | 0.545 |

Validation prevalence is 0.134; test prevalence is 0.063. Precision and AP depend on prevalence, so compare configurations within the same fixed split rather than treating the validation-to-test AP change as a pure measure of overfitting.

At the locked validation-derived thresholds, the adaptation detected 69 positive clips versus 68 for the unchanged CNN, but produced 564 false positives versus 276. AP describes overall ranking, while these counts describe one operating point. We retain the validation-selected configuration as this experiment's final adaptation; the evidence does not support warning deployment.

Adaptation minus unchanged-CNN test AP: +0.031; descriptive 95% paired storm-group bootstrap interval [-0.061, 0.155]. An interval containing zero does not establish a robust ranking improvement.

The final set contains 1,422 clips in only 10 storm groups. Paired storm-group bootstrap intervals and per-group metrics are in results/. Adjacent clips are correlated; these are descriptive uncertainty estimates rather than evidence of operational reliability.

Measured feature extraction: 201.0 seconds. Longest individual head fit: 1.5 seconds. These timings exclude downloading the two full archives (32.46 GB decimal) and include no claim about a production system.

The example figure uses the highest-scoring false positive, lowest-scoring false negative, and one seeded correct positive when those cases exist. Selection occurred after the final result was sealed; examples did not drive training decisions.

Limitations: current tornado detection, not advance forecasting; retrospective event-centered crops; two years and few independent storm groups; selected-case prevalence; uncertain warning labels; source-training exclusion relies on documented authors' practice rather than checkpoint sample metadata; other episodes on the same weather days may have been in pretraining. Class-weighted sigmoid scores are not calibrated tornado probabilities.

In the predeclared pool, 1,667/6,414 clips share a terminal calendar date with authors' source-training records (21/79 dates). This is date overlap, not evidence of duplicate samples or necessarily the same weather system; it limits claims about backbone exposure.

The committed pre-run configuration remains unchanged, including its historical planning status. Completion is documented by data_audit.json, feature_provenance.json, all twelve run rows, final_decision.json and final_test.json.

This is a technical study draft for team review. It does not supply signatures, personal contribution claims or peer evaluations. Both students must review the code and reasoning before submitting the report.

Sources and attribution:

- [A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data (2025)](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml)
- [Authors' accessible preprint](https://arxiv.org/html/2401.16437v1)
- [TorNet code and distribution notice](https://github.com/mit-ll/tornet)
- [Public pretrained checkpoint](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1)
- [2014 v1.1 data / CC BY 4.0](https://zenodo.org/records/12637032)
- [2015 v1.1 data / CC BY 4.0](https://zenodo.org/records/12655151)

MIT Lincoln Laboratory TorNet data: CC BY 4.0 as recorded by the official archive metadata; retain its release notice. TorNet software: MIT license. Checkpoint revision and upstream source commit are recorded in provenance.
