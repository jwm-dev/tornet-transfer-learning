# TorNet transfer learning experiment

Scientific design recorded before execution. Submission logistics and stale status text have been removed; the experimental choices below are preserved. See [RESULTS.md](RESULTS.md) for completion and measured outcomes. All implementation code is in [tornet_study.ipynb](tornet_study.ipynb).

Use the pretrained TorNet CNN to classify tornado presence in a radar observation, then test three small adaptations that could improve detection of non-tornadic lookalikes. The central temporal hypothesis is that averaging the current scan's features with those from roughly five minutes earlier suppresses transient false alarms. This is a retrospective detection study; the target is tornado presence at the current scan, not a tornado five minutes in the future.

Meteorologists and warning researchers benefit from understanding which adaptations improve discrimination while preserving tornado recall. The educational value is a clear comparison with identifiable causes, including adaptations that may perform worse.

## Model and dataset

Use [MIT TorNet version 1.1](https://github.com/mit-ll/tornet), its documented radar sequences and supplied frame labels, and the [public pretrained TorNet CNN](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1). This model accepts the radar variables directly. Prefer it over a general image encoder because its input representation and pretraining already match the radar task. Its smaller architecture makes it a practical choice for repeated head training. The [authors' model code](https://github.com/mit-ll/tornet/blob/main/tornet/models/keras/cnn_baseline.py) and [preprocessing code](https://github.com/mit-ll/tornet/blob/main/tornet/data/preprocess.py) are the implementation references.

Take the 2014 and 2015 source-test records from the downloaded catalog. Exclude source-test records whose episode or event appears anywhere in the authors' source-train partition, including their validation years. Also exclude candidates within 30 minutes and 0.25 degrees in both latitude and longitude of a source-training record, using catalog terminal times. This reapplies the proximity criterion in the [TorNet paper, Section 3.1](https://arxiv.org/html/2401.16437v1#S3.SS1). Our adaptation training set is drawn from this cleaned source-test pool; it is distinct from the dataset authors' training partition. This unusual choice protects evaluation from known checkpoint training exposure.

The catalog has three sampling categories: TOR, WRN and NUL. They describe tornado cases, warning cases and null cases. **They are not substitutes for the supplied terminal-frame binary label.** Read that label from each NetCDF file. A WRN example counts as a hard negative only if that actual label is zero. Category counts are verified now; positive-label counts remain to be audited.

The dataset is publicly released by MIT Lincoln Laboratory. Retain attribution and the distribution notice from the [2014 archive](https://zenodo.org/records/12637032) and [2015 archive](https://zenodo.org/records/12655151). The software repository carries an MIT license; keep the dataset's own release notice with it.

## Split and leakage controls

Keep complete episodes together. Connect episodes that share any UTC calendar date covered by a radar input window, including windows crossing midnight, and keep each connected group in one split. This prevents adjacent scans, the same episodes and the same radar dates from appearing in our adaptation training and evaluation sets.

The final counts and checksums are in [dataset_census.json](design/dataset_census.json), and every chosen file is in [planned_manifest.csv](design/planned_manifest.csv). The planning script selects a seeded split using only category and year counts. Its fixed objective balances both against approximate 60/20/20 targets, while requiring at least eight TOR-category groups per split. It examines 100,000 metadata assignments with seed 42 and retains the first assignment with the lowest objective. It never reads model scores or radar-frame target labels. Exact clip proportions can differ because groups remain intact.

From 6,448 candidate catalog records, the audit removes 33 records from four shared episodes and one additional nearby record, leaving **6,414 clips, 372 episodes and 48 connected storm-day groups**. The saved exclusion CSV files identify every removed record.

| Study split | Clips | Episodes | Storm-day groups | TOR category | WRN category | NUL category |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptation training | 3,667 | 203 | 28 | 438 | 1,333 | 1,896 |
| Validation | 1,325 | 93 | 10 | 177 | 519 | 629 |
| Final test | 1,422 | 76 | 10 | 89 | 473 | 860 |

Validation and test each contain eight TOR-category groups. These metadata counts are not verified positive-label counts. The largest test group contributes 703 clips, almost half the test set, which makes per-group results and group-level uncertainty particularly important. Both years occur in every split; the census records the year proportions.

Before fitting anything, audit the actual arrays and labels. Each split must contain both target classes; validation and test must each contain at least 40 positive clips, eight positive storm groups, and three groups containing WRN label-zero clips. Check strictly ordered timestamps, native input shapes and a prior-to-current interval between three and seven minutes. Exclude invalid-history examples identically from every configuration and reference model. Keep the assigned split of each surviving example; do not reshuffle to improve results. If these checks fail, stop and revise the data plan before training or examining model predictions.

Source-train exclusion follows the authors' [documented training procedure](https://github.com/mit-ll/tornet/blob/main/scripts/tornado_detection/train_tornado_keras.py). The checkpoint itself contains no embedded list of training samples. Other storms on the same weather days may have appeared in pretraining. Our grouped holdout therefore supports comparison of adaptations on held-out cases, with limited evidence of generalization to entirely unseen weather systems.

## Transfer learning setup

Pin the downloaded checkpoint to Hugging Face revision `b4d103aa9b2769d7c9258f84de89f876b0a22787`. Use a separate study environment with Keras and the PyTorch backend, the authors' custom layers and original normalization, missing-value handling, range-fold mask and coordinate channels. Preserve the six radar variables and two tilts. Do not convert this input into a color screenshot.

Freeze the CNN and disable dropout during feature extraction. Pool its `dropout_11` feature map with a spatial maximum, producing 384 features per scan. Cache both scans' features once, aligned by manifest filename. The downloaded architecture has approximately 4.7 million parameters; this inspection has not yet established runtime or inference correctness.

Fit a 384-dimensional standardizer on **current adaptation-training features only**. Apply that same standardizer to current and previous scans. No validation or test examples contribute to normalization. Each adapted model trains a new binary prediction head; the original CNN prediction head is retained only for an additional reference comparison. This is meaningful feature-extraction transfer learning.

## Four controlled configurations

Each experiment compares directly with the baseline. Do not carry earlier changes into later experiments.

| Run | Input and head | Only major change | Hypothesis and possible downside |
| --- | --- | --- | --- |
| Baseline | Current 384-dimensional features, linear head | First trained adaptation | A small head may transfer well with little data, but lack nonlinear discrimination. |
| Experiment 1 | Mean of current and previous feature vectors, same linear head | Temporal averaging | Persistent patterns may be easier to distinguish; averaging may suppress a newly emerging tornado signature. |
| Experiment 2 | Baseline input and head; double loss weight for WRN label-zero training cases | Hard-negative weighting | Lookalike warnings may generate fewer false alarms; stronger negative pressure may increase missed positives. |
| Experiment 3 | Baseline input; 64-unit ReLU head with dropout 0.2 | Head architecture | Nonlinear combinations may help; the larger head may overfit. |

The linear baseline and temporal head both have 384 inputs and **385 trainable parameters**. Temporal averaging deliberately tests persistence and smoothing; it does not learn the direction of storm evolution. Concatenating current, previous and difference vectors would increase active classifier capacity and confound that comparison. The MLP has 24,705 trainable parameters; changing capacity is precisely what Experiment 3 tests.

Hold the split, examples, frozen checkpoint, preprocessing, batch size 64, AdamW learning rate 0.001, weight decay 0.01 and 30-epoch budget constant. Use training-derived class-balanced binary cross-entropy. In Experiment 2, double only WRN label-zero weights, then normalize the complete training weight vector to mean one, as in every other run. Use the same shuffled sample order and linear-head initialization within each seed. Save the earliest epoch with the highest validation average precision. Plot weighted training loss separately from comparable, unweighted validation loss.

Run all four configurations with seeds 42, 43 and 44: twelve short head fits on cached features. Report every seed, not the best seed. Record timings and chronological errors in [experiment_log.csv](design/experiment_log.csv). A pilot must establish that one head run finishes within ten minutes. Feature extraction and archive downloads are separate measured costs; speed has not yet been demonstrated.

## Metrics and final decision

Use average precision as the primary ranking metric because positive cases are uncommon. Accuracy alone can reward predicting no tornado for every clip. Always show target prevalence and a precision-recall curve so AP has context. Select the configuration with the highest mean validation AP across three seeds. Among candidates within 0.01 absolute AP of the best, prefer fewer head parameters, then a single frame, then the declared run order. This practical tie rule is not a claim of statistical equivalence.

Average the three saved heads' sigmoid scores for the chosen configuration. Report both mean and standard deviation of individual seed AP and ensemble AP. For each scored model, choose the highest **observed** validation score whose `score >= threshold` predictions achieve at least 80% recall, retaining ties. Freeze the threshold before testing. This is a sensitivity target for comparison, not a promise of 80% test recall. Class-weighted scores are not calibrated operational probabilities.

Include three references in addition to the trained baseline: a majority-class predictor, the unchanged pretrained CNN, and a simple logistic classifier on predefined physical-unit radar statistics. The latter uses current-scan reflectivity, velocity spread, polarimetric summaries and radar range; exact statistics and fixed logistic settings are in [planned_config.json](planned_config.json). It has no pretrained image features. Fit its imputation, scaling and coefficients on adaptation training data only. Give every scored reference its own threshold chosen by the same validation rule. Compute the majority reference directly from the training majority label.

Before final evaluation, record the chosen configuration, all reference models, epoch checkpoints, scaler, thresholds, seeds and data/configuration hashes in a final-decision file. Evaluate the chosen configuration and the predeclared references on the final test set once. Keep unselected experimental comparisons on validation data. Save final predictions so plots and uncertainty estimates can be produced without repeated tuning.

Report AP, precision, recall, F1, confusion counts, overall false-positive rate and false-positive rate specifically among WRN label-zero cases. Show per-group results because long storms contribute many correlated clips. Use a paired bootstrap of whole test groups for 1,000 draws, seed 42, with 95% percentile intervals; report valid draw counts when a metric is undefined. With ten test groups, treat intervals as descriptive and avoid strong significance claims.

## Failure analysis and evidence

The first single-year grouped-split attempt was genuinely unsuitable: its training set had only 599 clips, while validation contained 1,269; test contained only 17 TOR-category clips. Large storms dominated individual folds. This failed development attempt motivated additional years and explicit metadata stratification. The source-partition audit also uncovered shared episode IDs. These are documented data-design findings, not failed trained models.

For any less successful trained adaptation, explain its intended benefit, observed validation result, plausible mechanism and the resulting decision. For temporal averaging, inspect correctly and incorrectly classified transitions; for warning weighting, compare misses with warning false alarms; for the MLP, compare training and validation curves across seeds. The pooled representation can also combine channel maxima from different radar locations; compare the unchanged CNN to see whether this reduction loses useful local relationships. These are hypotheses to investigate, not fabricated findings.

The notebook should contain the data/label audit, twelve-run comparison table, training and validation curves, precision-recall curves, final confusion matrix and examples of true positives, false positives and false negatives. Select example cases with a declared rule, such as highest-confidence errors and a seeded positive example. Do not choose showcase images as evidence of average performance.
