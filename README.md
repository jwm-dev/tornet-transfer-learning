# Executed TorNet transfer-learning study

Completed all twelve real-data fits, validation selection and one sealed final-test evaluation. The selected adaptation is **Two-scan mean / linear**. See [RESULTS.md](RESULTS.md) for measured comparisons, errors and limitations, [project.ipynb](project.ipynb) for executed outputs, and [the report](output/pdf/tornet_report.pdf).

Run from this directory with the isolated study Python:

```powershell
.\.venv\Scripts\python.exe run_study.py
```

The runner verifies and reuses complete data, feature, training and sealed test artifacts. It will reject changed locked inputs. Fresh experiments belong in a separate study copy; keep this final holdout intact. The unchanged pre-run design is in [PROTOCOL.md](PROTOCOL.md), [planned_config.json](planned_config.json), and design/planned_manifest.csv. Its historical planning status is intentionally preserved for auditability.

Official archive downloads total 32.46 GB. The shareable bundle excludes archives, raw radar files and virtual environments; it includes features, trained heads, manifests, provenance and the public checkpoint. Cached result inspection works without downloading radar archives. Re-extracting features or rebuilding raw-radar example figures requires the verified archives and selected files.

Install the recorded dependencies in Python 3.11 and use KERAS_BACKEND=torch. For this RTX 5060 Ti, PyTorch 2.8.0 uses the CUDA 12.8 wheel index https://download.pytorch.org/whl/cu128. The bundle includes the pinned authors' upstream/tornet source package; its commit is 92d9d97bbd174be2e5cea4896c62208c5be7900d. Full requirements and checkpoint revision are recorded.

For an external Jupyter installation, register the kernel from your own study environment before opening the notebook:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name tornet-study --display-name "TorNet study"
```

This is current detection on retrospectively selected storm chips. It is not an operational warning system or advance tornado forecast. The technical report is a draft for both students to review. Team signatures, contributions and peer ratings must be supplied truthfully by the students.
