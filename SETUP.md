# Run the single notebook

All first-party study code is in [tornet_study.ipynb](tornet_study.ipynb).
GitHub displays its saved results without installing anything.

## Windows / PowerShell

From the repository root, create an isolated Python 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tornet-study --display-name "TorNet study"
```

Open `tornet_study.ipynb` in VS Code or an existing Jupyter installation, select
**TorNet study**, and choose **Run All**. Keep the notebook's working directory
at the repository root. If you need a browser notebook editor:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyter lab tornet_study.ipynb
```

On Linux/macOS, create the environment with `python3.11 -m venv .venv` and use
`.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## What Run All does

1. Displays the saved split counts, measured comparisons, and precision-recall chart.
2. Loads the complete implementation from the notebook's source cells.
3. Restores the pinned public checkpoint when absent (**18.8 MB**) and checks its SHA-256.
4. Independently verifies all 12 fits, validation selection, thresholds, sealed
   predictions, and per-group metrics without making new test predictions.
5. Runs temporary synthetic tests for locks, thresholds, leakage checks,
   resampling, interruption recovery, and verified downloads.

The notebook writes its source modules into ignored `.study_runtime/`. This
allows the original feature-extraction and experiment source hashes to remain
verifiable while keeping the full implementation inside one notebook. Required
third-party TorNet code stays in `upstream/` with its MIT license.

After the checkpoint is downloaded, cached inspection and verification work
offline on CPU. A GPU is useful for extracting features from raw radar, but is
not required to verify this completed study.

## Optional execution paths

- Set `REPLAY_CACHED_PIPELINE = True` to check reuse of the frozen features,
  trained heads, and sealed test result. The real study is not refitted.
- Set `ACQUIRE_RAW_RADAR = True` only to download both original annual archives
  (**32.46 GB**, plus extraction space) and the **38.2 MB** public catalog. Input
  downloads are checksum-verified. Normal notebook use leaves this disabled.

The original run used PyTorch 2.8.0 with CUDA 12.8 on an RTX 5060 Ti. To match
that GPU wheel, install it before the other requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Study records

The locked configuration and dataset census retain their original planning
status. `results/final_test.json` documents completion; the notebook verifies the
measured artifacts against the original inputs. Historical absolute paths in
provenance describe the original machine and are not required for cached use.

Keep this completed study's inputs, model code, and holdout intact. New
hypotheses need a separate study and a new holdout. Removed packaging artifacts
and the original standalone scripts remain recoverable in Git history.
