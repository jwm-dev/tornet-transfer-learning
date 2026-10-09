# Fresh clone and cached replay

The executed notebook, four-page PDF, trained heads, frozen features and pinned public checkpoint are included. The original radar archives, temporary files and Python environments are excluded. The source bundle's inventory remains unchanged; repository additions are this guide, `.gitignore`, `.gitattributes` and the existing integrity tests.

Use Python 3.11. On Windows, run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:KERAS_BACKEND = "torch"
.\.venv\Scripts\python.exe run_study.py
```

Cached replay verifies and reuses the committed data audit, features, twelve fits and sealed test predictions. It does not need the 32.46 GB radar archives. It regenerates presentation artifacts from saved results. The retained absolute paths in provenance describe the original run; cached replay was verified after relocation.

For notebook access, register the interpreter from your new environment:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name tornet-study --display-name "TorNet study"
```

On Linux or Kaggle, use `python3.11 -m venv .venv`, `.venv/bin/python` in place of the Windows executable, and `export KERAS_BACKEND=torch`. Kaggle may already provide the required Python environment.

The original run used PyTorch 2.8.0 with CUDA 12.8 on an RTX 5060 Ti. To match that GPU wheel, install it before the remaining requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is the project dependency list. `requirements_lock.txt` records the complete original environment, including `torch==2.8.0+cu128`; its exact GPU build requires the CUDA wheel index. Cached inspection does not require the original GPU.

The integrity tests use temporary synthetic fixtures, separate from actual study results:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiments.py
```

The sealed holdout belongs to this completed study. Keep its configuration and model inputs intact; design fresh experiments in a separate study with a new holdout. Re-extracting radar features requires the official archives and raw selected clips.

See `README.md`, `RESULTS.md`, `PROTOCOL.md` and the provenance receipts for the research question, measured results, limitations, licenses and sources.
