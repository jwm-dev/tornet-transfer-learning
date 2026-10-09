"""Frozen TorNet features with the authors' scientific input representation.

This module never selects a model, fits a scaler, or evaluates test predictions.
Array row i always corresponds to runtime_manifest.csv row i. The checkpoint
already normalizes the six physical radar fields and fills their NaNs internally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))
VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]
INPUT_KEYS = VARIABLES + ["range_folded_mask", "coordinates"]
STAT_NAMES = ["DBZ_p95", "DBZ_fraction_ge40", "VEL_p95_minus_p05", "RHOHV_p05",
              "ZDR_median", "KDP_p95", "WIDTH_p95", "radar_range_midpoint_m"]
PINNED_UPSTREAM = "92d9d97bbd174be2e5cea4896c62208c5be7900d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nc_array(variable, selection=Ellipsis) -> np.ndarray:
    """Decode NetCDF masks/scale factors like the authors' xarray loader."""
    a = variable[selection]
    if np.ma.isMaskedArray(a):
        a = a.astype(np.float32).filled(np.nan)
    return np.asarray(a, dtype=np.float32)


def read_physical_frames(path: Path) -> tuple[dict, dict]:
    """Return current and previous physical arrays, without applying normalization."""
    from tornet.data.preprocess import compute_coordinates

    with Dataset(str(path), "r") as ds:
        raw = {name: nc_array(ds.variables[name], slice(-2, None)) for name in VARIABLES}
        raw["range_folded_mask"] = nc_array(ds.variables["range_folded_mask"], slice(-2, None))
        ranges = nc_array(ds.variables["range_limits"])
        azimuths = nc_array(ds.variables["azimuth_limits"])
    if any(raw[name].shape != (2, 120, 240, 2) for name in INPUT_KEYS[:-1]):
        raise ValueError(f"Unexpected input dimensions in {path}")
    metadata = {"rng_lower": ranges[:1], "rng_upper": ranges[1:],
                "az_lower": azimuths[:1], "az_upper": azimuths[1:]}
    metadata.update(raw)
    # This is the exact public coordinate implementation, without azimuth input.
    coords = compute_coordinates(metadata, include_az=False, tilt_last=True, backend=np)
    frames = []
    for index in [-1, -2]:
        frame = {name: np.asarray(raw[name][index], dtype=np.float32) for name in INPUT_KEYS[:-1]}
        frame["coordinates"] = np.asarray(coords, dtype=np.float32)
        frame["range_midpoint_m"] = float(np.mean(ranges))
        frames.append(frame)
    return frames[0], frames[1]


def radar_statistics(current: dict) -> np.ndarray:
    """Eight fixed physical summaries, no learned data-dependent preprocessing."""
    mask = np.isfinite(current["range_folded_mask"]) & (current["range_folded_mask"] == 0)
    values = {}
    for name in VARIABLES:
        a = current[name]
        values[name] = a[mask & np.isfinite(a)]

    def quantile(name, q):
        return float(np.quantile(values[name], q)) if values[name].size else np.nan

    return np.asarray([
        quantile("DBZ", .95),
        float(np.mean(values["DBZ"] >= 40)) if values["DBZ"].size else np.nan,
        quantile("VEL", .95) - quantile("VEL", .05),
        quantile("RHOHV", .05), quantile("ZDR", .5), quantile("KDP", .95),
        quantile("WIDTH", .95), current["range_midpoint_m"],
    ], dtype=np.float32)


def load_extractor(checkpoint: Path):
    import keras
    import torch
    from tornet.models.keras.layers import CoordConv2D, FillNaNs  # Registers serialization.

    if keras.backend.backend() != "torch":
        raise RuntimeError("This study requires KERAS_BACKEND=torch")
    keras.utils.set_random_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = keras.models.load_model(str(checkpoint), compile=False)
    model.trainable = False
    # One shared forward pass provides both pretrained decisions and features.
    pooled = keras.layers.GlobalMaxPooling2D(name="study_global_max_pool")(
        model.get_layer("dropout_11").output)
    extractor = keras.Model(inputs=model.input, outputs=[pooled, model.outputs[0]])
    extractor.trainable = False
    model.eval()
    extractor.eval()
    assert not any(p.requires_grad for p in extractor.parameters())
    return model, extractor


def predict_arrays(model, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    x = {name: np.stack([frame[name] for frame in frames]) for name in INPUT_KEYS}
    with torch.inference_mode():
        pooled, logits = model(x, training=False)
    return pooled.detach().cpu().numpy(), logits.detach().cpu().numpy().reshape(-1)


def author_reference_check(path: Path, model, extractor) -> dict:
    """Reproduce the published loader/preprocessing/inference on one train clip."""
    from tornet.data.loader import read_file
    from tornet.data.preprocess import add_coordinates
    import torch

    current, previous = read_physical_frames(path)
    reference = read_file(str(path), n_frames=1, tilt_last=True)
    add_coordinates(reference, include_az=False, tilt_last=True, backend=np)
    reference["coordinates"] = reference["coordinates"][None, ...]
    own = {name: current[name][None, ...] for name in INPUT_KEYS}
    max_input_difference = {}
    for name in INPUT_KEYS:
        expected = reference[name].astype(np.float32)
        np.testing.assert_allclose(own[name], expected, rtol=0, atol=0, equal_nan=True)
        finite = np.isfinite(expected)
        max_input_difference[name] = float(np.max(np.abs(own[name][finite] - expected[finite]))) if finite.any() else 0.
    with torch.inference_mode():
        ref = model({name: reference[name] for name in INPUT_KEYS}, training=False)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        ref_logit = ref.detach().cpu().numpy().reshape(-1)
    feature1, score1 = predict_arrays(extractor, [current])
    feature2, score2 = predict_arrays(extractor, [current])
    _, two_scores = predict_arrays(extractor, [current, previous])
    np.testing.assert_array_equal(feature1, feature2)
    np.testing.assert_array_equal(score1, score2)
    np.testing.assert_allclose(score1, ref_logit, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(two_scores[:1], score1, rtol=1e-4, atol=1e-4)
    if feature1.shape != (1, 384) or not np.isfinite(feature1).all():
        raise ValueError("Invalid feature output")
    return {"training_sample": str(path), "reference": "Public read_file + add_coordinates + original CNN",
            "input_max_absolute_differences": max_input_difference,
            "original_vs_joint_extractor_logit_difference": float(np.max(np.abs(score1-ref_logit))),
            "batch_vs_single_logit_difference": float(np.max(np.abs(two_scores[:1]-score1))),
            "repeat_feature_exactly_equal": True, "feature_shape": list(feature1.shape),
            "original_train_sample_logit": float(ref_logit[0]),
            "note": "Reference input only; its historical terminal-label override is not used as a study target."}


def resolve_sample(row, data_root: Path) -> Path:
    attempted = []
    for column in ["nc_path", "local_path", "path"]:
        if column in row and pd.notna(row[column]):
            p = Path(str(row[column]))
            if not p.is_absolute():
                p = ROOT / p
            attempted.append(p)
            if p.is_file():
                return p
    fallback = data_root / str(row["filename"])
    attempted.append(fallback)
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Radar file for {row['filename']} is unavailable; tried "
                            + "; ".join(str(p) for p in attempted))


def verify_sample_hash(row, path: Path) -> bool:
    """A relocated raw file must still be the physical clip that passed the audit."""
    if "data_sha256" in row and pd.notna(row["data_sha256"]):
        expected = str(row["data_sha256"]).strip().lower()
        if sha256(path) != expected:
            raise ValueError(f"Radar SHA256 mismatch for {row['filename']}: {path}")
        return True
    return False


def validated_cache(manifest_path: Path, checkpoint: Path) -> dict | None:
    """Verify saved rows and hashes before permitting a notebook cache reuse."""
    outdir = ROOT / "data"
    receipt_path = outdir / "feature_provenance.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed":
        return None
    expected = {"manifest_sha256": sha256(manifest_path),
                "checkpoint_sha256": sha256(checkpoint),
                "features_script_sha256": sha256(Path(__file__))}
    if any(receipt.get(k) != v for k, v in expected.items()):
        return None
    for source, digest in receipt["source_sha256"].items():
        if sha256(UPSTREAM / source) != digest:
            return None
    ids_path = outdir / "features_sample_ids.json"
    if not ids_path.exists() or sha256(ids_path) != receipt["sample_ids_sha256"]:
        return None
    manifest = pd.read_csv(manifest_path)
    if json.loads(ids_path.read_text(encoding="utf-8")) != manifest.filename.tolist():
        return None
    for filename, meta in receipt["files"].items():
        path = outdir / filename
        if not path.exists() or sha256(path) != meta["sha256"]:
            return None
        array = np.load(path, mmap_mode="r")
        if list(array.shape) != meta["shape"] or str(array.dtype) != meta["dtype"]:
            return None
    if receipt["feature_rows"] != len(manifest):
        return None
    return receipt


def extract(manifest_path: Path, data_root: Path, checkpoint: Path, batch_size: int):
    import keras
    import torch

    cached = validated_cache(manifest_path, checkpoint)
    if cached is not None:
        print(f"Reused checksum-verified frozen features for {cached['feature_rows']} aligned clips", flush=True)
        return cached
    if (ROOT/"results/final_decision.json").exists():
        raise RuntimeError("Feature cache is invalid after the final decision lock; refuse to change the sealed study")
    started = time.perf_counter()
    manifest = pd.read_csv(manifest_path)
    if manifest.filename.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if list(manifest.filename) != sorted(manifest.filename):
        raise ValueError("Runtime manifest must be sorted by filename")
    model, extractor = load_extractor(checkpoint)
    train_row = manifest.loc[manifest.study_split == "train"].iloc[0]
    train_path = resolve_sample(train_row, data_root)
    verify_sample_hash(train_row, train_path)
    qa = author_reference_check(train_path, model, extractor)
    n = len(manifest)
    arrays = {
        "features_current.npy": np.empty((n, 384), dtype=np.float32),
        "features_previous.npy": np.empty((n, 384), dtype=np.float32),
        "original_cnn_scores.npy": np.empty(n, dtype=np.float32),
        "radar_statistics.npy": np.empty((n, 8), dtype=np.float32),
    }
    outdir = ROOT / "data"
    outdir.mkdir(exist_ok=True)
    timings = []
    verified_raw_hashes = 0
    print(f"Feature QA passed; GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}; extracting {n} clips", flush=True)
    feature_started = time.perf_counter()
    for low in range(0, n, batch_size):
        high = min(low+batch_size, n)
        t = time.perf_counter()
        current, previous = [], []
        for i, row in manifest.iloc[low:high].iterrows():
            path = resolve_sample(row, data_root)
            verified_raw_hashes += int(verify_sample_hash(row, path))
            c, p = read_physical_frames(path)
            current.append(c)
            previous.append(p)
            arrays["radar_statistics.npy"][i] = radar_statistics(c)
        # Interleave current/previous to amortize kernels without changing order.
        interleaved = [frame for pair in zip(current, previous) for frame in pair]
        pooled, logits = predict_arrays(extractor, interleaved)
        if not np.isfinite(pooled).all() or not np.isfinite(logits).all():
            raise ValueError(f"Non-finite CNN output at rows {low}:{high}")
        arrays["features_current.npy"][low:high] = pooled[::2]
        arrays["features_previous.npy"][low:high] = pooled[1::2]
        arrays["original_cnn_scores.npy"][low:high] = 1./(1.+np.exp(-np.clip(logits[::2], -80., 80.)))
        timings.append(time.perf_counter()-t)
        if low == 0 or high == n or (low // batch_size) % 25 == 0:
            elapsed = time.perf_counter()-feature_started
            eta = elapsed/max(high, 1)*(n-high)
            print(f"Extracted {high}/{n}; elapsed {elapsed:.1f}s; estimated remaining {eta:.1f}s", flush=True)
    elapsed = time.perf_counter()-started
    for filename, array in arrays.items():
        temp = outdir / (filename + ".partial")
        with temp.open("wb") as f:
            np.save(f, array)
        temp.replace(outdir / filename)
    ids_path = outdir / "features_sample_ids.json"
    ids_temp = ids_path.with_suffix(".json.partial")
    ids_temp.write_text(json.dumps(manifest.filename.tolist(), indent=2), encoding="utf-8")
    ids_temp.replace(ids_path)
    if (UPSTREAM/".git").exists():
        commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    else:
        # Source-only submission archives deliberately omit git internals.
        commit = (ROOT/"upstream_revision.txt").read_text(encoding="utf-8").strip()
    if commit != PINNED_UPSTREAM:
        raise RuntimeError("Upstream source revision changed")
    version_packages = ["torch", "keras", "numpy", "pandas", "netCDF4", "xarray", "h5py", "scikit-learn"]
    source_names = ["tornet/models/keras/layers.py", "tornet/models/keras/cnn_baseline.py",
                    "tornet/data/preprocess.py", "tornet/data/loader.py"]
    provenance = {
        "status": "completed", "samples": n, "feature_rows": n, "feature_dimension": 384,
        "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "features_script_sha256": sha256(Path(__file__)),
        "checkpoint_repository": "tornet-ml/tornado_detector_baseline_v1",
        "checkpoint_revision": "b4d103aa9b2769d7c9258f84de89f876b0a22787",
        "upstream_revision": commit, "source_sha256": {s: sha256(UPSTREAM/s) for s in source_names},
        "keras_backend": keras.backend.backend(), "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {p: importlib.metadata.version(p) for p in version_packages},
        "python": sys.version, "backbone_parameters": model.count_params(), "backbone_trainable_parameters": 0,
        "feature_layer": "dropout_11", "pooling": "GlobalMaxPooling2D over spatial axes",
        "preprocessing": "Raw physical values; embedded Normalize_* and FillNaNs; original r/r_inverse coordinates; original range_folded_mask",
        "inference": "training=False, all weights frozen, torch.inference_mode, deterministic algorithms, TF32 disabled",
        "radar_statistics_columns": STAT_NAMES,
        "statistics_mask": "Exclude non-finite values and any cell whose range-fold mask is nonzero across the corresponding tilt",
        "statistics_range": "Mean of physical range_limits in meters",
        "original_cnn_scores": "Sigmoid of original model current-frame raw logit; no test metrics inspected",
        "row_alignment": "Exact runtime_manifest filename order; features_sample_ids.json stores row identifiers",
        "raw_data_hashes_verified": verified_raw_hashes,
        "qa": qa, "batch_size_clips": batch_size, "batch_size_frames": 2*batch_size,
        "development_attempts": [{"attempt": "First synthetic runtime smoke", "result": "Failed with AttributeError: list has no attribute detach",
                                  "reason": "The serialized checkpoint's model.output is a singleton list; nesting it in joint outputs produced a nested logit result",
                                  "repair": "Use model.outputs[0] for the joint feature/logit model and unwrap the original model's singleton output in QA",
                                  "verification": "Repaired synthetic shape test and public-input real-NetCDF reference inference both passed"}],
        "total_elapsed_seconds": elapsed, "extraction_seconds": time.perf_counter()-feature_started,
        "mean_batch_seconds": float(np.mean(timings)), "batches": len(timings),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "files": {name: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256(outdir/name)} for name,a in arrays.items()},
        "sample_ids_sha256": sha256(ids_path),
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    receipt_text = json.dumps(provenance, indent=2)
    for destination in [outdir/"feature_provenance.json", results/"feature_provenance.json"]:
        temp = destination.with_suffix(".json.partial")
        temp.write_text(receipt_text, encoding="utf-8")
        temp.replace(destination)
    (ROOT/"requirements_lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(f"Saved aligned feature arrays and provenance; {elapsed:.1f}s total", flush=True)
    return provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT/"data/runtime_manifest.csv")
    p.add_argument("--data-root", type=Path, default=ROOT/"data/raw")
    p.add_argument("--checkpoint", type=Path, default=ROOT/"probe/tornado_detector_baseline.keras")
    p.add_argument("--batch-size", type=int, default=16, help="Clips per batch; two radar frames each")
    p.add_argument("--qa-sample", type=Path)
    a = p.parse_args()
    if a.qa_sample:
        model, extractor = load_extractor(a.checkpoint)
        print(json.dumps(author_reference_check(a.qa_sample, model, extractor), indent=2))
    else:
        extract(a.manifest, a.data_root, a.checkpoint, a.batch_size)


if __name__ == "__main__":
    main()
