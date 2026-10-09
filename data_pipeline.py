"""Download checksum-verified TorNet archives, extract selected cases, audit labels.

This stage never loads a model or computes predictions. Study splits remain fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ARCHIVES = {
    2014: {"record": 12637032, "filename": "tornet_2014.tar.gz", "size": 15066513529,
           "md5": "db8bbdf9939153209f7b3d7cfe60dd7d"},
    2015: {"record": 12655151, "filename": "tornet_2015.tar.gz", "size": 17395624045,
           "md5": "7251feab3c571c71fcca8745d8431c7d"},
}
VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def download_archive(year, connections=8):
    meta = ARCHIVES[year]
    out = DATA / "archives" / meta["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    verified = out.with_suffix(out.suffix + ".verified.json")
    if out.exists() and out.stat().st_size == meta["size"] and verified.exists():
        prior = json.loads(verified.read_text())
        if prior["md5"] == meta["md5"] and prior.get("mtime_ns") == out.stat().st_mtime_ns:
            print(f"{year}: previously checksum-verified archive reused", flush=True)
            return out
        actual = digest(out, "md5")
        if actual != meta["md5"]:
            raise RuntimeError(f"Previously downloaded archive was modified: {out}")
        prior.update({"md5": actual, "verified_at": utc_now(), "mtime_ns": out.stat().st_mtime_ns})
        write_json(verified, prior)
        return out
    api = f"https://zenodo.org/api/records/{meta['record']}"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    entry = next(f for f in response.json()["files"] if f["key"] == meta["filename"])
    assert entry["size"] == meta["size"] and entry["checksum"] == "md5:" + meta["md5"]
    write_json(out.parent / f"{year}_zenodo_metadata.json", response.json())
    url = f"https://zenodo.org/records/{meta['record']}/files/{meta['filename']}?download=1"
    part = out.with_suffix(out.suffix + ".part")
    started = time.monotonic()
    prefix_size = part.stat().st_size if part.exists() else 0
    chunk_root = out.parent / (meta["filename"] + ".chunks")
    chunk_root.mkdir(exist_ok=True)
    plan_path = chunk_root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        prefix_size = plan["prefix_size"]
    else:
        plan = {"prefix_size": prefix_size, "chunk_size": 64 * 1024 * 1024,
                "official_size": meta["size"]}
        write_json(plan_path, plan)
    ranges = [(start, min(meta["size"] - 1, start + plan["chunk_size"] - 1))
              for start in range(prefix_size, meta["size"], plan["chunk_size"])]

    def fetch_chunk(bounds):
        start, end = bounds
        target = chunk_root / f"{start:012d}-{end:012d}.chunk"
        expected = end - start + 1
        if target.exists() and target.stat().st_size == expected:
            return target
        temporary = target.with_suffix(".chunk.part")
        attempt = 0
        while attempt < 10:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == expected:
                temporary.replace(target)
                return target
            try:
                # Zenodo large-response streams were strongly throttled in the actual
                # download pilot. Eight-MiB requests retain exact resumability while
                # staying below the service's documented response rate limit.
                request_end = min(end, start + offset + 8 * 1024 * 1024 - 1)
                with requests.get(url, headers={"Range": f"bytes={start + offset}-{request_end}"},
                                  stream=True, timeout=(60, 180)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or content_range != f"bytes {start + offset}-{request_end}/{meta['size']}":
                        raise RuntimeError(f"Incorrect range response: {response.status_code}, {content_range}")
                    with temporary.open("ab" if offset else "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                offset += len(chunk)
                                if offset > expected:
                                    raise RuntimeError("Range exceeded expected size")
                if temporary.stat().st_size != request_end - start + 1:
                    raise RuntimeError("Truncated range response")
                if temporary.stat().st_size == expected:
                    temporary.replace(target)
                    return target
                attempt = 0
            except (requests.RequestException, RuntimeError) as exc:
                print(f"{year}: range {start}-{end} attempt {attempt + 1}: {exc}", flush=True)
                if attempt == 9:
                    raise
                time.sleep(min(5 * (attempt + 1), 30))
                attempt += 1
        raise RuntimeError("Failed range")

    print(f"{year}: downloading {len(ranges)} ranges with {connections} connections; "
          f"reusing {prefix_size / 1e6:.1f} MB prefix", flush=True)
    paths = []
    completed_bytes = prefix_size
    last_report = 0
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures = {executor.submit(fetch_chunk, bounds): bounds for bounds in ranges}
        for future in as_completed(futures):
            path = future.result()
            paths.append(path)
            completed_bytes += path.stat().st_size
            now = time.monotonic()
            if now - last_report > 30:
                print(f"{year}: completed {completed_bytes / 1e9:.2f}/{meta['size'] / 1e9:.2f} GB "
                      f"({100 * completed_bytes / meta['size']:.1f}%), elapsed {now - started:.0f}s", flush=True)
                last_report = now
    # Assembly is resumable: keep the original prefix and truncate any partial assembly.
    with part.open("r+b" if part.exists() else "wb") as output:
        output.truncate(prefix_size)
        output.seek(prefix_size)
        for path in sorted(paths):
            with path.open("rb") as chunk:
                shutil.copyfileobj(chunk, output, 8 * 1024 * 1024)
    assert part.stat().st_size == meta["size"], "Truncated archive"
    print(f"{year}: checking full archive MD5", flush=True)
    actual = digest(part, "md5")
    if actual != meta["md5"]:
        raise RuntimeError(f"Archive checksum mismatch: {actual}, expected {meta['md5']}")
    part.replace(out)
    write_json(verified, {"year": year, "md5": actual, "official_size": meta["size"],
                         "verified_at": utc_now(), "source": url,
                         "mtime_ns": out.stat().st_mtime_ns,
                         "download_elapsed_seconds": time.monotonic() - started})
    # Remove only exact chunk files under the checked task-owned directory.
    chunk_root = chunk_root.resolve()
    if not chunk_root.is_relative_to(DATA.resolve()):
        raise RuntimeError("Chunk cleanup target escaped the task data directory")
    for path in paths:
        path.unlink()
    plan_path.unlink()
    chunk_root.rmdir()
    print(f"{year}: archive checksum passed", flush=True)
    return out


def extract_selected(archive, year, manifest):
    archive = Path(archive)
    receipt_path = archive.with_suffix(archive.suffix + ".verified.json")
    if not receipt_path.exists():
        raise RuntimeError("Selective extraction requires an official archive checksum receipt")
    proof = json.loads(receipt_path.read_text())
    if proof["md5"] != ARCHIVES[year]["md5"] or archive.stat().st_size != ARCHIVES[year]["size"]:
        raise RuntimeError("Archive receipt does not match official metadata")
    if proof.get("mtime_ns") != archive.stat().st_mtime_ns:
        actual = digest(archive, "md5")
        if actual != ARCHIVES[year]["md5"]:
            raise RuntimeError("Archive changed after checksum verification")
        proof.update({"mtime_ns": archive.stat().st_mtime_ns, "verified_at": utc_now()})
        write_json(receipt_path, proof)
    selected = set(manifest.loc[manifest.source_year == year, "filename"])
    extracted = set()
    root = (DATA / "netcdf").resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_report = started
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            # Archives may have one named root directory; only the exact test path is used.
            pieces = PurePosixPath(name).parts
            if "test" not in pieces:
                continue
            name = str(PurePosixPath(*pieces[pieces.index("test"):]))
            if name not in selected:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"Selected member is not a regular file: {member.name}")
            target = (root / Path(*PurePosixPath(name).parts)).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            prior_good = False
            receipt_path = target.with_suffix(".nc.sha256")
            if target.exists() and target.stat().st_size == member.size and receipt_path.exists():
                prior_good = digest(target) == receipt_path.read_text().strip()
            if not prior_good:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"Cannot read selected member: {member.name}")
                temporary = target.with_suffix(".nc.tmp")
                with temporary.open("wb") as f:
                    shutil.copyfileobj(stream, f, 1024 * 1024)
                assert temporary.stat().st_size == member.size
                temporary.replace(target)
                receipt_path.write_text(digest(target), encoding="utf-8")
            extracted.add(name)
            if len(extracted) == 1:
                print(f"{year}: first extracted sample {target}", flush=True)
                write_json(DATA / "first_sample.json", {"path": str(target), "filename": name,
                           "year": year, "created_at": utc_now()})
            now = time.monotonic()
            if now - last_report > 30:
                print(f"{year}: selected extraction {len(extracted)}/{len(selected)}, "
                      f"elapsed {now - started:.0f}s", flush=True)
                last_report = now
    missing = sorted(selected - extracted)
    if missing:
        raise RuntimeError(f"Archive missing {len(missing)} selected names: {missing[:5]}")
    print(f"{year}: extracted {len(extracted)} selected files", flush=True)
    return {"year": year, "files": len(extracted), "elapsed_seconds": time.monotonic() - started,
            "archive_md5": ARCHIVES[year]["md5"],
            "planned_manifest_sha256": digest(ROOT / "design" / "planned_manifest.csv")}


def record_extractions(records):
    """Merge independently completed years into one durable extraction ledger."""
    ledger = DATA / "extraction_log.json"
    prior = json.loads(ledger.read_text()) if ledger.exists() else []
    merged = {int(record["year"]): record for record in prior}
    merged.update({int(record["year"]): record for record in records})
    write_json(ledger, [merged[year] for year in sorted(merged)])


def audit_sample(row):
    import numpy as np
    import xarray as xr

    path = DATA / "netcdf" / Path(*PurePosixPath(row["filename"]).parts)
    out = {**row, "path": str(path.resolve()), "data_path": str(path.resolve()),
           "status": "valid", "exclusion_reason": ""}
    issues = []
    missing_fields = []
    try:
        with xr.open_dataset(path) as ds:
            times = np.asarray(ds.time.values).astype("datetime64[ns]")
            numeric = times.astype("int64")
            if len(times) < 2 or np.isnat(times).any() or (np.diff(numeric) <= 0).any():
                issues.append("timestamps_missing_or_not_strictly_ordered")
            gap = float((numeric[-1] - numeric[-2]) / 1e9) if len(times) >= 2 else -1
            if not 180 <= gap <= 420:
                issues.append("previous_to_current_gap_outside_3_to_7_minutes")
            labels = np.asarray(ds["frame_labels"].values)
            if labels.shape != (len(times),) or not np.isin(labels, [0, 1]).all():
                issues.append("frame_labels_missing_or_not_binary")
            for var in VARIABLES:
                if var not in ds or ds[var].shape != (len(times), 120, 240, 2):
                    issues.append("invalid_shape_" + var)
                    continue
                value = np.asarray(ds[var].values[-2:])
                if np.isinf(value).any():
                    issues.append("infinite_radar_values_" + var)
                # Missing values are supported by the authors' FillNaNs layer.
                # Even fully missing fields are retained, with an explicit audit.
                if not np.isfinite(value[-1]).any():
                    missing_fields.append(var + "_current")
                if not np.isfinite(value[-2]).any():
                    missing_fields.append(var + "_previous")
            mask_shape = tuple(ds["range_folded_mask"].shape)
            if mask_shape != (len(times), 120, 240, 2):
                issues.append("invalid_range_folded_mask_shape")
            for limit in ["azimuth_limits", "range_limits"]:
                a = np.asarray(ds[limit].values)
                if a.shape != (2,) or not np.isfinite(a).all():
                    issues.append("invalid_" + limit)
            category = str(ds.attrs.get("category", ""))
            if category != row["category"]:
                issues.append("netcdf_catalog_category_mismatch")
            site_match = str(ds.attrs.get("site_name", "")) == row["site_name"]
            if not site_match:
                issues.append("netcdf_catalog_site_mismatch")
            terminal_time_match = pd.Timestamp(row["end_time"]) == pd.Timestamp(times[-1])
            if not terminal_time_match:
                issues.append("netcdf_catalog_terminal_time_mismatch")
            event = int(ds.attrs.get("event_id", -999))
            episode = int(ds.attrs.get("episode_id", -999))
            event_mismatch = event != int(row["event_id"])
            episode_mismatch = episode != int(row["episode_id"])
            correction_applied = False
            # Upstream v1.1 documents corrected warning IDs. Official archives retain
            # their old attrs, while exact filenames match the corrected catalog IDs.
            # These attrs are provenance only; target labels and radar arrays stay raw.
            if category == "WRN" and (event_mismatch or episode_mismatch):
                token = re.search(r"_([0-9]+)n_", Path(row["filename"]).name)
                canonical_filename = token is not None and int(token.group(1)) == int(row["event_id"])
                if canonical_filename and category == row["category"] and site_match and terminal_time_match:
                    correction_applied = True
                else:
                    issues.append("warning_corrected_catalog_identity_not_confirmed")
            else:
                if event_mismatch:
                    issues.append("netcdf_catalog_event_mismatch")
                if episode_mismatch:
                    issues.append("netcdf_catalog_episode_mismatch")
            label = int(labels[-1])
            out.update({"label": label, "previous_label": int(labels[-2]), "gap_seconds": gap,
                        "frame_count": len(times), "current_time": str(times[-1]),
                        "previous_time": str(times[-2]), "stored_terminal_vs_ef_mismatch":
                        label != int(float(ds.attrs.get("ef_number", -1)) >= 0),
                        "input_window_start_time": str(times[0]),
                        "input_window_end_time": str(times[-1]),
                        "data_sha256": digest(path),
                        "nc_event_id": event, "nc_episode_id": episode,
                        "native_event_mismatch": event_mismatch,
                        "native_episode_mismatch": episode_mismatch,
                        "catalog_id_correction_applied": correction_applied,
                        "catalog_terminal_time_match": terminal_time_match,
                        "all_missing_variable_frames": json.dumps(missing_fields),
                        "all_missing_variable_frame_count": len(missing_fields),
                        "current_frame_index": len(times) - 1, "previous_frame_index": len(times) - 2,
                        "netcdf_bytes": path.stat().st_size})
    except Exception as exc:
        issues.append(f"read_error:{type(exc).__name__}:{exc}")
    if issues:
        out["status"] = "excluded"
        out["exclusion_reason"] = "|".join(issues)
    return out


def audit_data(manifest, workers=4):
    started = time.monotonic()
    rows = manifest.to_dict("records")
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(audit_sample, rows), 1):
            outputs.append(result)
            if i % 250 == 0:
                print(f"QA: {i}/{len(rows)} clips audited", flush=True)
    all_rows = pd.DataFrame(outputs)
    valid = all_rows[all_rows.status == "valid"].copy()
    invalid = all_rows[all_rows.status != "valid"].copy()
    for column in ["label", "previous_label", "frame_count", "current_frame_index", "previous_frame_index",
                   "all_missing_variable_frame_count"]:
        valid[column] = valid[column].astype("int64")
    expected_metadata = manifest.loc[manifest.filename.isin(valid.filename)].reset_index(drop=True)
    pd.testing.assert_frame_equal(valid[manifest.columns].reset_index(drop=True), expected_metadata,
                                  check_dtype=True, check_exact=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(DATA / "excluded_manifest.csv", index=False)
    valid.to_csv(DATA / "runtime_manifest.csv", index=False)
    census = {}
    failures = []
    if not valid.filename.is_unique:
        failures.append("Duplicate filenames in runtime manifest")
    for split in ["train", "val", "test"]:
        sub = valid[valid.study_split == split]
        positives = sub[sub.label == 1]
        negatives = sub[sub.label == 0]
        warnings = negatives[negatives.category == "WRN"]
        census[split] = {"samples": len(sub), "positive_samples": len(positives),
                        "negative_samples": len(negatives), "storm_groups": sub.storm_group.nunique(),
                        "positive_storm_groups": positives.storm_group.nunique(),
                        "warning_negative_samples": len(warnings),
                        "warning_negative_storm_groups": warnings.storm_group.nunique(),
                        "categories": sub.category.value_counts().to_dict()}
        if not len(positives) or not len(negatives):
            failures.append(split + " missing a label class")
        if split in ["val", "test"]:
            if len(positives) < 40 or positives.storm_group.nunique() < 8:
                failures.append(split + " below positive sample/group minimum")
            if warnings.storm_group.nunique() < 3:
                failures.append(split + " below warning-negative group minimum")
    # Recheck isolation using actual NetCDF times rather than catalog approximations.
    isolation = {"episode_overlap_pairs": [], "event_overlap_pairs": [], "utc_date_overlap_pairs": []}
    from itertools import combinations
    for left, right in combinations(["train", "val", "test"], 2):
        a = valid[valid.study_split == left]
        b = valid[valid.study_split == right]
        for column, key in [("episode_id", "episode_overlap_pairs"), ("event_id", "event_overlap_pairs")]:
            overlap = set(a[column]) & set(b[column])
            if overlap:
                isolation[key].append({"splits": [left, right], "ids": sorted(int(x) for x in overlap)})
                failures.append(f"{left}/{right} share {column}")
        def window_dates(sub):
            dates = set()
            for _, sample in sub.iterrows():
                dates.update(d.date().isoformat() for d in pd.date_range(
                    pd.Timestamp(sample.input_window_start_time).normalize(),
                    pd.Timestamp(sample.input_window_end_time).normalize(), freq="D"))
            return dates
        overlap = window_dates(a) & window_dates(b)
        if overlap:
            isolation["utc_date_overlap_pairs"].append({"splits": [left, right], "dates": sorted(overlap)})
            failures.append(f"{left}/{right} share actual input-window UTC dates")
    source_catalog = pd.read_csv(ROOT / "design" / "catalog.csv", parse_dates=["end_time"])
    source_train_dates = set(source_catalog.loc[source_catalog.type == "train", "end_time"].dt.date)
    selected_terminal_dates = pd.to_datetime(manifest.end_time).dt.date
    pretraining_weather_overlap = {
        "basis": "Catalog terminal UTC dates only; calendar-date overlap is not sample or necessarily weather-system overlap",
        "selected_clips_on_source_train_dates": int(selected_terminal_dates.isin(source_train_dates).sum()),
        "selected_clips": len(manifest), "shared_terminal_dates": len(set(selected_terminal_dates) & source_train_dates),
        "selected_terminal_dates": selected_terminal_dates.nunique(),
        "implication": "Do not claim all frozen-backbone weather environments are unseen"}
    from collections import Counter
    missingness_by_category = {}
    for category in ["NUL", "TOR", "WRN"]:
        sub = valid[valid.category == category]
        missingness_by_category[category] = {
            "samples": len(sub), "clips_with_any_all_missing_variable_frame":
                int((sub.all_missing_variable_frame_count > 0).sum()),
            "all_missing_variable_frame_counts": dict(Counter(
                flag for flags in sub.all_missing_variable_frames for flag in json.loads(flags)))}
    corrected = valid[valid.catalog_id_correction_applied]
    mapping_columns = ["filename", "category", "study_split", "storm_group", "episode_id", "event_id",
                       "nc_episode_id", "nc_event_id", "label", "previous_label", "current_time",
                       "previous_time", "catalog_terminal_time_match", "native_event_mismatch",
                       "native_episode_mismatch"]
    mapping = corrected[mapping_columns].rename(columns={"episode_id": "catalog_episode_id",
                                                         "event_id": "catalog_event_id"})
    mapping_path = DATA / "warning_metadata_remap.csv"
    mapping_tmp = mapping_path.with_suffix(".csv.tmp")
    mapping.to_csv(mapping_tmp, index=False)
    mapping_tmp.replace(mapping_path)
    legacy_cross_split = []
    for left, right in combinations(["train", "val", "test"], 2):
        a = valid[valid.study_split == left]
        b = valid[valid.study_split == right]
        legacy_cross_split.append({"splits": [left, right],
                                  "nc_event_id_overlap_count": len(set(a.nc_event_id) & set(b.nc_event_id)),
                                  "nc_episode_id_overlap_count": len(set(a.nc_episode_id) & set(b.nc_episode_id))})
    id_corrections = {
        "count": len(corrected), "category": "WRN",
        "authority": "Corrected v1.1 public catalog and exact planned filenames; original NC attrs retained as diagnostics",
        "evidence": {"source_url": "https://github.com/mit-ll/tornet",
                     "pinned_source_url": "https://github.com/mit-ll/tornet/blob/92d9d97bbd174be2e5cea4896c62208c5be7900d/README.md#L7",
                     "instruction": "Version 1.1 fixes event and episode IDs of the warning categories",
                     "filename_catalog_event_matches": len(corrected),
                     "all_category_site_terminal_time_matches": True},
        "no_label_changes": True, "no_radar_changes": True, "no_split_changes": True,
        "native_event_mismatch_count": int(valid.native_event_mismatch.sum()),
        "native_episode_mismatch_count": int(valid.native_episode_mismatch.sum()),
        "observed_mapping_summary": {
            "catalog_warning_event_ids": corrected.event_id.nunique(),
            "embedded_warning_event_ids": corrected.nc_event_id.nunique(),
            "catalog_warning_episode_ids": corrected.episode_id.nunique(),
            "embedded_warning_episode_ids": corrected.nc_episode_id.nunique(),
            "catalog_warning_events_with_multiple_embedded_ids":
                int((corrected.groupby("event_id").nc_event_id.nunique() > 1).sum()),
            "embedded_warning_events_with_multiple_catalog_ids":
                int((corrected.groupby("nc_event_id").event_id.nunique() > 1).sum())},
        "legacy_cross_split_diagnostics": legacy_cross_split,
        "legacy_overlap_interpretation": "Known erroneous IDs are diagnostics, not proof of shared samples or weather systems",
        "mapping_file": "data/warning_metadata_remap.csv",
        "mapping_sha256": digest(mapping_path),
        "initial_failed_audit": "results/development_audits/initial_warning_id_gate/data_audit.json"}
    audit = {"created_at": utc_now(), "status": "passed" if not failures else "failed",
             "failures": failures, "planned_samples": len(manifest), "valid_samples": len(valid),
             "excluded_samples": len(invalid), "exclusion_counts": invalid.exclusion_reason.value_counts().to_dict(),
             "split_census": census, "label_policy": "Use stored v1.1 terminal frame_labels unchanged",
             "split_isolation": isolation,
             "source_training_weather_date_overlap": pretraining_weather_overlap,
             "catalog_id_corrections": id_corrections,
             "radar_missingness": {
                 "policy": "Retain NaNs and all-missing variable frames; authors FillNaNs handles these values",
                 "by_category": missingness_by_category},
             "terminal_ef_mismatches": int(valid.stored_terminal_vs_ef_mismatch.sum()),
             "gap_seconds_min": float(valid.gap_seconds.min()),
             "gap_seconds_max": float(valid.gap_seconds.max()),
             "planned_manifest_sha256": digest(ROOT / "design" / "planned_manifest.csv"),
             "planned_config_sha256": digest(ROOT / "planned_config.json"),
             "data_pipeline_sha256": digest(Path(__file__)),
             "runtime_manifest_sha256": digest(DATA / "runtime_manifest.csv"),
             "archives": {str(y): {**ARCHIVES[y], "verified": json.loads((DATA / "archives" /
                        (ARCHIVES[y]["filename"] + ".verified.json")).read_text())} for y in ARCHIVES},
             "data_licenses": {str(y): json.loads((DATA / "archives" / f"{y}_zenodo_metadata.json").read_text())
                               .get("metadata", {}).get("license") for y in ARCHIVES},
             "elapsed_seconds": time.monotonic() - started}
    write_json(RESULTS / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    if failures:
        raise RuntimeError("Pre-training QA failed: " + "; ".join(failures))
    return audit


def prepare_data(workers=2):
    """Complete acquisition and QA, preserving the fixed planned split."""
    manifest = pd.read_csv(ROOT / "design" / "planned_manifest.csv")
    assert set(manifest.type) == {"test"}, "Only predeclared source-test examples allowed"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(download_archive, sorted(ARCHIVES)))
    records = [extract_selected(DATA / "archives" / ARCHIVES[year]["filename"], year, manifest)
               for year in sorted(ARCHIVES)]
    record_extractions(records)
    return audit_data(manifest, workers=1)


def download_status():
    """Read byte progress without touching downloads or accessing predictions."""
    progress = {}
    for year, meta in ARCHIVES.items():
        archive = DATA / "archives" / meta["filename"]
        receipt = archive.with_suffix(archive.suffix + ".verified.json")
        part = archive.with_suffix(archive.suffix + ".part")
        chunk_root = archive.parent / (meta["filename"] + ".chunks")
        plan_path = chunk_root / "plan.json"
        amount = part.stat().st_size if part.exists() else 0
        if plan_path.exists():
            amount = json.loads(plan_path.read_text())["prefix_size"]
            for path in chunk_root.glob("*chunk*"):
                try:
                    amount += path.stat().st_size
                except FileNotFoundError:
                    pass  # A finished chunk was atomically renamed while reading status.
        if archive.exists():
            amount = archive.stat().st_size
        amount = min(amount, meta["size"])
        progress[str(year)] = {"downloaded_bytes": amount, "official_bytes": meta["size"],
                               "percent": round(100 * amount / meta["size"], 2),
                               "checksum_verified": receipt.exists() and archive.exists()}
    out = {"updated_at": utc_now(), "progress": progress}
    write_json(DATA / "download_progress.json", out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["download", "extract", "audit", "all", "status"], default="all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--year", type=int, choices=[2014, 2015],
                   help="Limit download/extraction only; complete-study QA always uses both years")
    args = p.parse_args()
    if args.stage == "status":
        print(json.dumps(download_status(), indent=2), flush=True)
        return
    manifest = pd.read_csv(ROOT / "design" / "planned_manifest.csv")
    assert set(manifest.type) == {"test"}, "Only predeclared source-test examples allowed"
    if args.year is not None and args.stage in ["audit", "all"]:
        p.error("--year is allowed only with --stage download or --stage extract")
    years = [args.year] if args.year is not None else sorted(ARCHIVES)
    if args.stage in ["download", "all"]:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(download_archive, years))
    if args.stage in ["extract", "all"]:
        records = []
        for year in years:
            archive = DATA / "archives" / ARCHIVES[year]["filename"]
            records.append(extract_selected(archive, year, manifest))
        record_extractions(records)
    if args.stage in ["audit", "all"]:
        # netCDF C library thread safety varies by wheel: use one reader process.
        audit_data(manifest, workers=1)


if __name__ == "__main__":
    main()
