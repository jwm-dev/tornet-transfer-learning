"""Build a TorNet study manifest using catalog metadata, before any training.

Source categories are sampling strata, not per-frame target labels. Radar-label
and timestamp checks must still pass before the experiment can be executed.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from source_independence import proximity_exclusions

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CATEGORIES = ("NUL", "TOR", "WRN")


def window_dates(start, end):
    return pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize(), freq="D").strftime("%Y-%m-%d")


def group_storm_days(catalog: pd.DataFrame) -> np.ndarray:
    """Keep entire episodes and every shared input-window UTC date together."""
    episodes = sorted(catalog.episode_id.astype(str).unique())
    parent = {episode: episode for episode in episodes}

    def find(episode):
        while parent[episode] != episode:
            parent[episode] = parent[parent[episode]]
            episode = parent[episode]
        return episode

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    date_owner = {}
    for row in catalog.itertuples():
        episode = str(row.episode_id)
        for day in window_dates(row.start_time, row.end_time):
            if day in date_owner:
                union(episode, date_owner[day])
            else:
                date_owner[day] = episode
    return np.asarray([find(str(episode)) for episode in catalog.episode_id])


def stratify_groups(selected: pd.DataFrame, config: dict):
    """Choose the lowest declared category-count imbalance among fixed draws.

    Objective: equally weighted mean squared relative category-count and
    year-count deviation from targets. No radar arrays, target labels or model
    scores are read. Ties retain the first draw. This is dataset preparation.
    """
    cfg = config["data"]
    table = pd.crosstab(selected.storm_group, selected.category).reindex(
        columns=CATEGORIES, fill_value=0)
    counts = table.to_numpy()
    year_counts = pd.crosstab(selected.storm_group, selected.source_year).reindex(
        index=table.index, fill_value=0).to_numpy()
    sizes = cfg["split_group_counts"]
    if sum(sizes) != len(table):
        raise ValueError("Group count changed; revise the protocol before training.")
    target = np.asarray(cfg["split_target_fractions"])[:, None] * counts.sum(axis=0)
    year_target = np.asarray(cfg["split_target_fractions"])[:, None] * year_counts.sum(axis=0)
    rng = np.random.default_rng(config["split_seed"])
    best_score, best_assignment, best_trial = float("inf"), None, None
    valid = 0
    for trial in range(cfg["split_search_candidates"]):
        order = rng.permutation(len(table))
        assignment = np.empty(len(table), dtype=np.int64)
        cursor = 0
        for split_index, size in enumerate(sizes):
            assignment[order[cursor:cursor + size]] = split_index
            cursor += size
        split_counts = np.asarray([counts[assignment == i].sum(axis=0) for i in range(3)])
        tor_groups = [np.count_nonzero(counts[assignment == i, 1]) for i in range(3)]
        wrn_groups = [np.count_nonzero(counts[assignment == i, 2]) for i in (1, 2)]
        if min(tor_groups) < cfg["split_min_tornado_category_groups"]:
            continue
        if min(wrn_groups) < cfg["split_min_warning_category_eval_groups"]:
            continue
        if split_counts[1:, 1].min() < 40 or (split_counts == 0).any():
            continue
        valid += 1
        split_years = np.asarray([year_counts[assignment == i].sum(axis=0) for i in range(3)])
        score = float(0.5 * np.square((split_counts - target) / target).mean()
                      + 0.5 * np.square((split_years - year_target) / year_target).mean())
        if score < best_score:
            best_score, best_assignment, best_trial = score, assignment.copy(), trial
    if best_assignment is None:
        raise ValueError("No admissible metadata split. Revise before training.")
    mapping = dict(zip(table.index, (SPLITS[i] for i in best_assignment)))
    details = {"candidate_draws": cfg["split_search_candidates"],
               "admissible_draws": valid, "chosen_zero_based_draw": best_trial,
               "stratification_objective": best_score,
               "objective": "0.5*category mean squared relative deviation +0.5*year mean squared relative deviation from 60/20/20 targets"}
    return selected.storm_group.map(mapping), details


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan():
    config_path = ROOT / "planned_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    catalog_path = ROOT / "design/catalog.csv"
    data = pd.read_csv(catalog_path)
    years = pd.to_datetime(data.end_time).dt.year
    selected = data[(data["type"] == config["data"]["source_partition"])
                    & years.isin(config["data"]["years"])].copy()
    source_train = data[data["type"] == "train"]
    overlap = (selected.episode_id.isin(source_train.episode_id)
               | selected.event_id.isin(source_train.event_id))
    excluded = selected[overlap].copy()
    source_pool_size = len(selected)
    selected = selected[~overlap].copy()
    proximity_mask, proximity_evidence = proximity_exclusions(
        selected, source_train,
        minutes=config["data"]["source_train_proximity_minutes"],
        coordinate_tolerance_degrees=config["data"]["source_train_proximity_coordinate_degrees"])
    proximity_excluded = selected[proximity_mask].copy()
    selected = selected[~proximity_mask].copy()
    excluded.to_csv(ROOT / "design/source_episode_overlap.csv", index=False, lineterminator="\n")
    proximity_evidence.to_csv(ROOT / "design/source_proximity_overlap.csv", index=False, lineterminator="\n")
    assert not set(selected.episode_id) & set(source_train.episode_id)
    assert not set(selected.event_id) & set(source_train.event_id)
    selected = selected.sort_values("filename").reset_index(drop=True)
    selected["source_year"] = pd.to_datetime(selected.end_time).dt.year
    selected["storm_group"] = group_storm_days(selected)
    selected["study_split"], split_details = stratify_groups(selected, config)
    assert selected.filename.is_unique
    assert selected.study_split.notna().all()

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        l = selected[selected.study_split == left]
        r = selected[selected.study_split == right]
        assert not set(l.episode_id) & set(r.episode_id)
        assert not set(l.storm_group) & set(r.storm_group)
        l_dates = {day for row in l.itertuples() for day in window_dates(row.start_time, row.end_time)}
        r_dates = {day for row in r.itertuples() for day in window_dates(row.start_time, row.end_time)}
        assert not l_dates & r_dates

    output = ROOT / "design/planned_manifest.csv"
    selected.to_csv(output, index=False, lineterminator="\n")
    summary = []
    for split in SPLITS:
        section = selected[selected.study_split == split]
        item = {"split": split, "chips": len(section),
                "episodes": int(section.episode_id.nunique()),
                "storm_day_groups": int(section.storm_group.nunique()),
                "largest_group_chips": int(section.groupby("storm_group").size().max()),
                "years": {str(year): len(part) for year, part in section.groupby("source_year")},
                "categories": {}}
        for category, part in section.groupby("category"):
            item["categories"][category] = {"chips": len(part),
                "episodes": int(part.episode_id.nunique()),
                "storm_day_groups": int(part.storm_group.nunique())}
        summary.append(item)

    weights = ROOT / "probe/tornado_detector_baseline.keras"
    census = {"dataset_version": "1.1", "catalog_rows": len(data),
              "selected_years": config["data"]["years"],
              "source_partition": "authors official test", "samples": len(selected),
              "source_pool_before_overlap_filter": source_pool_size,
              "excluded_source_train_overlap_clips": len(excluded),
              "excluded_episode_ids": sorted(excluded.episode_id.unique().tolist()),
              "excluded_source_train_proximity_clips": len(proximity_excluded),
              "proximity_evidence_file": "source_proximity_overlap.csv",
              "episodes": int(selected.episode_id.nunique()),
              "storm_day_groups": int(selected.storm_group.nunique()),
              "splits": summary, "split_search": split_details,
              "catalog_sha256": sha256(catalog_path),
              "manifest_sha256": sha256(output), "config_sha256": sha256(config_path),
              "design_script_sha256": sha256(Path(__file__).resolve()),
              "proximity_script_sha256": sha256(ROOT / "source_independence.py"),
              "checkpoint_sha256": sha256(weights) if weights.exists() else None,
              "design_environment": {"python": platform.python_version(),
                  "numpy": np.__version__, "pandas": pd.__version__},
              "label_note": "TOR is a category; actual terminal-frame labels and positive group counts still require NetCDF audit.",
              "partition_basis": "Authors documented holdout protocol plus episode/event/proximity exclusion; checkpoint has no embedded training provenance. Other episodes on same weather days may have been seen during pretraining.",
              "grouping": "Complete episodes joined when they share any UTC date from start_time through end_time.",
              "status": "Catalog and split verified. No new training or final test inference performed."}
    (ROOT / "design/dataset_census.json").write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(census, indent=2))
    return output


if __name__ == "__main__":
    build_plan()
