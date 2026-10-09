"""Audit TorNet sample proximity to the pretrained model's source training data.

The TorNet paper, Section 3.1, reports removing training samples within
30 minutes and 0.25 degrees of latitude/longitude of any test sample:
https://arxiv.org/html/2401.16437v1#S3.SS1

This audit reapplies that documented criterion to the downloaded catalog,
using nominal terminal ``end_time`` timestamps and catalog chip locations.
It returns evidence so callers can exclude evaluation candidates rather than
changing the pretrained model's training data. It does not prove independence
of entire weather systems or establish which files the checkpoint saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED = ("end_time", "lat", "lon", "filename", "episode_id", "event_id")
_EVIDENCE_COLUMNS = (
    "selected_position", "selected_index", "selected_filename",
    "selected_episode_id", "selected_event_id", "selected_end_time",
    "selected_lat", "selected_lon", "source_train_position",
    "source_train_index", "source_train_filename", "source_train_episode_id",
    "source_train_event_id", "source_train_end_time", "source_train_lat",
    "source_train_lon", "time_difference_seconds", "latitude_difference_degrees",
    "longitude_difference_degrees",
)


def _coordinates_and_times(rows: pd.DataFrame, name: str):
    missing = set(_REQUIRED) - set(rows.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    times = pd.to_datetime(rows["end_time"], utc=True, errors="raise", format="mixed")
    if times.isna().any():
        raise ValueError(f"{name} contains missing end_time values")
    # Explicit nanosecond conversion also supports pandas datetime units other
    # than ns, while UTC conversion makes aware and naive catalog times agree.
    nanoseconds = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    latitudes = pd.to_numeric(rows["lat"], errors="raise").to_numpy(dtype=float)
    longitudes = pd.to_numeric(rows["lon"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError(f"{name} contains missing or non-finite coordinates")
    return times, nanoseconds, latitudes, longitudes


def proximity_exclusions(
    selected: pd.DataFrame,
    source_train: pd.DataFrame,
    *,
    minutes: float = 30.0,
    coordinate_tolerance_degrees: float = 0.25,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return a positional exclusion mask and every matching training pair.

    A pair matches only when all three inclusive conditions hold:
    ``abs(end_time difference) <= minutes`` and each absolute latitude and
    longitude difference is at most ``coordinate_tolerance_degrees``.

    ``mask[i]`` applies to ``selected.iloc[i]``; input index labels need not be
    consecutive or unique. Evidence includes both positions and original index
    labels. Source training rows are searched by sorted time, so comparisons
    are limited to the temporal neighborhood rather than the whole catalog.
    Neither input is mutated, and no files or network resources are accessed.
    Missing times or coordinates raise an error rather than silently treating
    an unauditable row as independent.
    """
    if not np.isfinite(minutes) or minutes < 0:
        raise ValueError("minutes must be finite and non-negative")
    if (not np.isfinite(coordinate_tolerance_degrees)
            or coordinate_tolerance_degrees < 0):
        raise ValueError("coordinate_tolerance_degrees must be finite and non-negative")
    window_ns = int(pd.Timedelta(minutes=minutes).value)
    selected_times, selected_ns, selected_lat, selected_lon = _coordinates_and_times(
        selected, "selected"
    )
    train_times, train_ns, train_lat, train_lon = _coordinates_and_times(
        source_train, "source_train"
    )
    mask = np.zeros(len(selected), dtype=bool)
    evidence = []
    order = np.argsort(train_ns, kind="stable")
    ordered_times = train_ns[order]
    for position, timestamp in enumerate(selected_ns):
        left = np.searchsorted(ordered_times, int(timestamp) - window_ns, side="left")
        right = np.searchsorted(ordered_times, int(timestamp) + window_ns, side="right")
        candidates = order[left:right]
        latitude_differences = np.abs(train_lat[candidates] - selected_lat[position])
        longitude_differences = np.abs(train_lon[candidates] - selected_lon[position])
        nearby = candidates[
            (latitude_differences <= coordinate_tolerance_degrees)
            & (longitude_differences <= coordinate_tolerance_degrees)
        ]
        if not len(nearby):
            continue
        mask[position] = True
        selected_row = selected.iloc[position]
        for train_position in nearby:
            train_row = source_train.iloc[train_position]
            evidence.append({
                "selected_position": position,
                "selected_index": selected.index[position],
                "selected_filename": selected_row["filename"],
                "selected_episode_id": selected_row["episode_id"],
                "selected_event_id": selected_row["event_id"],
                "selected_end_time": selected_times.iloc[position].isoformat(),
                "selected_lat": selected_lat[position],
                "selected_lon": selected_lon[position],
                "source_train_position": int(train_position),
                "source_train_index": source_train.index[train_position],
                "source_train_filename": train_row["filename"],
                "source_train_episode_id": train_row["episode_id"],
                "source_train_event_id": train_row["event_id"],
                "source_train_end_time": train_times.iloc[train_position].isoformat(),
                "source_train_lat": train_lat[train_position],
                "source_train_lon": train_lon[train_position],
                "time_difference_seconds": abs(int(train_ns[train_position]) - int(timestamp)) / 1e9,
                "latitude_difference_degrees": abs(train_lat[train_position] - selected_lat[position]),
                "longitude_difference_degrees": abs(train_lon[train_position] - selected_lon[position]),
            })
    return mask, pd.DataFrame(evidence, columns=_EVIDENCE_COLUMNS)
