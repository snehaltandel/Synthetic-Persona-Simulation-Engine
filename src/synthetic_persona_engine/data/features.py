"""Feature engineering utilities for clickstream datasets."""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List, Sequence

import pandas as pd


def _load_metadata_series(metadata_series: pd.Series) -> List[dict]:
    """Decode metadata JSON strings into dictionaries."""

    decoded: List[dict] = []
    for payload in metadata_series.fillna("{}"):  # type: ignore[arg-type]
        if isinstance(payload, dict):
            decoded.append(payload)
            continue
        try:
            decoded.append(json.loads(payload))
        except json.JSONDecodeError:
            decoded.append({})
    return decoded


def engineer_session_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into session-level features."""

    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Compute session duration and event counts.
    aggregations = events.groupby(["user_id", "session_id"]).agg(
        session_start=("timestamp", "min"),
        session_end=("timestamp", "max"),
        event_count=("event_type", "count"),
    )
    aggregations["session_duration_seconds"] = (
        aggregations["session_end"] - aggregations["session_start"]
    ).dt.total_seconds()

    # Compute per-session click frequency.
    aggregations["click_frequency_per_min"] = (
        aggregations["event_count"]
        / (aggregations["session_duration_seconds"].clip(lower=1) / 60.0)
    )

    aggregations = aggregations.reset_index()
    return aggregations


def engineer_path_features(events: pd.DataFrame, funnel_events: Sequence[str]) -> pd.DataFrame:
    """Compute path length and funnel step completion per session."""

    path_features = events.groupby(["user_id", "session_id"]).agg(
        path_length=("event_type", "count"),
        unique_events=("event_type", pd.Series.nunique),
        events=("event_type", lambda x: list(x)),
    )
    path_features["funnel_completion"] = path_features["events"].apply(
        lambda evts: _funnel_completion(evts, funnel_events)
    )
    path_features = path_features.drop(columns=["events"]).reset_index()
    return path_features


def _funnel_completion(events: Iterable[str], funnel_events: Sequence[str]) -> float:
    """Return the proportion of funnel steps completed in order."""

    iterator = iter(events)
    completed = 0
    for step in funnel_events:
        for event in iterator:
            if event == step:
                completed += 1
                break
        else:
            break
    return completed / max(len(funnel_events), 1)


def engineer_user_features(
    events: pd.DataFrame,
    session_features: pd.DataFrame,
    path_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine session-level features to create user-level aggregates."""

    session_metrics = session_features.groupby("user_id").agg(
        avg_session_duration=("session_duration_seconds", "mean"),
        sessions_per_user=("session_id", pd.Series.nunique),
        total_events=("event_count", "sum"),
    )

    path_metrics = path_features.groupby("user_id").agg(
        avg_path_length=("path_length", "mean"),
        avg_funnel_completion=("funnel_completion", "mean"),
    )

    return session_metrics.join(path_metrics, how="outer").fillna(0).reset_index()


def top_events(events: pd.DataFrame, *, n: int = 10) -> List[tuple[str, int]]:
    """Return the most common event types."""

    counter = Counter(events["event_type"].tolist())
    return counter.most_common(n)


def enrich_with_metadata(events: pd.DataFrame) -> pd.DataFrame:
    """Expand the metadata column into top-level columns."""

    metadata_dicts = _load_metadata_series(events["metadata"])
    metadata_df = pd.json_normalize(metadata_dicts)
    metadata_df.columns = [f"meta_{col}" for col in metadata_df.columns]
    return pd.concat([events.reset_index(drop=True), metadata_df], axis=1)
