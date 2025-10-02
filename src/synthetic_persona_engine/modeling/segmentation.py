"""Persona segmentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class SegmentationResult:
    labels: np.ndarray
    centroids: Optional[np.ndarray]
    cluster_sizes: List[int]


def encode_event_sequences(
    events: pd.DataFrame,
    *,
    session_delimiter: str = " ",
    event_column: str = "event_type",
) -> tuple[np.ndarray, List[str], CountVectorizer]:
    """Turn per-user event sequences into bag-of-events embeddings."""

    sequences = (
        events.sort_values("timestamp")
        .groupby("user_id")[event_column]
        .apply(lambda series: session_delimiter.join(series.astype(str)))
    )

    vectorizer = CountVectorizer()
    embeddings = vectorizer.fit_transform(sequences)
    return embeddings, list(sequences.index), vectorizer


def cluster_event_sequences(
    events: pd.DataFrame,
    *,
    n_clusters: int,
    clusterer: Optional[ClusterMixin] = None,
) -> SegmentationResult:
    """Cluster users by their event sequences using the provided clusterer."""

    embeddings, user_ids, _ = encode_event_sequences(events)
    clusterer = clusterer or KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = clusterer.fit_predict(embeddings)
    centroids = getattr(clusterer, "cluster_centers_", None)

    cluster_sizes = []
    for cluster_id in range(n_clusters):
        cluster_sizes.append(int(np.sum(labels == cluster_id)))

    return SegmentationResult(labels=labels, centroids=centroids, cluster_sizes=cluster_sizes)


def attach_clusters(events: pd.DataFrame, segmentation: SegmentationResult, user_ids: List[str]) -> pd.DataFrame:
    """Attach cluster labels to user-level events."""

    user_cluster_df = pd.DataFrame({"user_id": user_ids, "cluster": segmentation.labels})
    return events.merge(user_cluster_df, on="user_id", how="left")


def summarise_clusters(
    events: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
    top_n_events: int = 5,
) -> pd.DataFrame:
    """Produce summary statistics describing each cluster."""

    summaries = []
    for cluster_id, group in events.groupby(cluster_column):
        event_counts = group["event_type"].value_counts().head(top_n_events).to_dict()
        summaries.append(
            {
                "cluster": cluster_id,
                "user_count": group["user_id"].nunique(),
                "avg_session_duration": group.get("session_duration_seconds", pd.Series()).mean(),
                "top_events": event_counts,
            }
        )
    return pd.DataFrame(summaries)
