"""Visualization helpers using Plotly."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def sankey_from_sequences(sequences: Iterable[Iterable[str]]) -> go.Figure:
    """Create a Sankey diagram describing transitions between events."""

    transitions = Counter()
    for sequence in sequences:
        for a, b in zip(sequence, sequence[1:]):
            transitions[(a, b)] += 1

    labels = sorted({node for edge in transitions for node in edge})
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    sources = [label_to_index[a] for a, _ in transitions]
    targets = [label_to_index[b] for _, b in transitions]
    values = list(transitions.values())

    link = dict(source=sources, target=targets, value=values)
    node = dict(label=labels, pad=15, thickness=20)

    fig = go.Figure(data=[go.Sankey(link=link, node=node)])
    fig.update_layout(title="Persona Navigation Paths")
    return fig


def funnel_chart(metrics: pd.DataFrame, *, metric_column: str = "value") -> go.Figure:
    """Visualise funnel drop-off metrics as a bar chart."""

    fig = go.Figure(
        data=[
            go.Funnel(
                y=metrics["metric"],
                x=metrics[metric_column],
                text=metrics["variant"],
            )
        ]
    )
    fig.update_layout(title="Simulated Funnel Performance")
    return fig


def retention_curve(retention: pd.DataFrame) -> go.Figure:
    """Plot retention over time for each persona cluster."""

    fig = go.Figure()
    for cluster, group in retention.groupby("cluster"):
        fig.add_trace(
            go.Scatter(
                x=group["period"],
                y=group["retention_rate"],
                mode="lines+markers",
                name=f"Cluster {cluster}",
            )
        )
    fig.update_layout(
        title="Retention Curves by Persona",
        xaxis_title="Period",
        yaxis_title="Retention Rate",
        yaxis=dict(range=[0, 1]),
    )
    return fig
