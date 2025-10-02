"""Experiment runner for persona simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .environment import FlowModification, ProductFlowSimulator


@dataclass
class ExperimentResult:
    control: Dict[str, float]
    treatment: Dict[str, float]
    uplift: Dict[str, float]
    control_sequences: list[list[str]]
    treatment_sequences: list[list[str]]


def run_ab_experiment(
    simulator: ProductFlowSimulator,
    *,
    n_personas: int,
    modification: FlowModification,
    max_length: int = 20,
    random_state: Optional[int] = None,
) -> ExperimentResult:
    """Simulate a control vs treatment product flow."""

    control_outcome = simulator.run(
        n_personas=n_personas,
        max_length=max_length,
        random_state=random_state,
    )
    treatment_outcome = simulator.run(
        n_personas=n_personas,
        max_length=max_length,
        random_state=None if random_state is None else random_state + 1,
        flow_modification=modification,
    )

    control_metrics = {
        "drop_off_rate": control_outcome.drop_off_rate,
        "completion_rate": control_outcome.completion_rate,
        "avg_length": control_outcome.avg_length,
    }
    treatment_metrics = {
        "drop_off_rate": treatment_outcome.drop_off_rate,
        "completion_rate": treatment_outcome.completion_rate,
        "avg_length": treatment_outcome.avg_length,
    }
    uplift = {
        key: treatment_metrics[key] - control_metrics[key]
        for key in control_metrics
    }
    return ExperimentResult(
        control=control_metrics,
        treatment=treatment_metrics,
        uplift=uplift,
        control_sequences=control_outcome.sequences,
        treatment_sequences=treatment_outcome.sequences,
    )


def results_to_dataframe(result: ExperimentResult) -> pd.DataFrame:
    """Convert an experiment result into a tidy dataframe for reporting."""

    rows = []
    for variant, metrics in [("control", result.control), ("treatment", result.treatment)]:
        for metric, value in metrics.items():
            rows.append({"variant": variant, "metric": metric, "value": value})
    df = pd.DataFrame(rows)
    uplift_rows = [{"variant": "uplift", "metric": k, "value": v} for k, v in result.uplift.items()]
    return pd.concat([df, pd.DataFrame(uplift_rows)], ignore_index=True)
