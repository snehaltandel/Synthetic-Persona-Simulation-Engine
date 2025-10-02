"""Simulation environment for evaluating product flows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .persona_generator import MarkovPersonaGenerator


@dataclass
class FlowModification:
    """Describe a hypothetical product change."""

    description: str
    additional_steps: int = 0
    friction_factor: float = 0.0  # Increase in drop-off probability per step.
    reward_factor: float = 0.0  # Reduction in drop-off probability.


@dataclass
class SimulationOutcome:
    sequences: List[List[str]]
    drop_off_rate: float
    completion_rate: float
    avg_length: float


class ProductFlowSimulator:
    """Walk personas through a flow to estimate engagement."""

    def __init__(
        self,
        generator: MarkovPersonaGenerator,
        *,
        completion_events: Optional[Iterable[str]] = None,
        drop_events: Optional[Iterable[str]] = None,
    ) -> None:
        self.generator = generator
        self.completion_events = set(completion_events or [])
        self.drop_events = set(drop_events or [])

    def run(
        self,
        *,
        n_personas: int,
        flow_modification: Optional[FlowModification] = None,
        max_length: int = 20,
        random_state: Optional[int] = None,
    ) -> SimulationOutcome:
        rng = np.random.default_rng(random_state)
        sequences: List[List[str]] = []
        completions = 0
        drops = 0

        for i in range(n_personas):
            sequence = self.generator.sample_sequence(
                max_length=max_length,
                terminal_events=self.drop_events.union(self.completion_events),
                random_state=rng.integers(0, 1_000_000),
            )

            if flow_modification:
                sequence = self._apply_modification(sequence, flow_modification)

            sequences.append(sequence)
            if any(event in self.completion_events for event in sequence):
                completions += 1
            elif any(event in self.drop_events for event in sequence):
                drops += 1

        drop_off_rate = drops / n_personas
        completion_rate = completions / n_personas
        avg_length = float(np.mean([len(seq) for seq in sequences])) if sequences else 0.0
        return SimulationOutcome(
            sequences=sequences,
            drop_off_rate=drop_off_rate,
            completion_rate=completion_rate,
            avg_length=avg_length,
        )

    def _apply_modification(self, sequence: List[str], modification: FlowModification) -> List[str]:
        """Inject synthetic friction or reward events into a sequence."""

        new_sequence = sequence.copy()
        if modification.additional_steps > 0:
            new_sequence.extend(["synthetic_step"] * modification.additional_steps)

        if modification.friction_factor:
            drop_probability = min(1.0, modification.friction_factor * len(new_sequence))
            if np.random.rand() < drop_probability:
                new_sequence.append("friction_drop")

        if modification.reward_factor:
            reward_probability = min(1.0, modification.reward_factor)
            if np.random.rand() < reward_probability:
                new_sequence.append("reward_bonus")
        return new_sequence
