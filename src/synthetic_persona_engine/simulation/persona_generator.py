"""Synthetic persona sequence generators."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MarkovPersonaGenerator:
    """A simple first-order Markov chain generator for event sequences."""

    transition_matrix: Dict[str, Dict[str, float]]
    start_probabilities: Dict[str, float]

    @classmethod
    def from_events(cls, events: pd.DataFrame, *, event_column: str = "event_type") -> "MarkovPersonaGenerator":
        transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        start_counts: Dict[str, int] = defaultdict(int)

        for _, session in events.sort_values("timestamp").groupby(["user_id", "session_id"]):
            sequence = session[event_column].tolist()
            if not sequence:
                continue
            start_counts[sequence[0]] += 1
            for current_event, next_event in zip(sequence, sequence[1:]):
                transitions[current_event][next_event] += 1

        transition_matrix = {
            event: _normalise(counter)
            for event, counter in transitions.items()
        }
        start_probabilities = _normalise(start_counts)
        return cls(transition_matrix=transition_matrix, start_probabilities=start_probabilities)

    def sample_sequence(
        self,
        *,
        max_length: int = 20,
        terminal_events: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
    ) -> List[str]:
        rng = np.random.default_rng(random_state)
        if not self.start_probabilities:
            raise ValueError("Generator has no start probabilities")
        events, probs = zip(*self.start_probabilities.items())
        current = rng.choice(events, p=probs)
        sequence = [current]
        terminal_events = set(terminal_events or [])

        for _ in range(max_length - 1):
            if current in terminal_events:
                break
            next_probs = self.transition_matrix.get(current)
            if not next_probs:
                break
            events, probs = zip(*next_probs.items())
            current = rng.choice(events, p=probs)
            sequence.append(current)
        return sequence


def _normalise(counter: Dict[str, int]) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}
