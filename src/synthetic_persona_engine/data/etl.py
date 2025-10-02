"""ETL utilities for the Synthetic Persona Simulation Engine.

This module exposes composable helpers for loading raw clickstream events
from different storage backends and normalizing them into a canonical schema.
The helpers are intentionally lightweight so they can be reused inside
notebooks or batch pipelines.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)


CANONICAL_COLUMNS = [
    "user_id",
    "session_id",
    "timestamp",
    "event_type",
    "metadata",
]


@dataclass
class EventSchema:
    """Metadata describing the expected column names of an event dataset."""

    user_id: str = "user_id"
    session_id: str = "session_id"
    timestamp: str = "timestamp"
    event_type: str = "event_type"
    metadata: str = "metadata"

    def to_list(self) -> List[str]:
        return [
            self.user_id,
            self.session_id,
            self.timestamp,
            self.event_type,
            self.metadata,
        ]


def _ensure_metadata_serialised(df: pd.DataFrame, metadata_column: str) -> pd.DataFrame:
    """Normalise metadata columns to JSON strings."""

    if metadata_column not in df.columns:
        df[metadata_column] = "{}"
        return df

    def _convert(value: object) -> str:
        if isinstance(value, str):
            return value
        if pd.isna(value):
            return "{}"
        try:
            return json.dumps(value)
        except TypeError:
            return json.dumps(str(value))

    df[metadata_column] = df[metadata_column].apply(_convert)
    return df


def load_event_log_csv(
    path: Path | str,
    *,
    schema: Optional[EventSchema] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load a clickstream log from a CSV file.

    Args:
        path: Path to the CSV file.
        schema: Optional override for the expected column names. The file is
            required to contain the columns referenced in the schema.
        parse_dates: Whether to convert the timestamp column into ``datetime``.

    Returns:
        A pandas ``DataFrame`` following the canonical schema.
    """

    schema = schema or EventSchema()
    df = pd.read_csv(path)

    missing = set(schema.to_list()) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if parse_dates:
        df[schema.timestamp] = pd.to_datetime(df[schema.timestamp], utc=True)

    df = df.rename(
        columns={
            schema.user_id: "user_id",
            schema.session_id: "session_id",
            schema.timestamp: "timestamp",
            schema.event_type: "event_type",
            schema.metadata: "metadata",
        }
    )
    df = _ensure_metadata_serialised(df, "metadata")

    df = df[CANONICAL_COLUMNS].sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    logger.info("Loaded %s events from %s", len(df), path)
    return df


def load_event_logs(paths: Iterable[Path | str], **kwargs) -> pd.DataFrame:
    """Load and concatenate multiple CSV logs into a single dataframe."""

    frames = [load_event_log_csv(path, **kwargs) for path in paths]
    if not frames:
        raise ValueError("No event logs were provided")
    return pd.concat(frames, ignore_index=True).sort_values(["user_id", "timestamp"])


def from_dataframe(df: pd.DataFrame, schema: Optional[EventSchema] = None) -> pd.DataFrame:
    """Normalise an in-memory dataframe to the canonical schema."""

    schema = schema or EventSchema()
    missing = set(schema.to_list()) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.rename(
        columns={
            schema.user_id: "user_id",
            schema.session_id: "session_id",
            schema.timestamp: "timestamp",
            schema.event_type: "event_type",
            schema.metadata: "metadata",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = _ensure_metadata_serialised(df, "metadata")
    return df[CANONICAL_COLUMNS].sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def normalise_metadata_keys(
    df: pd.DataFrame,
    *,
    keys: Mapping[str, str],
) -> pd.DataFrame:
    """Rename keys inside the metadata JSON payload.

    Args:
        df: Canonical event log dataframe.
        keys: Mapping of existing key names to their new names.
    """

    def _rename(payload: str) -> str:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to decode metadata payload: %s", payload)
            return payload
        for old, new in keys.items():
            if old in data:
                data[new] = data.pop(old)
        return json.dumps(data)

    df = df.copy()
    df["metadata"] = df["metadata"].apply(_rename)
    return df
