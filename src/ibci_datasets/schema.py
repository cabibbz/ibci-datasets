"""The common in-memory form every reader produces and every writer consumes.

A reader turns one source file into one or more :class:`Recording` objects. Writers
(NWB, torch_brain) only ever see :class:`Recording`, so each dataset is parsed once.

Releases differ in bin size, threshold, normalization and layout, so a Recording
carries that provenance with it. Nothing here pools or re-normalizes data.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import numpy as np


@dataclasses.dataclass
class Channels:
    """One row per recording channel (electrode)."""

    ids: np.ndarray  # (C,) str, unique within the recording, e.g. "v6v-07"
    arrays: np.ndarray  # (C,) str, the array each channel belongs to, e.g. "v6v"
    locations: np.ndarray  # (C,) str, brain area, e.g. "ventral 6v"

    def __post_init__(self) -> None:
        n = len(self.ids)
        if len(self.arrays) != n or len(self.locations) != n:
            raise ValueError("Channels: ids, arrays and locations must have equal length")
        if len(set(self.ids.tolist())) != n:
            raise ValueError("Channels: ids must be unique")

    def __len__(self) -> int:
        return len(self.ids)


@dataclasses.dataclass
class Feature:
    """One binned neural feature over all channels, e.g. threshold crossings."""

    name: str  # "threshold_crossings" | "spike_band_power" | ...
    data: np.ndarray  # (T, C) float32, aligned with Recording.timestamps and .channels
    unit: str  # "count", "z-score", ...
    description: str
    threshold_rms: float | None = None  # e.g. -4.5 for -4.5 x RMS crossings
    normalization: str = "none"  # how the release normalized it, in words


@dataclasses.dataclass
class Signal:
    """A non-neural signal sampled on the recording's bins, e.g. cursor position."""

    name: str  # "cursor_position", "assist_amount", ...
    data: np.ndarray  # (T,) or (T, D), aligned with Recording.timestamps
    unit: str
    description: str
    components: tuple[str, ...] = ()  # names of the D columns, e.g. ("x", "y")


@dataclasses.dataclass
class Trials:
    """Trial table. ``start`` and ``end`` are seconds on the recording's timeline."""

    start: np.ndarray  # (N,) float64
    end: np.ndarray  # (N,) float64
    columns: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    descriptions: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.start)
        if len(self.end) != n:
            raise ValueError("Trials: start and end must have equal length")
        if np.any(self.end <= self.start):
            raise ValueError("Trials: every trial must end after it starts")
        for k, v in self.columns.items():
            if len(v) != n:
                raise ValueError(f"Trials: column {k!r} has {len(v)} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.start)


@dataclasses.dataclass
class Recording:
    """One session (or one session split) from one dataset."""

    dataset_id: str  # catalog id, e.g. "card2024_speech"
    session_id: str  # unique within the dataset, e.g. "t15.2023.08.11_train"
    participant_id: str  # consortium code, e.g. "T15"
    split: str | None  # release split ("train" / "val" / "test") or None
    bin_size_s: float
    timestamps: np.ndarray  # (T,) float64, bin centres in seconds
    channels: Channels
    features: list[Feature]
    trials: Trials | None = None
    session_date: dt.date | None = None
    post_implant_day: int | None = None
    task: str = ""
    timeline: str = "continuous"  # "continuous" or "concatenated-trials"
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
    behavior: list[Signal] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        t = len(self.timestamps)
        if t and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Recording: timestamps must be strictly increasing")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("Recording: feature names must be unique")
        for f in self.features:
            if f.data.shape != (t, len(self.channels)):
                raise ValueError(
                    f"Recording: feature {f.name!r} has shape {f.data.shape}, "
                    f"expected {(t, len(self.channels))}"
                )
        names = [s.name for s in self.behavior]
        if len(set(names)) != len(names):
            raise ValueError("Recording: behavior signal names must be unique")
        for s in self.behavior:
            if s.data.ndim not in (1, 2) or s.data.shape[0] != t:
                raise ValueError(
                    f"Recording: behavior signal {s.name!r} has shape {s.data.shape}, expected ({t}, ...)"
                )
            if s.components and (s.data.ndim != 2 or len(s.components) != s.data.shape[1]):
                raise ValueError(f"Recording: behavior signal {s.name!r} components do not match its columns")

    def feature(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)
