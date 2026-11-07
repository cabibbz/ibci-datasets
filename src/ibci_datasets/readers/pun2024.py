"""Reader for Pun et al. 2024 (Commun Biol), recording instability (MINDFUL).

Source: https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn5s (CC0-1.0).
Companion code (MATLAB): https://github.com/ewinapun/MINDFUL

One zip, ``MINDFUL_Data/<participant>/day_<n>/Block_<nn>/{data,task,info}.mat`` (MATLAB v5),
plus ``T11(additional)/{personal_use,random_targets}/day_<n>/Block_<nn>/``. Participants
T5 (6 sessions, post-implant days 2121-2149, 21 blocks, Fitts random-target task, no
click) and T11 (15 sessions, days 658-800, 29 blocks, center-out-and-back with click),
each with two 96-channel arrays in the hand knob area of left precentral gyrus. Every
block is continuous, in 20 ms bins, decoded online by a fixed decoder.

Findings from the data, beyond the README:

* ``nctx`` is a rate in Hz, not a count: in T5 every value is a multiple of 50 (count per
  20 ms bin x 50); in T11, 86-98% are, and the rest are fractions of the form 50 k / n,
  which is what linear interpolation over noise-outlier bins (the lab's
  ``RemoveOutliers.m``) produces. ``info.prctOutliers`` marks those bins.
* ``startStops`` are 1-based inclusive MATLAB indices (the companion code uses them as
  ``start:stop``); with that reading, trial length equals ``timeToTarget`` exactly in
  every block.
* ``useClick`` is one value per block, not per trial.
* Channels 0-95 and 96-191 are the two arrays (within-array correlation exceeds
  across-array, against an interleaved control). Which physical array is which is not
  stated, so they are labelled ``a1`` and ``a2``.
* Personal-use blocks have no intention labels or per-trial task fields; their
  ``startStops`` segments (1-27 s) are kept as trials, but what they delimit is not documented.
* T11 day 689 block 14 has 76 rows of ``trialSuccess``, ``pathEfficiency`` and
  ``orthChanges`` for 75 trials. In every other T11 block success equals "acquired before
  the 10 s timeout"; dropping the last row restores that here (dropping the first does
  not), so the surplus trailing row is dropped and recorded in the provenance.

Not converted: per-channel ``prctOutliers`` (sparse) and per-block summary metrics other
than ``percentCorrect`` (kept as a task parameter). Subject ages in the README are ages at
enrollment, years before these sessions, so they are not written as recording ages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Channels, Feature, Recording, Signal, Trials

DATASET_ID = "pun2024_instability"
DOI = "doi:10.5061/dryad.n2z34tn5s"
BIN_S = 0.020
N_PER_ARRAY = 96
LOCATION = "hand knob, precentral gyrus"

_BLOCK_RE = re.compile(
    r"(?P<pid>T5|T11)(?P<extra>\(additional\)[/\\](?P<kind>personal_use|random_targets))?"
    r"[/\\]day_(?P<day>\d+)[/\\]Block_(?P<block>\d+)$"
)

_TASKS = {
    "fitts": "2D random-target cursor task (Fitts), fixed decoder",
    "circleofcircles": "2D point-and-click center-out-and-back cursor task, fixed decoder",
    "personal use": "Personal use: self-paced web browsing with the BCI cursor, fixed decoder",
}


def _loadmat(path: Path) -> dict:
    import scipy.io

    return scipy.io.loadmat(str(path))


def find_blocks(root: str | Path) -> list[Path]:
    """Block folders (holding data/task/info.mat), sorted by participant, kind, day, block."""
    found = []
    for data in Path(root).rglob("data.mat"):
        folder = data.parent
        m = _BLOCK_RE.search(folder.as_posix())
        if m and (folder / "task.mat").exists() and (folder / "info.mat").exists():
            key = (m["pid"], m["kind"] or "", int(m["day"]), int(m["block"]))
            found.append((key, folder))
    return [f for _, f in sorted(found)]


def channels() -> Channels:
    ids, arrays = [], []
    for a in ("a1", "a2"):
        for i in range(N_PER_ARRAY):
            ids.append(f"{a}-{i:02d}")
            arrays.append(a)
    return Channels(ids=np.array(ids), arrays=np.array(arrays), locations=np.full(len(ids), LOCATION))


def _col(m: dict, key: str, n: int | None = None) -> np.ndarray | None:
    if key not in m:
        return None
    v = np.asarray(m[key]).ravel()
    if n is not None and len(v) != n:
        raise ValueError(f"{key}: {len(v)} rows, expected {n}")
    return v


def _xy(m: dict, key: str, n: int) -> np.ndarray | None:
    if key not in m:
        return None
    v = np.asarray(m[key], dtype=np.float32)
    if v.shape != (n, 2):
        raise ValueError(f"{key}: shape {v.shape}, expected {(n, 2)}")
    return v


def read_block(folder: str | Path) -> Recording:
    folder = Path(folder)
    m = _BLOCK_RE.search(folder.as_posix())
    if not m:
        raise ValueError(f"not a MINDFUL block folder: {folder}")
    pid, kind, day, block = m["pid"], m["kind"], int(m["day"]), int(m["block"])
    data = _loadmat(folder / "data.mat")
    task = _loadmat(folder / "task.mat")
    info = _loadmat(folder / "info.mat")

    nctx = np.asarray(data["nctx"], dtype=np.float32)
    n, c = nctx.shape
    ch = channels()
    if c != len(ch):
        raise ValueError(f"{folder}: nctx has {c} channels, expected {len(ch)}")
    if int(np.asarray(task["nPointsPerBlock"]).ravel()[0]) != n:
        raise ValueError(f"{folder}: nPointsPerBlock does not match the data")

    features = [Feature(
        name="threshold_crossings", data=nctx, unit="Hz",
        description="Non-causal -3.5 x RMS threshold crossing rate per electrode per 20 ms bin "
                    "(count x 50); noise-outlier bins linearly interpolated by the lab",
        threshold_rms=-3.5, normalization="none (rates)")]
    if "spikePower" in data:
        sbp = np.asarray(data["spikePower"], dtype=np.float32)
        if sbp.shape != (n, c):
            raise ValueError(f"{folder}: spikePower shape {sbp.shape}")
        features.append(Feature(
            name="spike_band_power", data=sbp, unit="a.u.",
            description="Spike band power (250-5000 Hz) per electrode per 20 ms bin",
            normalization="none (raw)"))

    behavior = []
    for key, name, unit, desc, comps in (
        ("cursorPos", "cursor_position", "screen units", "Cursor position", ("x", "y")),
        ("targetPos", "target_position", "screen units", "Target position; NaN when no target is shown", ("x", "y")),
    ):
        v = _xy(info, key, n)
        if v is not None:
            behavior.append(Signal(name, v, unit, desc, comps))
    for key, name, unit, desc in (
        ("cursorVel", "cursor_decoder_output", "screen units per bin", "Decoded 2D velocity output of the online decoder"),
        ("labels", "intended_direction", "screen units", "Inferred cursor-to-target vector (intention label)"),
    ):
        v = _xy(data, key, n)
        if v is not None:
            behavior.append(Signal(name, v, unit, desc, ("x", "y")))
    for src, key, name, unit, desc in (
        (info, "magEst", "decoded_magnitude", "a.u.", "Adjusted decoded cursor-to-target magnitude"),
        (info, "angleError", "angle_error", "degree", "Instantaneous angle between decoded and intended direction"),
        (info, "avgOutliers", "outlier_percent", "percent",
         "Maximum, across the two arrays, of the average percent of noise-outlier samples per bin"),
    ):
        v = _col(src, key, n)
        if v is not None:
            behavior.append(Signal(name, v.astype(np.float32), unit, desc))

    ss = np.asarray(task["startStops"], dtype=np.int64).reshape(-1, 2)
    trials = None
    if len(ss):
        if ss.min() < 1 or ss.max() > n or np.any(ss[:, 1] < ss[:, 0]):
            raise ValueError(f"{folder}: startStops out of range")
        start = (ss[:, 0] - 1) * BIN_S  # 1-based inclusive -> leading edge of first bin
        end = ss[:, 1] * BIN_S  # trailing edge of last bin
        k = len(ss)
        columns, descriptions = {}, {}
        trimmed = []
        for src, key, name, dtype, desc in (
            (task, "excludeTrials", "exclude", bool, "True if mean noise exceeded 5% during the trial"),
            (info, "trialSuccess", "success", bool, "True if the target was acquired"),
            (info, "timeToTarget", "time_to_target_s", np.float64, "Time to target, seconds"),
            (info, "pathEfficiency", "path_efficiency", np.float64, "Path efficiency"),
            (info, "orthChanges", "orthogonal_changes", np.int64, "Number of orthogonal direction changes"),
            (info, "angleErrorPerTrial", "angle_error_deg", np.float64, "Median angular error, degrees"),
        ):
            v = _col(src, key)
            if v is not None and len(v) == k + 1:
                # One release block (T11 day 689 block 14) has one surplus trailing row in
                # three info metrics; dropping the last row restores the success/timeout
                # agreement seen in every other block, dropping the first does not.
                v = v[:k]
                trimmed.append(key)
            if v is not None:
                if len(v) != k:
                    raise ValueError(f"{folder}: {key} has {len(v)} rows for {k} trials")
                columns[name] = v.astype(dtype)
                descriptions[name] = desc
        mdv = _xy(task, "moveDirVect", k) if "moveDirVect" in task else None
        if mdv is not None:
            columns["move_dir_x"], columns["move_dir_y"] = mdv[:, 0], mdv[:, 1]
            descriptions["move_dir_x"] = descriptions["move_dir_y"] = "Movement direction vector component"
        trials = Trials(start=start.astype(np.float64), end=end.astype(np.float64),
                        columns=columns, descriptions=descriptions)

    name = str(np.asarray(task["name"]).ravel()[0]).strip()
    params = {"task_name": name, "use_click": bool(np.asarray(task["useClick"]).ravel()[0])}
    if "percentCorrect" in info:
        params["percent_correct"] = float(np.asarray(info["percentCorrect"]).ravel()[0])

    suffix = f"_{kind}" if kind else ""
    extra_prov = {"trimmed_trailing_trial_rows": trimmed} if trials is not None and trimmed else {}
    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"{pid.lower()}_day{day:04d}_block{block:02d}{suffix}",
        participant_id=pid,
        split=None,
        bin_size_s=BIN_S,
        timestamps=BIN_S * (np.arange(n) + 0.5),
        channels=ch,
        features=features,
        trials=trials,
        post_implant_day=day,
        task=_TASKS.get(name.lower(), name),
        timeline="continuous",
        behavior=behavior,
        provenance={
            "source": f"https://datadryad.org/dataset/{DOI}",
            "license": "CC0-1.0",
            "file": folder.as_posix().split("MINDFUL_Data/")[-1],
            "block": block,
            "subset": kind or "main",
            **extra_prov,
            "task_parameters": params,
            "time_origin": "start of the block",
            "subject_sex": "M",
            "citation": "Pun TK et al. Measuring instability in chronic human intracortical neural "
                        "recordings towards stable, long-term brain-computer interfaces. "
                        "Commun Biol 2024.",
            "doi": "doi:10.1038/s42003-024-06784-4",
            "experimenters": ["Pun, Tsam Kiu", "Hochberg, Leigh R."],
            "experiment_description": (
                f"BrainGate2 participant {pid} controlled a 2D cursor with a fixed decoder from two "
                "96-channel arrays in the hand knob area, at home, across weeks to months."
            ),
            "keywords": ["brain-computer interface", "cursor control", "recording instability",
                         "BrainGate", "intracortical"],
        },
    )


def iter_recordings(root: str | Path) -> Iterator[Recording]:
    blocks = find_blocks(root)
    if not blocks:
        raise FileNotFoundError(f"no MINDFUL block folders under {root}")
    for b in blocks:
        yield read_block(b)


def read(root: str | Path) -> list[Recording]:
    return list(iter_recordings(root))


# Uniform per-file access, used by tools that select files before loading them.
def find_sources(root: str | Path) -> list[Path]:
    return find_blocks(root)


def read_source(path: str | Path) -> Recording:
    return read_block(path)
