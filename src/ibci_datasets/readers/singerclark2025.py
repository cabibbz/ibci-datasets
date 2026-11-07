"""Reader for Singer-Clark et al. 2025 (J Neural Eng), cursor control from speech motor cortex.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxzq (CC0-1.0).
Companion code and field documentation:
https://github.com/Neuroprosthetics-Lab/BCI-cursor-control-from-speech-motor-cortex

One MATLAB v5 ``.mat`` file per 3-5 minute block, named
``t15_day<DDDDD>_block<BB>_<task>.mat``, from post-implant days 39, 202 and 468. Three
tasks: ``radial8_calibration_task``, ``grid_evaluation_task`` and
``simultaneous_speech_and_cursor_task``. Each block is continuous, in 10 ms bins:

* ``threshold_crossings`` and ``spike_band_power`` (T x 256), raw (not normalized); the
  crossing threshold is not stated in the release;
* cursor and target position, online cursor (and click) decoder output, assist level;
* per-trial bin indices (0-based, as used by the companion code) and, in the speech
  task, the prompted word.

``t15_day00202_words.mat`` is undocumented and is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..schema import Channels, Feature, Recording, Signal, Trials

DATASET_ID = "singerclark2025_cursor"
DOI = "doi:10.5061/dryad.prr4xgxzq"
BIN_S = 0.010

AREAS = {"v6v": "ventral 6v", "4": "area 4", "55b": "area 55b", "d6v": "dorsal 6v"}

TASKS = {
    "radial8_calibration_task": "Radial8 calibration: center-out cursor task, assist fading from open to closed loop",
    "grid_evaluation_task": "Grid evaluation: closed-loop cursor and click selection on a grid of square targets",
    "simultaneous_speech_and_cursor_task": "Simultaneous speech and cursor: closed-loop cursor control while speaking a prompted word",
}

_FILE_RE = re.compile(r"^(t15)_day(\d{5})_block(\d{2})_(" + "|".join(TASKS) + r")\.mat$")

# Block-level scalars kept as task parameters.
_BLOCK_SCALARS = ("target_radius", "cursor_radius", "dwell_requirement_sec",
                  "grid_num_rows", "grid_total_height", "is_control_block")

_SCREEN = "screen heights"  # README: 1.0 = the shortest screen dimension (the height)


def _loadmat(path: Path) -> dict:
    try:
        import scipy.io
    except ImportError as e:  # pragma: no cover
        raise ImportError("this reader needs scipy: pip install scipy") from e
    return scipy.io.loadmat(str(path))


def find_files(root: str | Path) -> list[Path]:
    """All block files under ``root``, sorted by day then block."""
    found = []
    for p in Path(root).rglob("t15_day*_block*.mat"):
        m = _FILE_RE.match(p.name)
        if m:
            found.append((int(m.group(2)), int(m.group(3)), p))
    return [p for _, _, p in sorted(found)]


def channels(labels) -> Channels:
    labels = [str(s).strip() for s in np.asarray(labels).ravel()]
    unknown = sorted(set(labels) - set(AREAS))
    if unknown:
        raise ValueError(f"unknown array labels {unknown}")
    counts: dict[str, int] = {}
    ids = []
    for a in labels:
        ids.append(f"{a}-{counts.get(a, 0):02d}")
        counts[a] = counts.get(a, 0) + 1
    return Channels(ids=np.array(ids), arrays=np.array(labels),
                    locations=np.array([AREAS[a] for a in labels]))


def _vec(d: dict, key: str) -> np.ndarray:
    return np.asarray(d[key]).ravel()


def _scalar(d: dict, key: str):
    v = np.asarray(d[key]).ravel()
    if v.size != 1:
        raise ValueError(f"{key}: expected one value, got {v.size}")
    x = v[0].item() if hasattr(v[0], "item") else v[0]
    return bool(x) if key == "is_control_block" else x


def _strings(d: dict, key: str) -> list[str]:
    out = []
    for cell in np.asarray(d[key]).ravel():
        flat = np.asarray(cell).ravel()
        out.append(" ".join(str(s).strip() for s in flat) if flat.size else "")
    return out


def read_file(path: str | Path) -> Recording:
    """Read one block file into a continuous Recording."""
    path = Path(path)
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Singer-Clark 2025 block file: {path.name}")
    participant, day, block, task = m.group(1).upper(), int(m.group(2)), int(m.group(3)), m.group(4)
    d = _loadmat(path)

    ts = _vec(d, "timestamp_sec").astype(np.float64)
    n = len(ts)
    if n < 2 or not np.allclose(np.diff(ts), BIN_S, atol=1e-6):
        raise ValueError(f"{path.name}: timestamp_sec is not a regular {BIN_S * 1000:.0f} ms grid")
    # timestamp_sec starts at 0 and marks the start of each bin; the schema stores centres.
    timestamps = ts + BIN_S / 2
    block_end = float(ts[-1] + BIN_S)

    tx = np.asarray(d["threshold_crossings"])
    sbp = np.asarray(d["spike_band_power"])
    ch = channels(d["array_label_by_electrode"])
    for name, x in (("threshold_crossings", tx), ("spike_band_power", sbp)):
        if x.shape != (n, len(ch)):
            raise ValueError(f"{path.name}: {name} has shape {x.shape}, expected {(n, len(ch))}")

    features = [
        Feature(name="threshold_crossings", data=tx.astype(np.float32), unit="count",
                description="Threshold crossings per electrode per 10 ms bin "
                            "(electrode-specific threshold; level not stated in the release)",
                threshold_rms=None, normalization="none (raw counts)"),
        Feature(name="spike_band_power", data=sbp.astype(np.float32),
                unit="a.u. (mean squared filtered signal)",
                description="Spike band power: mean squared value of the filtered signal per 10 ms bin",
                normalization="none (raw)"),
    ]

    behavior = [
        Signal("cursor_position", np.asarray(d["cursor_position"], dtype=np.float32), _SCREEN,
               "Cursor centre on screen; (0, 0) is the screen centre, +x right, +y up", ("x", "y")),
        Signal("target_position", np.asarray(d["target_position"], dtype=np.float32), _SCREEN,
               "Centre of the cued target for the current trial", ("x", "y")),
        Signal("cursor_decoder_output", np.asarray(d["cursor_decoder_output"], dtype=np.float32),
               f"{_SCREEN} per second", "Online cursor decoder velocity output", ("vx", "vy")),
        Signal("assist_amount", _vec(d, "assist_amount").astype(np.float32), "fraction",
               "Computer assistance weight: 0 = fully neural (closed loop), 1 = fully assisted (open loop)"),
    ]
    if "click_decoder_output" in d:
        behavior.append(Signal("click_decoder_output", _vec(d, "click_decoder_output").astype(np.uint8),
                               "boolean", "Online click decoder output (1 = click)"))
    if "click_assist" in d:
        behavior.append(Signal("click_assist", _vec(d, "click_assist").astype(np.uint8), "boolean",
                               "1 = target selection by computer assistance after dwell, 0 = by the click decoder"))

    trials = _trials(d, ts, block_end, path.name)
    params = {k: _scalar(d, k) for k in _BLOCK_SCALARS if k in d}

    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"t15_day{day:05d}_block{block:02d}",
        participant_id=participant,
        split=None,
        bin_size_s=BIN_S,
        timestamps=timestamps,
        channels=ch,
        features=features,
        trials=trials,
        session_date=None,
        post_implant_day=day,
        task=TASKS[task],
        timeline="continuous",
        behavior=behavior,
        provenance={
            "source": f"https://datadryad.org/dataset/{DOI}",
            "license": "CC0-1.0",
            "file": path.name,
            "task_name": task,
            "block": block,
            "task_parameters": params,
            "time_origin": "start of the block (timestamp_sec = 0)",
            "citation": "Singer-Clark T et al. Speech motor cortex enables BCI cursor control "
                        "and click. J Neural Eng 2025.",
            "doi": "doi:10.1088/1741-2552/add0e5",
            "experimenters": ["Singer-Clark, Tyler", "Stavisky, Sergey D.", "Brandman, David M."],
            "institution": "University of California, Davis",
            "experiment_description": (
                "BrainGate2 participant T15 (ALS) controlled a computer cursor and click with "
                "decoders reading 256 electrodes on four arrays in speech motor cortex."
            ),
            "keywords": ["brain-computer interface", "cursor control", "BrainGate",
                         "intracortical", "speech motor cortex", "ALS"],
        },
    )


def _trials(d: dict, ts: np.ndarray, block_end: float, name: str) -> Trials:
    n = len(ts)
    starts = _vec(d, "trial_start_bin").astype(np.int64)
    if len(starts) == 0:
        raise ValueError(f"{name}: no trials")
    if starts.min() < 0 or starts.max() >= n or np.any(np.diff(starts) <= 0):
        raise ValueError(f"{name}: trial_start_bin out of range or not increasing")

    def t_start(bins):  # time of the leading edge of each bin
        return ts[bins]

    has_end = "trial_end_bin" in d
    if has_end:
        end_bins = _vec(d, "trial_end_bin").astype(np.int64)
        if len(end_bins) != len(starts) or np.any(end_bins < starts) or end_bins.max() >= n:
            raise ValueError(f"{name}: trial_end_bin inconsistent with trial_start_bin")
        end = ts[end_bins] + BIN_S
        truncated = np.zeros(len(starts), dtype=bool)
    else:
        # A trial runs until the next one starts; the last one is cut off by the block end.
        end = np.append(t_start(starts[1:]), block_end)
        truncated = np.zeros(len(starts), dtype=bool)
        truncated[-1] = True

    target = np.asarray(d["target_position"], dtype=np.float32)
    trial_idx = _vec(d, "trial_idx").astype(np.int64)
    target_bins = _vec(d, "target_presentation_bin").astype(np.int64) if "target_presentation_bin" in d else starts

    columns = {
        "trial": trial_idx[starts],
        "target_x": target[target_bins, 0],
        "target_y": target[target_bins, 1],
        "truncated": truncated,
    }
    descriptions = {
        "trial": "Trial counter (trial_idx) within the block",
        "target_x": "Cued target centre x, screen heights",
        "target_y": "Cued target centre y, screen heights",
        "truncated": "True if the trial has no recorded end and was cut off by the end of the block",
    }
    for key, col, desc in (
        ("target_presentation_bin", "target_presentation_time", "Target and prompted word shown"),
        ("cursor_go_cue_bin", "cursor_go_cue_time", "Cursor go cue (target turned green)"),
        ("speech_go_cue_bin", "speech_go_cue_time", "Speech go cue (beep); NaN on trials without a beep"),
    ):
        if key in d:
            bins = _vec(d, key).astype(np.int64)
            if len(bins) != len(starts) or bins.max() >= n:
                raise ValueError(f"{name}: {key} inconsistent with trial_start_bin")
            t = np.full(len(bins), np.nan)
            ok = bins >= 0
            t[ok] = ts[bins[ok]]
            columns[col] = t
            descriptions[col] = f"{desc}, seconds from block start"
    if "speech_prompt" in d:
        prompts = _strings(d, "speech_prompt")
        if len(prompts) != len(starts):
            raise ValueError(f"{name}: speech_prompt has {len(prompts)} entries for {len(starts)} trials")
        columns["speech_prompt"] = np.array(prompts, dtype=str)
        descriptions["speech_prompt"] = "Word prompted above the target"

    return Trials(start=t_start(starts).astype(np.float64), end=end.astype(np.float64),
                  columns=columns, descriptions=descriptions)


def read(root: str | Path) -> list[Recording]:
    """Read every block under ``root`` (one Recording per block)."""
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no t15_day*_block*_<task>.mat files under {root}")
    return [read_file(p) for p in files]
