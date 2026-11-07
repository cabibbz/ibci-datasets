"""Reader for Rigotti-Thompson et al. 2026, preparatory activity in human motor cortex.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvr9 (CC0-1.0).
Companion code: https://github.com/snel-repo/human-preparatory-activity-paper

HDF5 files grouped by paper figure, for T5 (2 x 96-channel arrays), T11 (2 x 96) and T16
(4 x 64). Heavily preprocessed by the authors (README): threshold crossings and spike band
power, binned at 10-20 ms, then

* ``fig{1,1c,2,3,4}_{P}_{delay,move}.h5``: trial windows aligned to delay start or to
  movement, smoothed acausally (Gaussian, sigma 100 ms), channels under 2 Hz dropped
  (``keep_chans_all``, 0-based). The delay and move files hold the **same trials**, as two
  overlapping windows; each file becomes its own Recording.
* ``fig5_{P}_continuous.h5``: one continuous closed-loop block (T11, T16), all channels,
  z-scored after causal exponential smoothing (tau 28-70 ms), with cursor, decoder and
  trial tables.

Checked on the data: in the continuous files the first half of ``neural_data_norm`` is
threshold crossings and the second half spike band power for the same electrodes, in the
same order (same-electrode correlation 0.16 and 0.39 vs about 0.01 for other pairs); T16
is binned at 20 ms and T11 at 10 ms (one 11 ms step). The release does not name the
arrays, so they are labelled ``a1``... in channel order. Channel ids keep the original
electrode index (``a2-05`` = array 2, electrode 5), also for the dropped-channel files.

Array correspondence (see ERRATA.md): the ``fig1-4`` kept channels concentrate on Deo 2026's
most active channels with the columns taken straight, suggesting ``a1`` / ``a2`` = Deo's
``d2`` / ``d1`` for T5 and T11. Channel-correlation comparisons of the continuous blocks with
Deo, Pun and Srinivasan were inconclusive; for T11 the two scorings of ``ibci
verify-mapping`` disagreed, which is how the block-centred scoring came to be needed.

Not converted: ``fig5_*_start*.h5`` (no neural data) and ``fig5_*_onset.h5`` (onset-aligned
windows, mostly duplicating training blocks not otherwise released as streams). Session
dates are not in the release.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Channels, Feature, Recording, Signal, Trials

DATASET_ID = "rigottithompson2026_preparatory"
DOI = "doi:10.5061/dryad.brv15dvr9"
TRIAL_GAP_S = 1.0
ARRAY_SIZE = {"T5": 96, "T11": 96, "T16": 64}
N_CHANNELS = {"T5": 192, "T11": 192, "T16": 256}

_TRIAL_RE = re.compile(r"^(fig1c|fig[1-4])_(T5|T11|T16)_(delay|move)\.h5$")
_CONT_RE = re.compile(r"^fig5_(T11|T16)_continuous\.h5$")

_TASKS = {
    "fig1": "Open-loop instructed-delay radial-8 task, attempted wrist movements",
    "fig1c": "Open-loop instructed-delay radial-4 task, wrist movements, regular vs suppressed visuals",
    "fig2": "Open-loop instructed-delay radial-8 task, attempted finger movements",
    "fig3": "Open-loop instructed-delay curved radial-8 task, attempted wrist movements",
    "fig4": "Open-loop instructed-delay Pac-Man task, attempted wrist movements",
    "fig5": "Closed-loop radial-8 cursor control driven by preparatory activity",
}


def find_files(root: str | Path) -> list[Path]:
    files = [p for p in Path(root).rglob("*.h5") if _TRIAL_RE.match(p.name) or _CONT_RE.match(p.name)]
    return sorted(files, key=lambda p: p.name)


def channels(pid: str, keep: np.ndarray | None = None) -> Channels:
    size = ARRAY_SIZE[pid]
    idx = np.arange(N_CHANNELS[pid]) if keep is None else np.asarray(keep, dtype=np.int64)
    if idx.min() < 0 or idx.max() >= N_CHANNELS[pid] or np.any(np.diff(idx) <= 0):
        raise ValueError(f"{pid}: channel indices out of range or not increasing")
    arrays = np.array([f"a{i // size + 1}" for i in idx])
    ids = np.array([f"a{i // size + 1}-{i % size:02d}" for i in idx])
    loc = "precentral gyrus (array not named in the release)"
    return Channels(ids=ids, arrays=arrays, locations=np.full(len(idx), loc))


def _provenance(path: Path, pid: str, **extra) -> dict:
    return {
        "source": f"https://datadryad.org/dataset/{DOI}",
        "license": "CC0-1.0",
        "file": path.name,
        "citation": "Rigotti-Thompson M et al. Preparatory encoding of intended movement in human "
                    "motor cortex and implications for brain-computer interfaces. 2026.",
        "experimenters": ["Rigotti-Thompson, Mattia", "AuYong, Nicholas", "Pandarinath, Chethan"],
        "experiment_description": (
            f"BrainGate2 participant {pid}: instructed-delay movement tasks and closed-loop cursor "
            "control, to separate preparatory from movement-related activity."),
        "keywords": ["brain-computer interface", "preparatory activity", "motor cortex", "BrainGate",
                     "intracortical"],
        **extra,
    }


def read_trial_file(path: str | Path) -> Recording:
    import h5py

    path = Path(path)
    m = _TRIAL_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Rigotti-Thompson trial file: {path.name}")
    fig, pid, align = m.groups()
    with h5py.File(path, "r") as f:
        d = {k: f[k][()] for k in f.keys()}
    t = np.asarray(d["t_data"], dtype=np.float64) / 1000.0
    dt = np.unique(np.round(np.diff(t), 6))
    if len(dt) != 1:
        raise ValueError(f"{path.name}: irregular t_data")
    bin_s = float(dt[0])
    spk, sbp = np.asarray(d["spike_data_all"]), np.asarray(d["sbp_data_all"])
    n_tr, n_bins, n_ch = spk.shape
    if sbp.shape != spk.shape:
        raise ValueError(f"{path.name}: spike and sbp shapes differ")
    ch = channels(pid, d["keep_chans_all"])
    if len(ch) != n_ch:
        raise ValueError(f"{path.name}: keep_chans_all has {len(ch)} entries for {n_ch} channels")

    seg_len = n_bins * bin_s
    starts = np.arange(n_tr) * (seg_len + TRIAL_GAP_S)
    ends = starts + seg_len
    timestamps = (starts[:, None] + bin_s * (np.arange(n_bins) + 0.5)[None, :]).ravel()
    flat = lambda a: a.reshape(n_tr * n_bins, *a.shape[2:])

    columns = {"event_time": starts - t[0]}
    descriptions = {"event_time": f"Alignment event ({'delay start' if align == 'delay' else 'movement'}) "
                                  "on the concatenated timeline"}
    for key, v in d.items():
        v = np.asarray(v)
        if not key.endswith("_all") or key in ("spike_data_all", "sbp_data_all", "keep_chans_all") or v.shape[0] != n_tr:
            continue
        name = key[: -len("_all")].replace("_data", "")
        if v.ndim == 1:
            v = v.astype(np.float64)
            if name.endswith("_bool"):
                name = name[: -len("_bool")]
                if np.isfinite(v).all() and set(np.unique(v)) <= {0.0, 1.0}:
                    v = v.astype(bool)  # 0/1 flags; kept as float only where the release has NaN
            columns[name] = v
            descriptions[name] = f"{key} from the release"
        elif v.ndim == 2 and v.shape[1] == 2:
            columns[f"{name}_x"], columns[f"{name}_y"] = v[:, 0].astype(np.float64), v[:, 1].astype(np.float64)
            descriptions[f"{name}_x"] = descriptions[f"{name}_y"] = f"{key} from the release, px"

    behavior = []
    for key, name, unit, desc in (("cursor_data_all", "cursor_position", "px", "Cursor position [x, y]"),
                                  ("emg_data_all", "emg", "a.u.", "Surface EMG, 2 channels (T16)"),
                                  ("gaze_data_all", "gaze_position", "px", "Eye-gaze position [x, y]")):
        if key in d:
            v = np.asarray(d[key], dtype=np.float32)
            if v.shape[:2] == (n_tr, n_bins):
                behavior.append(Signal(name, flat(v), unit, desc, ("x", "y") if name != "emg" else ("ch1", "ch2")))

    smooth = "acausal Gaussian smoothing (sigma 100 ms) by the authors; channels under 2 Hz dropped"
    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"{fig}_{pid.lower()}_{align}",
        participant_id=pid,
        split=None,
        bin_size_s=bin_s,
        timestamps=timestamps,
        channels=ch,
        features=[
            Feature("threshold_crossings", flat(spk).astype(np.float32), "spikes per bin (smoothed)",
                    "Smoothed binned threshold crossings (threshold not stated)", normalization=smooth),
            Feature("spike_band_power", flat(sbp).astype(np.float32), "a.u.",
                    "Smoothed spike band power", normalization=smooth),
        ],
        trials=Trials(start=starts, end=ends, columns=columns, descriptions=descriptions),
        task=_TASKS[fig],
        timeline="concatenated-trials",
        behavior=behavior,
        provenance=_provenance(path, pid, figure=fig, alignment=align, trial_gap_s=TRIAL_GAP_S,
                               window_s=[float(t[0]), float(t[-1] + bin_s)],
                               note="The delay and move files of a figure hold the same trials."),
    )


def read_continuous(path: str | Path) -> Recording:
    import h5py

    path = Path(path)
    m = _CONT_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Rigotti-Thompson continuous file: {path.name}")
    pid = m.group(1)
    with h5py.File(path, "r") as f:
        data = {k: f["data"][k][()] for k in f["data"].keys()}
        info = {k: f["trial_info"][k][()] for k in f["trial_info"].keys()}
        keep = f["keep_chans"][()]
    x = np.asarray(data["neural_data_norm"], dtype=np.float32)
    n = N_CHANNELS[pid]
    if x.shape[1] != 2 * n:
        raise ValueError(f"{path.name}: {x.shape[1]} columns, expected {2 * n} (spikes + sbp)")
    clock = np.asarray(data["clock_time"], dtype=np.float64) / 1000.0
    steps = np.diff(clock)
    bin_s = round(float(np.median(steps)), 6)
    t0 = clock[0]
    ts = clock - t0
    norm = "z-scored by the authors after causal exponential smoothing (tau 28-70 ms)"

    behavior = [
        Signal("cursor_position", np.asarray(data["cursor_data"], np.float32), "px", "Cursor position", ("x", "y")),
        Signal("cursor_decoder_output", np.asarray(data["control_samples"], np.float32), "px per bin",
               "Decoded cursor velocity used for control", ("vx", "vy")),
        Signal("move_state_probability", np.asarray(data["dec_move_state"], np.float32).ravel(), "probability",
               "Online movement-onset decoder output"),
    ]
    if "emg_data" in data:
        behavior.append(Signal("emg", np.asarray(data["emg_data"], np.float32), "a.u.", "Surface EMG", ("ch1", "ch2")))

    start = np.asarray(info["start_time"], np.float64) / 1000.0 - t0
    end = np.asarray(info["end_time"], np.float64) / 1000.0 - t0
    onset_bin = np.asarray(info["dec_movement_onset_time"]).astype(np.int64)
    onset = np.where(onset_bin >= 0, ts[np.clip(onset_bin, 0, len(ts) - 1)], np.nan)
    target = np.asarray(info["target_data"], np.float64)
    trials = Trials(
        start=start, end=end,
        columns={"go_cue_time": np.asarray(info["go_cue_time"], np.float64) / 1000.0 - t0,
                 "delay_s": np.asarray(info["delay_data"], np.float64) / 1000.0,
                 "no_go": np.asarray(info["no_go_bool"]).astype(bool),
                 "decoded_onset_time": onset,
                 "target_x": target[:, 0], "target_y": target[:, 1]},
        descriptions={"go_cue_time": "Go cue; NaN on no-go trials", "delay_s": "Instructed delay, s",
                      "no_go": "No-go (catch) trial", "decoded_onset_time": "Decoded movement onset; NaN if none",
                      "target_x": "Target x, px", "target_y": "Target y, px"},
    )
    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"fig5_{pid.lower()}_continuous",
        participant_id=pid,
        split=None,
        bin_size_s=bin_s,
        timestamps=ts,
        channels=channels(pid),
        features=[
            Feature("threshold_crossings", x[:, :n], "z-score", "Threshold crossings (first half of neural_data_norm)",
                    normalization=norm),
            Feature("spike_band_power", x[:, n:], "z-score", "Spike band power (second half of neural_data_norm)",
                    normalization=norm),
        ],
        trials=trials,
        task=_TASKS["fig5"],
        timeline="continuous",
        behavior=behavior,
        provenance=_provenance(path, pid, figure="fig5", time_origin="first bin of the block (clock_time - clock_time[0])",
                               preparatory_decoder_channels=np.asarray(keep).tolist(),
                               bin_steps_ms=sorted({int(round(s * 1000)) for s in steps})),
    )


def read_source(path: str | Path) -> Recording:
    path = Path(path)
    return read_continuous(path) if _CONT_RE.match(path.name) else read_trial_file(path)


def find_sources(root: str | Path) -> list[Path]:
    return find_files(root)


def iter_recordings(root: str | Path) -> Iterator[Recording]:
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no fig*.h5 files under {root}")
    for p in files:
        yield read_source(p)


def read(root: str | Path) -> list[Recording]:
    return list(iter_recordings(root))
