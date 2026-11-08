"""Reader for Srinivasan et al. 2026, speech modes and loudness in ventral precentral gyrus.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.2547d7x5w (CC0-1.0).
Companion code: https://github.com/Neuroprosthetics-Lab/srinivasan-speech-modes

Three zips of per-block MATLAB v5 files, ``<participant>_<task>_block_<n>.mat``:

* ``t15_word-loudness`` (5 blocks) and ``t16_word-loudness`` (9 blocks): one word per trial
  at a cued loudness, e.g. ``"LOUD: have"`` (MIME / WHISPER / NORMAL / LOUD x six words).
* ``t15_sentence-loudness`` (2 closed-loop evaluation blocks): sentences with emphasised
  words, e.g. ``"i CERTAINLY HOPE so"``, with the online brain-to-text output.

Per the Dryad README, features are -4.5 x RMS threshold crossings and spike band power on
256 electrodes (four 64-electrode arrays), binned every 10 ms, then normalized and
smoothed (``threshcross``, ``spikepow``). Word-loudness files also carry unsmoothed
``raw_threshcross`` counts.

Trial layout (word-loudness), established from the companion analysis scripts and checked
on the data: every per-trial array starts 1 s (100 bins) before the delay period; the go
cue follows the delay (``delay_duration_ms``); ``speech_onsets`` / ``speech_offsets`` are
30 kHz sample counts from the go cue. ``raw_threshcross`` covers 3 s after the go cue (4 s in
most T16 blocks), the smoothed features 3 s more; both start at the same time (smoothed vs smoothed-raw correlation
0.97 at lag 0, negative at lag 300). Raw counts are NaN-padded to the smoothed length.

Electrode order differs between files. The lab's pipeline has a
``using_correct_electrode_mapping`` flag; the companion scripts give T15's word-loudness
session the flag-0 order (area 4, v6v, d6v, 55b) and later sessions the flag-1 order
(v6v, area 4, 55b, d6v), the order documented for the Card 2024 release. Checked with
``ibci verify-mapping`` (see ERRATA.md): the sentence-loudness files are consistent with
Card's order. For the word-loudness files the reader uses the companion scripts' flag-0
order; the data are consistent with it (Deo 2026's T15 columns, also 2023, match these
one to one; against Card sessions 0-15 days away Card's order is never the best) but no
comparison gives a verdict. Within arrays the channel order may also differ from Card's
(channel-for-channel comparison matches only the 55b block), so electrode ids in these
files may not correspond to Card's electrodes. T16's order (55b/PEF, 6v, then two hand-knob arrays) is from
the companion scripts; comparisons with Deo 2026, Rigotti-Thompson 2026 and Kunz 2025 T16
data are near-ties.

Not converted: the sentence files' phoneme ``logits`` and partial decoder outputs (20 ms
decoder clock, aligned through tracking ids the neural arrays lack) and
``predicted_amplitude``. Session dates are not in the release. Two T15 trials are
``DO NOTHING`` controls (not in the README); they get an empty word and NaN onset/offset,
since their release onsets (3-5 ms after the go cue) are placeholders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Channels, Feature, Recording, Trials

DATASET_ID = "srinivasan2026_speech_modes"
DOI = "doi:10.5061/dryad.2547d7x5w"
BIN_S = 0.010
TRIAL_GAP_S = 1.0
PRE_DELAY_BINS = 100  # each trial array starts 1 s before the delay period
AUDIO_HZ = 30000
LOUDNESS = ("MIME", "WHISPER", "NORMAL", "LOUD")
CONTROL_CUE = "DO NOTHING"  # rare control trials (2 of 1140 word trials, both T15)

# (array id, location) in feature order, 64 electrodes each. See the module docstring.
ARRAYS = {
    ("t15", "word-loudness"): (("4", "area 4"), ("v6v", "ventral 6v"), ("d6v", "dorsal 6v"), ("55b", "area 55b")),
    ("t15", "sentence-loudness"): (("v6v", "ventral 6v"), ("4", "area 4"), ("55b", "area 55b"), ("d6v", "dorsal 6v")),
    ("t16", "word-loudness"): (("55b", "area 55b / premotor eye field"), ("6v", "area 6v"),
                               ("hk1", "hand knob"), ("hk2", "hand knob")),
}
_MAPPING_SOURCE = {
    ("t15", "word-loudness"): "companion scripts (flag 0, 2023 sessions); consistent with the data, not proven "
                              "by them: matches Deo 2026 T15 columns one to one; vs Card 0-15 days away no verdict",
    ("t15", "sentence-loudness"): "verify-mapping consistent vs Card 2024-2025 sessions (flag-1 order)",
    ("t16", "word-loudness"): "companion scripts (psth.py) only; verify-mapping vs Deo 2026, "
                              "Rigotti-Thompson 2026 and Kunz 2025 T16 inconclusive (near-ties)",
}

_FILE_RE = re.compile(r"^(t1[56])_(word-loudness|sentence-loudness)_block_(\d+)\.mat$")

_EXPERIMENT = {
    "word-loudness": "Attempted speech of single words at a cued loudness or mode (mime, whisper, normal, loud)",
    "sentence-loudness": "Closed-loop brain-to-text with decoded loudness: sentences with emphasised words",
}


def _loadmat(path: Path) -> dict:
    import scipy.io

    return scipy.io.loadmat(str(path))


def find_files(root: str | Path) -> list[Path]:
    found = []
    for p in Path(root).rglob("t1*_*-loudness_block_*.mat"):
        m = _FILE_RE.match(p.name)
        if m:
            found.append((m.group(1), m.group(2), int(m.group(3)), p))
    return [p for *_, p in sorted(found)]


def channels(participant: str, task: str) -> Channels:
    ids, arrays, locations = [], [], []
    for array_id, location in ARRAYS[(participant, task)]:
        for i in range(64):
            ids.append(f"{array_id}-{i:02d}")
            arrays.append(array_id)
            locations.append(location)
    return Channels(ids=np.array(ids), arrays=np.array(arrays), locations=np.array(locations))


def _cells(d: dict, key: str) -> list[np.ndarray]:
    return [np.asarray(c) for c in np.asarray(d[key]).ravel()]


def _text(cell) -> str:
    flat = np.asarray(cell).ravel()
    return " ".join(str(s).strip() for s in flat).strip()


def _sample(cell) -> float:
    flat = np.asarray(cell).ravel()
    return float(flat[0]) if flat.size else np.nan


def read_file(path: str | Path) -> Recording:
    path = Path(path)
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Srinivasan 2026 block file: {path.name}")
    participant, task, block = m.group(1), m.group(2), int(m.group(3))
    d = _loadmat(path)
    ch = channels(participant, task)
    word_task = task == "word-loudness"

    tx = _cells(d, "threshcross")
    sbp = _cells(d, "spikepow")
    raw = _cells(d, "raw_threshcross") if word_task else None
    cues = [_text(c) for c in np.asarray(d["cue"]).ravel()]
    n = len(tx)
    if len(sbp) != n or len(cues) != n or (raw is not None and len(raw) != n):
        raise ValueError(f"{path.name}: per-trial fields have different trial counts")

    tx_out, sbp_out, raw_out, starts, ends = [], [], [], [], []
    t = 0.0
    for i in range(n):
        a, b = tx[i], sbp[i]
        if a.ndim != 2 or a.shape[1] != len(ch) or b.shape != a.shape:
            raise ValueError(f"{path.name} trial {i}: feature shapes {a.shape} / {b.shape}")
        tx_out.append(a.astype(np.float32))
        sbp_out.append(b.astype(np.float32))
        if raw is not None:
            r = raw[i]
            if r.ndim != 2 or r.shape[1] != len(ch) or r.shape[0] > a.shape[0]:
                raise ValueError(f"{path.name} trial {i}: raw_threshcross shape {r.shape}")
            pad = np.full((a.shape[0], len(ch)), np.nan, dtype=np.float32)
            pad[: r.shape[0]] = r
            raw_out.append(pad)
        starts.append(t)
        ends.append(t + a.shape[0] * BIN_S)
        t = ends[-1] + TRIAL_GAP_S

    start, end = np.array(starts), np.array(ends)
    timestamps = np.concatenate([s + BIN_S * (np.arange(len(x)) + 0.5) for s, x in zip(start, tx_out)])
    norm = "normalized and smoothed by the authors (method not stated in the release)"
    features = []
    if raw is not None:
        delay_bins = np.round(np.asarray(d["delay_duration_ms"], dtype=np.float64).ravel() / (1000 * BIN_S))
        after = sorted({int(len(r) - PRE_DELAY_BINS - db) for r, db in zip(raw, delay_bins)})
        window = (f"{after[0] * BIN_S:g} s" if len(after) == 1 or after[-1] - after[0] <= 1
                  else f"{after[0] * BIN_S:g}-{after[-1] * BIN_S:g} s")
        features.append(Feature(
            name="threshold_crossings", data=np.concatenate(raw_out), unit="count",
            description="Unsmoothed binned -4.5 x RMS threshold crossings (raw_threshcross). "
                        f"Released only up to {window} after the go cue in this file; NaN after that",
            threshold_rms=-4.5, normalization="none (raw counts)"))
    features += [
        Feature(name="threshold_crossings_smoothed", data=np.concatenate(tx_out), unit="a.u.",
                description="Binned -4.5 x RMS threshold crossings, normalized and smoothed (threshcross)",
                threshold_rms=-4.5, normalization=norm),
        Feature(name="spike_band_power_smoothed", data=np.concatenate(sbp_out), unit="a.u.",
                description="Binned spike band power, normalized and smoothed (spikepow)",
                normalization=norm),
    ]

    columns: dict[str, np.ndarray] = {"cue": np.array(cues, dtype=str)}
    descriptions = {"cue": "Cue shown to the participant"}
    if word_task:
        loud, words = [], []
        for c in cues:
            level, _, word = c.partition(":")
            if c == CONTROL_CUE:
                loud.append(CONTROL_CUE)
                words.append("")
            elif level.strip() in LOUDNESS and word.strip():
                loud.append(level.strip())
                words.append(word.strip())
            else:
                raise ValueError(f"{path.name}: unexpected cue {c!r}")
        control = np.array([c == CONTROL_CUE for c in cues])
        delay_s = np.asarray(d["delay_duration_ms"], dtype=np.float64).ravel() / 1000.0
        delay_start = start + PRE_DELAY_BINS * BIN_S
        go = delay_start + delay_s
        onset = go + np.array([_sample(c) for c in _cells(d, "speech_onsets")]) / AUDIO_HZ
        offset = go + np.array([_sample(c) for c in _cells(d, "speech_offsets")]) / AUDIO_HZ
        # Control trials carry a 3-5 ms placeholder onset, not a speech annotation.
        onset[control] = np.nan
        offset[control] = np.nan
        columns.update(loudness=np.array(loud, dtype=str), word=np.array(words, dtype=str),
                       delay_duration_s=delay_s, delay_start_time=delay_start, go_cue_time=go,
                       speech_onset_time=onset, speech_offset_time=offset)
        descriptions.update(
            loudness="Cued loudness or mode: MIME, WHISPER, NORMAL or LOUD; DO NOTHING on control trials",
            word="Cued word",
            delay_duration_s="Delay period before the go cue, seconds",
            delay_start_time="Start of the delay period (1 s after the trial segment starts)",
            go_cue_time="Go cue (end of the delay period)",
            speech_onset_time="Speech onset, from the 30 kHz onset annotation relative to the go cue; NaN on control trials",
            speech_offset_time="Speech offset, from the 30 kHz offset annotation relative to the go cue",
        )
    else:
        columns["decoded_text"] = np.array([_text(c) for c in _cells(d, "decoder_final_output")], dtype=str)
        columns["decoded_text_with_loudness"] = np.array(
            [_text(c) for c in _cells(d, "updated_decoder_final_output")], dtype=str)
        descriptions.update(
            decoded_text="Final closed-loop brain-to-text output",
            decoded_text_with_loudness="Final output with emphasis (upper case) from the closed-loop loudness decoder",
        )
        for k in ("decoded_text", "decoded_text_with_loudness"):
            if len(columns[k]) != n:
                raise ValueError(f"{path.name}: {k} has {len(columns[k])} entries for {n} trials")

    key = (participant, task)
    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"{participant}_{task}_block{block:02d}",
        participant_id=participant.upper(),
        split=None,
        bin_size_s=BIN_S,
        timestamps=timestamps,
        channels=ch,
        features=features,
        trials=Trials(start=start, end=end, columns=columns, descriptions=descriptions),
        task=_EXPERIMENT[task],
        timeline="concatenated-trials",
        provenance={
            "source": f"https://datadryad.org/dataset/{DOI}",
            "license": "CC0-1.0",
            "file": path.name,
            "task_name": task,
            "block": block,
            "trial_gap_s": TRIAL_GAP_S,
            "electrode_mapping_source": _MAPPING_SOURCE[key],
            "citation": "Srinivasan A et al. Encoding of speech modes and loudness in ventral "
                        "precentral gyrus. Nat Commun 2026.",
            "doi": "doi:10.1038/s41467-026-71284-4",
            "experimenters": ["Srinivasan, Aparna", "Brandman, David M.", "Stavisky, Sergey D."],
            # T15 is at UC Davis; T16's site is not stated in the release or the companion repo.
            **({"institution": "University of California, Davis"} if participant == "t15" else {}),
            "experiment_description": (
                f"BrainGate2 participant {participant.upper()}: {_EXPERIMENT[task].lower()}; "
                "256 electrodes on four arrays in left precentral gyrus."
            ),
            "keywords": ["brain-computer interface", "speech", "loudness", "BrainGate", "intracortical"],
        },
    )


def iter_recordings(root: str | Path) -> Iterator[Recording]:
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no t1*_*-loudness_block_*.mat files under {root}")
    for p in files:
        yield read_file(p)


def read(root: str | Path) -> list[Recording]:
    return list(iter_recordings(root))


# Uniform per-file access, used by tools that select files before loading them.
def find_sources(root: str | Path) -> list[Path]:
    return find_files(root)


def read_source(path: str | Path) -> Recording:
    return read_file(path)
