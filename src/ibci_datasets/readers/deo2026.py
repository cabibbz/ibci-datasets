"""Reader for Deo et al. 2026, a mosaic of whole-body representations on the precentral gyrus.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.mpg4f4rg5 (CC0-1.0).
Companion code (MATLAB/Python): https://github.com/d-r-deo/pcg-mosaic

One MATLAB file per participant (T5, T11, T12, T15, T16, T17), plus ``T12_control.mat``.
All but T17 are MATLAB v5; ``T17.mat`` is v7.3 (HDF5, matrices stored transposed).
Each holds one session of an instructed-delay task with 46-48 movement cues (face,
head, limbs, speech), in 20 ms bins, as a struct ``DataMat`` (``T12_control`` has
``DataMat.neural`` and ``DataMat.opticalFlow``).

* ``tx`` (T x F) holds threshold-crossing counts for the multi-unit channels **and** the
  spike-sorted units, concatenated. ``chanSets`` / ``chanSetNames`` (1-based column
  indices) split them; sets named ``...Sorted`` are sorted units.
* ``blockNum`` marks blocks; neural data between blocks was not recorded, so each block is
  one continuous Recording.
* ``state`` (0 delay, 1 go, 2 return; NaN at some block edges) gives the trial structure;
  ``goCue`` (1-based row indices) and ``trialCue`` (index into ``cueList``) give one row
  per trial.
* ``details`` gives the session date and blocks, e.g. ``t15.2023.08.27: blocks [6 7 8 9]``.

Findings from the data, beyond the README:

* ``goCue`` is 1-based: in T5, T11, T12, T12_control and T16 it marks the first Go bin of
  ``state`` in every trial. In T17 it is one bin (20 ms) earlier in every trial; in T15,
  in 48% of trials, evenly across blocks. Trials here follow ``state``; ``goCue`` is kept
  as its own column.
* A few blocks end during a trial's Go state (8 trials in T12_control, 3 in T15, 7 in
  T17); those trials are kept, end at the block end and are marked ``truncated``.
* The arrays are named with the lab's codes, apparently position along the gyrus
  (``v`` ventral, ``m`` middle, ``d`` dorsal), not brain areas. The reader keeps those
  names. For T15, ``ibci verify-mapping`` rejects Card 2024's column order against Card's
  same-day session (2023-08-27), and Deo's T15 columns match Srinivasan 2026's word-loudness
  columns one to one; Card's order is rejected for those too, and their companion scripts
  give area 4, v6v, d6v, 55b. Hence ``v2`` = area 4, ``v3`` = ventral 6v, ``v1`` = dorsal 6v,
  ``m1`` = 55b. T15 channel locations name these inferred areas and point to the evidence.
  For T5 and T11, Pun 2024's arrays ``a1`` / ``a2`` match Deo's ``d2`` / ``d1``. T16 could
  not be resolved (near-ties against Srinivasan 2026 and Rigotti-Thompson 2026). See
  ERRATA.md.

Not converted: sorted units (they come from the same electrodes; the schema has no units
table yet), ``tx_blkMeanSub`` (derivable), and ``T12_control.opticalFlow`` (movie-frame
clock). Participants C1 and C2 are hosted on DABI, not Dryad.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Channels, Feature, Recording, Trials

DATASET_ID = "deo2026_mosaic"
DOI = "doi:10.5061/dryad.mpg4f4rg5"
BIN_S = 0.020
STATE_DELAY, STATE_GO, STATE_RETURN = 0, 1, 2

# ibci verify-mapping vs Card 2024 (see the module docstring and ERRATA.md).
T15_AREA_EVIDENCE = {
    "basis": "Card 2024 order rejected vs Card's same-day session (2023-08-27); columns match Srinivasan 2026 "
             "word-loudness one to one (both scorings); Card order rejected for those too; their companion "
             "scripts give the flag-0 order area 4, v6v, d6v, 55b",
    "v2": "area 4", "v3": "ventral 6v", "v1": "dorsal 6v", "m1": "area 55b",
}
_T15_AREA = {"m1": "area 55b", "v1": "dorsal 6v", "v2": "area 4", "v3": "ventral 6v"}

_FILE_RE = re.compile(r"^(T5|T11|T12|T15|T16|T17)(_control)?\.mat$")
_DETAILS_RE = re.compile(r"t(\d+)\.(\d{4})\.(\d{2})\.(\d{2})")


# ---- loading (MATLAB v5 via scipy, v7.3 via h5py) ----

def _load_v5(path: Path) -> dict:
    import scipy.io

    top = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)["DataMat"]
    s = top.neural if hasattr(top, "neural") else top
    out = {f: getattr(s, f) for f in s._fieldnames}
    for k in ("cueList", "chanSetNames", "stateNames"):
        if k in out:
            out[k] = [str(x) for x in np.atleast_1d(out[k])]
    out["chanSets"] = [np.atleast_1d(np.asarray(x)).astype(np.int64) for x in np.atleast_1d(out["chanSets"])]
    return out


def _load_v73(path: Path) -> dict:
    import h5py

    def text(ds) -> str:
        return "".join(chr(c) for c in np.asarray(ds).ravel())

    with h5py.File(path, "r") as f:
        g = f["DataMat"]
        if "neural" in g:
            g = g["neural"]
        out = {}
        for k in g:
            ds = g[k]
            if ds.dtype == object:  # cell array of references
                items = [f[r] for r in np.asarray(ds).ravel()]
                if k in ("cueList", "chanSetNames", "stateNames"):
                    out[k] = [text(x) for x in items]
                else:
                    out[k] = [np.asarray(x).ravel().astype(np.int64) for x in items]
            elif k in ("details", "hemisphere"):
                out[k] = text(ds)
            else:
                v = np.asarray(ds)
                out[k] = v.T if v.ndim == 2 and v.shape[0] != 1 and v.shape[1] != 1 else v.ravel()
        if np.ndim(out.get("binWidth")):
            out["binWidth"] = float(np.asarray(out["binWidth"]).ravel()[0])
    return out


def _load(path: Path) -> dict:
    with open(path, "rb") as fh:
        head = fh.read(128)
    return _load_v73(path) if b"MATLAB 7.3" in head else _load_v5(path)


# ---- reading ----

def find_files(root: str | Path) -> list[Path]:
    return sorted((p for p in Path(root).rglob("*.mat") if _FILE_RE.match(p.name)),
                  key=lambda p: (int(_FILE_RE.match(p.name).group(1)[1:]), p.name))


def _parse_details(details: str) -> dt.date | None:
    m = _DETAILS_RE.search(details or "")
    return dt.date(int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None


def _channel_layout(sets, names, pid: str, hemisphere: str):
    """Multi-unit channel sets in column order: (columns, Channels)."""
    cols, ids, arrays, locs = [], [], [], []
    for idx, name in zip(sets, names):
        if name.endswith("Sorted"):
            continue
        array = name.split("-", 1)[1] if name.startswith(f"{pid}-") else name
        for i, c in enumerate(idx):
            cols.append(int(c) - 1)  # 1-based MATLAB indices
            ids.append(f"{array}-{i:02d}")
            arrays.append(array)
            if pid == "T15" and array in _T15_AREA:
                locs.append(f"{_T15_AREA[array]}, {hemisphere.lower()} precentral gyrus "
                            f"(inferred; see provenance['array_area_evidence'])")
            else:
                locs.append(f"precentral gyrus, {hemisphere.lower()} hemisphere (array '{array}' as named in the release)")
    if not cols:
        raise ValueError("no multi-unit channel sets")
    return np.array(cols), Channels(np.array(ids), np.array(arrays), np.array(locs))


def _trials_from_state(state: np.ndarray):
    """(start, go, end, truncated) bin indices of each Delay-Go-Return sequence in one block.

    A final Delay-Go that runs into the end of the block (no Return recorded) is kept,
    ending at the block end and marked truncated."""
    s = np.where(np.isnan(state), -1, state).astype(int)
    runs = []  # (value, start, stop)
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[i - 1]:
            runs.append((s[start], start, i))
            start = i
    out = []
    for k in range(len(runs) - 2):
        (a, a0, _), (b, b0, _), (c, _, c1) = runs[k], runs[k + 1], runs[k + 2]
        if (a, b, c) == (STATE_DELAY, STATE_GO, STATE_RETURN):
            out.append((a0, b0, c1, False))
    if len(runs) >= 2 and (runs[-2][0], runs[-1][0]) == (STATE_DELAY, STATE_GO):
        out.append((runs[-2][1], runs[-1][1], runs[-1][2], True))
    return out


def read_file(path: str | Path) -> list[Recording]:
    """Read one participant file into one Recording per block."""
    path = Path(path)
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Deo 2026 file: {path.name}")
    pid, control = m.group(1), bool(m.group(2))
    d = _load(path)
    if abs(float(d["binWidth"]) - BIN_S) > 1e-9:
        raise ValueError(f"{path.name}: binWidth {d['binWidth']}, expected {BIN_S}")

    tx = np.asarray(d["tx"])
    n = tx.shape[0]
    block = np.asarray(d["blockNum"]).ravel().astype(np.int64)
    state = np.asarray(d["state"], dtype=np.float64).ravel()
    go_cue = np.asarray(d["goCue"]).ravel().astype(np.int64) - 1  # 1-based -> 0-based
    cue = np.asarray(d["trialCue"]).ravel().astype(np.int64)
    cue_list = list(d["cueList"])
    if len(block) != n or len(state) != n or len(go_cue) != len(cue):
        raise ValueError(f"{path.name}: field lengths disagree")
    if cue.min() < 1 or cue.max() > len(cue_list):
        raise ValueError(f"{path.name}: trialCue outside cueList")

    hemisphere = str(d.get("hemisphere", "")).strip() or "unknown"
    cols, ch = _channel_layout(d["chanSets"], list(d["chanSetNames"]), pid, hemisphere)
    if cols.max() >= tx.shape[1]:
        raise ValueError(f"{path.name}: chanSets index beyond tx columns")
    date = _parse_details(str(d.get("details", "")))
    go_set = {int(g): i for i, g in enumerate(go_cue)}

    recs = []
    for b in np.unique(block):
        rows = np.flatnonzero(block == b)
        if np.any(np.diff(rows) != 1):
            raise ValueError(f"{path.name}: block {b} is not contiguous")
        r0 = rows[0]
        x = tx[rows][:, cols].astype(np.float32)
        seqs = _trials_from_state(state[rows])
        starts, ends, cols_out = [], [], {k: [] for k in ("cue_id", "cue", "go_time", "go_cue_time", "return_time", "truncated")}
        for a0, g0, c1, cut in seqs:
            gabs = r0 + g0
            # goCue of this trial: same bin, or (T15) one bin earlier
            i = go_set.get(gabs, go_set.get(gabs - 1))
            if i is None:
                continue  # sequence without a cue row (none seen in the release)
            starts.append(a0 * BIN_S)
            ends.append(c1 * BIN_S)
            ret = np.nan if cut else g0 + int(np.argmax(state[rows][g0:] == STATE_RETURN))
            cols_out["cue_id"].append(int(cue[i]))
            cols_out["cue"].append(cue_list[cue[i] - 1])
            cols_out["go_time"].append(g0 * BIN_S)
            cols_out["go_cue_time"].append((go_cue[i] - r0) * BIN_S)
            cols_out["return_time"].append(ret * BIN_S)
            cols_out["truncated"].append(cut)
        n_cues = int(np.sum((go_cue >= rows[0]) & (go_cue <= rows[-1])))
        if len(starts) != n_cues:
            raise ValueError(f"{path.name} block {b}: {len(starts)} state trials for {n_cues} goCue rows")
        trials = Trials(
            start=np.array(starts), end=np.array(ends),
            columns={"cue_id": np.array(cols_out["cue_id"]), "cue": np.array(cols_out["cue"], dtype=str),
                     "go_time": np.array(cols_out["go_time"]), "go_cue_time": np.array(cols_out["go_cue_time"]),
                     "return_time": np.array(cols_out["return_time"]),
                     "truncated": np.array(cols_out["truncated"], dtype=bool)},
            descriptions={
                "cue_id": "Movement condition number (index into the release's cueList, 1-based)",
                "cue": "Movement condition name",
                "go_time": "Start of the Go state (square turns green)",
                "go_cue_time": "Go time from the release's goCue field (in T15, one bin earlier in about half the trials)",
                "return_time": "Start of the Return state; NaN if the block ended first",
                "truncated": "True if the block ended before the trial's Return state",
            },
        ) if starts else None
        recs.append(Recording(
            dataset_id=DATASET_ID,
            session_id=f"{pid.lower()}_{date.isoformat() if date else 'nodate'}_block{int(b):02d}"
                       + ("_control" if control else ""),
            participant_id=pid,
            split=None,
            bin_size_s=BIN_S,
            timestamps=BIN_S * (np.arange(len(rows)) + 0.5),
            channels=ch,
            features=[Feature(
                name="threshold_crossings", data=x, unit="count",
                description="Binned threshold crossings per multi-unit channel per 20 ms bin "
                            "(threshold not stated in the release)",
                threshold_rms=None, normalization="none (raw counts)")],
            trials=trials,
            session_date=date,
            task="Instructed-delay cued movements across the whole body (text cue, then go)"
                 + (" (control experiment, limbs only)" if control else ""),
            timeline="continuous",
            provenance={
                "source": f"https://datadryad.org/dataset/{DOI}",
                "license": "CC0-1.0",
                "file": path.name,
                "block": int(b),
                "details": str(d.get("details", "")),
                "hemisphere": hemisphere,
                **({"array_area_evidence": T15_AREA_EVIDENCE} if pid == "T15" else {}),
                "time_origin": "start of the block (gaps between blocks were not recorded)",
                "citation": "Deo DR et al. A mosaic of whole-body representations on the human "
                            "precentral gyrus. 2026.",
                "experimenters": ["Deo, Darrel R."],
                "experiment_description": (
                    f"BrainGate2 participant {pid} attempted or made text-cued movements of the face, "
                    "head, arms, hands, legs and feet, and spoke cued words, after an instructed delay."
                ),
                "keywords": ["brain-computer interface", "precentral gyrus", "somatotopy", "BrainGate",
                             "intracortical"],
            },
        ))
    return recs


def iter_recordings(root: str | Path) -> Iterator[Recording]:
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no T*.mat files under {root}")
    for p in files:
        yield from read_file(p)


def read(root: str | Path) -> list[Recording]:
    return list(iter_recordings(root))


# Uniform per-file access. One file holds several blocks, so read_source returns a list.
def find_sources(root: str | Path) -> list[Path]:
    return find_files(root)


def read_source(path: str | Path) -> list[Recording]:
    return read_file(path)
