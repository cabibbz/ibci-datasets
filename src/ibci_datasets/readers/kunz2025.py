"""Reader for Kunz et al. 2025, inner speech in motor cortex.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j (CC0-1.0). No companion
code is linked. Eleven task zips, each with a readme inside; MATLAB v5 files named
``t<P>.<YYYY.MM.DD>_<description>.mat`` for T12, T15, T16 and T17.

Two layouts, both with ``binnedTX`` (threshold crossings) and ``spikePow`` (spike band
power, uV^2), ``goTrialEpochs`` / ``delayTrialEpochs`` (1-based inclusive bin ranges),
``chanSets`` / ``chanSetNames`` (arrays, named by the HCP parcellation) and ``binSize``:

* task folders (isolated / interleaved verbal behaviours, conjunctive counting, six
  sequence-recall tasks): ``trialCues`` + ``cueList``, or ``numTargets``;
* sentence folders (``sentenceData``, ``chittychittybangbang``): ``sentenceText``,
  ``decodedSentenceText``, ``sentenceDurations`` and a convenience copy
  ``sentenceFeatures`` (not converted).

Features are denoised by linear-regression referencing; the sequence-recall files are
also block-mean subtracted. Most files have ``blockNum``; each block becomes one
continuous Recording (data between blocks was not recorded). The sequence-recall files
have no block markers, so each is one Recording whose block boundaries are unknown.

Findings from the data, beyond the readmes (see ERRATA.md):

* **uint8 wraparound.** Three T12 sequence-recall files store block-mean-subtracted
  crossings as uint8, so -1..-7 became 255..249 (about 20% of values). The reader reads
  them back as int8 (detected by the empty gap between small and near-255 values) and
  records the fraction.
* **Copied array labels.** The T12 ``chittychittybangbang`` files label their four
  64-channel blocks ``i6v, M1, 55b, s6v``, T15's names. Every other T12 file names only
  ``i6v`` (1-64) and ``s6v`` (65-128); T12 has no M1 or 55b array. Matching channel-
  correlation structure against two other T12 files puts the block labelled ``M1`` at
  ``s6v`` (both scorings, margin ~0.15). The reader relabels blocks 2-4 as ``s6v``,
  ``unnamed-1``, ``unnamed-2`` and records the release's labels.
* **Bin size metadata.** ``t12.2024.12.19_..._eval.mat`` says 25 ms and
  ``t16.2024.04.29_imagined_50words_eval.mat`` 33 ms, against 20 ms for the other files of
  those sessions (the readmes say 10 or 20). Crossings per bin match the 20 ms files
  (T16: 1.06x, where 33 ms bins would give 1.65x) and durations then line up, so 20 ms is
  likely; the reader keeps the stated value and flags it in the provenance.
* **T12 sentenceData files (2023-11-28).** ``binnedTX`` holds only the 128 named
  channels, and ``sentenceFeatures`` (256 channels) does not match it at the go cue,
  contrary to the readme. In the other sentence files it does.
* ``t16.2024.12.18_imagined_125kwords.mat`` has 123 decoded sentences for 125 trials, with
  no way to tell which trials lack one; that column is omitted for this file.
* ``isolatedVerbalBehaviors/*_allBehaviors_raw.mat`` repeats the seven per-behaviour files of
  its session exactly (same blocks, features and epochs; cues suffixed ``-<behaviour>``).
  ``find_files`` skips it when those seven files are present, so nothing is converted twice.
* Go epochs are fixed-length windows that often run into the next trial's delay; some
  first or last trials start before or end after the recording and are clipped and
  marked ``truncated``.

Array ids are made consistent per participant (originals in provenance): T16 ``PEF`` and
``PEF/55b`` -> ``PEF-55b``; ``ventral6v`` -> ``i6v``; the 128-channel ``6d`` set -> ``i6d``
(129-192) and ``s6d`` (193-256), as the sequence-recall files name them. Other 128-channel
sets are split into ``<name>-1`` / ``<name>-2`` (assuming two 64-channel arrays). Columns
outside any set are ``unnamed-1``, ``unnamed-2``...
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Channels, Feature, Recording, Trials

DATASET_ID = "kunz2025_inner_speech"
DOI = "doi:10.5061/dryad.gf1vhhn1j"
ARRAY = 64

_FILE_RE = re.compile(r"^t(12|15|16|17)\.(\d{4})\.(\d{2})\.(\d{2})_(.+)\.mat$")
_RENAME = {"PEF": "PEF-55b", "PEF/55b": "PEF-55b", "ventral6v": "i6v"}
_SPLIT = {("T16", "6d"): ("i6d", "s6d")}
# The T12 keyword-task files carry T15's labels; see the module docstring.
_T12_KEYWORD_LABELS = ("i6v", "s6v", "unnamed-1", "unnamed-2")
# allBehaviors_raw.mat is these seven files' blocks concatenated, byte for byte, with "-<behaviour>" on each cue.
_BEHAVIOURS = ("attempted", "mouthed", "imagined1stMotor", "imagined1stAuditory", "imaginedListening", "listening",
               "reading")
_BIN_SIZE_SUSPECT = {"t12.2024.12.19_chittychittybangbang_50words_eval.mat",
                     "t16.2024.04.29_imagined_50words_eval.mat"}


def _loadmat(path: Path) -> dict:
    import scipy.io

    return scipy.io.loadmat(str(path))


def find_files(root: str | Path) -> list[Path]:
    """Release files, minus AppleDouble forks and ``allBehaviors`` files whose per-behaviour files are present."""
    files = [p for p in Path(root).rglob("*.mat") if _FILE_RE.match(p.name) and not p.name.startswith("._")]
    names = {(p.parent, p.name) for p in files}

    def duplicate(p: Path) -> bool:
        if not p.name.endswith("_allBehaviors_raw.mat"):
            return False
        prefix = p.name[: -len("allBehaviors_raw.mat")]
        return all((p.parent, f"{prefix}{b}_raw.mat") in names for b in _BEHAVIOURS)

    return sorted((p for p in files if not duplicate(p)), key=lambda p: (p.parent.name, p.name))


def _strings(v) -> list[str]:
    out = []
    for x in np.asarray(v).ravel():
        flat = np.asarray(x).ravel()
        out.append(" ".join(str(s) for s in flat).strip() if flat.size else "")
    return out


def _fix_wraparound(tx: np.ndarray) -> tuple[np.ndarray, float | None]:
    """uint8 that should be int8: small values plus a cluster just below 256, nothing between."""
    if tx.dtype != np.uint8:
        return tx, None
    v = np.unique(tx)
    lo, hi = v[v < 128], v[v >= 128]
    if len(hi) and len(lo) and hi.min() - lo.max() > 64:
        return tx.view(np.int8), float((tx >= 128).mean())
    return tx, None


def _layout(pid: str, names: list[str], sets: list[np.ndarray], n_cols: int, keyword_t12: bool):
    """Return (column order, array ids, original label per column)."""
    blocks = []  # (array id, 0-based columns, original label)
    if keyword_t12:
        if [len(s) for s in sets] != [ARRAY] * 4 or n_cols != 4 * ARRAY:
            raise ValueError("T12 keyword-task file with an unexpected layout")
        for lab, s, orig in zip(_T12_KEYWORD_LABELS, sets, names):
            blocks.append((lab, np.asarray(s) - 1, orig))
    else:
        for name, s in zip(names, sets):
            s = np.asarray(s).ravel() - 1
            nm = _RENAME.get(name, name)
            if len(s) == ARRAY:
                blocks.append((nm, s, name))
            elif len(s) == 2 * ARRAY:
                a, b = _SPLIT.get((pid, name), (f"{nm}-1", f"{nm}-2"))
                blocks += [(a, s[:ARRAY], name), (b, s[ARRAY:], name)]
            else:
                raise ValueError(f"array set {name!r} has {len(s)} channels")
        used = np.concatenate([b for _, b, _ in blocks])
        rest = np.setdiff1d(np.arange(n_cols), used)
        for k in range(0, len(rest), ARRAY):
            blocks.append((f"unnamed-{k // ARRAY + 1}", rest[k:k + ARRAY], "(not named in the release)"))
    cols = np.concatenate([b for _, b, _ in blocks])
    if len(np.unique(cols)) != len(cols) or cols.max() >= n_cols:
        raise ValueError("chanSets overlap or exceed the feature columns")
    ids, arrays, locs, origs = [], [], [], {}
    for array, b, orig in blocks:
        origs[array] = orig
        for i in range(len(b)):
            ids.append(f"{array}-{i:02d}")
            arrays.append(array)
            locs.append(f"{array} (HCP-parcellation name from the release)" if not array.startswith("unnamed")
                        else "array not named in the release")
    return cols, Channels(np.array(ids), np.array(arrays), np.array(locs)), origs


def read_file(path: str | Path) -> list[Recording]:
    path = Path(path)
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a Kunz 2025 file: {path.name}")
    pid = f"T{m.group(1)}"
    date = dt.date(int(m.group(2)), int(m.group(3)), int(m.group(4)))
    desc, folder = m.group(5), path.parent.name
    d = _loadmat(path)

    tx_raw = np.asarray(d["binnedTX"])
    tx, wrapped = _fix_wraparound(tx_raw)
    sbp = np.asarray(d["spikePow"])
    n, n_cols = tx.shape
    if sbp.shape != tx.shape:
        raise ValueError(f"{path.name}: binnedTX {tx.shape} vs spikePow {sbp.shape}")
    bin_s = float(np.asarray(d["binSize"]).ravel()[0]) / 1000.0
    names = [s.strip() for s in _strings(d["chanSetNames"])]
    cs = np.asarray(d["chanSets"])
    sets = [np.asarray(x).ravel().astype(np.int64) for x in (cs.ravel() if cs.dtype == object else cs)]
    keyword_t12 = pid == "T12" and folder == "chittychittybangbang"
    cols, ch, origs = _layout(pid, names, sets, n_cols, keyword_t12)

    delay = np.asarray(d["delayTrialEpochs"]).reshape(-1, 2).astype(np.int64)
    go = np.asarray(d["goTrialEpochs"]).reshape(-1, 2).astype(np.int64)
    if len(delay) != len(go):
        raise ValueError(f"{path.name}: {len(delay)} delay epochs, {len(go)} go epochs")
    k = len(go)
    per_trial: dict[str, tuple[np.ndarray, str]] = {}
    if "trialCues" in d and "cueList" in d:
        cue_ids = np.asarray(d["trialCues"]).ravel().astype(np.int64)
        cue_list = _strings(d["cueList"])
        if len(cue_ids) != k or cue_ids.min() < 1 or cue_ids.max() > len(cue_list):
            raise ValueError(f"{path.name}: trialCues inconsistent with cueList / epochs")
        per_trial["cue_id"] = (cue_ids, "Cue number (1-based index into the release's cueList)")
        per_trial["cue"] = (np.array([cue_list[i - 1] for i in cue_ids], dtype=str), "Cue text")
    if "sentenceText" in d:
        per_trial["sentence"] = (np.array(_strings(d["sentenceText"]), dtype=str), "Cued sentence")
    if "decodedSentenceText" in d:
        per_trial["decoded_sentence"] = (np.array(_strings(d["decodedSentenceText"]), dtype=str),
                                         "Final closed-loop decoded sentence; empty in open-loop blocks")
    if "sentenceDurations" in d:
        per_trial["sentence_duration_s"] = (np.asarray(d["sentenceDurations"]).ravel().astype(np.float64) * bin_s,
                                            "Go-period (sentence production) duration from sentenceDurations")
    if "numTargets" in d:
        per_trial["num_targets"] = (np.asarray(d["numTargets"]).ravel().astype(np.int64), "Correct count for the trial")
    dropped = {}
    if "decoded_sentence" in per_trial and len(per_trial["decoded_sentence"][0]) != k:
        # Seen once (t16.2024.12.18_imagined_125kwords: 123 rows for 125 trials). Which trials
        # lack an entry is not recorded, so aligning them would be a guess; drop the column.
        dropped["decoded_sentence"] = f"decodedSentenceText has {len(per_trial['decoded_sentence'][0])} rows for {k} trials; omitted"
        del per_trial["decoded_sentence"]
    for key, (v, _) in per_trial.items():
        if len(v) != k:
            raise ValueError(f"{path.name}: {key} has {len(v)} rows for {k} trials")

    block = np.asarray(d["blockNum"]).ravel().astype(np.int64) if "blockNum" in d else np.zeros(n, np.int64)
    if len(block) != n:
        raise ValueError(f"{path.name}: blockNum length {len(block)} != {n}")
    has_blocks = "blockNum" in d
    go0 = go[:, 0] - 1  # 0-based first go bin

    provenance_base = {
        "source": f"https://datadryad.org/dataset/{DOI}",
        "license": "CC0-1.0",
        "file": f"{folder}/{path.name}",
        "task_folder": folder,
        "condition": desc,
        "release_array_labels": origs,
        "citation": "Kunz EM et al. Inner speech in motor cortex and implications for speech "
                    "neuroprostheses. Cell 2025.",
        "experimenters": ["Kunz, Erin M.", "Willett, Francis R."],
        "experiment_description": (f"BrainGate2 participant {pid}: instructed-delay inner, attempted and "
                                   "perceived speech tasks and cognitive tasks."),
        "keywords": ["brain-computer interface", "inner speech", "speech", "BrainGate", "intracortical"],
    }
    if wrapped is not None:
        provenance_base["uint8_wraparound_fixed"] = {"fraction_of_values": round(wrapped, 4),
                                                     "note": "binnedTX stored as uint8; read as int8"}
    if keyword_t12:
        provenance_base["relabelled"] = ("release labels i6v, M1, 55b, s6v are T15's; blocks 2-4 relabelled "
                                         "s6v, unnamed-1, unnamed-2 (verify-mapping vs other T12 files)")
    if path.name in _BIN_SIZE_SUSPECT:
        provenance_base["bin_size_suspect"] = ("binSize disagrees with the session's other files (20 ms); "
                                               "crossings per bin and durations suggest 20 ms")
    if not has_blocks:
        provenance_base["block_boundaries"] = "not marked in the release (blocks concatenated)"
    if dropped:
        provenance_base["dropped_columns"] = dropped

    recs = []
    for b in np.unique(block):
        rows = np.flatnonzero(block == b)
        if np.any(np.diff(rows) != 1):
            raise ValueError(f"{path.name}: block {b} is not contiguous")
        r0, r1 = rows[0], rows[-1] + 1
        mine = np.flatnonzero((go0 >= r0) & (go0 < r1))
        trials = None
        if len(mine):
            s = delay[mine, 0] - 1  # 0-based first delay bin
            e = go[mine, 1]  # 1-based inclusive end == 0-based exclusive end
            cut = (s < r0) | (e > r1)
            s, e = np.clip(s, r0, r1), np.clip(e, r0, r1)
            columns = {"go_time": (go0[mine] - r0) * bin_s, "truncated": cut}
            descriptions = {"go_time": "Go cue (start of goTrialEpochs)",
                            "truncated": "Delay or go window extends past the recording and was clipped"}
            for key, (v, desc_k) in per_trial.items():
                columns[key] = v[mine]
                descriptions[key] = desc_k
            trials = Trials(start=(s - r0) * bin_s, end=(e - r0) * bin_s, columns=columns, descriptions=descriptions)
        x_tx = tx[r0:r1][:, cols].astype(np.float32)
        x_sp = sbp[r0:r1][:, cols].astype(np.float32)
        mean_sub = folder.startswith("seqRecall")
        norm = "linear-regression referenced" + ("; block-wise mean subtracted" if mean_sub else "")
        recs.append(Recording(
            dataset_id=DATASET_ID,
            session_id=f"{pid.lower()}_{date.isoformat()}_{folder}_{desc}" + (f"_block{int(b):02d}" if has_blocks else ""),
            participant_id=pid,
            split=None,
            bin_size_s=bin_s,
            timestamps=bin_s * (np.arange(r1 - r0) + 0.5),
            channels=ch,
            features=[
                Feature("threshold_crossings", x_tx, "count" + (" (mean-subtracted)" if mean_sub else ""),
                        "Binned threshold crossings (-3.5 or -4.5 x RMS, per participant)", normalization=norm),
                Feature("spike_band_power", x_sp, "uV^2" + (" (mean-subtracted)" if mean_sub else ""),
                        "Spike band power: mean squared high-passed (250 Hz) voltage per bin", normalization=norm),
            ],
            trials=trials,
            session_date=date,
            task=f"{folder}: {desc}",
            timeline="continuous",
            provenance={**provenance_base, "block": int(b) if has_blocks else None,
                        "time_origin": "start of the block" if has_blocks else "start of the file"},
        ))
    return recs


def iter_recordings(root: str | Path) -> Iterator[Recording]:
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no t<P>.<date>_*.mat files under {root}")
    for p in files:
        yield from read_file(p)


def read(root: str | Path) -> list[Recording]:
    return list(iter_recordings(root))


def find_sources(root: str | Path) -> list[Path]:
    return find_files(root)


def read_source(path: str | Path) -> list[Recording]:
    return read_file(path)
