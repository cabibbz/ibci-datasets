"""Reader for Card et al. 2024 (NEJM), the Brain-to-Text '25 dataset.

Source: https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85 (CC0-1.0).
Layout after unzipping ``t15_copyTask_neuralData.zip``::

    hdf5_data_final/t15.YYYY.MM.DD/data_{train,val,test}.hdf5

Each HDF5 file holds one group per trial with ``input_features`` (T x 512), and, except
in the unlabeled test split, ``seq_class_ids`` (phoneme ids) and ``transcription``
(zero-terminated character codes). Group attributes: ``n_time_steps``, ``seq_len``,
``block_num``, ``trial_num``, ``session``, ``sentence_label``.

Per the dataset README, the 512 features are -4.5 x RMS threshold crossings and spike
band power for 256 electrodes on four 64-electrode arrays, in 20 ms bins, z-scored
against the preceding 20 trials. Features 0-255 are threshold crossings and 256-511 spike
band power, each in the array order ventral 6v, area 4, 55b, dorsal 6v.

Trials are released as separate segments, so a Recording lays them end to end on one
timeline with a gap between trials (``timeline = "concatenated-trials"``). Times are
therefore not wall-clock times.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from ..schema import Channels, Feature, Recording, Trials

DATASET_ID = "card2024_speech"
DOI = "doi:10.5061/dryad.dncjsxm85"
BIN_S = 0.020
TRIAL_GAP_S = 1.0
N_ELECTRODES = 256

# (array id, brain area), in feature order
ARRAYS = (
    ("v6v", "ventral 6v"),
    ("4", "area 4"),
    ("55b", "area 55b"),
    ("d6v", "dorsal 6v"),
)

# Phoneme id -> symbol, as used by seq_class_ids. Index 0 is the CTC blank and
# "|" marks a word boundary. Order follows the companion repo's model_training code.
PHONEMES = (
    "BLANK",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    "|",
)

_FILE_RE = re.compile(r"(t15\.\d{4}\.\d{2}\.\d{2})[/\\]data_(train|val|test)\.hdf5$")


def channels() -> Channels:
    ids, arrays, locations = [], [], []
    for array_id, area in ARRAYS:
        for i in range(N_ELECTRODES // len(ARRAYS)):
            ids.append(f"{array_id}-{i:02d}")
            arrays.append(array_id)
            locations.append(area)
    return Channels(ids=np.array(ids), arrays=np.array(arrays), locations=np.array(locations))


def find_files(root: str | Path) -> list[Path]:
    """All ``data_<split>.hdf5`` files under ``root``, sorted by session then split."""
    order = {"train": 0, "val": 1, "test": 2}
    found = [p for p in Path(root).rglob("data_*.hdf5") if _FILE_RE.search(p.as_posix())]

    def key(p: Path):
        m = _FILE_RE.search(p.as_posix())
        return (m.group(1), order[m.group(2)])

    return sorted(found, key=key)


def load_block_table(csv_path: str | Path) -> dict[tuple[str, int], dict[str, str]]:
    """Read the companion repo's ``t15_copyTaskData_description.csv`` (optional).

    It gives the post-implant day and the text corpus of each block. Returns
    ``{(YYYY-MM-DD, block): row}``.
    """
    table: dict[tuple[str, int], dict[str, str]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            table[(row["Date"], int(row["Block number"]))] = row
    return table


def _decode_text(codes: np.ndarray) -> str:
    codes = np.asarray(codes).astype(np.int64).ravel()
    end = np.flatnonzero(codes == 0)
    codes = codes[: end[0]] if len(end) else codes
    return "".join(chr(c) for c in codes)


def _decode_phonemes(ids: np.ndarray, seq_len: int | None) -> str:
    ids = np.asarray(ids).astype(np.int64).ravel()
    if seq_len is not None:
        ids = ids[: int(seq_len)]
    out = []
    for i in ids:
        if i == 0:
            continue
        if not 0 < i < len(PHONEMES):
            raise ValueError(f"phoneme id {i} out of range")
        out.append(PHONEMES[i])
    return " ".join(out)


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return "".join(v.decode() if isinstance(v, bytes) else str(v) for v in value.ravel())
    return str(value)


def read_file(path: str | Path, block_table: dict | None = None) -> Recording:
    """Read one ``data_<split>.hdf5`` file into a Recording."""
    path = Path(path)
    m = _FILE_RE.search(path.as_posix())
    if not m:
        raise ValueError(f"not a Card 2024 data file: {path}")
    session, split = m.group(1), m.group(2)
    date = dt.date(*map(int, session.split(".")[1:]))

    chunks, starts, ends = [], [], []
    cols: dict[str, list] = {"block": [], "trial": [], "sentence": [], "phonemes": [], "corpus": []}
    t = 0.0
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: (int(re.sub(r"\D", "", k) or 0), k))
        for key in keys:
            g = f[key]
            x = np.asarray(g["input_features"], dtype=np.float32)
            n = int(g.attrs.get("n_time_steps", x.shape[0]))
            if x.ndim != 2 or x.shape[1] != 2 * N_ELECTRODES or n != x.shape[0]:
                raise ValueError(f"{path}:{key}: unexpected input_features shape {x.shape}")
            if _attr_str(g.attrs.get("session", session)) != session:
                raise ValueError(f"{path}:{key}: session attribute does not match the folder")
            block = int(g.attrs["block_num"])
            chunks.append(x)
            starts.append(t)
            ends.append(t + n * BIN_S)
            t += n * BIN_S + TRIAL_GAP_S
            cols["block"].append(block)
            cols["trial"].append(int(g.attrs["trial_num"]))
            if "sentence_label" in g.attrs:
                sentence = _attr_str(g.attrs["sentence_label"])
            elif "transcription" in g:
                sentence = _decode_text(g["transcription"][()])
            else:
                sentence = ""
            cols["sentence"].append(sentence)
            if "seq_class_ids" in g:
                seq_len = g.attrs.get("seq_len")
                cols["phonemes"].append(_decode_phonemes(g["seq_class_ids"][()], seq_len))
            else:
                cols["phonemes"].append("")
            row = (block_table or {}).get((date.isoformat(), block), {})
            cols["corpus"].append(row.get("Corpus", ""))

    if not chunks:
        raise ValueError(f"{path}: no trials found")
    x = np.concatenate(chunks)
    start, end = np.array(starts), np.array(ends)
    timestamps = np.concatenate(
        [s + BIN_S * (np.arange(len(c)) + 0.5) for s, c in zip(start, chunks)]
    )

    post_implant_day = None
    if block_table:
        days = {r["Post-implant day"] for (d, _), r in block_table.items() if d == date.isoformat()}
        if len(days) == 1:
            post_implant_day = int(days.pop())

    norm = "z-scored against the preceding 20 trials (per the dataset README)"
    labeled = split != "test"
    trials = Trials(
        start=start,
        end=end,
        columns={
            "block": np.array(cols["block"], dtype=np.int64),
            "trial": np.array(cols["trial"], dtype=np.int64),
            "sentence": np.array(cols["sentence"], dtype=str),
            "phonemes": np.array(cols["phonemes"], dtype=str),
            "corpus": np.array(cols["corpus"], dtype=str),
        },
        descriptions={
            "block": "Block number within the session",
            "trial": "Trial number within the block",
            "sentence": "Cued sentence (empty in the unlabeled test split)",
            "phonemes": "ARPAbet phoneme sequence, '|' between words (empty in the test split)",
            "corpus": "Sentence corpus of the block, from the companion repo's block table",
        },
    )
    return Recording(
        dataset_id=DATASET_ID,
        session_id=f"{session}_{split}",
        participant_id="T15",
        split=split,
        bin_size_s=BIN_S,
        timestamps=timestamps,
        channels=channels(),
        features=[
            Feature(
                name="threshold_crossings",
                data=np.ascontiguousarray(x[:, :N_ELECTRODES]),
                unit="z-score",
                description="Binned -4.5 x RMS threshold crossings per electrode, 20 ms bins",
                threshold_rms=-4.5,
                normalization=norm,
            ),
            Feature(
                name="spike_band_power",
                data=np.ascontiguousarray(x[:, N_ELECTRODES:]),
                unit="z-score",
                description="Spike band power per electrode, 20 ms bins",
                normalization=norm,
            ),
        ],
        trials=trials,
        session_date=date,
        post_implant_day=post_implant_day,
        task="attempted speech, copy task",
        timeline="concatenated-trials",
        provenance={
            "source": f"https://datadryad.org/dataset/{DOI}",
            "license": "CC0-1.0",
            "file": f"{session}/data_{split}.hdf5",
            "labeled": labeled,
            "trial_gap_s": TRIAL_GAP_S,
            "citation": "Card NS et al. An accurate and rapidly calibrating speech "
                        "neuroprosthesis. N Engl J Med 2024;391:609-618.",
            "doi": "doi:10.1056/NEJMoa2314132",
            "experimenters": ["Card, Nicholas S.", "Stavisky, Sergey D.", "Brandman, David M."],
            "institution": "University of California, Davis",
            "experiment_description": (
                "BrainGate2 participant T15 (ALS) attempted to speak cued sentences "
                "(copy task) while 256 electrodes on four arrays recorded from speech motor cortex."
            ),
            "keywords": ["brain-computer interface", "speech neuroprosthesis", "BrainGate",
                         "intracortical", "ALS"],
        },
    )


def iter_recordings(root: str | Path, block_table_csv: str | Path | None = None) -> Iterator[Recording]:
    """Yield each session and split under ``root``, one file in memory at a time."""
    table = load_block_table(block_table_csv) if block_table_csv else None
    files = find_files(root)
    if not files:
        raise FileNotFoundError(f"no t15.*/data_*.hdf5 files under {root}")
    for p in files:
        yield read_file(p, table)


def read(root: str | Path, block_table_csv: str | Path | None = None) -> list[Recording]:
    """Read every session and split under ``root``."""
    return list(iter_recordings(root, block_table_csv))


# Uniform per-file access, used by tools that select files before loading them.
def find_sources(root: str | Path) -> list[Path]:
    return find_files(root)


def read_source(path: str | Path, block_table_csv: str | Path | None = None) -> Recording:
    return read_file(path, load_block_table(block_table_csv) if block_table_csv else None)
