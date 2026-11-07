"""Synthetic fixtures that mirror the documented release layouts. No real data is used."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

SENTENCES = ["hello world", "how are you", "i am fine"]
# ARPAbet ids from the reader's PHONEMES table: HH=16 AH=3 L=21 OW=25 |=40 W=36 ER=12 D=9
PHONEME_IDS = [16, 3, 21, 25, 40, 36, 12, 21, 9]


def _write_card_file(path, session, labeled, lengths, block=2, seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i, n in enumerate(lengths):
            g = f.create_group(f"trial_{i:04d}")
            g.create_dataset("input_features", data=rng.standard_normal((n, 512)).astype(np.float32))
            g.attrs["n_time_steps"] = n
            g.attrs["block_num"] = block
            g.attrs["trial_num"] = i
            g.attrs["session"] = session
            if labeled:
                text = SENTENCES[i % len(SENTENCES)]
                codes = np.zeros(500, dtype=np.int32)
                codes[: len(text)] = [ord(c) for c in text]
                g.create_dataset("transcription", data=codes)
                ids = np.zeros(500, dtype=np.int32)
                ids[: len(PHONEME_IDS)] = PHONEME_IDS
                g.create_dataset("seq_class_ids", data=ids)
                g.attrs["seq_len"] = len(PHONEME_IDS)
                g.attrs["sentence_label"] = text


@pytest.fixture
def card_root(tmp_path):
    root = tmp_path / "hdf5_data_final"
    _write_card_file(root / "t15.2023.08.11" / "data_train.hdf5", "t15.2023.08.11", True, [50, 30, 40])
    _write_card_file(root / "t15.2023.08.11" / "data_val.hdf5", "t15.2023.08.11", True, [20], block=3)
    _write_card_file(root / "t15.2023.08.11" / "data_test.hdf5", "t15.2023.08.11", False, [25, 35], block=4)
    csv = tmp_path / "blocks.csv"
    csv.write_text(
        "Date,Post-implant day,Block number,Number of sentences,Corpus,Split\n"
        "2023-08-11,25,2,3,50-Word,Train\n"
        "2023-08-11,25,3,1,Switchboard,Val\n"
        "2023-08-11,25,4,2,Switchboard,Test\n"
    )
    return root, csv
