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


# ---- Singer-Clark 2025: one MATLAB v5 file per block, fields as in the dataset README ----

SC_LABELS = ["v6v"] * 64 + ["4"] * 64 + ["55b"] * 64 + ["d6v"] * 64


def _write_sc_block(path, n_bins, trial_starts, speech=False, grid=False, seed=0):
    import scipy.io

    rng = np.random.default_rng(seed)
    trial_idx = np.zeros(n_bins, dtype=np.int16)
    for i, s in enumerate(trial_starts):
        trial_idx[s:] = i
    targets = rng.uniform(-0.4, 0.4, (len(trial_starts), 2)).astype(np.float32)
    target_position = targets[trial_idx]
    d = {
        "timestamp_sec": (np.arange(n_bins) * 0.01)[None, :],
        "threshold_crossings": rng.integers(0, 6, (n_bins, 256)).astype(np.int16),
        "spike_band_power": rng.uniform(50, 5000, (n_bins, 256)).astype(np.float32),
        "assist_amount": np.linspace(1, 0, n_bins)[None, :],
        "cursor_position": rng.uniform(-0.4, 0.4, (n_bins, 2)).astype(np.float32),
        "target_position": target_position,
        "trial_idx": trial_idx[None, :],
        "cursor_decoder_output": rng.standard_normal((n_bins, 2)).astype(np.float32),
        "trial_start_bin": np.array(trial_starts, dtype=np.int16)[None, :],
        "cursor_radius": np.array([[0.025]]),
        "array_label_by_electrode": np.array(SC_LABELS),
    }
    if grid:
        d["click_decoder_output"] = rng.integers(0, 2, n_bins).astype(np.uint8)[None, :]
        d["click_assist"] = np.zeros((1, n_bins), dtype=np.uint8)
        d["grid_num_rows"] = np.array([[6.0]])
        d["grid_total_height"] = np.array([[0.9]])
    else:
        d["target_radius"] = np.array([[0.05]])
        d["dwell_requirement_sec"] = np.array([[1.0]])
    if speech:
        n = len(trial_starts)
        starts = np.array(trial_starts)
        d["target_presentation_bin"] = (starts + 10).astype(np.int16)[None, :]
        d["cursor_go_cue_bin"] = (starts + 20).astype(np.int16)[None, :]
        beep = (starts + 30).astype(np.int16)
        beep[1::2] = -1
        d["speech_go_cue_bin"] = beep[None, :]
        ends = np.append(starts[1:] - 1, n_bins - 1)
        d["trial_end_bin"] = ends.astype(np.int16)[None, :]
        prompts = np.empty((1, n), dtype=object)
        for i, w in enumerate(["kite", "choice", "though", "word"][:n]):
            prompts[0, i] = np.array([w])
        d["speech_prompt"] = prompts
        d["is_control_block"] = np.array([[1]], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), d)


@pytest.fixture
def sc_root(tmp_path):
    root = tmp_path / "dryad_files"
    _write_sc_block(root / "t15_day00039_block00_radial8_calibration_task.mat", 400, [0, 120, 250])
    _write_sc_block(root / "t15_day00202_block02_simultaneous_speech_and_cursor_task.mat", 500,
                    [32, 150, 300, 420], speech=True, seed=1)
    _write_sc_block(root / "t15_day00468_block03_grid_evaluation_task.mat", 300, [0, 100, 200],
                    grid=True, seed=2)
    (root / "t15_day00202_words.mat").write_bytes(b"undocumented")  # must be skipped
    return root


# ---- Srinivasan 2026: per-trial cell arrays in one MATLAB v5 file per block ----

def _cellrow(items):
    out = np.empty((1, len(items)), dtype=object)
    for i, x in enumerate(items):
        out[0, i] = x
    return out


def _write_sri_word(path, cues, seed=0):
    import scipy.io

    rng = np.random.default_rng(seed)
    n = len(cues)
    smooth_len = [900, 901, 900][:n] + [900] * max(0, n - 3)
    d = {
        "cue": np.array(cues),
        "delay_duration_ms": np.full((1, n), 2000, dtype=np.int64),
        "threshcross": _cellrow([rng.standard_normal((L, 256)).astype(np.float32) for L in smooth_len]),
        "spikepow": _cellrow([rng.standard_normal((L, 256)) for L in smooth_len]),  # float64, as in T16
        "raw_threshcross": _cellrow([rng.integers(0, 5, (L - 300, 256)).astype(np.int16) for L in smooth_len]),
        "speech_onsets": _cellrow([np.array([[45000 if c != "DO NOTHING" else 91]], dtype=np.int64) for c in cues]),
        "speech_offsets": _cellrow([np.array([[72000 if c != "DO NOTHING" else 120]], dtype=np.int64) for c in cues]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), d)


def _write_sri_sentence(path, seed=0):
    import scipy.io

    rng = np.random.default_rng(seed)
    lens = [537, 686]
    d = {
        "cue": np.array(["i mean we ARE talking.", "POTATOES and ONIONS please."]),
        "threshcross": _cellrow([rng.standard_normal((L, 256)).astype(np.float32) for L in lens]),
        "spikepow": _cellrow([rng.standard_normal((L, 256)).astype(np.float32) for L in lens]),
        "decoder_final_output": _cellrow([np.array(["i mean, we are talking."]), np.array(["potatoes and onions, please."])]),
        "updated_decoder_final_output": _cellrow([np.array(["i mean, we ARE talking."]), np.array(["POTATOES and ONIONS, please."])]),
        "logits": _cellrow([rng.standard_normal((L // 2, 41)).astype(np.float32) for L in lens]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), d)


@pytest.fixture
def sri_root(tmp_path):
    root = tmp_path / "srinivasan"
    _write_sri_word(root / "t15_word-loudness_block_6.mat", ["WHISPER: have", "LOUD: do", "DO NOTHING"])
    _write_sri_word(root / "t16_word-loudness_block_3.mat", ["MIME: be", "NORMAL: going"], seed=1)
    _write_sri_sentence(root / "t15_sentence-loudness_block_11.mat", seed=2)
    return root
