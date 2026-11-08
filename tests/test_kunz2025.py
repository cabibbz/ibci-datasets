import numpy as np
import pytest
import scipy.io

from ibci_datasets import catalog
from ibci_datasets.readers import kunz2025


def _cells(items):
    out = np.empty((len(items), 1), dtype=object)
    for i, x in enumerate(items):
        out[i, 0] = np.asarray(x)
    return out


def _write(path, n=600, n_cols=256, sets=None, names=None, binsize=20, blocks=True, tx=None, extra=None,
           delay=((1, 100), (151, 250), (301, 400)), go=((101, 150), (251, 300), (401, 450))):
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "binnedTX": rng.integers(0, 5, (n, n_cols)).astype(np.uint8) if tx is None else tx,
        "spikePow": rng.uniform(10, 1e4, (n, n_cols)).astype(np.float32),
        "binSize": np.array([[binsize]]),
        "chanSets": np.array(sets if sets is not None else [np.arange(1, 65), np.arange(65, 129),
                                                            np.arange(129, 193), np.arange(193, 257)]),
        "chanSetNames": np.array(names if names is not None else ["i6v", "M1", "55b", "s6v"]),
        "delayTrialEpochs": np.array(delay, dtype=np.int64),
        "goTrialEpochs": np.array(go, dtype=np.int64),
    }
    if blocks:
        d["blockNum"] = np.repeat([3, 4], n // 2)[:, None]
    d.update(extra or {"trialCues": np.array([[1, 2, 1]]), "cueList": _cells(["kite", "Do Nothing"])})
    scipy.io.savemat(str(path), d)
    return path


def test_task_file_blocks_trials_and_channels(tmp_path):
    p = _write(tmp_path / "isolatedVerbalBehaviors" / "t15.2024.04.07_attempted_raw.mat")
    recs = kunz2025.read_file(p)
    assert [r.session_id for r in recs] == ["t15_2024-04-07_isolatedVerbalBehaviors_attempted_raw_block03",
                                            "t15_2024-04-07_isolatedVerbalBehaviors_attempted_raw_block04"]
    r3, r4 = recs
    assert r3.channels.ids[[0, 64, 128, 192]].tolist() == ["i6v-00", "M1-00", "55b-00", "s6v-00"]
    assert r3.bin_size_s == 0.02 and r3.session_date.isoformat() == "2024-04-07"
    np.testing.assert_allclose(r3.trials.start, [0.0, 3.0])  # 1-based inclusive epochs
    np.testing.assert_allclose(r3.trials.columns["go_time"], [2.0, 5.0])
    np.testing.assert_allclose(r3.trials.end, [3.0, 6.0])
    assert r3.trials.columns["cue"].tolist() == ["kite", "Do Nothing"]
    assert len(r4.trials) == 1 and r4.trials.columns["go_time"][0] == pytest.approx(2.0)


def test_uint8_wraparound_is_repaired(tmp_path):
    true = np.random.default_rng(1).integers(-6, 9, (600, 256)).astype(np.int8)
    p = _write(tmp_path / "seqRecallUninstructed3ElementLines" / "t12.2023.09.19_binnedTX_spikePow_20ms_lines.mat",
               tx=true.view(np.uint8), blocks=False, sets=[np.arange(1, 65), np.arange(65, 129)], names=["i6v", "s6v"])
    (rec,) = kunz2025.read_file(p)
    np.testing.assert_array_equal(rec.feature("threshold_crossings").data[:, :64], true[:, :64].astype(np.float32))
    assert rec.provenance["uint8_wraparound_fixed"]["fraction_of_values"] > 0.3
    assert rec.channels.arrays.tolist()[128] == "unnamed-1" and "block_boundaries" in rec.provenance


def test_t16_names_normalised_and_6d_split(tmp_path):
    p = _write(tmp_path / "sentenceData" / "t16.2024.04.29_attempted_50words.mat",
               sets=np.array([np.arange(1, 65), np.arange(65, 129), np.arange(129, 257)], dtype=object),
               names=np.array(["PEF", "ventral6v", "6d"], dtype=object),
               extra={"sentenceText": _cells(["a", "b", "c"]), "decodedSentenceText": _cells(["", "", ""]),
                      "sentenceDurations": np.array([[50], [50], [50]])})
    r = kunz2025.read_file(p)[0]
    assert list(dict.fromkeys(r.channels.arrays.tolist())) == ["PEF-55b", "i6v", "i6d", "s6d"]
    assert r.provenance["release_array_labels"]["i6d"] == "6d"
    assert r.trials.columns["sentence"].tolist() == ["a", "b"]


def test_t12_keyword_labels_replaced(tmp_path):
    p = _write(tmp_path / "chittychittybangbang" / "t12.2024.12.06_chittychittybangbang_50words.mat",
               extra={"sentenceText": _cells(["a", "b", "c"]), "decodedSentenceText": _cells(["", "", ""]),
                      "sentenceDurations": np.array([[50], [50], [50]])})
    r = kunz2025.read_file(p)[0]
    assert list(dict.fromkeys(r.channels.arrays.tolist())) == ["i6v", "s6v", "unnamed-1", "unnamed-2"]
    assert r.provenance["release_array_labels"]["s6v"] == "M1" and "relabelled" in r.provenance


def test_edge_trials_clipped_and_marked(tmp_path):
    p = _write(tmp_path / "seqRecallSpeech" / "t16.2024.07.17_binnedTX_spikePow_20ms.mat", blocks=False,
               tx=np.zeros((600, 256)), delay=((-49, 50), (301, 400)), go=((51, 200), (401, 650)),
               extra={"trialCues": np.array([[1, 2]]), "cueList": _cells(["UP", "Do Nothing"])})
    tr = kunz2025.read_file(p)[0].trials
    assert tr.columns["truncated"].tolist() == [True, True]
    np.testing.assert_allclose([tr.start[0], tr.end[1]], [0.0, 12.0])


def test_misaligned_decoded_text_is_dropped(tmp_path):
    p = _write(tmp_path / "sentenceData" / "t16.2024.12.18_imagined_125kwords.mat",
               extra={"sentenceText": _cells(["a", "b", "c"]), "decodedSentenceText": _cells(["x", "y"]),
                      "sentenceDurations": np.array([[50], [50], [50]])})
    r = kunz2025.read_file(p)[0]
    assert "decoded_sentence" not in r.trials.columns and "dropped_columns" in r.provenance


def test_skips_appledouble_files(tmp_path):
    _write(tmp_path / "sentenceData" / "t15.2024.03.03_attempted_50words.mat",
           extra={"sentenceText": _cells(["a", "b", "c"]), "sentenceDurations": np.array([[5], [5], [5]])})
    (tmp_path / "sentenceData" / "._t15.2024.03.03_attempted_50words.mat").write_bytes(b"\x00\x05\x16\x07")
    assert [p.name for p in kunz2025.find_files(tmp_path)] == ["t15.2024.03.03_attempted_50words.mat"]


def test_allbehaviors_skipped_only_when_per_behaviour_files_exist(tmp_path):
    folder = tmp_path / "isolatedVerbalBehaviors"
    _write(folder / "t16.2024.03.04_allBehaviors_raw.mat")
    assert [p.name for p in kunz2025.find_files(tmp_path)] == ["t16.2024.03.04_allBehaviors_raw.mat"]
    for b in kunz2025._BEHAVIOURS:
        _write(folder / f"t16.2024.03.04_{b}_raw.mat")
    names = [p.name for p in kunz2025.find_files(tmp_path)]
    assert len(names) == 7 and not any("allBehaviors" in n for n in names)


def test_catalog_points_at_reader():
    assert catalog.get("kunz2025_inner_speech").reader == "kunz2025"
