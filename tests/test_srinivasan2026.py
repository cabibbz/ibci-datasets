import numpy as np
import pytest

from ibci_datasets import catalog
from ibci_datasets.readers import srinivasan2026 as sri


def _by_id(root):
    return {r.session_id: r for r in sri.read(root)}


def test_word_block_layout_and_events(sri_root):
    rec = _by_id(sri_root)["t15_word-loudness_block06"]
    assert rec.participant_id == "T15" and rec.timeline == "concatenated-trials"
    tr = rec.trials
    np.testing.assert_allclose(tr.start, [0.0, 10.0, 20.01])  # 9 s / 9.01 s trials + 1 s gaps
    c = tr.columns
    assert c["loudness"].tolist() == ["WHISPER", "LOUD", "DO NOTHING"]
    assert c["word"].tolist() == ["have", "do", ""]
    np.testing.assert_allclose(c["go_cue_time"] - tr.start, 3.0)  # 1 s pre-delay + 2 s delay
    np.testing.assert_allclose(c["speech_onset_time"][:2] - c["go_cue_time"][:2], 1.5)  # 45000 / 30 kHz
    assert np.isnan(c["speech_onset_time"][2]) and np.isnan(c["speech_offset_time"][2])


def test_raw_counts_padded_to_smoothed_length(sri_root):
    rec = _by_id(sri_root)["t15_word-loudness_block06"]
    raw = rec.feature("threshold_crossings").data
    sm = rec.feature("threshold_crossings_smoothed").data
    assert raw.shape == sm.shape and sm.dtype == np.float32
    first = raw[:900]
    assert not np.isnan(first[:600]).any() and np.isnan(first[600:]).all()
    assert rec.feature("spike_band_power_smoothed").data.dtype == np.float32  # float64 in source


def test_electrode_order_per_file(sri_root):
    recs = _by_id(sri_root)
    first_ids = lambda r: r.channels.ids[[0, 64, 128, 192]].tolist()
    assert first_ids(recs["t15_word-loudness_block06"]) == ["4-00", "v6v-00", "d6v-00", "55b-00"]
    assert first_ids(recs["t15_sentence-loudness_block11"]) == ["v6v-00", "4-00", "55b-00", "d6v-00"]
    assert first_ids(recs["t16_word-loudness_block03"]) == ["55b-00", "6v-00", "hk1-00", "hk2-00"]
    for r in recs.values():
        assert r.provenance["electrode_mapping_source"]


def test_sentence_block(sri_root):
    rec = _by_id(sri_root)["t15_sentence-loudness_block11"]
    assert [f.name for f in rec.features] == ["threshold_crossings_smoothed", "spike_band_power_smoothed"]
    c = rec.trials.columns
    assert c["decoded_text_with_loudness"].tolist() == ["i mean, we ARE talking.", "POTATOES and ONIONS, please."]
    assert len(rec.timestamps) == 537 + 686


def test_rejects_unknown_cue(tmp_path):
    from conftest import _write_sri_word

    _write_sri_word(tmp_path / "t15_word-loudness_block_1.mat", ["SHOUT: be"])
    with pytest.raises(ValueError, match="unexpected cue"):
        sri.read(tmp_path)


def test_catalog_points_at_reader():
    assert catalog.get("srinivasan2026_speech_modes").reader == "srinivasan2026"


def test_nwb_roundtrip(sri_root, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from ibci_datasets.writers import nwb

    rec = _by_id(sri_root)["t15_word-loudness_block06"]
    path = nwb.write(rec, tmp_path)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        raw = f.processing["ecephys"]["threshold_crossings"].data[:]
        np.testing.assert_array_equal(raw, rec.feature("threshold_crossings").data)  # NaNs survive
        assert f.trials.to_dataframe()["loudness"].tolist() == ["WHISPER", "LOUD", "DO NOTHING"]
