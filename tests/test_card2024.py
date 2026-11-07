import numpy as np
import pytest

from ibci_datasets.readers import card2024


def test_reads_all_splits_in_order(card_root):
    root, csv = card_root
    recs = card2024.read(root, block_table_csv=csv)
    assert [r.session_id for r in recs] == [
        "t15.2023.08.11_train", "t15.2023.08.11_val", "t15.2023.08.11_test"
    ]


def test_train_split_contents(card_root):
    root, csv = card_root
    rec = card2024.read(root, block_table_csv=csv)[0]
    assert rec.participant_id == "T15"
    assert rec.post_implant_day == 25
    assert rec.bin_size_s == 0.02
    assert len(rec.channels) == 256
    assert rec.channels.ids[0] == "v6v-00" and rec.channels.ids[-1] == "d6v-63"
    tx, sbp = rec.feature("threshold_crossings"), rec.feature("spike_band_power")
    assert tx.data.shape == sbp.data.shape == (120, 256)
    assert tx.threshold_rms == -4.5
    assert list(rec.trials.columns["sentence"]) == ["hello world", "how are you", "i am fine"]
    assert rec.trials.columns["phonemes"][0] == "HH AH L OW | W ER L D"
    assert set(rec.trials.columns["corpus"]) == {"50-Word"}


def test_features_split_without_reordering(card_root, tmp_path):
    import h5py

    root, _ = card_root
    path = root / "t15.2023.08.11" / "data_train.hdf5"
    with h5py.File(path, "r") as f:
        first = f["trial_0000"]["input_features"][()]
    rec = card2024.read_file(path)
    np.testing.assert_array_equal(rec.feature("threshold_crossings").data[:50], first[:, :256])
    np.testing.assert_array_equal(rec.feature("spike_band_power").data[:50], first[:, 256:])


def test_trials_are_laid_end_to_end_with_gaps(card_root):
    root, _ = card_root
    rec = card2024.read(root)[0]
    tr = rec.trials
    np.testing.assert_allclose(tr.end - tr.start, [1.0, 0.6, 0.8])
    np.testing.assert_allclose(tr.start[1:] - tr.end[:-1], card2024.TRIAL_GAP_S)
    # every bin centre falls inside exactly one trial
    inside = (rec.timestamps[:, None] > tr.start) & (rec.timestamps[:, None] < tr.end)
    assert (inside.sum(axis=1) == 1).all()
    assert rec.timeline == "concatenated-trials"


def test_test_split_is_unlabeled(card_root):
    root, _ = card_root
    rec = card2024.read(root)[2]
    assert rec.split == "test"
    assert rec.provenance["labeled"] is False
    assert list(rec.trials.columns["sentence"]) == ["", ""]
    assert list(rec.trials.columns["phonemes"]) == ["", ""]


def test_rejects_wrong_feature_width(tmp_path):
    import h5py

    path = tmp_path / "t15.2024.01.01" / "data_train.hdf5"
    path.parent.mkdir()
    with h5py.File(path, "w") as f:
        g = f.create_group("trial_0000")
        g.create_dataset("input_features", data=np.zeros((10, 256), np.float32))
        g.attrs.update(n_time_steps=10, block_num=1, trial_num=0, session="t15.2024.01.01")
    with pytest.raises(ValueError, match="unexpected input_features shape"):
        card2024.read_file(path)


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        card2024.read(tmp_path)
