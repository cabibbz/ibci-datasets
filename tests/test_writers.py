import numpy as np
import pytest

from ibci_datasets.readers import card2024


@pytest.fixture
def train_rec(card_root):
    root, csv = card_root
    return card2024.read(root, block_table_csv=csv)[0]


def test_nwb_roundtrip(train_rec, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from ibci_datasets.writers import nwb

    path = nwb.write(train_rec, tmp_path)
    assert path.name == "sub-T15_ses-t15-2023-08-11-train.nwb"
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        assert f.subject.subject_id == "T15"
        tx = f.processing["ecephys"]["threshold_crossings"]
        np.testing.assert_allclose(tx.data[:], train_rec.feature("threshold_crossings").data)
        np.testing.assert_allclose(tx.timestamps[:], train_rec.timestamps)
        trials = f.trials.to_dataframe()
        assert list(trials["sentence"]) == list(train_rec.trials.columns["sentence"])
        assert len(f.electrodes) == 256
        assert f.electrodes["channel_id"][0] == "v6v-00"


def test_torch_brain_roundtrip(train_rec, tmp_path):
    pytest.importorskip("torch_brain")
    import h5py
    from torch_brain.data import Data
    from ibci_datasets.writers import torch_brain as tb

    path = tb.write(train_rec, tmp_path)
    with h5py.File(path, "r") as f:
        data = Data.from_hdf5(f, lazy=False)
        assert data.session.id == "t15.2023.08.11_train"
        np.testing.assert_allclose(data.neural.threshold_crossings, train_rec.feature("threshold_crossings").data)
        assert len(data.trials) == 3
        assert len(data.train_domain) == 3 and len(data.test_domain) == 0
        # slicing one trial returns only that trial's bins
        t0, t1 = float(train_rec.trials.start[1]), float(train_rec.trials.end[1])
        assert len(data.slice(t0, t1).neural) == 30
