import numpy as np
import pytest

from ibci_datasets import catalog
from ibci_datasets.readers import pun2024


def _by_id(root):
    return {r.session_id: r for r in pun2024.read(root)}


def test_finds_all_blocks(pun_root):
    assert sorted(_by_id(pun_root)) == [
        "t11_day0658_block08_personal_use", "t11_day0689_block14", "t5_day2121_block01"]


def test_t5_block(pun_root):
    rec = _by_id(pun_root)["t5_day2121_block01"]
    assert rec.participant_id == "T5" and rec.post_implant_day == 2121 and rec.bin_size_s == 0.02
    assert [f.name for f in rec.features] == ["threshold_crossings"]
    tx = rec.feature("threshold_crossings")
    assert tx.unit == "Hz" and tx.threshold_rms == -3.5 and tx.data.dtype == np.float32
    assert rec.channels.ids[0] == "a1-00" and rec.channels.ids[96] == "a2-00" and len(rec.channels) == 192
    tr = rec.trials
    # 1-based inclusive [10, 60] -> bins 9..59 -> 0.18 s to 1.20 s
    np.testing.assert_allclose(tr.start, [0.18, 1.58, 4.58])
    np.testing.assert_allclose(tr.end, [1.20, 4.00, 9.60])
    np.testing.assert_allclose(tr.end - tr.start, tr.columns["time_to_target_s"])
    assert rec.provenance["task_parameters"] == {"task_name": "Fitts", "use_click": False, "percent_correct": 100.0}
    assert rec.provenance["subject_sex"] == "M"


def test_surplus_trailing_row_is_dropped_and_recorded(pun_root):
    rec = _by_id(pun_root)["t11_day0689_block14"]
    assert len(rec.trials.columns["success"]) == 3
    assert rec.provenance["trimmed_trailing_trial_rows"] == ["trialSuccess", "pathEfficiency", "orthChanges"]
    assert {"threshold_crossings", "spike_band_power"} == {f.name for f in rec.features}


def test_two_surplus_rows_is_an_error(tmp_path):
    from conftest import _write_pun_block

    _write_pun_block(tmp_path / "T11" / "day_700" / "Block_01", trials=((10, 60), (80, 200)), extra_row=True)
    # extra_row adds one row to k=2 -> 3 rows; force a second mismatch by editing angleErrorPerTrial
    import scipy.io
    info = scipy.io.loadmat(str(tmp_path / "T11" / "day_700" / "Block_01" / "info.mat"))
    info = {k: v for k, v in info.items() if not k.startswith("__")}
    info["angleErrorPerTrial"] = np.zeros((5, 1))
    scipy.io.savemat(str(tmp_path / "T11" / "day_700" / "Block_01" / "info.mat"), info)
    with pytest.raises(ValueError, match="angleErrorPerTrial"):
        pun2024.read(tmp_path)


def test_personal_use_block(pun_root):
    rec = _by_id(pun_root)["t11_day0658_block08_personal_use"]
    assert rec.provenance["subset"] == "personal_use"
    assert "intended_direction" not in {s.name for s in rec.behavior}
    assert len(rec.trials) == 2 and rec.trials.columns == {}


def test_catalog_points_at_reader():
    assert catalog.get("pun2024_instability").reader == "pun2024"


def test_nwb_roundtrip(pun_root, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from ibci_datasets.writers import nwb

    rec = _by_id(pun_root)["t11_day0689_block14"]
    path = nwb.write(rec, tmp_path)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        np.testing.assert_array_equal(f.processing["ecephys"]["spike_band_power"].data[:],
                                      rec.feature("spike_band_power").data)
        assert f.processing["ecephys"]["threshold_crossings"].rate == pytest.approx(50.0)
        assert f.subject.sex == "M"
        assert len(f.trials) == 3
