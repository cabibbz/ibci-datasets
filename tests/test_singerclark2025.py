import numpy as np
import pytest

from ibci_datasets import catalog
from ibci_datasets.readers import singerclark2025 as sc


def test_finds_blocks_in_order_and_skips_words_file(sc_root):
    names = [p.name for p in sc.find_files(sc_root)]
    assert names == [
        "t15_day00039_block00_radial8_calibration_task.mat",
        "t15_day00202_block02_simultaneous_speech_and_cursor_task.mat",
        "t15_day00468_block03_grid_evaluation_task.mat",
    ]


def test_radial8_block(sc_root):
    rec = sc.read(sc_root)[0]
    assert rec.session_id == "t15_day00039_block00"
    assert rec.participant_id == "T15" and rec.post_implant_day == 39
    assert rec.timeline == "continuous" and rec.bin_size_s == 0.01
    np.testing.assert_allclose(rec.timestamps[:2], [0.005, 0.015])
    assert rec.channels.ids[0] == "v6v-00" and rec.channels.ids[64] == "4-00"
    assert rec.channels.ids[-1] == "d6v-63"
    tx = rec.feature("threshold_crossings")
    assert tx.data.dtype == np.float32 and tx.unit == "count" and tx.threshold_rms is None
    tr = rec.trials
    np.testing.assert_allclose(tr.start, [0.0, 1.2, 2.5])
    np.testing.assert_allclose(tr.end, [1.2, 2.5, 4.0])  # next start; last runs to block end
    assert tr.columns["truncated"].tolist() == [False, False, True]
    assert tr.columns["trial"].tolist() == [0, 1, 2]
    assert {s.name for s in rec.behavior} == {
        "cursor_position", "target_position", "cursor_decoder_output", "assist_amount"}
    assert rec.provenance["task_parameters"] == {
        "target_radius": 0.05, "cursor_radius": 0.025, "dwell_requirement_sec": 1.0}


def test_speech_block_trial_events(sc_root):
    rec = sc.read(sc_root)[1]
    tr = rec.trials
    assert tr.columns["speech_prompt"].tolist() == ["kite", "choice", "though", "word"]
    np.testing.assert_allclose(tr.columns["target_presentation_time"], [0.42, 1.6, 3.1, 4.3])
    beep = tr.columns["speech_go_cue_time"]
    assert np.isnan(beep[1]) and np.isnan(beep[3])
    np.testing.assert_allclose(beep[[0, 2]], [0.62, 3.3])
    np.testing.assert_allclose(tr.end, [1.5, 3.0, 4.2, 5.0])  # end of trial_end_bin
    assert not tr.columns["truncated"].any()
    assert rec.provenance["task_parameters"]["is_control_block"] is True


def test_grid_block_has_click_signals(sc_root):
    rec = sc.read(sc_root)[2]
    names = [s.name for s in rec.behavior]
    assert "click_decoder_output" in names and "click_assist" in names
    assert rec.provenance["task_parameters"]["grid_num_rows"] == 6.0


def test_catalog_points_at_reader():
    assert catalog.get("singerclark2025_cursor").reader == "singerclark2025"


def test_nwb_roundtrip_with_behavior(sc_root, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from ibci_datasets.writers import nwb

    rec = sc.read(sc_root)[1]
    path = nwb.write(rec, tmp_path)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        cur = f.processing["behavior"]["cursor_position"]
        np.testing.assert_allclose(cur.data[:], rec.behavior[0].data)
        tx = f.processing["ecephys"]["threshold_crossings"]
        assert tx.timestamps is None  # regular grid stored as starting_time + rate
        assert tx.rate == pytest.approx(100.0) and tx.starting_time == pytest.approx(0.005)
        trials = f.trials.to_dataframe()
        assert list(trials["speech_prompt"]) == ["kite", "choice", "though", "word"]


def test_torch_brain_continuous_domain(sc_root, tmp_path):
    pytest.importorskip("torch_brain")
    import h5py
    from torch_brain.data import Data
    from ibci_datasets.writers import torch_brain as tb

    rec = sc.read(sc_root)[0]
    path = tb.write(rec, tmp_path)
    with h5py.File(path, "r") as f:
        data = Data.from_hdf5(f, lazy=False)
        # continuous recording: one domain interval over the whole block, not one per trial
        assert len(data.domain) == 1
        assert len(data.trials) == 3
        np.testing.assert_allclose(data.behavior.cursor_position, rec.behavior[0].data)


def test_cli_convert(sc_root, tmp_path):
    pytest.importorskip("pynwb")
    from ibci_datasets.cli import main

    assert main(["convert", "singerclark2025_cursor", "--source", str(sc_root),
                 "--to", "nwb", "--out", str(tmp_path / "nwb")]) == 0
    assert len(list((tmp_path / "nwb").rglob("*.nwb"))) == 3
