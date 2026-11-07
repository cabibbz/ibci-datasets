import h5py
import numpy as np
import pytest

from ibci_datasets import catalog
from ibci_datasets.readers import rigottithompson2026 as rt


@pytest.fixture
def rt_root(tmp_path):
    rng = np.random.default_rng(0)
    n_tr, n_bins, keep = 4, 150, np.array([3, 10, 95, 96, 150, 191])
    with h5py.File(tmp_path / "fig1_T11_delay.h5", "w") as f:
        spk = rng.uniform(0, 2, (n_tr, n_bins, len(keep)))
        spk[1, 140:] = np.nan  # a short trial, NaN-padded
        f["spike_data_all"] = spk
        f["sbp_data_all"] = rng.uniform(50, 4000, (n_tr, n_bins, len(keep)))
        f["keep_chans_all"] = keep
        f["cursor_data_all"] = rng.uniform(-400, 400, (n_tr, n_bins, 2))
        f["target_data_all"] = rng.uniform(-400, 400, (n_tr, 2))
        f["start_data_all"] = np.zeros((n_tr, 2))
        f["delay_data_all"] = np.array([0.4, 0.8, 1.2, 0.2])
        f["no_go_bool_all"] = np.array([0, 0, 1, 0])
        f["trial_duration_data_all"] = np.array([1.0, 1.4, 2.0, 0.9])
        f["block_id_data_all"] = np.array([7, 7, 8, 8])
        f["t_data"] = np.arange(-1000, 2000, 20)
    n = 300
    with h5py.File(tmp_path / "fig5_T16_continuous.h5", "w") as f:
        g = f.create_group("data")
        g["clock_time"] = 47 + 20 * np.arange(n)
        g["neural_data_norm"] = rng.standard_normal((n, 512))
        g["control_samples"] = rng.standard_normal((n, 2)).astype(np.float32)
        g["dec_move_state"] = rng.uniform(0, 1, (n, 1))
        g["cursor_data"] = rng.uniform(-400, 400, (n, 2))
        g["emg_data"] = rng.uniform(0, 1e-3, (n, 2))
        t = f.create_group("trial_info")
        t["start_time"] = np.array([447.0, 2447.0])
        t["go_cue_time"] = np.array([1447.0, np.nan])
        t["end_time"] = np.array([2047.0, 4047.0])
        t["delay_data"] = np.array([1000.0, 1200.0])
        t["dec_movement_onset_time"] = np.array([60, -1])
        t["no_go_bool"] = np.array([0, 1])
        t["start_data"] = np.zeros((2, 2))
        t["target_data"] = np.array([[400.0, 0.0], [0.0, -400.0]])
        f["keep_chans"] = np.array([5, 70, 300])
    (tmp_path / "fig5_T16_start.h5").write_bytes(b"")  # no neural data: skipped
    return tmp_path


def test_finds_neural_files_only(rt_root):
    assert [p.name for p in rt.find_sources(rt_root)] == ["fig1_T11_delay.h5", "fig5_T16_continuous.h5"]


def test_trial_file(rt_root):
    rec = rt.read_source(rt_root / "fig1_T11_delay.h5")
    assert rec.session_id == "fig1_t11_delay" and rec.timeline == "concatenated-trials"
    assert rec.channels.ids.tolist() == ["a1-03", "a1-10", "a1-95", "a2-00", "a2-54", "a2-95"]
    assert rec.bin_size_s == pytest.approx(0.02) and len(rec.timestamps) == 4 * 150
    tr = rec.trials
    np.testing.assert_allclose(tr.start, [0.0, 4.0, 8.0, 12.0])  # 3 s windows + 1 s gaps
    np.testing.assert_allclose(tr.columns["event_time"], [1.0, 5.0, 9.0, 13.0])  # t_data starts at -1 s
    assert tr.columns["no_go"].dtype == bool and tr.columns["no_go"].tolist() == [False, False, True, False]
    assert "target_x" in tr.columns
    assert np.isnan(rec.feature("threshold_crossings").data[150 + 145]).all()  # padding kept as NaN


def test_continuous_file(rt_root):
    rec = rt.read_source(rt_root / "fig5_T16_continuous.h5")
    assert rec.timeline == "continuous" and rec.bin_size_s == 0.02 and len(rec.channels) == 256
    assert rec.channels.ids[64] == "a2-00"
    assert rec.feature("threshold_crossings").data.shape == (300, 256)
    tr = rec.trials
    np.testing.assert_allclose(tr.start, [0.4, 2.4])
    assert np.isnan(tr.columns["go_cue_time"][1]) and np.isnan(tr.columns["decoded_onset_time"][1])
    assert tr.columns["decoded_onset_time"][0] == pytest.approx(1.2)
    assert {s.name for s in rec.behavior} >= {"cursor_position", "emg", "move_state_probability"}
    assert rec.provenance["preparatory_decoder_channels"] == [5, 70, 300]


def test_catalog_points_at_reader():
    assert catalog.get("rigottithompson2026_preparatory").reader == "rigottithompson2026"
