import numpy as np
import pytest

from ibci_datasets import catalog
from ibci_datasets.readers import deo2026

CUES = ["Do Nothing", "EYEBROWS - Raise", "HEAD - Turn Down"]


def _session(pid, early_go=()):
    """Two blocks of 300 bins. Block 6: NaN edges and 2 full trials; block 7: 2 full trials
    plus one that the block end cuts off during Go."""
    D, G, R = 0.0, 1.0, 2.0
    trial = [D] * 50 + [G] * 40 + [R] * 30
    b6 = [np.nan] * 30 + trial * 2 + [np.nan] * 30
    b7 = trial * 2 + [D] * 30 + [G] * 30
    state = np.array(b6 + b7)
    block = np.array([6] * 300 + [7] * 300, dtype=np.uint8)
    go0 = np.flatnonzero((state[1:] == G) & (state[:-1] == D)) + 1  # 0-based first Go bins
    go_cue = go0 + 1  # 1-based, as in the release
    for i in early_go:
        go_cue[i] -= 1  # T15/T17-style one-bin-early goCue
    rng = np.random.default_rng(0)
    n_mu, n_sorted = 8, 2
    tx = rng.integers(0, 5, (600, n_mu + n_sorted)).astype(np.uint8)
    return {
        "tx": tx,
        "tx_blkMeanSub": tx - tx.mean(0),
        "blockNum": block,
        "state": state,
        "goCue": go_cue.astype(np.int32),
        "trialCue": np.array([1, 2, 3, 2, 1][: len(go0)], dtype=np.uint8),
        "cueList": CUES,
        "binWidth": 0.02,
        "chanSets": [np.arange(1, 5), np.arange(5, 9), np.array([9, 10])],
        "chanSetNames": [f"{pid}-v2", f"{pid}-m1", f"{pid}-v2Sorted"],
        "hemisphere": "Left",
        "details": f"{pid.lower()}.2023.08.27: blocks [6 7]",
        "sorted": tx[:, 8:],
        "sortedChanIDs": np.array([3, 6]),
    }


def _write_v5(path, d):
    import scipy.io

    m = dict(d)
    for k in ("cueList", "chanSetNames", "chanSets"):
        arr = np.empty(len(d[k]), dtype=object)
        arr[:] = d[k]
        m[k] = arr
    scipy.io.savemat(str(path), {"DataMat": m})


def _write_v73(path, d):
    """MATLAB v7.3 layout: HDF5 with a 512-byte header, transposed matrices, cells as refs."""
    import h5py

    with h5py.File(path, "w", userblock_size=512) as f:
        g = f.create_group("DataMat")
        refs = f.create_group("#refs#")
        k = [0]

        def ref(data):
            k[0] += 1
            return refs.create_dataset(f"r{k[0]}", data=data).ref

        def text(s):
            return np.array([ord(c) for c in s], dtype=np.uint16)[:, None]

        for key, v in d.items():
            if key in ("cueList", "chanSetNames"):
                g.create_dataset(key, data=np.array([[ref(text(s))] for s in v], dtype=h5py.ref_dtype))
            elif key == "chanSets":
                g.create_dataset(key, data=np.array([[ref(np.asarray(x, float)[None, :])] for x in v],
                                                    dtype=h5py.ref_dtype))
            elif key in ("details", "hemisphere"):
                g.create_dataset(key, data=text(v))
            else:
                a = np.asarray(v, dtype=float)
                g.create_dataset(key, data=a.T if a.ndim == 2 else a[None, :] if a.ndim == 1 else a.reshape(1, 1))
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: test HDF5 schema 1.00 .")


@pytest.fixture
def deo_root(tmp_path):
    _write_v5(tmp_path / "T15.mat", _session("T15", early_go=(1,)))
    _write_v73(tmp_path / "T17.mat", _session("T17", early_go=(0, 1, 2, 3, 4)))
    return tmp_path


def test_blocks_trials_and_channels(deo_root):
    recs = deo2026.read_file(deo_root / "T15.mat")
    assert [r.session_id for r in recs] == ["t15_2023-08-27_block06", "t15_2023-08-27_block07"]
    r6, r7 = recs
    assert r6.session_date.isoformat() == "2023-08-27" and r6.participant_id == "T15"
    assert r6.channels.ids.tolist()[:5] == ["v2-00", "v2-01", "v2-02", "v2-03", "m1-00"]
    assert len(r6.channels) == 8  # sorted units excluded
    assert r6.feature("threshold_crossings").data.shape == (300, 8)
    tr6 = r6.trials
    np.testing.assert_allclose(tr6.start, [0.6, 3.0])
    np.testing.assert_allclose(tr6.columns["go_time"], [1.6, 4.0])
    np.testing.assert_allclose(tr6.columns["go_cue_time"] - tr6.columns["go_time"], [0.0, -0.02])
    assert tr6.columns["cue"].tolist() == ["Do Nothing", "EYEBROWS - Raise"]
    tr7 = r7.trials
    assert tr7.columns["truncated"].tolist() == [False, False, True]
    assert np.isnan(tr7.columns["return_time"][2]) and tr7.end[2] == pytest.approx(6.0)
    assert "array_area_evidence" in r6.provenance
    assert r6.channels.locations[0].startswith("area 4,") and r6.channels.locations[4].startswith("area 55b,")


def test_v73_matches_v5(deo_root, tmp_path):
    v73 = deo2026.read_file(deo_root / "T17.mat")
    (tmp_path / "v5").mkdir()
    _write_v5(tmp_path / "v5" / "T17.mat", _session("T17", early_go=(0, 1, 2, 3, 4)))
    v5 = deo2026.read_file(tmp_path / "v5" / "T17.mat")
    assert len(v73) == len(v5) == 2
    for a, b in zip(v73, v5):
        np.testing.assert_array_equal(a.feature("threshold_crossings").data, b.feature("threshold_crossings").data)
        np.testing.assert_allclose(a.trials.start, b.trials.start)
        np.testing.assert_allclose(a.trials.columns["go_cue_time"], b.trials.columns["go_cue_time"])
        assert a.channels.ids.tolist() == b.channels.ids.tolist()
    assert v73[0].session_date.isoformat() == "2023-08-27"


def test_goCue_without_state_trial_is_an_error(tmp_path):
    d = _session("T15")
    d["goCue"] = np.append(d["goCue"], 200).astype(np.int32)  # a cue with no Delay-Go-Return
    d["trialCue"] = np.append(d["trialCue"], 1).astype(np.uint8)
    _write_v5(tmp_path / "T15.mat", d)
    with pytest.raises(ValueError, match="goCue rows"):
        deo2026.read_file(tmp_path / "T15.mat")


def test_catalog_points_at_reader():
    assert catalog.get("deo2026_mosaic").reader == "deo2026"


def test_nwb_roundtrip(deo_root, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from ibci_datasets.writers import nwb

    rec = deo2026.read_file(deo_root / "T15.mat")[1]
    path = nwb.write(rec, tmp_path / "out")
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        np.testing.assert_array_equal(f.processing["ecephys"]["threshold_crossings"].data[:],
                                      rec.feature("threshold_crossings").data)
        assert f.trials.to_dataframe()["truncated"].tolist() == [False, False, True]
