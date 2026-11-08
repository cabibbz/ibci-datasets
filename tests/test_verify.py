import numpy as np
import pytest

from ibci_datasets.schema import Channels, Feature, Recording
from ibci_datasets.verify import verify

ARRAYS = ("v6v", "4", "55b", "d6v")
PER = 16


def _structure(seed=0):
    """Fixed channel-level structure: loadings on per-array and cross-array latents."""
    rng = np.random.default_rng(seed)
    n_lat = 4 + 3
    load = np.zeros((4 * PER, n_lat))
    for a in range(4):
        load[a * PER:(a + 1) * PER, a] = rng.uniform(0.5, 1.5, PER)
    # cross-array latents with distinct pairs, so array pairs are distinguishable
    for k, (a, b) in enumerate(((0, 1), (1, 2), (0, 3))):
        load[a * PER:(a + 1) * PER, 4 + k] = rng.uniform(0.3, 1.0, PER)
        load[b * PER:(b + 1) * PER, 4 + k] = rng.uniform(0.3, 1.0, PER) * (1.0 - 0.2 * k)
    return load


def _rec(physical_order, labels, seed, load=None, n=20000, pid="T15", signal=True):
    rng = np.random.default_rng(seed)
    load = _structure() if load is None else load
    lat = rng.standard_normal((n, load.shape[1]))
    from scipy.ndimage import gaussian_filter1d
    lat = gaussian_filter1d(lat, 5, axis=0)
    x = (lat @ load.T if signal else 0) + 0.5 * rng.standard_normal((n, 4 * PER))
    # rearrange physical arrays into the column order of this release
    x = np.concatenate([x[:, a * PER:(a + 1) * PER] for a in physical_order], axis=1)
    ids = [f"{lab}-{i:02d}" for lab in labels for i in range(PER)]
    arrays = [lab for lab in labels for _ in range(PER)]
    return Recording(
        dataset_id="synthetic", session_id=f"s{seed}", participant_id=pid, split=None, bin_size_s=0.01,
        timestamps=0.01 * (np.arange(n) + 0.5),
        channels=Channels(np.array(ids), np.array(arrays), np.array(arrays)),
        features=[Feature("threshold_crossings", x.astype(np.float32), "count", "synthetic")],
    )


def test_consistent_when_orders_agree():
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2)
    r = verify([src], [ref], n_bootstrap=50)
    assert r.verdict == "consistent", r.summary()
    assert r.best == ARRAYS and r.support >= 0.95


def test_detects_mislabelled_blocks():
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    # columns hold physical arrays 1, 0, 3, 2 but are labelled in reference order
    src = _rec((1, 0, 3, 2), ARRAYS, seed=3)
    r = verify([src], [ref], n_bootstrap=50)
    assert r.verdict == "mismatch", r.summary()
    assert r.best == ("4", "v6v", "d6v", "55b")
    assert r.score_claimed < r.score_best


def test_correct_labels_on_reordered_columns_are_consistent():
    # a release may order arrays differently; if its labels say so, that is fine
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    src = _rec((1, 0, 3, 2), ("4", "v6v", "d6v", "55b"), seed=4)
    r = verify([src], [ref], n_bootstrap=50)
    assert r.verdict == "consistent", r.summary()


def test_no_structure_is_inconclusive():
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1, signal=False, n=3000)
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2, signal=False, n=3000)
    r = verify([src], [ref], n_bootstrap=50)
    assert r.verdict == "inconclusive", r.summary()


def test_nan_rows_are_skipped():
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2)
    src.features[0].data[5000:9000] = np.nan
    r = verify([src], [ref], n_bootstrap=20)
    assert r.bins_source == 16000 and r.best == ARRAYS


def test_rejects_mixed_participants():
    with pytest.raises(ValueError, match="one participant"):
        verify([_rec((0, 1, 2, 3), ARRAYS, 1, pid="T5")], [_rec((0, 1, 2, 3), ARRAYS, 2)])


def test_cli_verify_mapping_json(sc_root, capsys):
    import json

    from ibci_datasets.cli import main

    code = main(["verify-mapping", "singerclark2025_cursor", "--source", str(sc_root),
                 "--source-files", "*day00039*", "--reference-dataset", "singerclark2025_cursor",
                 "--reference", str(sc_root), "--reference-files", "*day00202*",
                 "--bootstrap", "5", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] in ("consistent", "mismatch", "inconclusive")
    assert out["claimed"] == ["v6v", "4", "55b", "d6v"] and out["n_bootstrap"] == 5
    assert code == (0 if out["verdict"] == "consistent" else 1)


def test_small_margin_or_distant_reference_is_inconclusive():
    import dataclasses
    import datetime as dt

    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2)
    r = verify([src], [ref], n_bootstrap=20)
    assert r.verdict == "consistent" and r.margin >= 0.03
    tight = dataclasses.replace(r, min_margin=r.margin + 0.01)
    assert tight.verdict == "inconclusive" and "margin" in tight.reasons()[0]
    far = dataclasses.replace(r, gap_days=200.0)
    assert far.verdict == "inconclusive" and "200 days" in far.reasons()[0]
    # the gap is computed from dates when both sides carry them
    src.session_date, ref.session_date = dt.date(2024, 1, 1), dt.date(2024, 3, 1)
    assert verify([src], [ref], n_bootstrap=5).gap_days == 60


def test_both_scorings_must_agree():
    import dataclasses

    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2)
    r = verify([src], [ref], n_bootstrap=20)
    assert r.best_centred == r.best and r.margin_centred >= 0.03 and r.verdict == "consistent"
    split = dataclasses.replace(r, best_centred=("4", "v6v", "55b", "d6v"))
    assert split.verdict == "inconclusive" and "scorings disagree" in split.reasons()[0]
    weak = dataclasses.replace(r, margin_centred=0.01)
    assert weak.verdict == "inconclusive" and "block-centred margin" in weak.reasons()[0]
    plain_only = verify([src], [ref], n_bootstrap=5, consensus=False)
    assert plain_only.best_centred is None and plain_only.verdict == "consistent"


def test_centring_ignores_block_level_strength():
    """Damping one array's shared signal (as referencing can) must not flip the plain match into
    a confident wrong consensus verdict."""
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    load = _structure()
    load[:PER, 0] *= 0.05  # array 0 loses almost all of its own shared signal in the source
    src = _rec((0, 1, 2, 3), ARRAYS, seed=2, load=load)
    r = verify([src], [ref], n_bootstrap=30)
    assert r.verdict in ("consistent", "inconclusive"), r.summary()


def test_trailing_claim_is_a_diagnostic_not_a_verdict():
    ref = _rec((0, 1, 2, 3), ARRAYS, seed=1)
    right = verify([_rec((0, 1, 2, 3), ARRAYS, seed=2)], [ref], n_bootstrap=10)
    assert not right.claimed_rejected and right.deficit_claimed == 0.0
    wrong = verify([_rec((1, 0, 3, 2), ARRAYS, seed=3)], [ref], n_bootstrap=10)
    assert wrong.verdict == "mismatch"
    assert not wrong.claimed_rejected  # no dates: rejection needs a known gap
    import dataclasses
    wrong = dataclasses.replace(wrong, gap_days=10.0)
    assert wrong.claimed_rejected
    # best not settled: the diagnostic still fires, but it no longer turns into a verdict
    undetermined = dataclasses.replace(wrong, margin_centred=0.001)
    assert undetermined.claimed_rejected and undetermined.verdict == "inconclusive"


def test_single_array_is_inconclusive():
    r = verify([_rec((0,), ARRAYS[:1], seed=2)], [_rec((0,), ARRAYS[:1], seed=1)], n_bootstrap=5)
    assert r.verdict == "inconclusive" and "fewer than two arrays" in r.reasons()[0]
