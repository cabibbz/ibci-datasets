import importlib

import pytest

from ibci_datasets import catalog
from ibci_datasets.cli import main


def test_ids_unique_and_well_formed():
    ids = [d.id for d in catalog.DATASETS]
    assert len(ids) == len(set(ids))
    for d in catalog.DATASETS:
        assert d.id == d.id.lower() and " " not in d.id
        assert d.host in ("dryad", "dandi")
        if d.host == "dryad":
            assert d.source_id.startswith("doi:10.5061/dryad.")
        assert d.license in ("CC0-1.0", "CC-BY-4.0")
        assert d.effort in ("S", "M", "L", "done")


def test_every_declared_reader_exists():
    for d in catalog.DATASETS:
        if d.reader:
            mod = importlib.import_module(f"ibci_datasets.readers.{d.reader}")
            assert callable(mod.read)


def test_unknown_id_is_helpful():
    with pytest.raises(KeyError, match="ibci list"):
        catalog.get("nope")


def test_cli_list_and_info(capsys):
    assert main(["list"]) == 0
    assert "card2024_speech" in capsys.readouterr().out
    assert main(["info", "card2024_speech"]) == 0
    assert "dncjsxm85" in capsys.readouterr().out
    assert main(["list", "--markdown"]) == 0
    assert capsys.readouterr().out.startswith("| Dataset |")


def test_cli_convert_nwb(card_root, tmp_path, capsys):
    pytest.importorskip("pynwb")
    root, csv = card_root
    rc = main(["convert", "card2024_speech", "--source", str(root), "--to", "nwb",
               "--out", str(tmp_path / "nwb"), "--block-table", str(csv)])
    assert rc == 0
    assert len(list((tmp_path / "nwb").rglob("*.nwb"))) == 3
