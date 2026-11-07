"""``ibci`` command line: browse the catalog, download from Dryad, convert."""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import sys
from pathlib import Path

from . import catalog
from .dryad import DryadClient, DryadError, safe_extract


def _cmd_list(args) -> int:
    if args.markdown:
        print(catalog.as_markdown())
        return 0
    for d in catalog.DATASETS:
        status = "reader" if d.reader else ("nwb" if d.in_nwb else "planned")
        print(f"{d.id:34s} {d.size_gb:7.2f} GB  {status:8s} {d.title}")
    return 0


def _cmd_info(args) -> int:
    d = catalog.get(args.dataset)
    for field in ("id", "title", "participants", "host", "source_id", "license", "size_gb",
                  "bin_ms", "threshold", "layout", "companion_repo", "effort", "reader",
                  "torch_brain", "in_nwb", "notes"):
        print(f"{field:15s} {getattr(d, field)}")
    print(f"{'url':15s} {d.url}")
    return 0


def _dryad_files(d, client, include):
    if d.host != "dryad":
        raise SystemExit(f"{d.id} is on DANDI; use `dandi download https://dandiarchive.org/dandiset/{d.source_id}`")
    files = client.files(d.source_id)
    if include:
        files = [f for f in files if any(fnmatch.fnmatch(f.path, p) for p in include)]
    return files


def _cmd_files(args) -> int:
    d = catalog.get(args.dataset)
    for f in _dryad_files(d, DryadClient.from_env(), args.include):
        print(f"{f.size / 1e9:9.3f} GB  {f.path}")
    return 0


def _cmd_download(args) -> int:
    d = catalog.get(args.dataset)
    client = DryadClient.from_env()
    files = _dryad_files(d, client, args.include)
    dest = Path(args.dest) / d.id
    try:
        for f in files:
            print(f"downloading {f.path} ({f.size / 1e9:.2f} GB)", file=sys.stderr)
            path = client.download(f, dest / "raw")
            if args.extract and path.suffix == ".zip":
                suffixes = args.extract_suffix or None
                n = len(safe_extract(path, dest / "extracted", suffixes))
                print(f"  extracted {n} files", file=sys.stderr)
    except DryadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(dest)
    return 0


def _cmd_convert(args) -> int:
    d = catalog.get(args.dataset)
    if not d.reader:
        print(f"error: no reader for {d.id} yet", file=sys.stderr)
        return 2
    reader = importlib.import_module(f"ibci_datasets.readers.{d.reader}")
    kwargs = {}
    if args.block_table:
        kwargs["block_table_csv"] = args.block_table
    recordings = reader.read(args.source, **kwargs)
    if args.to == "nwb":
        from .writers import nwb as writer
    else:
        from .writers import torch_brain as writer
    for rec in recordings:
        print(writer.write(rec, args.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ibci", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list catalogued datasets")
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(func=_cmd_list)

    s = sub.add_parser("info", help="show one dataset")
    s.add_argument("dataset")
    s.set_defaults(func=_cmd_info)

    s = sub.add_parser("files", help="list a Dryad dataset's files (no login needed)")
    s.add_argument("dataset")
    s.add_argument("--include", action="append", help="glob on file path; repeatable")
    s.set_defaults(func=_cmd_files)

    s = sub.add_parser("download", help="download from Dryad (needs credentials)")
    s.add_argument("dataset")
    s.add_argument("--dest", default="ibci_data")
    s.add_argument("--include", action="append", help="glob on file path; repeatable")
    s.add_argument("--extract", action="store_true", help="unzip downloaded .zip files")
    s.add_argument("--extract-suffix", action="append",
                   help="only extract members ending in this suffix, e.g. .hdf5; repeatable")
    s.set_defaults(func=_cmd_download)

    s = sub.add_parser("convert", help="convert a downloaded dataset")
    s.add_argument("dataset")
    s.add_argument("--source", required=True, help="folder holding the extracted release")
    s.add_argument("--to", choices=["nwb", "torch_brain"], default="nwb")
    s.add_argument("--out", required=True)
    s.add_argument("--block-table", help="card2024 only: t15_copyTaskData_description.csv")
    s.set_defaults(func=_cmd_convert)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
