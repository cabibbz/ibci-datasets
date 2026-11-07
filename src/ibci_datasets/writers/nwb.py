"""Write a Recording to an NWB file, the format the DANDI Archive expects.

Layout:

* one ``Device`` and one ``ElectrodeGroup`` per array, and an electrodes table with one
  row per channel (column ``channel_id``);
* each binned feature as a ``TimeSeries`` in the ``ecephys`` processing module, columns
  in electrodes-table order, with explicit timestamps (trial-concatenated recordings
  have gaps);
* the trial table as NWB trials, with every Trials column as a trial column.

Provenance (source, license, normalization, timeline) goes in the file's
``notes``/``data_collection`` fields so it survives without this package.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np

from ..schema import Recording

_NS = uuid.UUID("6f1d3a52-8c1e-4b7e-9d2a-3f1c2b4e5a60")


def to_nwbfile(rec: Recording):
    try:
        from pynwb import NWBFile, TimeSeries
        from pynwb.file import Subject
    except ImportError as e:  # pragma: no cover
        raise ImportError("NWB output needs pynwb: pip install 'ibci-datasets[nwb]'") from e

    date = rec.session_date or dt.date(1900, 1, 1)
    start = dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc)
    notes = {
        "dataset_id": rec.dataset_id,
        "timeline": rec.timeline,
        "bin_size_s": rec.bin_size_s,
        "split": rec.split,
        "post_implant_day": rec.post_implant_day,
        "time_of_day_known": False,
        **rec.provenance,
    }
    nwb = NWBFile(
        session_description=f"{rec.task or 'BCI session'}; split={rec.split}",
        identifier=str(uuid.uuid5(_NS, f"{rec.dataset_id}/{rec.session_id}")),
        session_start_time=start,
        session_id=rec.session_id,
        experimenter=list(rec.provenance.get("experimenters", [])) or None,
        institution=rec.provenance.get("institution"),
        experiment_description=rec.provenance.get("experiment_description") or rec.task or None,
        keywords=list(rec.provenance.get("keywords", [])) or None,
        related_publications=rec.provenance.get("doi"),
        notes=json.dumps(notes, default=str),
        data_collection=(
            f"Converted by ibci-datasets from {rec.provenance.get('source', 'the original release')}. "
            "Session start is the recording date at 00:00 UTC; time of day is not in the release. "
            + ("Trials were released as separate segments and are laid end to end with gaps, "
               "so times are not wall-clock times." if rec.timeline == "concatenated-trials" else "")
        ),
    )
    nwb.subject = Subject(
        subject_id=rec.participant_id,
        species="Homo sapiens",
        sex=rec.provenance.get("subject_sex", "U"),
        age=rec.provenance.get("subject_age"),  # ISO 8601, e.g. "P45Y"; DANDI requires it
        description="Clinical trial participant; consortium participant code.",
    )

    device = nwb.create_device(name="microelectrode_arrays", description="Intracortical microelectrode arrays")
    groups = {}
    for array_id, location in dict(zip(rec.channels.arrays, rec.channels.locations)).items():
        groups[array_id] = nwb.create_electrode_group(
            name=f"array_{array_id}", description=f"Array {array_id}", location=str(location), device=device
        )
    nwb.add_electrode_column(name="channel_id", description="Channel id used by ibci-datasets")
    for cid, array_id, location in zip(rec.channels.ids, rec.channels.arrays, rec.channels.locations):
        nwb.add_electrode(group=groups[array_id], location=str(location), channel_id=str(cid))

    # A regular grid is stored as starting_time + rate (NWB best practice); gappy
    # timelines (concatenated trials) need explicit timestamps.
    timing = _timing(rec)

    module = nwb.create_processing_module(name="ecephys", description="Binned neural features")
    for f in rec.features:
        desc = f"{f.description}. Normalization: {f.normalization}. Columns follow the electrodes table."
        if f.threshold_rms is not None:
            desc += f" Threshold: {f.threshold_rms} x RMS."
        module.add(
            TimeSeries(
                name=f.name,
                data=_compressed(f.data),
                unit=f.unit,
                description=desc,
                **timing,
            )
        )

    if rec.behavior:
        bmod = nwb.create_processing_module(name="behavior", description="Task and decoder signals")
        for s in rec.behavior:
            desc = s.description
            if s.components:
                desc += f" Columns: {', '.join(s.components)}."
            bmod.add(TimeSeries(name=s.name, data=_compressed(s.data), unit=s.unit, description=desc, **timing))

    if rec.trials is not None and len(rec.trials):
        tr = rec.trials
        for name, values in tr.columns.items():
            nwb.add_trial_column(name=name, description=tr.descriptions.get(name, name))
        for i in range(len(tr)):
            row = {k: _scalar(v[i]) for k, v in tr.columns.items()}
            nwb.add_trial(start_time=float(tr.start[i]), stop_time=float(tr.end[i]), **row)
    return nwb


def _compressed(data: np.ndarray):
    """gzip with chunking: lossless, and readable by any HDF5 tool (DANDI recommends it)."""
    from hdmf.backends.hdf5.h5_utils import H5DataIO

    return H5DataIO(data=data, compression="gzip", compression_opts=4, shuffle=True)


def _timing(rec: Recording) -> dict:
    ts = rec.timestamps
    if rec.timeline == "continuous" and len(ts) > 1:
        expected = ts[0] + rec.bin_size_s * np.arange(len(ts))
        if np.allclose(ts, expected, rtol=0, atol=1e-6):
            return {"starting_time": float(ts[0]), "rate": 1.0 / rec.bin_size_s}
    return {"timestamps": ts}


def _scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def write(rec: Recording, out_dir: str | Path) -> Path:
    """Write ``<out_dir>/sub-<participant>/sub-<participant>_ses-<session>.nwb``."""
    from pynwb import NWBHDF5IO

    out_dir = Path(out_dir)
    sub = _bids_label(rec.participant_id)
    ses = _bids_label(rec.session_id)
    path = out_dir / f"sub-{sub}" / f"sub-{sub}_ses-{ses}.nwb"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(to_nwbfile(rec))
    return path


def _bids_label(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")
