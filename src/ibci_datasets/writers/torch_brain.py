"""Write a Recording as a torch_brain ``Data`` object (one ``.h5`` per recording).

Binned features are stored as an ``IrregularTimeSeries`` named ``neural`` with one
attribute per feature (e.g. ``neural.threshold_crossings``, shape (T, C)), timestamps at
bin centres and a domain of one interval per trial. Irregular rather than regular,
because trial-concatenated releases have gaps between trials.

The release split becomes the recording's train/valid/test domain. Unlabeled test
splits get an empty train/valid domain and keep their trials for inference only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import Recording

_SPLIT_TO_DOMAIN = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def to_data(rec: Recording, brainset_id: str | None = None):
    try:
        from torch_brain.data import (
            ArrayDict,
            BrainsetDescription,
            Data,
            DeviceDescription,
            Interval,
            IrregularTimeSeries,
            SessionDescription,
            SubjectDescription,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError("torch_brain output needs torch_brain: pip install 'ibci-datasets[torch-brain]'") from e
    import datetime as dt

    if rec.trials is not None and len(rec.trials):
        domain = Interval(start=rec.trials.start.astype(np.float64), end=rec.trials.end.astype(np.float64))
    else:
        half = rec.bin_size_s / 2
        domain = Interval(start=np.array([rec.timestamps[0] - half]), end=np.array([rec.timestamps[-1] + half]))

    neural = IrregularTimeSeries(
        timestamps=rec.timestamps.astype(np.float64),
        domain=domain,
        **{f.name: f.data.astype(np.float32) for f in rec.features},
    )
    channels = ArrayDict(
        id=rec.channels.ids.astype("S"),
        array=rec.channels.arrays.astype("S"),
        location=rec.channels.locations.astype("S"),
    )
    parts = dict(neural=neural, channels=channels, domain=domain)
    if rec.trials is not None and len(rec.trials):
        cols = {}
        for k, v in rec.trials.columns.items():
            cols[k] = v.astype("S") if v.dtype.kind in "US" else v
        parts["trials"] = Interval(start=rec.trials.start.astype(np.float64),
                                   end=rec.trials.end.astype(np.float64), **cols)

    recording_date = None
    if rec.session_date is not None:
        recording_date = dt.datetime(rec.session_date.year, rec.session_date.month, rec.session_date.day)
    data = Data(
        brainset=BrainsetDescription(
            id=brainset_id or rec.dataset_id,
            origin_version=str(rec.provenance.get("source", "")),
            derived_version="ibci-datasets",
            source=str(rec.provenance.get("source", "")),
            description=f"{rec.task}. Converted by ibci-datasets; timeline={rec.timeline}.",
        ),
        subject=SubjectDescription(id=rec.participant_id, species="HOMO_SAPIENS"),
        session=SessionDescription(id=rec.session_id, recording_date=recording_date, task=rec.task),
        device=DeviceDescription(id=f"{rec.participant_id}_arrays",
                                 recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS"),
        **parts,
    )
    empty = Interval(start=np.array([]), end=np.array([]))
    which = _SPLIT_TO_DOMAIN.get(rec.split or "", "train")
    for name in ("train", "valid", "test"):
        setattr(data, f"{name}_domain", domain if name == which else empty)
    return data


def write(rec: Recording, out_dir: str | Path, brainset_id: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec.session_id}.h5"
    to_data(rec, brainset_id).save(path)
    return path
