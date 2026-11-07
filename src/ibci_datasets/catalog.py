"""Catalog of public human intracortical BCI datasets.

Facts were checked against the Dryad and DANDI APIs and each dataset's README in
November 2026. ``None`` means "not stated in the public docs we could read", not zero.
Effort: S = one documented schema; M = several file types or missing docs;
L = heterogeneous and under-documented.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    participants: tuple[str, ...]
    host: str  # "dryad" | "dandi"
    source_id: str  # Dryad DOI ("doi:10.5061/dryad.x") or DANDI id ("000950")
    license: str  # SPDX id
    size_gb: float
    bin_ms: str | None  # "10", "20", "10-20" or None
    threshold: str | None  # e.g. "-4.5 RMS"
    layout: str  # "trials" | "continuous" | "units"
    companion_repo: str | None
    effort: str  # "S" | "M" | "L" | "done"
    reader: str | None = None  # module in ibci_datasets.readers, once implemented
    torch_brain: str = "none"  # "merged" | "draft" | "none"
    in_nwb: bool = False
    notes: str = ""

    @property
    def url(self) -> str:
        if self.host == "dryad":
            return f"https://datadryad.org/dataset/{self.source_id}"
        return f"https://dandiarchive.org/dandiset/{self.source_id}"


_D = Dataset

DATASETS: tuple[Dataset, ...] = (
    _D("card2024_speech", "Card 2024 speech neuroprosthesis (Brain-to-Text '25 data)",
       ("T15",), "dryad", "doi:10.5061/dryad.dncjsxm85", "CC0-1.0", 11.59, "20", "-4.5 RMS",
       "trials", "Neuroprosthetics-Lab/nejm-brain-to-text", "S", reader="card2024",
       torch_brain="draft",
       notes="Per-trial HDF5; 45 sessions; features z-scored over the preceding 20 trials. "
             "Test split has no labels. Skip the .pkl files and the pretrained model."),
    _D("willett2023_speech", "Willett 2023 speech neuroprosthesis (Brain-to-Text '24 data)",
       ("T12",), "dryad", "doi:10.5061/dryad.x69p8czpq", "CC0-1.0", 80.44, "20", None,
       "trials", "fwillett/speechBCI", "M", torch_brain="draft",
       notes="About 60 GB of the release is language models and derived files. "
             "Thresholds for tx2-tx4 are documented only in sub-readmes."),
    _D("willett2021_handwriting", "Willett 2021 handwriting BCI",
       ("T5",), "dryad", "doi:10.5061/dryad.wh70rxwmv", "CC0-1.0", 1.41, "10", None,
       "trials", "fwillett/handwritingBCI", "M",
       notes="Data dictionary is a PDF; rebuild it from the companion notebooks."),
    _D("fan2023_corp", "Fan 2023 plug-and-play handwriting stability (CORP)",
       ("T5",), "dryad", "doi:10.5061/dryad.hqbzkh1p6", "CC0-1.0", 3.57, "20", "-4.5 RMS",
       "trials", "cffan/CORP", "S",
       notes="FALCON H2 repackages these sessions plus five withheld ones."),
    _D("kunz2025_inner_speech", "Kunz 2025 inner speech",
       ("T12", "T15", "T16", "T17"), "dryad", "doi:10.5061/dryad.gf1vhhn1j", "CC0-1.0", 10.80,
       None, None, "trials", None, "L",
       notes="Eleven task folders; per-folder readmes are inside the zips."),
    _D("wairagkar2025_voice", "Wairagkar 2025 instantaneous voice synthesis",
       ("T15",), "dryad", "doi:10.5061/dryad.2280gb64f", "CC0-1.0", 26.87, "10", None,
       "trials", "Neuroprosthetics-Lab/brain-to-voice-2025", "M",
       notes="Features are log-transformed, normalized and smoothed. Never load the Keras models."),
    _D("srinivasan2026_speech_modes", "Srinivasan 2026 speech modes and loudness",
       ("T15", "T16"), "dryad", "doi:10.5061/dryad.2547d7x5w", "CC0-1.0", 2.47, "10", "-4.5 RMS",
       "trials", "Neuroprosthetics-Lab/srinivasan-speech-modes", "S", reader="srinivasan2026",
       notes="Per-trial .mat; smoothed+normalized features plus raw crossings (word task). "
             "Electrode order differs between files (see the reader docstring)."),
    _D("jude2026_anarthria", "Jude 2026 speech after long-term anarthria",
       ("T17",), "dryad", "doi:10.5061/dryad.vq83bk481", "CC0-1.0", 18.98, "10", "-3.5 SD",
       "continuous", "justin-jude/anarthria-speech-bci", "M",
       notes="Continuous per-block streams with NSP and Redis clocks; T17 has 6 arrays "
             "but files hold 512 features."),
    _D("jude2026_typing", "Jude 2026 bimanual typing",
       ("T17", "T18"), "dryad", "doi:10.5061/dryad.cz8w9gjjk", "CC0-1.0", 38.02, "10", "-3.5 SD",
       "continuous", "justin-jude/typingBCI", "M",
       notes="Same block schema as jude2026_anarthria; 18.6 GB is language models."),
    _D("singerclark2025_cursor", "Singer-Clark 2025 cursor control from speech motor cortex",
       ("T15",), "dryad", "doi:10.5061/dryad.prr4xgxzq", "CC0-1.0", 1.47, "10", None,
       "continuous", "Neuroprosthetics-Lab/BCI-cursor-control-from-speech-motor-cortex", "S",
       reader="singerclark2025",
       notes="One .mat per block, raw features and cursor/click behaviour; threshold level "
             "not stated. t15_day00202_words.mat is undocumented and skipped."),
    _D("hahn2026_arrays", "Hahn 2026 microelectrode array performance",
       ("T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"), "dryad",
       "doi:10.5061/dryad.x0k6djj1h", "CC0-1.0", 84.73, "10", "six thresholds",
       "continuous", "nptl-stanford/array-paper", "M",
       notes="Decoding subset covers 9 participants; the yield subset (14 participants) "
             "is not time-aligned across arrays."),
    _D("deo2026_mosaic", "Deo 2026 whole-body mosaic in precentral gyrus",
       ("T5", "T11", "T12", "T15", "T16", "T17"), "dryad", "doi:10.5061/dryad.mpg4f4rg5",
       "CC0-1.0", 0.38, "20", None, "continuous", "d-r-deo/pcg-mosaic", "S", reader="deo2026",
       notes="Multi-unit channels and sorted units share one matrix; split them with chanSets. "
             "Array names are lab codes; T15 columns are not in Card 2024's order (see ERRATA.md). "
             "T17.mat is MATLAB v7.3. Participants C1/C2 are hosted separately on DABI."),
    _D("deo2024_bimanual", "Deo 2024 bimanual cursor control",
       ("T5",), "dryad", "doi:10.5061/dryad.sn02v6xbb", "CC0-1.0", 0.14, "20", None,
       "trials", "d-r-deo/bimanualBCI", "S"),
    _D("willsey2024_finger", "Willsey 2024 finger decoding and quadcopter control",
       ("T5",), "dryad", "doi:10.5061/dryad.1jwstqk4f", "CC0-1.0", 0.18, None, None,
       "continuous", "WillseyBCILab/BCI_Finger_Decoding_Quadcopter", "S",
       notes="Skip the PyTorch .pt decoders."),
    _D("rigottithompson2026_preparatory", "Rigotti-Thompson 2026 preparatory activity",
       ("T5", "T11", "T16"), "dryad", "doi:10.5061/dryad.brv15dvr9", "CC0-1.0", 1.54, "10-20",
       None, "trials", "snel-repo/human-preparatory-activity-paper", "M", reader="rigottithompson2026",
       notes="Figure-specific, heavily smoothed files; low value as training data. The fig5 continuous "
             "blocks keep all channels and are the most useful part."),
    _D("gusman2026_anthropomorphicity", "Gusman 2026 observation and anthropomorphicity",
       ("T11", "T17"), "dryad", "doi:10.5061/dryad.37pvmcw08", "CC0-1.0", 0.23, "20", None,
       "units", "jgusman/anthropomorphicity", "S"),
    _D("gusman2025_drag_drop", "Gusman 2025 drag-and-drop control",
       ("T11", "T5"), "dryad", "doi:10.5061/dryad.98sf7m0v1", "CC0-1.0", 8.23, "20", None,
       "continuous", "jgusman/drag-and-drop", "M",
       notes="1,344 features per bin include 5 LFP bands; MATLAB categorical fields."),
    _D("vargasirwin2025_gestures", "Vargas-Irwin 2025 gesture encoding",
       ("T11", "T5"), "dryad", "doi:10.5061/dryad.cvdncjtcq", "CC0-1.0", 1.72, "20", None,
       "trials", "cvargasi/GestureEncoding", "M",
       notes="Only significantly modulated features are kept, so channel identity is unclear."),
    _D("wilson2025_recalibration", "Wilson 2025 long-term unsupervised recalibration",
       ("T5",), "dryad", "doi:10.5061/dryad.1jwstqk6g", "CC0-1.0", 1.69, None, None,
       "continuous", "guyhwilson/nonstationarities", "M",
       notes="Unnamed positional struct fields; one day has its arrays swapped."),
    _D("pun2024_instability", "Pun 2024 recording instability (MINDFUL)",
       ("T5", "T11"), "dryad", "doi:10.5061/dryad.n2z34tn5s", "CC0-1.0", 0.41, "20", "-3.5 RMS",
       "continuous", "ewinapun/MINDFUL", "S", reader="pun2024",
       notes="nctx is a rate in Hz (count x 50) with outlier bins interpolated; two 96-channel "
             "arrays; one block has a surplus trailing metrics row (see the reader docstring)."),
    _D("falcon_h1", "FALCON H1 7-DoF reach and grasp",
       ("HumanPitt",), "dandi", "000954", "CC-BY-4.0", 0.10, "20", None, "continuous",
       "snel-repo/falcon-challenge", "done", torch_brain="merged", in_nwb=True,
       notes="Merged on torch_brain main, not yet in a release."),
    _D("falcon_h2", "FALCON H2 handwriting",
       ("T5",), "dandi", "000950", "CC-BY-4.0", 1.22, "20", None, "trials",
       "snel-repo/falcon-challenge", "S", in_nwb=True,
       notes="Already NWB; needs only a torch_brain pipeline."),
    _D("caltech_ppc_finger", "Caltech PPC finger movements",
       ("P1",), "dandi", "000147", "CC-BY-4.0", 0.08, None, None, "units", None, "S", in_nwb=True),
    _D("caltech_finger_rl", "Caltech finger movements, left and right",
       ("P1", "N1"), "dandi", "000252", "CC-BY-4.0", 0.04, None, None, "units", None, "S",
       in_nwb=True),
)

_BY_ID = {d.id: d for d in DATASETS}
if len(_BY_ID) != len(DATASETS):
    raise RuntimeError("duplicate dataset id in catalog")


def get(dataset_id: str) -> Dataset:
    try:
        return _BY_ID[dataset_id]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset_id!r}; run `ibci list` to see the catalog"
        ) from None


def as_markdown() -> str:
    rows = [
        "| Dataset | Participants | Size (GB) | Bin (ms) | Layout | Reader | torch_brain |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in sorted(DATASETS, key=lambda d: (d.reader is None, d.effort, d.id)):
        rows.append(
            f"| [{d.id}]({d.url}) | {', '.join(d.participants)} | {d.size_gb:.2f} | "
            f"{d.bin_ms or '?'} | {d.layout} | {'yes' if d.reader else 'planned'} | "
            f"{d.torch_brain} |"
        )
    return "\n".join(rows)
