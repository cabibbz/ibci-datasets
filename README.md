# ibci-datasets

One reader per public human intracortical BCI dataset, with writers for [NWB](https://www.nwb.org) (the DANDI Archive's format) and [torch_brain](https://github.com/neuro-galaxy/torch_brain).

About two dozen human Utah-array BCI datasets are public, almost all from the BrainGate consortium and released CC0 on Dryad. Each ships in its own format (.mat, HDF5, pickles) with a paper-specific loader, and none of the Dryad releases is on DANDI as NWB. Dryad now also requires a login token for every download, which broke most of the papers' download scripts. This package fixes those three things:

- **A catalog** of every known dataset, with license, size, bin size, threshold and layout (`ibci list`).
- **A Dryad client** that handles tokens, resumes interrupted downloads and checks SHA-256 digests.
- **Readers** that parse each release once into a common `Recording`, and **writers** that turn a `Recording` into NWB or a torch_brain `Data` file.

> Status: pre-alpha. Seven readers, each run on the full real release from Dryad (November 2026): Card 2024 (Brain-to-Text '25), Singer-Clark 2025 (cursor), Srinivasan 2026 (speech modes), Pun 2024 (MINDFUL), Deo 2026 (whole-body mosaic, six participants), Rigotti-Thompson 2026 (preparatory activity) and Kunz 2025 (inner speech, four participants). NWB output is bit-identical to the source arrays and passes nwbinspector apart from fields the releases lack (subject age; some dates and institutions). Release discrepancies, including T15 releases whose electrode order differs from Card 2024's, are in [ERRATA.md](ERRATA.md).

## Install

```bash
pip install "ibci-datasets[nwb] @ git+https://github.com/cabibbz/ibci-datasets"
# torch_brain output too:
pip install "ibci-datasets[nwb,torch-brain] @ git+https://github.com/cabibbz/ibci-datasets"
```

## Quick start

```bash
ibci list                                   # the catalog
ibci info card2024_speech                   # one dataset
ibci files card2024_speech                  # its files on Dryad (no login needed)

export DRYAD_CLIENT_ID=... DRYAD_CLIENT_SECRET=...
ibci download card2024_speech --include "t15_copyTask_neuralData.zip" \
    --extract --extract-suffix .hdf5

ibci convert card2024_speech --source ibci_data/card2024_speech/extracted \
    --to nwb --out nwb/ \
    --block-table nejm-brain-to-text/data/t15_copyTaskData_description.csv
```

Or from Python:

```python
from ibci_datasets.readers import card2024
from ibci_datasets.writers import nwb

for rec in card2024.read("ibci_data/card2024_speech/extracted"):
    print(rec.session_id, rec.feature("threshold_crossings").data.shape, len(rec.trials))
    nwb.write(rec, "nwb/")
```

The block table is optional. It comes from the companion repo [Neuroprosthetics-Lab/nejm-brain-to-text](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) and adds each trial's sentence corpus and each session's post-implant day.

## Dryad credentials

Listing files is public, but downloads need a token. Create a Dryad account linked to your ORCID, then request an API account ([Dryad's instructions](https://github.com/datadryad/dryad-app/blob/main/documentation/apis/api_accounts.md)). Set `DRYAD_CLIENT_ID` and `DRYAD_CLIENT_SECRET`, and the client fetches and refreshes tokens itself (each lasts about 10 hours). `DRYAD_TOKEN` also works if you already have one.

## Design

```
release files ──reader──▶ Recording ──writer──▶ NWB (.nwb)
                                     └─writer──▶ torch_brain (.h5)
```

A `Recording` is one session (or one session split). It holds binned features (T × channels), a channel table (id, array, brain area), optional behaviour signals on the same bins (cursor position, decoder output), an optional trial table, and provenance: bin size, threshold, how the release normalized the data, and whether the timeline is continuous or made of trials laid end to end.

Releases differ in bin size (10 vs 20 ms), threshold (−3.5 vs −4.5 RMS) and normalization (raw counts vs z-scores). **Nothing here pools or re-normalizes data across datasets.** That choice belongs to whoever trains the model, so the provenance travels with every file.

## Datasets

Generated with `ibci list --markdown`. "torch_brain: draft" means an unmerged pipeline exists on a brainsets branch.

| Dataset | Participants | Size (GB) | Bin (ms) | Layout | Reader | torch_brain |
| --- | --- | --- | --- | --- | --- | --- |
| [kunz2025_inner_speech](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j) | T12, T15, T16, T17 | 10.80 | 10-20 | continuous | yes | none |
| [rigottithompson2026_preparatory](https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvr9) | T5, T11, T16 | 1.54 | 10-20 | trials | yes | none |
| [card2024_speech](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) | T15 | 11.59 | 20 | trials | yes | draft |
| [deo2026_mosaic](https://datadryad.org/dataset/doi:10.5061/dryad.mpg4f4rg5) | T5, T11, T12, T15, T16, T17 | 0.38 | 20 | continuous | yes | none |
| [pun2024_instability](https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn5s) | T5, T11 | 0.41 | 20 | continuous | yes | none |
| [singerclark2025_cursor](https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxzq) | T15 | 1.47 | 10 | continuous | yes | none |
| [srinivasan2026_speech_modes](https://datadryad.org/dataset/doi:10.5061/dryad.2547d7x5w) | T15, T16 | 2.47 | 10 | trials | yes | none |
| [gusman2025_drag_drop](https://datadryad.org/dataset/doi:10.5061/dryad.98sf7m0v1) | T11, T5 | 8.23 | 20 | continuous | planned | none |
| [hahn2026_arrays](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h) | T2, T3, T5, T6, T7, T8, T9, T10, T11 | 84.73 | 10 | continuous | planned | none |
| [jude2026_anarthria](https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk481) | T17 | 18.98 | 10 | continuous | planned | none |
| [jude2026_typing](https://datadryad.org/dataset/doi:10.5061/dryad.cz8w9gjjk) | T17, T18 | 38.02 | 10 | continuous | planned | none |
| [vargasirwin2025_gestures](https://datadryad.org/dataset/doi:10.5061/dryad.cvdncjtcq) | T11, T5 | 1.72 | 20 | trials | planned | none |
| [wairagkar2025_voice](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f) | T15 | 26.87 | 10 | trials | planned | none |
| [willett2021_handwriting](https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv) | T5 | 1.41 | 10 | trials | planned | none |
| [willett2023_speech](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) | T12 | 80.44 | 20 | trials | planned | draft |
| [wilson2025_recalibration](https://datadryad.org/dataset/doi:10.5061/dryad.1jwstqk6g) | T5 | 1.69 | ? | continuous | planned | none |
| [caltech_finger_rl](https://dandiarchive.org/dandiset/000252) | P1, N1 | 0.04 | ? | units | planned | none |
| [caltech_ppc_finger](https://dandiarchive.org/dandiset/000147) | P1 | 0.08 | ? | units | planned | none |
| [deo2024_bimanual](https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6xbb) | T5 | 0.14 | 20 | trials | planned | none |
| [falcon_h2](https://dandiarchive.org/dandiset/000950) | T5 | 1.22 | 20 | trials | planned | none |
| [fan2023_corp](https://datadryad.org/dataset/doi:10.5061/dryad.hqbzkh1p6) | T5 | 3.57 | 20 | trials | planned | none |
| [gusman2026_anthropomorphicity](https://datadryad.org/dataset/doi:10.5061/dryad.37pvmcw08) | T11, T17 | 0.23 | 20 | units | planned | none |
| [willsey2024_finger](https://datadryad.org/dataset/doi:10.5061/dryad.1jwstqk4f) | T5 | 0.18 | ? | continuous | planned | none |
| [falcon_h1](https://dandiarchive.org/dandiset/000954) | HumanPitt | 0.10 | 20 | continuous | planned | merged |

Not included: Caltech dandiset 000401 (empty), and Paulk et al.'s acute human Neuropixels recordings (raw broadband, no BCI task).

## Roadmap

Readers are added smallest and best-documented first:

1. `card2024_speech` (done)
2. `singerclark2025_cursor` (done), `srinivasan2026_speech_modes` (done), `pun2024_instability` (done), `deo2026_mosaic` (done), `rigottithompson2026_preparatory` (done), `kunz2025_inner_speech` (done), `gusman2026_anthropomorphicity`, `deo2024_bimanual`, `willsey2024_finger`
3. `falcon_h2` and the Caltech sets (already NWB; torch_brain pipelines only)
4. `fan2023_corp`, `willett2023_speech`, `jude2026_anarthria` + `jude2026_typing` (shared block format)
5. The rest

Each torch_brain pipeline is contributed upstream to [neuro-galaxy/torch_brain](https://github.com/neuro-galaxy/torch_brain), using this package for parsing. Before sending one, coordinate with the maintainers: unmerged drafts for Willett 2023 and Card 2024 exist on brainsets branches `brain2text` and `brain2text25`. NWB copies go to DANDI only with the original lab's agreement.

## Checking electrode order across releases

Releases from the same participant do not always order electrodes the same way (see [ERRATA.md](ERRATA.md)). `ibci verify-mapping` checks one release's array order against another's, using channel-correlation structure that, for T15 at least, persists across days and tasks:

```bash
ibci verify-mapping singerclark2025_cursor --source ibci_data/singerclark2025_cursor/raw \
    --source-files "*day00039*" \
    --reference-dataset card2024_speech --reference ibci_data/card2024_speech/extracted \
    --reference-files "*t15.2023.08.25/data_train*"
```

```
verdict      consistent
claimed      v6v, 4, 55b, d6v  r=0.236  trails best by 0.000 / 0.000 (plain / centred)
best         v6v, 4, 55b, d6v  r=0.236
runner-up    r=0.148  (margin 0.088)
centred      v6v, 4, 55b, d6v  (margin 0.099)
support      1.00 over 200 bootstrap resamples
gap          unknown; verdicts are calibrated for references within 120 days
data         source 107856 bins of threshold_crossings; reference 70550 bins of threshold_crossings
```

(Singer-Clark dates its files by post-implant day and Card by calendar date, so the tool cannot compute the gap here. It is 0 days: Singer-Clark day 39 is 2023-08-25.)

It scores every assignment of the source's arrays to the reference's in two ways: plain correlation-structure matching, and the same after subtracting each array-pair block's mean (preprocessing such as common-reference removal changes how strongly each array's channels correlate overall, which can fool the plain scoring). It gives `consistent`, `mismatch` or `matched` only when both scorings pick the same assignment, each with a margin of at least 0.03, with the same assignment winning both scorings in at least 95% of bootstrap resamples over chunks of time, and with the reference within 120 days when both sides are dated; otherwise `inconclusive`, with the reason. It also reports how far the claimed order trails the best; that is a diagnostic, not a test (an earlier rule that rejected trailing orders fired on unrelated data). The verdict rules were set and checked on the same 1,215 T15 comparisons (Card 2024 against itself across 45 sessions; Singer-Clark 2025 and Srinivasan 2026 against Card): no false verdicts, but every correct verdict was for a correct claimed order, so detecting a real mismatch has not been demonstrated, and other participants are outside the calibration. The bootstrap captures noise within a recording, not differences between days or releases. Treat `inconclusive` as the common outcome and a verdict as one line of evidence. It checks which block of channels is which array, not channel order within an array. Python: `ibci_datasets.verify.verify(source_recordings, reference_recordings)`.

## Errata

Building readers against the full releases turned up places where files differ from their documentation, including an electrode order that changes between files of one release. See [ERRATA.md](ERRATA.md).

## Safety

Several releases include pickles and Keras, PyTorch or TensorFlow model files. Readers never open them, and `--extract-suffix` keeps them out of the working tree. Downloads refuse file names and zip members that would escape the target folder.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: a reader is one module in `src/ibci_datasets/readers/` exposing `iter_recordings(root, **options)` and `read(root, **options) -> list[Recording]`, with tests that build synthetic files to the documented layout.

## Citing the data

Cite the original paper and Dryad record of every dataset you use; each `Recording` carries both in its provenance. The data are CC0 or CC-BY-4.0. This package is MIT-licensed.
