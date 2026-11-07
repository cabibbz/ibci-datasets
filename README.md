# ibci-datasets

One reader per public human intracortical BCI dataset, with writers for [NWB](https://www.nwb.org) (the DANDI Archive's format) and [torch_brain](https://github.com/neuro-galaxy/torch_brain).

About two dozen human Utah-array BCI datasets are public, almost all from the BrainGate consortium and released CC0 on Dryad. Each ships in its own format (.mat, HDF5, pickles) with a paper-specific loader, and none of the Dryad releases is on DANDI as NWB. Dryad now also requires a login token for every download, which broke most of the papers' download scripts. This package fixes those three things:

- **A catalog** of every known dataset, with license, size, bin size, threshold and layout (`ibci list`).
- **A Dryad client** that handles tokens, resumes interrupted downloads and checks SHA-256 digests.
- **Readers** that parse each release once into a common `Recording`, and **writers** that turn a `Recording` into NWB or a torch_brain `Data` file.

> Status: pre-alpha. One reader (Card 2024, the Brain-to-Text '25 data) is implemented and tested against synthetic files built to the documented layout. It has not yet been run on the real release, because that needs Dryad credentials.

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

A `Recording` is one session (or one session split). It holds binned features (T × channels), a channel table (id, array, brain area), an optional trial table, and provenance: bin size, threshold, how the release normalized the data, and whether the timeline is continuous or made of trials laid end to end.

Releases differ in bin size (10 vs 20 ms), threshold (−3.5 vs −4.5 RMS) and normalization (raw counts vs z-scores). **Nothing here pools or re-normalizes data across datasets.** That choice belongs to whoever trains the model, so the provenance travels with every file.

## Datasets

Generated with `ibci list --markdown`. "torch_brain: draft" means an unmerged pipeline exists on a brainsets branch.

| Dataset | Participants | Size (GB) | Bin (ms) | Layout | Reader | torch_brain |
| --- | --- | --- | --- | --- | --- | --- |
| [card2024_speech](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85) | T15 | 11.59 | 20 | trials | yes | draft |
| [kunz2025_inner_speech](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j) | T12, T15, T16, T17 | 10.80 | ? | trials | planned | none |
| [gusman2025_drag_drop](https://datadryad.org/dataset/doi:10.5061/dryad.98sf7m0v1) | T11, T5 | 8.23 | 20 | continuous | planned | none |
| [hahn2026_arrays](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h) | T2, T3, T5, T6, T7, T8, T9, T10, T11 | 84.73 | 10 | continuous | planned | none |
| [jude2026_anarthria](https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk481) | T17 | 18.98 | 10 | continuous | planned | none |
| [jude2026_typing](https://datadryad.org/dataset/doi:10.5061/dryad.cz8w9gjjk) | T17, T18 | 38.02 | 10 | continuous | planned | none |
| [rigottithompson2026_preparatory](https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvr9) | T5, T11, T16 | 1.54 | 10-20 | trials | planned | none |
| [vargasirwin2025_gestures](https://datadryad.org/dataset/doi:10.5061/dryad.cvdncjtcq) | T11, T5 | 1.72 | 20 | trials | planned | none |
| [wairagkar2025_voice](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f) | T15 | 26.87 | 10 | trials | planned | none |
| [willett2021_handwriting](https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv) | T5 | 1.41 | 10 | trials | planned | none |
| [willett2023_speech](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) | T12 | 80.44 | 20 | trials | planned | draft |
| [wilson2025_recalibration](https://datadryad.org/dataset/doi:10.5061/dryad.1jwstqk6g) | T5 | 1.69 | ? | continuous | planned | none |
| [caltech_finger_rl](https://dandiarchive.org/dandiset/000252) | P1, N1 | 0.04 | ? | units | planned | none |
| [caltech_ppc_finger](https://dandiarchive.org/dandiset/000147) | P1 | 0.08 | ? | units | planned | none |
| [deo2024_bimanual](https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6xbb) | T5 | 0.14 | 20 | trials | planned | none |
| [deo2026_mosaic](https://datadryad.org/dataset/doi:10.5061/dryad.mpg4f4rg5) | T5, T11, T12, T15, T16, T17 | 0.38 | 20 | continuous | planned | none |
| [falcon_h2](https://dandiarchive.org/dandiset/000950) | T5 | 1.22 | 20 | trials | planned | none |
| [fan2023_corp](https://datadryad.org/dataset/doi:10.5061/dryad.hqbzkh1p6) | T5 | 3.57 | 20 | trials | planned | none |
| [gusman2026_anthropomorphicity](https://datadryad.org/dataset/doi:10.5061/dryad.37pvmcw08) | T11, T17 | 0.23 | 20 | units | planned | none |
| [pun2024_instability](https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn5s) | T5, T11 | 0.41 | 20 | continuous | planned | none |
| [singerclark2025_cursor](https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxzq) | T15 | 1.47 | 10 | continuous | planned | none |
| [srinivasan2026_speech_modes](https://datadryad.org/dataset/doi:10.5061/dryad.2547d7x5w) | T15, T16 | 2.47 | 10 | trials | planned | none |
| [willsey2024_finger](https://datadryad.org/dataset/doi:10.5061/dryad.1jwstqk4f) | T5 | 0.18 | ? | continuous | planned | none |
| [falcon_h1](https://dandiarchive.org/dandiset/000954) | HumanPitt | 0.10 | 20 | continuous | planned | merged |

Not included: Caltech dandiset 000401 (empty), and Paulk et al.'s acute human Neuropixels recordings (raw broadband, no BCI task).

## Roadmap

Readers are added smallest and best-documented first:

1. `card2024_speech` (done)
2. `singerclark2025_cursor`, `srinivasan2026_speech_modes`, `pun2024_instability`, `deo2026_mosaic`, `gusman2026_anthropomorphicity`, `deo2024_bimanual`, `willsey2024_finger`
3. `falcon_h2` and the Caltech sets (already NWB; torch_brain pipelines only)
4. `fan2023_corp`, `willett2023_speech`, `jude2026_anarthria` + `jude2026_typing` (shared block format)
5. The rest

Each torch_brain pipeline is contributed upstream to [neuro-galaxy/torch_brain](https://github.com/neuro-galaxy/torch_brain), using this package for parsing. Before sending one, coordinate with the maintainers: unmerged drafts for Willett 2023 and Card 2024 exist on brainsets branches `brain2text` and `brain2text25`. NWB copies go to DANDI only with the original lab's agreement.

## Safety

Several releases include pickles and Keras, PyTorch or TensorFlow model files. Readers never open them, and `--extract-suffix` keeps them out of the working tree. Downloads refuse file names and zip members that would escape the target folder.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: a reader is one module in `src/ibci_datasets/readers/` exposing `read(root, **options) -> list[Recording]`, with tests that build synthetic files to the documented layout.

## Citing the data

Cite the original paper and Dryad record of every dataset you use; each `Recording` carries both in its provenance. The data are CC0 or CC-BY-4.0. This package is MIT-licensed.
