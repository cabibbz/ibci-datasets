# Contributing

## Adding a reader

1. Pick a dataset from `ibci list` with no reader. Check its row in `src/ibci_datasets/catalog.py`, its Dryad README, and its companion repo's loading code.
2. Add `src/ibci_datasets/readers/<name>.py` exposing `iter_recordings(root, **options)` (a generator, one file in memory at a time) and `read(root, **options) -> list[Recording]`. Put the source layout, units, bin size, threshold and normalization in the module docstring, citing where each fact came from.
3. Build a synthetic fixture in `tests/conftest.py` that mirrors the documented layout, and test the reader against it. Never commit real data.
4. Set `reader="<name>"` on the catalog entry.
5. If you can, run the reader on the real release and say so in the pull request, with the version number you used.

## Rules

- Don't re-normalize, resample or pool. Record what the release did in `Feature.normalization` and `Recording.provenance`.
- Never open pickles or model files (`.pkl`, `.h5` Keras models, `.pt`, TensorFlow checkpoints). If a dataset's labels live only in a pickle, leave them out and note it.
- Keep participant codes exactly as released (`T5`, `T15`, ...). They are shared across datasets.
- Times are seconds. If a release has no continuous clock, lay trials end to end and set `timeline="concatenated-trials"`.

## Development

```bash
pip install -e ".[dev]"      # add [torch-brain] to run those tests too
pytest
```
