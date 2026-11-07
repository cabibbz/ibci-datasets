# Release notes and errata

Things we found while building readers against the full public releases, where the files differ from, or go beyond, their documentation. Each reader handles these, and its module docstring has the details. Every entry was checked on the real data (November 2026). Electrode-order checks use `ibci verify-mapping` (see the README); the full audit is summarized at the end. None of this is a criticism of the releases: they are unusually well documented, and these are the gaps that only show up when you read every file.

If you are an author of one of these datasets and something here is wrong, please open an issue.

## Srinivasan 2026, speech modes and loudness (doi:10.5061/dryad.2547d7x5w)

- **Electrode order differs between files.** T15 word-loudness files use the order area 4, v6v, d6v, 55b; T15 sentence-loudness files use v6v, area 4, 55b, d6v, the order documented for Card 2024. The companion analysis scripts give the first order for the 2023 word-loudness session (flag `using_correct_electrode_mapping = 0`). We checked both with `ibci verify-mapping` against Card 2024 sessions recorded by the same arrays (the method recovers the known order of Singer-Clark 2025 files, which label every electrode). The sentence-loudness order is clearly supported. The word-loudness order is supported against one same-era Card session at a margin just over the threshold, and inconclusive against three others, so it rests mainly on the companion scripts (see the audit table below). The reader assigns each file its own order and records the evidence in `provenance["electrode_mapping_source"]`. T16's order (55b/PEF, 6v, two hand-knob arrays) comes from the companion scripts only and is not yet independently checked.
- **Smoothed and raw features have different lengths.** `threshcross` and `spikepow` run 3 s longer per trial than `raw_threshcross`; both start at the same time. The reader NaN-pads the raw counts.
- **Trial timing.** Each trial array starts 1 s before the delay period; `speech_onsets` and `speech_offsets` are 30 kHz sample counts from the go cue (from the companion scripts, confirmed on the data).
- **Control trials.** Two T15 trials are cued `DO NOTHING`, which the README does not mention. Their onset values (3-5 ms after the go cue) are placeholders; the reader sets onset and offset to NaN.

## Pun 2024, recording instability / MINDFUL (doi:10.5061/dryad.n2z34tn5s)

- **`nctx` is a rate in Hz, not a count.** In T5 every value is a multiple of 50 (count per 20 ms bin x 50). In T11, 86-98% are; the rest have the form 50 k / n, which is what linear interpolation over noise-outlier bins produces (the lab's `RemoveOutliers.m`; `info.prctOutliers` marks those bins).
- **`startStops` are 1-based, inclusive.** Read that way, trial length equals `timeToTarget` exactly in every block.
- **`useClick` is one value per block**, not one per trial as the README describes.
- **One block has a surplus row.** T11 day 689 block 14 has 76 rows of `trialSuccess`, `pathEfficiency` and `orthChanges` for 75 trials. In every other T11 block, success equals "acquired before the 10 s timeout"; dropping the last row restores that here (dropping the first does not). The reader drops it and records that in `provenance["trimmed_trailing_trial_rows"]`.
- **Array split not labelled.** Channels 0-95 and 96-191 behave as the two arrays (within-array correlation exceeds across-array, against an interleaved control), but which physical array is which is not stated. The reader labels them `a1` and `a2`.
- **Ages are at enrollment.** The README's ages (65 and 37) are at enrollment, years before these sessions, so they are not written as the subject's age at recording.

## Singer-Clark 2025, cursor control from speech motor cortex (doi:10.5061/dryad.prr4xgxzq)

- **Crossing threshold not stated.** The README describes an electrode-specific threshold without giving the level. The reader leaves `threshold_rms` unset.
- **`t15_day00202_words.mat` is undocumented.** The reader skips it.
- **Bin indices are 0-based** (the companion code indexes NumPy arrays with them directly).

## Card 2024, Brain-to-Text '25 (doi:10.5061/dryad.dncjsxm85)

- No discrepancies found. The documented electrode order agrees with Singer-Clark 2025's per-electrode labels on a shared recording day (2023-08-25). Not every session has a test split.

## Electrode-order audit (November 2026)

`ibci verify-mapping` over every release with a reader, with the calibrated verdict rules (support >= 0.95, margin >= 0.03, reference within 120 days):

| Comparison | Pairs | Result |
| --- | --- | --- |
| Card 2024, each session vs the previous one (2023-08 to 2025-04) | 44 | 43 consistent; 1 inconclusive only because the sessions are 166 days apart (margin 0.097) |
| Pun 2024, each day vs the previous day (T5, T11) | 19 | 19 consistent |
| Singer-Clark 2025 day 39 and day 202 vs Card sessions 0-37 days away | 3 | 3 consistent |
| Singer-Clark 2025 day 468 vs the nearest Card sessions (75 and 91 days away) | 2 | inconclusive (margins about 0.01) |
| Srinivasan 2026 T15 sentence-loudness vs Card (2024-2025) | 4 | 4 consistent |
| Srinivasan 2026 T15 word-loudness (reader's order) vs Card, Oct-Nov 2023 | 4 | 1 consistent (2023-11-04, margin 0.033), 3 inconclusive |

No mismatches. The only order difference found is the Srinivasan word-loudness one above, which the reader already handles; its support from the fingerprint alone is weak, and rests mainly on the companion scripts. Singer-Clark day 468 relies on its per-electrode labels.

## Dryad downloads (all datasets)

- Dryad answers file downloads with a redirect to a pre-signed S3 URL. A client that forwards its `Authorization` header to S3 gets HTTP 400. `ibci_datasets.dryad` drops the token on cross-host redirects.
