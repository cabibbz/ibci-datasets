# Release notes and errata

Things we found while building readers against the full public releases, where the files differ from, or go beyond, their documentation. Each reader handles these, and its module docstring has the details. Every entry was checked on the real data (November 2026). Electrode-order checks use `ibci verify-mapping` (see the README); the full audit is summarized at the end. None of this is a criticism of the releases: they are unusually well documented, and these are the gaps that only show up when you read every file.

If you are an author of one of these datasets and something here is wrong, please open an issue.

## Srinivasan 2026, speech modes and loudness (doi:10.5061/dryad.2547d7x5w)

- **Electrode order differs between files.** T15 word-loudness files use the order area 4, v6v, d6v, 55b; T15 sentence-loudness files use v6v, area 4, 55b, d6v, the order documented for Card 2024. The companion analysis scripts give the first order for the 2023 word-loudness session (flag `using_correct_electrode_mapping = 0`). We checked both with `ibci verify-mapping` against Card 2024 sessions recorded by the same arrays (the method recovers the known order of Singer-Clark 2025 files, which label every electrode). The sentence-loudness order is clearly supported. Against Card, the word-loudness order is supported by one same-era session at a margin just over the threshold and inconclusive against three others; but Deo 2026's T15 file, which uses the same older column order, matches the word-loudness columns one to one (margin 0.074), which confirms it. The reader assigns each file its own order and records the evidence in `provenance["electrode_mapping_source"]`. T16's order (55b/PEF, 6v, two hand-knob arrays) comes from the companion scripts; a check against Deo 2026's T16 file was inconclusive (near-tie).
- **Smoothed and raw features have different lengths.** `threshcross` and `spikepow` run 3 s longer per trial than `raw_threshcross`; both start at the same time. The reader NaN-pads the raw counts.
- **Trial timing.** Each trial array starts 1 s before the delay period; `speech_onsets` and `speech_offsets` are 30 kHz sample counts from the go cue (from the companion scripts, confirmed on the data).
- **Control trials.** Two T15 trials are cued `DO NOTHING`, which the README does not mention. Their onset values (3-5 ms after the go cue) are placeholders; the reader sets onset and offset to NaN.

## Deo 2026, whole-body mosaic (doi:10.5061/dryad.mpg4f4rg5)

- **T15 columns are not in Card 2024's order.** Deo names arrays with lab codes (apparently position along the gyrus: `v` ventral, `m` middle, `d` dorsal). T15's columns are `v2, v3, v1, m1`. Against Card 2024, including a session on the same day (2023-08-27), every reference puts dorsal 6v at `v1` and 55b at `m1`; two references (same day, and 9 days away) pass the verdict rules for the full assignment, four are inconclusive. Srinivasan 2026's word-loudness files match Deo's T15 columns one to one (margin 0.074), confirming `v2` = area 4 and `v3` = ventral 6v. So the columns are in the order (area 4, v6v, d6v, 55b), the same older order as Srinivasan's word-loudness files, while Card's own session that day uses (v6v, 4, 55b, d6v). Pooling Deo T15 with Card by column position would mix all four arrays. The reader keeps Deo's codes as array ids, names the inferred area in each T15 channel's location, and records the evidence in `provenance["array_area_evidence"]`.
- **`goCue` vs `state`.** `goCue` (1-based) marks the first Go bin of `state` in every trial for T5, T11, T12, T12_control and T16. In T17 it is one bin (20 ms) earlier in every trial; in T15, in 48% of trials, evenly across blocks. The reader defines trials from `state` and keeps `goCue` as its own column.
- **Trials cut off by block ends.** 8 trials in T12_control, 3 in T15 and 7 in T17 have Go running into the end of a block, with no Return recorded. The reader keeps them, marked `truncated`.
- **`T17.mat` is MATLAB v7.3** (HDF5, matrices transposed), unlike the other files; the README says all load with `scipy.io.loadmat`. The reader handles both.
- **Array correspondence with Pun 2024.** Pun's `a1` / `a2` match Deo's `d2` / `d1` for T5 (margins 0.18-0.21) and T11 (margin 0.057 at day 800; inconclusive at day 658).

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
| Deo 2026 T15 (lab codes) vs Card, 0-9 days away | 6 | 2 matched (4, v6v, d6v, 55b); 4 inconclusive, 3 of them with the same best assignment |
| Srinivasan 2026 T15 word-loudness vs Deo 2026 T15 | 1 | matched one to one (margin 0.074) |
| Srinivasan 2026 T16 vs Deo 2026 T16 | 1 | inconclusive (near-tie) |
| Pun 2024 vs Deo 2026 (T5 first and last day; T11 first and last day) | 4 | 3 matched (`a1`=`d2`, `a2`=`d1`); 1 inconclusive |

No mismatches within any release. Two releases use a column order different from Card 2024's for T15: Srinivasan 2026 word-loudness and Deo 2026 (both the older order, area 4, v6v, d6v, 55b). Both readers handle it. Singer-Clark day 468 relies on its per-electrode labels. ("Matched" means the assignment passed the rules but the two releases name arrays differently, so there is no claimed order to compare against.)

## Dryad downloads (all datasets)

- Dryad answers file downloads with a redirect to a pre-signed S3 URL. A client that forwards its `Authorization` header to S3 gets HTTP 400. `ibci_datasets.dryad` drops the token on cross-host redirects.
