# Release notes and errata

Things we found while building readers against the full public releases, where the files differ from, or go beyond, their documentation. Each reader handles these, and its module docstring has the details. Every entry was checked on the real data (November 2026). Electrode-order checks use `ibci verify-mapping` (see the README); the full audit is summarized at the end. None of this is a criticism of the releases: they are unusually well documented, and these are the gaps that only show up when you read every file.

If you are an author of one of these datasets and something here is wrong, please open an issue.

## Srinivasan 2026, speech modes and loudness (doi:10.5061/dryad.2547d7x5w)

- **Electrode order differs between files.** T15 sentence-loudness files use v6v, area 4, 55b, d6v, the order documented for Card 2024 (consistent with Card against all four references tested). T15 word-loudness files do not: against all six Card sessions within two weeks of their session, Card's order is rejected (trails the best by 0.10-0.22 / 0.04-0.09 under the plain / block-centred scorings, in every bootstrap resample; the session date 2023-11-04 is taken from the companion scripts, since the files carry none). The companion scripts give these files the lab's flag-0 order, area 4, v6v, d6v, 55b (`using_correct_electrode_mapping = 0`), and the reader uses it; the Card comparisons favour that order under the plain scoring, but the data alone do not settle the exact assignment under the consensus rule. The reader records the evidence in `provenance["electrode_mapping_source"]`. T16's order (55b/PEF, 6v, two hand-knob arrays) comes from the companion scripts; checks against Deo 2026 and Rigotti-Thompson 2026 T16 data were inconclusive (near-ties).
- **Smoothed and raw features have different lengths.** `threshcross` and `spikepow` run 3 s longer per trial than `raw_threshcross`; both start at the same time. The reader NaN-pads the raw counts.
- **Trial timing.** Each trial array starts 1 s before the delay period; `speech_onsets` and `speech_offsets` are 30 kHz sample counts from the go cue (from the companion scripts, confirmed on the data).
- **Control trials.** Two T15 trials are cued `DO NOTHING`, which the README does not mention. Their onset values (3-5 ms after the go cue) are placeholders; the reader sets onset and offset to NaN.

## Deo 2026, whole-body mosaic (doi:10.5061/dryad.mpg4f4rg5)

- **T15 columns are not in Card 2024's order.** Deo names arrays with lab codes (apparently position along the gyrus: `v` ventral, `m` middle, `d` dorsal); T15's columns are `v2, v3, v1, m1`. Two lines of evidence: (1) against Card's session on the same day (2023-08-27), Card's order is rejected for these columns (trails by 0.139 / 0.050, in every resample); against sessions 2-9 days away it trails clearly under the plain scoring only. (2) Deo's T15 columns match Srinivasan 2026's word-loudness columns one to one under both scorings (margins 0.074 / 0.051), and Card's order is rejected for those (above). So Deo T15 uses the same non-Card column order as Srinivasan word-loudness, which its companion scripts give as area 4, v6v, d6v, 55b; hence `v2` = area 4, `v3` = ventral 6v, `v1` = dorsal 6v, `m1` = 55b (55b being mid-gyrus fits the `m` code). Pooling Deo T15 with Card by column position would mix the arrays. The reader keeps Deo's codes as array ids, names the inferred area in each T15 channel's location, and records the evidence in `provenance["array_area_evidence"]`.
- **`goCue` vs `state`.** `goCue` (1-based) marks the first Go bin of `state` in every trial for T5, T11, T12, T12_control and T16. In T17 it is one bin (20 ms) earlier in every trial; in T15, in 48% of trials, evenly across blocks. The reader defines trials from `state` and keeps `goCue` as its own column.
- **Trials cut off by block ends.** 8 trials in T12_control, 3 in T15 and 7 in T17 have Go running into the end of a block, with no Return recorded. The reader keeps them, marked `truncated`.
- **`T17.mat` is MATLAB v7.3** (HDF5, matrices transposed), unlike the other files; the README says all load with `scipy.io.loadmat`. The reader handles both.
- **Array correspondence with Pun 2024.** Pun's `a1` / `a2` match Deo's `d2` / `d1` for T5 (both scorings, margins 0.18-0.21) and T11 (day 800, margins 0.057 / 0.072; inconclusive at day 658).

## Rigotti-Thompson 2026, preparatory activity (doi:10.5061/dryad.brv15dvr9)

- **Arrays are not named.** The reader labels them `a1`... in channel order. In the `fig1-4` files, the channels the authors kept (firing above 2 Hz) concentrate on Deo 2026's most active channels with Rigotti-Thompson's columns taken straight (T5 and T11), which suggests `a1` / `a2` correspond to Deo's `d2` / `d1`; the continuous T11 block's own channel-correlation comparison with Deo is inconclusive (the two scorings disagree, see the audit below).
- **The `delay` and `move` files of a figure hold the same trials**, as two overlapping windows. Each file is converted separately and says so in its provenance.
- **In the continuous `fig5` files, `neural_data_norm` is threshold crossings then spike band power** for the same electrodes in the same order (same-electrode correlation 0.16 for T16 and 0.39 for T11, against about 0.01 otherwise). The README lists both without saying which half is which.

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

`ibci verify-mapping` over every release with a reader. A verdict needs two scorings to agree (plain correlation-structure matching, and the same after subtracting each array-pair block's mean), each with margin >= 0.03 and bootstrap support >= 0.95, with the reference within 120 days when dates are known. On 1,215 comparisons with known answers it gave no false verdicts (and 483 correct ones with the reference within 120 days). An earlier, plain-only version of the rule produced a confident wrong match (Rigotti-Thompson T11 vs Deo T11, margin 0.100), because preprocessing changed which array had the stronger internal correlation; the block-centred scoring was added for that reason. A claimed order is *rejected* when it trails the best by >= 0.03 under both scorings in >= 95% of resamples, with a known gap <= 120 days.

| Comparison | Pairs | Result |
| --- | --- | --- |
| Card 2024, each session vs the previous one (2023-08 to 2025-04) | 44 | 43 consistent; 1 inconclusive only because the sessions are 166 days apart |
| Pun 2024, each day vs the previous day (T5, T11) | 19 | 19 consistent |
| Singer-Clark 2025 day 39 and day 202 vs Card sessions 0-37 days away | 3 | 3 consistent |
| Singer-Clark 2025 day 468 vs the nearest Card sessions (75 and 91 days away) | 2 | inconclusive (margins about 0.01) |
| Srinivasan 2026 T15 sentence-loudness vs Card (2024-2025) | 4 | 4 consistent |
| Srinivasan 2026 T15 word-loudness vs Card, within two weeks | 6 | Card's order rejected in all 6; best assignment not settled (inconclusive) |
| Deo 2026 T15 (lab codes) vs Card, 0-9 days away | 6 | Card's order rejected against the same-day session; best assignment inconclusive in all 6 |
| Srinivasan 2026 T15 word-loudness vs Deo 2026 T15 | 1 | matched one to one (both scorings) |
| Pun 2024 vs Deo 2026 (T5 first and last day; T11 first and last day) | 4 | 3 matched (`a1`=`d2`, `a2`=`d1`); 1 inconclusive |
| Rigotti-Thompson 2026 T11 continuous vs Deo T11 and Pun T11 | 2 | inconclusive (scorings disagree / small margins) |
| T16: Srinivasan vs Deo, Rigotti-Thompson vs Deo, Srinivasan vs Rigotti-Thompson | 3 | inconclusive (near-ties) |

No mismatches within any release. Two releases use a T15 column order other than Card 2024's: Srinivasan 2026 word-loudness and Deo 2026; both readers handle it. T16's array order across releases is unresolved: in every T16 comparison several orders score almost the same. Singer-Clark day 468 relies on its per-electrode labels. ("Matched" means the assignment passed the rules but the releases name arrays differently, so there is no claimed order to compare.) Reproduce with `scoping/scripts/run_audit.py`, `calibrate_consensus.py` and `reject_card_order.py` in the project bundle.

## Dryad downloads (all datasets)

- Dryad answers file downloads with a redirect to a pre-signed S3 URL. A client that forwards its `Authorization` header to S3 gets HTTP 400. `ibci_datasets.dryad` drops the token on cross-host redirects.
