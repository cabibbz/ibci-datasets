# Release notes and errata

Things we found while building readers against the full public releases, where the files differ from, or go beyond, their documentation. Each reader handles these, and its module docstring has the details. Every entry was checked on the real data (November 2026). Electrode-order checks use `ibci verify-mapping` (see the README); the full audit is summarized at the end. None of this is a criticism of the releases: they are unusually well documented, and these are the gaps that only show up when you read every file.

If you are an author of one of these datasets and something here is wrong, please open an issue.

## Kunz 2025, inner speech (doi:10.5061/dryad.gf1vhhn1j)

- **Negative counts stored as uint8.** Three T12 sequence-recall files (`t12.2023.05.16_..._joystickSeq.mat`, `t12.2023.05.16_..._joystickSingle.mat`, `t12.2023.09.19_..._lineSequenes.mat`) hold block-mean-subtracted crossings as `uint8`, so -1 to -7 became 255 to 249 (20-23% of values). Read as stored, the per-channel mean is about 60; read as `int8`, about 0.2, like the other block-mean-subtracted files. The reader detects the empty gap between small and near-255 values, reinterprets as `int8` and records the fraction in `provenance["uint8_wraparound_fixed"]`. The repaired file is consistent with a T12 file stored as `int8` nine days later (same array order best under both scorings, every resample; margin 0.014 / 0.029, just under the verdict threshold).
- **T12 keyword-task files carry T15's array labels.** The four `chittychittybangbang` files (T12, December 2024) label their four 64-channel blocks `i6v, M1, 55b, s6v`, T15's names. Every other T12 file names only `i6v` (channels 1-64) and `s6v` (65-128), and T12 has no area 4 or 55b array. Matching channel-correlation structure against two other T12 files puts the block labelled `M1` at `s6v` (plain r 0.41 vs 0.24, block-centred 0.39 vs 0.23, against the interleaved task; similar against sentences); the audit run (below) agrees with margin 0.067 in every resample, but the references are more than 120 days away, so this is not a calibrated verdict. The reader labels the blocks `i6v, s6v, unnamed-1, unnamed-2` and keeps the release's labels in `provenance["release_array_labels"]`.
- **Bin size metadata.** 34 files use 10 ms bins and 33 use 20 ms, which the readmes cover. Two more state other values: `t12.2024.12.19_chittychittybangbang_50words_eval.mat` says 25 ms and `t16.2024.04.29_imagined_50words_eval.mat` 33 ms, against 20 ms for the other files of those sessions. Crossings per bin match the 20 ms files (T16 1.06x, where 33 ms bins would give 1.65x; T12 1.12x against 1.25x), so 20 ms is likely. The reader keeps the stated value and flags it in `provenance["bin_size_suspect"]`.
- **`sentenceFeatures` does not match `binnedTX` in the T12 `sentenceData` files.** In the three 2023-11-28 T12 files, `binnedTX` has 128 channels and `sentenceFeatures` 256 per feature, and they are uncorrelated at the go cue (r about -0.03), contrary to the readme. In the other 16 sentence files (T12 keyword task, T15, T16), `sentenceFeatures` row 0 equals `binnedTX` at the go bin. The reader converts `binnedTX` / `spikePow` and skips `sentenceFeatures`.
- **Epochs are 1-based, inclusive**, as MATLAB indices (`sentenceFeatures` row 0 equals `binnedTX[go - 1]` in those 16 files).
- **Go windows are fixed length** and often overlap the next trial's delay period. In five T16 sequence-recall files the first trial starts before the recording and the last ends after it; the reader clips these and marks them `truncated`.
- **Decoded text is short in one file.** `t16.2024.12.18_imagined_125kwords.mat` has 123 decoded sentences for 125 trials, with nothing to say which two are missing. The reader drops that column for this file and records why in `provenance["dropped_columns"]`.
- **Array names vary within a participant.** T16 is `PEF`, `ventral6v` or `i6v`, and a 128-channel `6d` in most folders, but `PEF/55b`, `i6v`, `i6d`, `s6d` in the sequence-recall folders. The reader uses the last form throughout (originals in provenance). T15 uses `i6v, M1, 55b, s6v`; same-day comparisons with Card 2024 (2024-03-03 and 2024-06-14) match these to Card's v6v, area 4, 55b, d6v one to one (margins 0.449 and 0.185, every resample). T17 is `s6v-v`, `s6v-d` and a 128-channel `55b`.
- **`allBehaviors` files duplicate the per-behaviour files.** In `isolatedVerbalBehaviors`, each participant's `*_allBehaviors_raw.mat` is the seven per-behaviour files of that session concatenated: the same blocks, with identical features and epochs, and each cue suffixed `-<behaviour>` (checked for all four participants). The readme does not mention it. Converting the folder as is would count those trials twice; the reader skips `allBehaviors` when the seven files are present. In T12's `reading` file the rest cue is blank where every other file has `DO_NOTHING`; the reader keeps it as an empty string.
- **No block markers in sequence-recall files.** These files lack `blockNum`; each becomes one Recording, and the provenance says the block boundaries are unknown.
- **AppleDouble files.** The zips include `__MACOSX/._*.mat` resource forks, which are not MATLAB files. The reader skips them.

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
| Kunz 2025 T15 (`i6v, M1, 55b, s6v`) vs Card, same day (2024-03-03, 2024-06-14) | 2 | 2 matched: `i6v`=v6v, `M1`=4, `55b`=55b, `s6v`=d6v (margins 0.449, 0.185) |
| Kunz 2025 T12: wrap-repaired file vs an `int8` file 9 days later; keyword task (relabelled) vs sequence recall 435 days earlier | 2 | inconclusive; best order = the reader's in both, every resample (margin 0.014; the other too far apart) |
| T16: Srinivasan vs Deo, Rigotti-Thompson vs Deo, Srinivasan vs Rigotti-Thompson | 3 | inconclusive (near-ties) |
| T16: Kunz 2025 (`PEF-55b, i6v, i6d, s6d`) vs Deo, Srinivasan and Rigotti-Thompson; Kunz sentences and sequence recall vs Kunz isolated (35-56 days) | 5 | inconclusive (near-ties, margins 0.000-0.027), even within Kunz |

No mismatches within any release. Two releases use a T15 column order other than Card 2024's: Srinivasan 2026 word-loudness and Deo 2026; both readers handle it. Kunz 2025's T15 order matches Card's. T16's array order across releases is unresolved: in every T16 comparison several orders score almost the same, including between Kunz 2025 files that name their arrays identically, so the method cannot tell T16's arrays apart. Singer-Clark day 468 relies on its per-electrode labels. ("Matched" means the assignment passed the rules but the releases name arrays differently, so there is no claimed order to compare.) Reproduce with `scoping/scripts/run_audit.py`, `calibrate_consensus.py` and `reject_card_order.py` in the project bundle.

## Dryad downloads (all datasets)

- Dryad answers file downloads with a redirect to a pre-signed S3 URL. A client that forwards its `Authorization` header to S3 gets HTTP 400. `ibci_datasets.dryad` drops the token on cross-host redirects.
