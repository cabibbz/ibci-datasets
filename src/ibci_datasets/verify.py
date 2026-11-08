"""Check that two sets of recordings from the same participant agree on electrode-array order.

Releases from one participant do not always order electrodes the same way (see
ERRATA.md). Electrodes on the same array, and particular pairs of arrays, have
characteristic correlation structure that survives across days and tasks. So:

1. Smooth one neural feature (threshold crossings by default) within each contiguous
   segment and compute the channel x channel correlation matrix of the source and of
   the reference recordings.
2. Treat each array as a block of channels. For every assignment of source blocks to
   reference arrays, score the correlation between the rearranged source matrix and the
   reference matrix (off-diagonal entries).
3. Compare the best assignment with the one the source's channel labels claim.
4. Bootstrap over contiguous chunks of time (independently in source and reference) and
   report how often the same assignment wins.

It tests which block is which array, not channel order inside an array. Correlation
ignores per-channel scale, so counts, rates and z-scored features compare directly.

Two scorings, because each fails differently:

* **plain**: correlate the rearranged matrices as they are. It can pair arrays by their
  overall within-array correlation strength, which preprocessing changes (Rigotti-Thompson
  2026 T11 vs Deo 2026 T11 gave a confident answer, margin 0.100, that two indirect checks
  contradict).
* **block-centred**: subtract each array-pair block's mean first, so only which channel
  pairs correlate more than others counts. Immune to that, but noisy when the
  identifying structure is weak.

A verdict needs both to pick the same assignment, each with margin >= 0.03, bootstrap
support >= 0.95 (a resample counts only if both scorings pick that assignment), and, when
the recordings carry dates or post-implant days, a reference within 120 days.

Calibration (scoping/scripts/calibrate_consensus.py, November 2026): 1,215 T15 comparisons
(Card 2024 vs itself, 990 pairs; Singer-Clark 2025 and Srinivasan 2026 vs Card, 225), point
estimates only, thresholds chosen on the same data. No false verdicts; 483 correct ones
(462 within 120 days, 21 undated), all for correct claimed orders. 90 of the reference
answers (Srinivasan) are themselves inferred, and the rule has not been shown to detect a
known mismatch in real data. In a later review, unrelated data (Deo 2026 T16 labelled as
T15) gave no confident verdict against any of 45 Card sessions. All of this is T15.

The "claimed order trails the best" numbers are diagnostics, not a test. An earlier version
rejected the claimed order when it trailed by >= 0.03 under both scorings; review showed
that fires on data with no usable channel identity (Deo T16 labelled as T15: 18 of 26 Card
sessions within 120 days) and rejected both candidate orders for Srinivasan word-loudness,
so it no longer affects verdicts.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Sequence

import numpy as np

from .schema import Recording

DEFAULT_FEATURES = ("threshold_crossings", "threshold_crossings_smoothed")
MIN_MARGIN = 0.03  # per scoring; no false consensus verdicts in calibration at 0.02 or above
MIN_SUPPORT = 0.95
MAX_GAP_DAYS = 120  # kept as a guard: the calibration is mostly one participant (T15)
# Diagnostic threshold for how far the claimed order trails the best (both scorings). It does
# NOT feed the verdict: see the module docstring.
REJECT_DEFICIT = 0.03


@dataclasses.dataclass
class Result:
    reference_arrays: tuple[str, ...]  # reference array labels, in reference block order
    claimed: tuple[str, ...] | None  # array each source block claims to be (None if labels differ)
    best: tuple[str, ...]  # array each source block matches best
    score_best: float
    score_runner_up: float
    score_claimed: float | None
    support: float  # fraction of bootstrap resamples where `best` wins
    n_bootstrap: int
    feature_source: str
    feature_reference: str
    bins_source: int
    bins_reference: int
    ranking: list[tuple[tuple[str, ...], float]]
    gap_days: float | None = None  # between source and reference, when both carry dates
    min_margin: float = MIN_MARGIN
    best_centred: tuple[str, ...] | None = None  # None if only the plain scoring was run
    margin_centred: float | None = None
    deficit_claimed: float | None = None  # plain: best score minus the claimed order's score
    deficit_claimed_centred: float | None = None
    reject_support: float | None = None  # fraction of resamples where both deficits >= REJECT_DEFICIT

    @property
    def claimed_rejected(self) -> bool:
        """Diagnostic only, not a calibrated test and not used in ``verdict``: the claimed order
        trails the best by >= REJECT_DEFICIT under both scorings in >= 95% of resamples, with a
        known gap <= MAX_GAP_DAYS. It also fires on data that carry no usable channel identity."""
        if self.deficit_claimed is None or self.deficit_claimed_centred is None:
            return False
        # Unlike agreement verdicts, rejections did go wrong beyond 120 days in calibration
        # (a correct order trailed by up to 0.071), so the gap must be known, not just unknown.
        if self.gap_days is None or self.gap_days > MAX_GAP_DAYS:
            return False
        if self.reject_support is not None and self.reject_support < MIN_SUPPORT:
            return False
        return min(self.deficit_claimed, self.deficit_claimed_centred) >= REJECT_DEFICIT

    @property
    def margin(self) -> float:
        """Plain-scoring margin of the best assignment over the runner-up."""
        return self.score_best - self.score_runner_up

    def reasons(self) -> list[str]:
        """Why a verdict is inconclusive (empty if it is not)."""
        out = []
        if len(self.reference_arrays) < 2 or not np.isfinite(self.margin):
            return ["fewer than two arrays: nothing to compare"]
        if self.best_centred is not None and self.best_centred != self.best:
            out.append(f"scorings disagree (block-centred picks {_fmt(self.best_centred)})")
        if self.support < MIN_SUPPORT:
            out.append(f"support {self.support:.2f} < {MIN_SUPPORT}")
        if self.margin < self.min_margin:
            out.append(f"margin {self.margin:.3f} < {self.min_margin}")
        if self.margin_centred is not None and self.margin_centred < self.min_margin:
            out.append(f"block-centred margin {self.margin_centred:.3f} < {self.min_margin}")
        if self.gap_days is not None and self.gap_days > MAX_GAP_DAYS:
            out.append(f"reference {self.gap_days:.0f} days away (> {MAX_GAP_DAYS})")
        return out

    @property
    def verdict(self) -> str:
        if self.reasons():
            return "inconclusive"
        if self.claimed is None:
            return "matched (labels not comparable)"
        return "consistent" if self.best == self.claimed else "mismatch"

    def summary(self) -> str:
        lines = [
            f"verdict      {self.verdict}",
            f"claimed      {_fmt(self.claimed)}" + (f"  r={self.score_claimed:.3f}" if self.score_claimed is not None else "")
            + (f"  trails best by {self.deficit_claimed:.3f} / {self.deficit_claimed_centred:.3f} (plain / centred)"
               + (f"  (diagnostic, not a test: trails by >= {REJECT_DEFICIT} in {self.reject_support:.2f} of resamples)"
                  if self.claimed_rejected and self.reject_support is not None else
                  ("  (rejection needs a known gap within 120 days)"
                   if self.gap_days is None and min(self.deficit_claimed, self.deficit_claimed_centred) >= REJECT_DEFICIT
                   else ""))
               if self.deficit_claimed is not None and self.deficit_claimed_centred is not None else ""),
            f"best         {_fmt(self.best)}  r={self.score_best:.3f}",
            f"runner-up    r={self.score_runner_up:.3f}  (margin {self.margin:.3f})",
            *([f"centred      {_fmt(self.best_centred)}  (margin {self.margin_centred:.3f})"]
              if self.best_centred is not None else []),
            f"support      {self.support:.2f} over {self.n_bootstrap} bootstrap resamples",
            "gap          " + (f"{self.gap_days:.0f} days" if self.gap_days is not None
                               else f"unknown; verdicts are calibrated for references within {MAX_GAP_DAYS} days"),
            f"data         source {self.bins_source} bins of {self.feature_source}; "
            f"reference {self.bins_reference} bins of {self.feature_reference}",
        ]
        if self.reasons():
            lines.insert(1, f"why          {'; '.join(self.reasons())}")
        return "\n".join(lines)


def _fmt(order) -> str:
    return "-" if order is None else ", ".join(order)


def _gap_days(source: Sequence[Recording], reference: Sequence[Recording]) -> float | None:
    """Days between the median source and median reference recording, if both are dated."""
    for attr, conv in (("session_date", lambda d: d.toordinal()), ("post_implant_day", float)):
        vs = [getattr(r, attr) for r in source]
        vr = [getattr(r, attr) for r in reference]
        if all(v is not None for v in vs + vr):
            return abs(float(np.median([conv(v) for v in vs])) - float(np.median([conv(v) for v in vr])))
    return None


def _blocks(rec: Recording) -> list[tuple[str, np.ndarray]]:
    """Contiguous runs of channels sharing an array label, in channel order."""
    arrays = [str(a) for a in rec.channels.arrays]
    out: list[tuple[str, list[int]]] = []
    for i, a in enumerate(arrays):
        if out and out[-1][0] == a:
            out[-1][1].append(i)
        else:
            if any(name == a for name, _ in out):
                raise ValueError(f"array {a!r} is not one contiguous block of channels")
            out.append((a, [i]))
    return [(a, np.array(ix)) for a, ix in out]


def _pick_feature(rec: Recording, name: str | None) -> str:
    names = [f.name for f in rec.features]
    if name:
        if name not in names:
            raise KeyError(f"{rec.session_id}: no feature {name!r} (has {names})")
        return name
    for candidate in DEFAULT_FEATURES:
        if candidate in names:
            return candidate
    raise KeyError(f"{rec.session_id}: none of {DEFAULT_FEATURES} (has {names})")


def _segments(rec: Recording, x: np.ndarray) -> Iterable[np.ndarray]:
    """Runs of finite rows that are contiguous in time."""
    finite = np.isfinite(x).all(axis=1)
    gap = np.diff(rec.timestamps) > 1.5 * rec.bin_size_s
    breaks = np.flatnonzero(gap | ~finite[1:] | ~finite[:-1]) + 1
    for seg in np.split(np.arange(len(x)), breaks):
        if len(seg) and finite[seg[0]]:
            yield x[seg]


def _chunk_stats(recs: Sequence[Recording], feature: str | None, sigma_s: float, chunk_s: float):
    """Per-chunk sums (n, sum x, sum x x^T) of the smoothed feature, plus the order of blocks."""
    from scipy.ndimage import gaussian_filter1d

    blocks = _blocks(recs[0])
    labels = [a for a, _ in blocks]
    used = None
    n_list, s1_list, s2_list = [], [], []
    acc = None  # running (n, s1, s2); short segments (trials) accumulate into one chunk

    def flush():
        nonlocal acc
        if acc is not None and acc[0] >= 2:
            n_list.append(acc[0])
            s1_list.append(acc[1])
            s2_list.append(acc[2])
        acc = None

    for rec in recs:
        if [a for a, _ in _blocks(rec)] != labels:
            raise ValueError(f"{rec.session_id}: array blocks differ from {recs[0].session_id}")
        used = _pick_feature(rec, feature)
        x = rec.feature(used).data
        sigma = sigma_s / rec.bin_size_s
        chunk = max(2, int(round(chunk_s / rec.bin_size_s)))
        for seg in _segments(rec, x):
            seg = seg.astype(np.float64)
            if sigma > 0:
                seg = gaussian_filter1d(seg, sigma, axis=0, mode="nearest")
            for start in range(0, len(seg), chunk):
                c = seg[start:start + chunk]
                if acc is None:
                    acc = [0, np.zeros(c.shape[1]), np.zeros((c.shape[1], c.shape[1]))]
                acc[0] += len(c)
                acc[1] += c.sum(axis=0)
                acc[2] += c.T @ c
                if acc[0] >= chunk:
                    flush()
        flush()  # chunks never span recordings
    if not n_list:
        raise ValueError("no usable data")
    return blocks, used, np.array(n_list, float), np.array(s1_list), np.array(s2_list)


def _corr(w: np.ndarray, n: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    total = w @ n
    mean = (w @ s1) / total
    cov = np.tensordot(w, s2, axes=1) / total - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = cov / np.outer(sd, sd)
    c[~np.isfinite(c)] = 0.0  # silent channels carry no information
    np.fill_diagonal(c, 0.0)
    return c


def _centre(c: np.ndarray, blocks) -> np.ndarray:
    """Subtract each array-pair block's mean (diagonal excluded).

    How strongly an array's channels correlate overall depends on preprocessing (common-
    average or linear-regression referencing removes shared signal) and on the day; which
    channel pairs correlate more than others is what identifies the array. Without this,
    two-array comparisons can pair the more correlated array with the more correlated array
    regardless of identity (seen: Rigotti-Thompson 2026 T11 vs Deo 2026 T11). Centring
    commutes with moving whole blocks, so it is applied once per matrix.
    """
    c = c.copy()
    for _, bi in blocks:
        for _, bj in blocks:
            sub = c[np.ix_(bi, bj)]
            mask = ~np.eye(len(bi), dtype=bool) if bi[0] == bj[0] else np.ones(sub.shape, bool)
            sub[mask] -= sub[mask].mean()
            c[np.ix_(bi, bj)] = sub
    return c


def _rank(cs: np.ndarray, cr: np.ndarray, src_blocks, perms) -> np.ndarray:
    iu = np.triu_indices(cr.shape[0], 1)
    ref = cr[iu]
    scores = np.empty(len(perms))
    for k, perm in enumerate(perms):  # perm[j] = source block placed at reference position j
        idx = np.concatenate([src_blocks[p][1] for p in perm])
        scores[k] = np.corrcoef(cs[np.ix_(idx, idx)][iu], ref)[0, 1]
    return scores


def verify(source: Sequence[Recording], reference: Sequence[Recording], feature: str | None = None,
           reference_feature: str | None = None, sigma_s: float = 0.05, chunk_s: float = 10.0,
           n_bootstrap: int = 200, seed: int = 0, min_margin: float = MIN_MARGIN,
           consensus: bool = True) -> Result:
    """Compare the array order of ``source`` recordings with ``reference`` recordings.

    With ``consensus`` (the default) both the plain and the block-centred scoring must agree;
    ``consensus=False`` runs the plain scoring only (not calibrated as a verdict rule)."""
    if not source or not reference:
        raise ValueError("need at least one source and one reference recording")
    pids = {r.participant_id for r in [*source, *reference]}
    if len(pids) != 1:
        raise ValueError(f"source and reference must be one participant, got {sorted(pids)}")

    sb, sfeat, sn, s1, s2 = _chunk_stats(source, feature, sigma_s, chunk_s)
    rb, rfeat, rn, r1, r2 = _chunk_stats(reference, reference_feature or feature, sigma_s, chunk_s)
    sizes_s, sizes_r = [len(ix) for _, ix in sb], [len(ix) for _, ix in rb]
    if len(sb) != len(rb) or len(set(sizes_s + sizes_r)) != 1:
        raise ValueError(f"block sizes differ: source {sizes_s}, reference {sizes_r}")
    if len(sb) > 8:
        raise ValueError("more than 8 arrays: exhaustive search is too large")
    if [ix.tolist() for _, ix in rb] != [list(range(i * sizes_r[0], (i + 1) * sizes_r[0])) for i in range(len(rb))]:
        raise ValueError("reference arrays must be contiguous blocks in channel order")

    ref_labels = tuple(a for a, _ in rb)
    perms = list(itertools.permutations(range(len(sb))))

    def assignment(perm) -> tuple[str, ...]:  # array each source block is matched to
        return tuple(ref_labels[perm.index(i)] for i in range(len(sb)))

    def score(w_s, w_r):
        """(plain scores, centred scores or None) for one weighting of the chunks."""
        cs, cr = _corr(w_s, sn, s1, s2), _corr(w_r, rn, r1, r2)
        plain = _rank(cs, cr, sb, perms)
        centred = _rank(_centre(cs, sb), _centre(cr, rb), sb, perms) if consensus else None
        return plain, centred

    scores, scores_c = score(np.ones(len(sn)), np.ones(len(rn)))
    order = np.argsort(-scores)
    best = assignment(perms[order[0]])
    best_c = margin_c = None
    if scores_c is not None:
        order_c = np.argsort(-scores_c)
        best_c = assignment(perms[order_c[0]])
        margin_c = float(scores_c[order_c[0]] - scores_c[order_c[1]]) if len(order_c) > 1 else float("nan")

    claimed = tuple(a for a, _ in sb)
    score_claimed = deficit = deficit_c = None
    if set(claimed) == set(ref_labels):
        k = perms.index(tuple(claimed.index(a) for a in ref_labels))
        score_claimed = float(scores[k])
        deficit = float(scores.max() - scores[k])
        if scores_c is not None:
            deficit_c = float(scores_c.max() - scores_c[k])
    else:
        claimed = None

    rng = np.random.default_rng(seed)
    wins = rejects = 0
    k_claimed = perms.index(tuple(claimed.index(a) for a in ref_labels)) if claimed is not None else None
    for _ in range(n_bootstrap):
        ws = np.bincount(rng.integers(0, len(sn), len(sn)), minlength=len(sn)).astype(float)
        wr = np.bincount(rng.integers(0, len(rn), len(rn)), minlength=len(rn)).astype(float)
        b, bc = score(ws, wr)
        win = assignment(perms[int(np.argmax(b))]) == best
        if bc is not None:
            win = win and assignment(perms[int(np.argmax(bc))]) == best
            if k_claimed is not None:
                rejects += min(b.max() - b[k_claimed], bc.max() - bc[k_claimed]) >= REJECT_DEFICIT
        wins += win

    return Result(
        reference_arrays=ref_labels,
        claimed=claimed,
        best=best,
        score_best=float(scores[order[0]]),
        score_runner_up=float(scores[order[1]]) if len(order) > 1 else float("nan"),
        score_claimed=score_claimed,
        support=wins / n_bootstrap if n_bootstrap else float("nan"),
        n_bootstrap=n_bootstrap,
        feature_source=sfeat,
        feature_reference=rfeat,
        bins_source=int(sn.sum()),
        bins_reference=int(rn.sum()),
        ranking=[(assignment(perms[k]), float(scores[k])) for k in order],
        gap_days=_gap_days(source, reference),
        min_margin=min_margin,
        best_centred=best_c,
        margin_centred=margin_c,
        deficit_claimed=deficit,
        deficit_claimed_centred=deficit_c,
        reject_support=(rejects / n_bootstrap if n_bootstrap and k_claimed is not None and consensus else None),
    )
