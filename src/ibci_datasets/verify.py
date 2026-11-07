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
Calibration (scoping/scripts/calibrate_margin.py, November 2026): Card 2024 has one
constant order across 45 sessions, so every pair of its sessions has a known answer.
Over those 990 pairs plus 225 comparisons of other T15 releases against Card, no wrong
assignment beat the correct one by more than 0.028 when the reference was within 120
days; further apart, wrong assignments reached 0.071. Hence the verdict needs bootstrap
support >= 0.95, a margin >= 0.03 over the runner-up, and, when the recordings carry dates
or post-implant days, a reference within 120 days.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Sequence

import numpy as np

from .schema import Recording

DEFAULT_FEATURES = ("threshold_crossings", "threshold_crossings_smoothed")
MIN_MARGIN = 0.03  # above the largest wrong-assignment margin seen in calibration (0.028)
MIN_SUPPORT = 0.95
MAX_GAP_DAYS = 120  # calibration showed no wrong verdicts above MIN_MARGIN within this


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

    @property
    def margin(self) -> float:
        return self.score_best - self.score_runner_up

    def reasons(self) -> list[str]:
        """Why a verdict is inconclusive (empty if it is not)."""
        out = []
        if self.support < MIN_SUPPORT:
            out.append(f"support {self.support:.2f} < {MIN_SUPPORT}")
        if self.margin < self.min_margin:
            out.append(f"margin {self.margin:.3f} < {self.min_margin}")
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
            f"claimed      {_fmt(self.claimed)}" + (f"  r={self.score_claimed:.3f}" if self.score_claimed is not None else ""),
            f"best         {_fmt(self.best)}  r={self.score_best:.3f}",
            f"runner-up    r={self.score_runner_up:.3f}  (margin {self.margin:.3f})",
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
           n_bootstrap: int = 200, seed: int = 0, min_margin: float = MIN_MARGIN) -> Result:
    """Compare the array order of ``source`` recordings with ``reference`` recordings."""
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

    cs = _corr(np.ones(len(sn)), sn, s1, s2)
    cr = _corr(np.ones(len(rn)), rn, r1, r2)
    scores = _rank(cs, cr, sb, perms)
    order = np.argsort(-scores)
    best = assignment(perms[order[0]])

    claimed = tuple(a for a, _ in sb)
    score_claimed = None
    if set(claimed) == set(ref_labels):
        claimed_perm = tuple(claimed.index(a) for a in ref_labels)
        score_claimed = float(scores[perms.index(claimed_perm)])
    else:
        claimed = None

    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_bootstrap):
        ws = np.bincount(rng.integers(0, len(sn), len(sn)), minlength=len(sn)).astype(float)
        wr = np.bincount(rng.integers(0, len(rn), len(rn)), minlength=len(rn)).astype(float)
        b = _rank(_corr(ws, sn, s1, s2), _corr(wr, rn, r1, r2), sb, perms)
        wins += assignment(perms[int(np.argmax(b))]) == best

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
    )
