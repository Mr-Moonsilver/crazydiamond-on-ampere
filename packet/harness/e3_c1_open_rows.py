"""C1-aware progress and ring rules (alpha-next-c1b, DESIGN-c1-next (a); lead accepted 3 Nov ~10:10 CET). Pure functions, no I/O.

``progress-line-open-rows-carried`` supersedes ``progress-line-with-open-rows`` (e3_native41_progress_box.progress_receipts) on
overlap arms (C1 on, C1 defer) by name; on C1-off arms (C0 local, serial) the old rule stays exactly as it is.

Why: the C1 driver commits chunk n while n+1 is already submitting. OverlapSegmentDriver._reset copies progress.native before
C0's end() flushes n's line with ``open_rows = len(native)`` and restores the copy into n+1's line. An open row at n's flush is a
successor command in flight at the commit instant; it is recorded in n+1's line. Box run q18a-alpha-next-c1: 99 of 99 flagged
flushes were exactly that (results/hub/alpha-next-c1/box-inc/inc-open-rows.json).

For each raw engine flush L of an overlap arm (before any regroup; the arm's engine flushes in line order) with
k = identity.open_rows > 0, every check must hold, else INC with the named reason:
  1. L is not the arm's last engine flush                                             c1-open-rows-at-last-line
  2. the next flush N is txn L.txn+1 and L's stage-overlap line names it as successor  c1-open-rows-successor-mismatch
  3. F = the largest int t_apply over L's 'n' rows (the commit's own release apply); the rows of N whose int t_submit < F
     (the carried set) number exactly k                                               c1-open-rows-not-carried
  4. every carried sequence is in N.txn's stage-overlap seqs and in none of L.txn's    c1-open-row-not-successor  (INC ENGINE:
     a chunk committed with its own command unapplied; the lead's classification for now, not FAIL)
A non-int open_rows is ``c1-open-rows-unreadable``.

``ring_increments``: on overlap arms the stage-local line's ``headroom_blocked_ns`` is a running total across chunks
(OverlapSegmentDriver._maybe_commit re-does C0's per-chunk resets of _both_ns and _headroom_chunk for a successor that began before
the commit, not _ring_blocked_chunk_ns; C0's _begin is where it is reset). The per-chunk value is the rise from the previous line
(a drop is a reset: the value itself). Their sum equals the status counter's blocked_ns delta (box run q18a-alpha-next-c1: exact on
every C1 arm, results/hub/alpha-next-c1/box-inc/inc-ring-lines-2.json).
"""

RULE = 'progress-line-open-rows-carried'
SUPERSEDES = 'progress-line-with-open-rows'
REASONS = ('c1-open-rows-unreadable', 'c1-open-rows-at-last-line', 'c1-open-rows-successor-mismatch', 'c1-open-rows-not-carried',
           'c1-open-row-not-successor')
ENGINE_REASONS = ('c1-open-row-not-successor',)


def overlap_index(texts, parse):
    """txn -> (set of seqs, named successor or None) from stage-overlap line texts; ``parse(text)`` -> (txn, seqs, successor)."""
    out = {}
    for text in texts:
        parsed = parse(text)
        if parsed is None:
            continue
        txn, seqs, successor = parsed
        cell = out.setdefault(txn, [set(), None])
        cell[0].update(seqs)
        cell[1] = successor
    return {t: (frozenset(v[0]), v[1]) for t, v in out.items()}


def _ints(rows, key):
    return [r[key] for r in rows if type(r.get(key)) is int]


def check_arm(flushes, overlap):
    """The rule over one overlap arm. ``flushes``: the arm's raw engine flushes (dicts with 'line', 'identity', 'rows'), any order;
    ``overlap``: overlap_index(...) of the arm's stage-overlap lines. Returns dict(reasons, flagged, checked)."""
    ordered = sorted(flushes, key=lambda f: f.get('line') or 0)
    reasons, flagged = [], []
    for i, flush in enumerate(ordered):
        identity = flush.get('identity') or {}
        k = identity.get('open_rows')
        if type(k) is not int:
            reasons.append('c1-open-rows-unreadable')
            continue
        if k == 0:
            continue
        txn = identity.get('txn')
        cell = dict(line=flush.get('line'), txn=txn, open_rows=k, reasons=[])
        flagged.append(cell)
        if i + 1 >= len(ordered):
            cell['reasons'].append('c1-open-rows-at-last-line')
            continue
        nxt = ordered[i + 1]
        n_txn = (nxt.get('identity') or {}).get('txn')
        cell['next_txn'] = n_txn
        named = (overlap.get(txn) or (frozenset(), None))[1]
        if type(txn) is not int or n_txn != txn + 1 or named != n_txn:
            cell['reasons'].append('c1-open-rows-successor-mismatch')
            continue
        applies = _ints((flush.get('rows') or {}).get('n', []), 't_apply')
        if not applies:
            cell['reasons'].append('c1-open-rows-not-carried')
            continue
        f_ns = max(applies)
        carried = [r for r in (nxt.get('rows') or {}).get('n', []) if type(r.get('t_submit')) is int and r['t_submit'] < f_ns]
        cell['carried_seqs'] = [r.get('seq') for r in carried]
        if len(carried) != k:
            cell['reasons'].append('c1-open-rows-not-carried')
        own = (overlap.get(txn) or (frozenset(), None))[0]
        successor = (overlap.get(n_txn) or (frozenset(), None))[0]
        if any(r.get('seq') not in successor or r.get('seq') in own for r in carried):
            cell['reasons'].append('c1-open-row-not-successor')
    for cell in flagged:
        reasons += cell['reasons']
    return dict(rule=RULE, reasons=sorted(set(reasons)), flagged=flagged, checked=len(ordered))


def ring_increments(values):
    """Per-chunk blocked ns from a running total (in commit order): the rise from the previous value, or the value after a drop
    (a reset). None values are skipped. Returns dict(per_chunk, total, max_chunk, resets)."""
    per, prev, resets = [], None, 0
    for v in values:
        if type(v) is not int:
            continue
        if prev is None or v < prev:
            resets += int(prev is not None)
            per.append(v)
        else:
            per.append(v - prev)
        prev = v
    return dict(per_chunk=per, total=sum(per), max_chunk=max(per) if per else None, resets=resets)
