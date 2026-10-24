"""Verdict logic for leak_partial_tail.py, kept pure so harness/selftest.py can check it offline.

First verdict of one answer (kept forever, never overwritten):
  LEAK     names another session's identifier (any round)
  OK       names its own identifier
  SUSPECT  wrong while the uncached (cold) control is right
  MISS     wrong in both (model failure, not a cache symptom)
A SUSPECT is settled by retries that RECREATE its cache path: prime a fresh cache_salt with the prompt's first H+1
tokens (H = the original hit length, a multiple of the match unit), then ask under that salt; only retries whose hit
length equals H count as the same path. Settled verdict:
  LEAK        any retry text (warm or cold) names a foreign identifier
  COPYFAIL    every same-path retry is wrong while the cold control is right at least twice
  NOISE       at least two same-path retries, all right (not reproducible on the identical path)
  INCONCLUSIVE anything else (e.g. the path was not recreated) -- blocks a clearance
Run verdict: FAIL on LEAK / COPYFAIL / errors / missing results / an engine restart during the run; INCONCLUSIVE on
any INCONCLUSIVE case, MISS above MISS_MAX of the answers, or too little partial-hit coverage; PASS otherwise.
"""


def first_verdict(answer, own, foreign_ids, cold):
    if any(x in answer for x in foreign_ids):
        return 'LEAK'
    if own in answer:
        return 'OK'
    if cold is not None and own in cold:
        return 'SUSPECT'
    return 'MISS'


def settle(retries, own, foreign_ids, hit):
    """retries: [{'warm': text, 'warm_cached': int, 'cold': text}, ...]"""
    if any(x in (r['warm'] or '') + (r['cold'] or '') for r in retries for x in foreign_ids):
        return 'LEAK'
    same = [r for r in retries if r['warm_cached'] == hit]
    cold_ok = sum(own in (r['cold'] or '') for r in retries)
    if same and all(own not in r['warm'] for r in same) and cold_ok >= 2:
        return 'COPYFAIL'
    if len(same) >= 2 and all(own in r['warm'] for r in same):
        return 'NOISE'
    return 'INCONCLUSIVE'


def run_verdict(cases, expected, errors, restarted=False, miss_max=0.05, min_partial=0.8, need_partial=True):
    """cases: one dict per graded answer with 'first', 'settled' (or None), 'partial' (bool: partial-page hit)."""
    final = [c['settled'] or c['first'] for c in cases]
    counts = {v: final.count(v) for v in sorted(set(final))}
    fails = list(errors)
    if restarted:
        fails.append('engine container changed during the run')
    if len(cases) != expected:
        fails.append(f'graded {len(cases)} of {expected} expected answers')
    if counts.get('LEAK') or counts.get('COPYFAIL'):
        fails.append(f"LEAK {counts.get('LEAK', 0)} COPYFAIL {counts.get('COPYFAIL', 0)}")
    inconclusive = []
    if counts.get('INCONCLUSIVE'):
        inconclusive.append(f"{counts['INCONCLUSIVE']} unsettled suspect answers")
    if cases and counts.get('MISS', 0) > miss_max * len(cases):
        inconclusive.append(f"MISS {counts['MISS']} > {miss_max:.0%} of answers")
    if need_partial and cases and sum(c['partial'] for c in cases) < min_partial * len(cases):
        inconclusive.append(f"partial-page hits on {sum(c['partial'] for c in cases)} of {len(cases)} answers")
    verdict = 'FAIL' if fails else ('INCONCLUSIVE' if inconclusive else 'PASS')
    return {'verdict': verdict, 'counts': counts, 'fails': fails, 'inconclusive': inconclusive}
