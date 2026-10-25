#!/usr/bin/env python3
"""Independent freshness check for the E2 drained-state proof (Codex, 25 Oct 2026 16:00 UTC, from the external f020e3a
review; validation freshness, not observed runtime leakage).

e2_state_grade.py drained grades the LAST FLASH_E2_STATE record of every worker, but runner_lifecycle.py emits a STATE
record only when the frame is retired or empty, while EVERY RPC return first emits 'FLASH_E2_COW seq=N ...' with the same
mirror sequence (block_mirror.cow_line / counters: rpc_sequence). So a COW line with a HIGHER seq than the final STATE's
rpc_sequence, or any COW line AFTER the final STATE line, means work completed after the proof: the proof is STALE.

Per worker (the four TP/PP ranks): FRESH when the last STATE record's rpc_sequence equals the highest COW seq and no COW
line follows it; STALE otherwise; missing/malformed STATE or no COW line at all -> cannot prove freshness. Verdict: FRESH
(rc 0) only when all four ranks are fresh; INCONCLUSIVE (rc 2) otherwise. Never FAIL: a completed sequence alone cannot
rule out in-flight work either (a terminal fenced observation is a separate, future requirement), so this check can only
withdraw a drained PASS, not confirm leakage. Nonzero conservative fields (pending_producer_pages, metadata_pages) are
reported, never judged here.

usage: e2_drain_freshness.py ENGINE.log            -> JSON on stdout, rc 0 FRESH / 2 INCONCLUSIVE
"""
import json, re, sys

RANK = re.compile(r'Worker_PP\d+_TP\d+')
COW = re.compile(r'FLASH_E2_COW seq=(\d+) ')
RANKS = ('Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1')


def scan(lines):
    per = {r: dict(state=None, state_index=None, cow_max=None, cow_max_index=None, cow_last_index=None, cow_count=0,
                   state_count=0, malformed_state=0, unranked=0) for r in RANKS}
    for i, line in enumerate(lines):
        if 'FLASH_E2_COW seq=' not in line and 'FLASH_E2_STATE ' not in line:
            continue
        w = RANK.search(line)
        if not w or w.group(0) not in per:
            for r in per: per[r]['unranked'] += 1
            continue
        d = per[w.group(0)]
        m = COW.search(line)
        if m:
            seq = int(m.group(1)); d['cow_count'] += 1; d['cow_last_index'] = i
            if d['cow_max'] is None or seq >= d['cow_max']:
                d['cow_max'], d['cow_max_index'] = seq, i
            continue
        d['state_count'] += 1
        try:
            rec = json.loads(line.split('FLASH_E2_STATE ', 1)[1])
            seq = int(rec['rpc_sequence'])
        except (ValueError, KeyError, TypeError, IndexError):
            d['malformed_state'] += 1; d['state'], d['state_index'] = None, i; continue
        d['state'], d['state_index'] = rec, i
    return per


def grade(lines):
    per = scan(lines); inconclusive, workers = [], {}
    for r in RANKS:
        d = per[r]; rec = d['state']
        row = dict(state_records=d['state_count'], cow_records=d['cow_count'], malformed_state_records=d['malformed_state'],
                   cow_max_seq=d['cow_max'])
        if d['unranked']: inconclusive.append(r + ': %d E2 lines without a rank marker' % d['unranked'])
        if rec is None:
            row['freshness'] = 'missing'
            inconclusive.append(r + ': no well-formed final FLASH_E2_STATE record'); workers[r] = row; continue
        row.update(final_rpc_sequence=int(rec['rpc_sequence']), final_state=rec.get('state'),
                   pending_producer_pages=rec.get('pending_producer_pages'), metadata_pages=rec.get('metadata_pages'))
        if d['cow_max'] is None:
            row['freshness'] = 'unproven'
            inconclusive.append(r + ': no FLASH_E2_COW line, freshness of the final state cannot be shown'); workers[r] = row; continue
        stale = []
        if d['cow_max'] > row['final_rpc_sequence']:
            stale.append('latest completed COW seq %d > final STATE rpc_sequence %d' % (d['cow_max'], row['final_rpc_sequence']))
        elif d['cow_max'] < row['final_rpc_sequence']:
            # Codex (16:05): the proof must match a COMPLETED sequence; a STATE ahead of every COW line lacks completion evidence.
            stale.append('final STATE rpc_sequence %d has no completed COW seq (latest %d)' % (row['final_rpc_sequence'], d['cow_max']))
        if d['cow_last_index'] > d['state_index']:
            stale.append('a COW line (seq %d) follows the final STATE line' % d['cow_max'])
        row['freshness'] = 'stale' if stale else 'fresh'
        for s in stale: inconclusive.append(r + ': ' + s)
        workers[r] = row
    return dict(verdict='INCONCLUSIVE' if inconclusive else 'FRESH', inconclusive=inconclusive, workers=workers,
                scope='proof freshness of the drained-state record vs completed COW sequences; completed seq alone cannot rule '
                      'out in-flight work (terminal fenced observation is a separate requirement); never a FAIL')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    out = grade(open(sys.argv[1], errors='replace').read().splitlines())
    print(json.dumps(out, sort_keys=True))
    sys.exit(0 if out['verdict'] == 'FRESH' else 2)
