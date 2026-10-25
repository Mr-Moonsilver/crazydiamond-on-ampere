#!/usr/bin/env python3
"""Grade the E2 runtime's own ledger lines in an engine log: startup geometry and the drained state.

startup LOG [--verify 0|1]: every one of the four workers (Worker_PP?_TP? marker) emitted exactly one
    FLASH_E2_STARTUP record with arena_pages 24, page_tokens 4048, incremental_kv_bytes 0, baseline_arena_bytes
    99483648 == arena_budget_bytes // page bytes * page bytes, six local QSA layers at the PP24/24 positions, and (when
    --verify is given) the matching exactness_enabled flag. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE (a worker without
    a record: the engine may not have reached the hook).
drained LOG: the LAST FLASH_E2_STATE record of every worker shows registered_requests, active_owners,
    source_leases, pending_dma, pending_consumers and valid_arena_pages all 0, state idle, rpc_active false, failed
    false, quarantined false. pending_producer_pages and metadata_pages are reported, not required to be 0 (bounded
    cached-source metadata / conservative stream provenance, per the package review). Exit 0 PASS, 1 FAIL,
    2 INCONCLUSIVE (a worker without any state record).
Payloads are never echoed; the summary prints counters only.
"""
import argparse, json, re, sys

RANKS = ('Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1')
RANK = re.compile(r'Worker_PP\d+_TP\d+')
PAGE_BYTES = 4048 * 1024
ZERO = ('registered_requests', 'active_owners', 'source_leases', 'pending_dma', 'pending_consumers', 'valid_arena_pages')


def records(lines, marker):
    out = {}
    for line in lines:
        if marker + ' ' not in line:
            continue
        w = RANK.search(line)
        if not w:
            out.setdefault('?', []).append(None); continue
        try:
            out.setdefault(w.group(0), []).append(json.loads(line.split(marker + ' ', 1)[1]))
        except ValueError:
            out.setdefault(w.group(0), []).append(None)
    return out


def startup(lines, verify=None):
    recs = records(lines, 'FLASH_E2_STARTUP'); fails, missing, summary = [], [], {}
    if '?' in recs: fails.append('startup record without a rank marker')
    for rank in RANKS:
        rs = recs.get(rank, [])
        if not rs: missing.append(rank + ': no startup record'); continue
        if len(rs) != 1 or rs[0] is None: fails.append(rank + ': duplicate or malformed startup record'); continue
        d = rs[0]; pp = int(rank[9])
        try:
            expected = set(range(3 + 24 * pp, 24 + 24 * pp, 4))
            layers = {int(re.findall(r'(?:^|\.)layers\.(\d+)\.', n)[0]) for n in d['local_qsa_layers']}
            ok = (d['pp'] == pp and d['arena_pages'] == 24 and d['page_tokens'] == 4048 and d['incremental_kv_bytes'] == 0 and
                  d['baseline_arena_bytes'] == 99483648 and d['arena_budget_bytes'] // PAGE_BYTES * PAGE_BYTES == d['baseline_arena_bytes'] and
                  layers == expected and len(d['local_qsa_groups']) == 6 and
                  (verify is None or bool(d['exactness_enabled']) == bool(verify)))
        except (KeyError, TypeError, ValueError, IndexError):
            ok = False
        if not ok: fails.append(rank + ': startup geometry/declaration mismatch')
        summary[rank] = {k: d.get(k) for k in ('arena_pages', 'arena_budget_bytes', 'baseline_arena_bytes', 'incremental_kv_bytes',
                                               'pool_blocks', 'host_sources_bytes', 'dma_env_enabled', 'exactness_enabled')}
    return verdict(fails, missing, summary)


def drained(lines):
    recs = records(lines, 'FLASH_E2_STATE'); fails, missing, summary = [], [], {}
    if '?' in recs: fails.append('state record without a rank marker')
    for rank in RANKS:
        rs = recs.get(rank, [])
        if not rs: missing.append(rank + ': no state record'); continue
        d = rs[-1]
        if d is None: fails.append(rank + ': malformed final state record'); continue
        try:
            ok = (all(d[k] == 0 for k in ZERO) and d['state'] == 'idle' and d['rpc_active'] is False and
                  d['failed'] is False and d['quarantined'] is False)
        except (KeyError, TypeError):
            ok = False
        if not ok: fails.append(rank + ': final state not drained/idle')
        summary[rank] = {k: d.get(k) for k in (*ZERO, 'state', 'rpc_active', 'failed', 'quarantined', 'pending_producer_pages',
                                               'metadata_pages', 'rpc_sequence', 'cow_pairs', 'arena_epoch')}
    return verdict(fails, missing, summary)


def verdict(fails, missing, summary):
    v = 'FAIL' if fails else 'INCONCLUSIVE' if missing else 'PASS'
    return dict(verdict=v, fails=fails, missing=missing, workers=summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('startup', 'drained')); p.add_argument('log')
    p.add_argument('--verify', type=int, choices=(0, 1))
    a = p.parse_args()
    lines = open(a.log, errors='replace').read().splitlines()
    out = startup(lines, a.verify) if a.cmd == 'startup' else drained(lines)
    print(json.dumps(out, sort_keys=True))
    sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[out['verdict']])
