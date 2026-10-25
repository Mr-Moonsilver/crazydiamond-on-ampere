#!/usr/bin/env python3
"""Matched CoW copy-count grader for the E2 candidate (counter contract: reviews/codex-20261025ai-block-mirror.md).

The candidate's mirror emits, on each rank after every successful frame (zero deltas included):
    FLASH_E2_COW seq=N pairs=N views=N cumulative_pairs=N cumulative_views=N
pairs = runner-filtered host-QSA physical copy pairs of the frame, views = pair applications across the local host-QSA
layer views (= pairs x local host layers), cumulative counters reset only on worker creation. The rank comes from the
Worker_PP?_TP? marker in the engine log line, as for cow_invariants.py. `cow_invariants.py` does not gate CoW
frequency; this grader does, on the same instrumented engine with the feature off and on under one scripted request set.

phase LOG [--layers L]: coverage and consistency of one phase: all four ranks present, seq contiguous (+1) in log order,
    cumulative_pairs/views equal to the running sums, views == pairs x L on every frame (L given, or inferred per rank as
    the constant ratio of the nonzero frames). Exit 0 PASS, 1 FAIL.
compare OFF.log ON.log [--layers L] [--expected-pairs N]: both phases must pass; per rank the phase delta of pairs
    (final cumulative minus the cumulative before the first frame) must be equal off vs on, and equal to
    --expected-pairs when given. Exit 0 PASS (equal), 2 INSPECT (valid counters, unequal totals: evidence to inspect,
    not proof of mutated bytes), 1 FAIL (coverage or consistency broken on either side).
"""
import argparse, re, sys

RANKS = ('Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1')
LINE = re.compile(r'FLASH_E2_COW seq=(\d+) pairs=(\d+) views=(\d+) cumulative_pairs=(\d+) cumulative_views=(\d+)')
RANK = re.compile(r'Worker_PP\d+_TP\d+')


def parse(lines):
    """-> ({rank: [(seq, pairs, views, cum_pairs, cum_views), ...] in log order}, [problems])"""
    out, problems = {}, []
    for l in lines:
        m = LINE.search(l)
        if not m:
            continue
        w = RANK.search(l)
        if not w:
            problems.append(f'counter line without a rank marker: {l[:120]!r}')
            continue
        out.setdefault(w.group(0), []).append(tuple(int(x) for x in m.groups()))
    return out, problems


def phase(lines, layers=None):
    """-> (ok, per_rank, msg); per_rank[rank] = dict(frames, nonzero, delta_pairs, delta_views, final_pairs, layers)"""
    recs, problems = parse(lines)
    missing = [r for r in RANKS if r not in recs]
    extra = [r for r in recs if r not in RANKS]
    if missing or extra:
        problems.append(f'ranks missing {missing} extra {extra}')
    per = {}
    for rank, rs in recs.items():
        seqs = [r[0] for r in rs]
        if any(b != a + 1 for a, b in zip(seqs, seqs[1:])):
            problems.append(f'{rank}: seq not contiguous {seqs[:12]}{"..." if len(seqs) > 12 else ""}')
        base_p, base_v = rs[0][3] - rs[0][1], rs[0][4] - rs[0][2]
        if rs[0][0] == 1 and (base_p or base_v):
            problems.append(f'{rank}: seq 1 starts with cumulative {rs[0][3]}/{rs[0][4]} not equal to its own deltas')
        cp, cv = base_p, base_v
        for seq, p, v, c_p, c_v in rs:
            cp += p; cv += v
            if (c_p, c_v) != (cp, cv):
                problems.append(f'{rank}: seq {seq} cumulative {c_p}/{c_v} but running sums {cp}/{cv}')
                break
        ratios = {v // p for _, p, v, _, _ in rs if p and v % p == 0}
        bad_ratio = [seq for seq, p, v, _, _ in rs if (p == 0) != (v == 0) or (p and v % p)]
        if bad_ratio:
            problems.append(f'{rank}: views not a multiple of pairs at seq {bad_ratio[:8]}')
        L = layers if layers is not None else (ratios.pop() if len(ratios) == 1 else None)
        if layers is not None and any(v != p * layers for _, p, v, _, _ in rs):
            problems.append(f'{rank}: views != pairs x {layers}')
        elif layers is None and len(ratios) > 1:
            problems.append(f'{rank}: inconsistent views/pairs ratios {sorted(ratios)}')
        per[rank] = dict(frames=len(rs), nonzero=sum(1 for r in rs if r[1]), delta_pairs=rs[-1][3] - base_p,
                         delta_views=rs[-1][4] - base_v, final_pairs=rs[-1][3], layers=L, first_seq=rs[0][0], last_seq=rs[-1][0])
    ok = not problems
    msg = ('; '.join(problems) if problems else
           ' '.join(f'{r}:frames={d["frames"]},nonzero={d["nonzero"]},pairs={d["delta_pairs"]},views={d["delta_views"]},L={d["layers"]}'
                    for r, d in sorted(per.items())))
    return ok, per, msg


def compare(off_lines, on_lines, layers=None, expected_pairs=None):
    """-> (code, msg); code 0 PASS, 2 INSPECT, 1 FAIL"""
    ok_off, off, m_off = phase(off_lines, layers)
    ok_on, on, m_on = phase(on_lines, layers)
    if not (ok_off and ok_on):
        return 1, f'FAIL off[{"ok" if ok_off else m_off}] on[{"ok" if ok_on else m_on}]'
    diff = {r: (off[r]['delta_pairs'], on[r]['delta_pairs']) for r in RANKS if off[r]['delta_pairs'] != on[r]['delta_pairs']}
    exp = {r: on[r]['delta_pairs'] for r in RANKS if expected_pairs is not None and on[r]['delta_pairs'] != expected_pairs}
    totals = ' '.join(f'{r}:off={off[r]["delta_pairs"]},on={on[r]["delta_pairs"]}' for r in RANKS)
    if diff or exp:
        return 2, f'INSPECT unequal pairs off/on {diff} expected {expected_pairs} mismatches {exp}; {totals}'
    return 0, f'PASS equal pairs per rank; {totals}; views off[{m_off}] on[{m_on}]'


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('phase'); p.add_argument('log'); p.add_argument('--layers', type=int)
    c = sub.add_parser('compare'); c.add_argument('off'); c.add_argument('on'); c.add_argument('--layers', type=int)
    c.add_argument('--expected-pairs', type=int)
    a = ap.parse_args(argv)
    if a.cmd == 'phase':
        ok, _, msg = phase(open(a.log, errors='replace').read().splitlines(), a.layers)
        print('PASS' if ok else 'FAIL', msg); return 0 if ok else 1
    code, msg = compare(open(a.off, errors='replace').read().splitlines(), open(a.on, errors='replace').read().splitlines(),
                        a.layers, a.expected_pairs)
    print(msg); return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
