#!/usr/bin/env python3
"""Steady-state reader for replay arms of any length (long-arm packets, lead 14 Nov 2026; also j2/cx 480 s packets).

  python3 -I -B tools/q2024p_j2_long_reader.py --packet <fetched QNAME dir> [--packet ...] [--json-out <fresh.json>]

Per packet the replay pairs are found by index (``sp.g2-live-<i>`` with ``n3.g2-live-<i>``, same seed and length; the
turn files ``arm-<label>.json.turns.jsonl[.zst]``). Turns with an error or no first output are dropped, as in the
review's scripts (investigate-scratch-20261113/review-work-normalised/{lib,norm,steady}.py), and W is the arm's own
``replay_config['DURATION']`` (480 or 1800 s).

The primary statistic (the review's (i), "per-agent time to the same completed turns"): per agent the two arms' turns
are walked in order while kind and reply target agree and both arms completed the turn by W; K is the last such turn.
R = sum over agent streams of T_sp(K) / sum of T_n3(K) (stock time over engine time; > 1 = the engine faster).
Reported per pair, per packet (its pairs pooled: one j2 block = 16 agent streams) and pooled over all packets given
(longa + longb = 16 agent streams on seeds 1 and 2), with G = geometric mean of the packet values. Then
  * ratio (i) the whole arm, (ii) from 180 s on (per agent the first matched turn both arms ended after 180 s is the
    origin: sum (T_sp(K) - T_sp(k0)) / sum (T_n3(K) - T_n3(k0))), (iii) per 300 s slice (a, b]: origin as (ii) at a,
    end the last matched turn both arms completed by b;
  * per agent: the time each arm completed the K-th matched turn for every K, the matched length, whether the arms
    diverged (and at which turn), and the unmatched completion times (k-th completed turn of each arm);
  * cumulative completed turns and work at 60 s steps per arm and agent (work = prefill rows x C + output tokens,
    C = 0.0218 as the review; an open turn's rows count when its first output is in, its output by its arrivals, a
    turn not yet at first output linearly), and per 300 s slice the turn and work ratios n3/sp (descriptive).
CPU only, read-only: no box command, GPU, VM, deletion or push.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

C = 0.0218
STEP_S = 60
SLICE_S = 300
FROM_S = 180
PAIR_PREFIXES = ('sp', 'n3')


def load(path):
    """(header, good turns sorted by send time) from a turns file, the review's lib.load + norm.arm."""
    path = Path(path)
    if path.exists():
        lines = path.read_text().splitlines()
    else:
        zst = Path(str(path) + '.zst')
        lines = subprocess.run(['zstd', '-dc', str(zst)], capture_output=True, check=True).stdout.decode().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    head = rows[0] if rows and rows[0].get('header') else None
    turns = [r for r in rows if not r.get('header')]
    return head, prepare(turns)


def prepare(turns):
    turns = sorted((dict(t) for t in turns), key=lambda r: r['t'])
    prev = {}
    good = []
    for t in turns:
        a = t['agent']
        if 'error' in t or t.get('ttft') is None:
            continue
        t['end'] = t['t'] + t['seconds']
        t['ft'] = t['t'] + t['ttft']
        t['need'] = t['prompt_tokens'] if t['kind'] == 'fresh' or a not in prev else t['prompt_tokens'] - prev[a]
        prev[a] = t['prompt_tokens']
        good.append(t)
    return good


def by_agent(turns):
    d = {}
    for t in turns:
        d.setdefault(t['agent'], []).append(t)
    return d


def matched(x, y, window):
    """(K, diverged_at or None) for one agent: the review's prefix walk."""
    k, diverged = 0, None
    for i in range(min(len(x), len(y))):
        if x[i]['kind'] != y[i]['kind'] or x[i]['reply_target'] != y[i]['reply_target']:
            diverged = i + 1
            break
        if x[i]['end'] <= window and y[i]['end'] <= window:
            k = i + 1
        else:
            break
    return k, diverged


def agents_of(sp, n3, window):
    A, B = by_agent(sp), by_agent(n3)
    out = []
    for a in sorted(A):
        x, y = A[a], B.get(a, [])
        k, diverged = matched(x, y, window)
        out.append(dict(agent=a, K=k, diverged_at=diverged, sp=[round(t['end'], 3) for t in x[:k]],
                        n3=[round(t['end'], 3) for t in y[:k]],
                        sp_completed=[round(e, 3) for e in sorted(t['end'] for t in x if t['end'] <= window)],
                        n3_completed=[round(e, 3) for e in sorted(t['end'] for t in y if t['end'] <= window)]))
    return out


def span(agent, start_s, end_s):
    """(k0, k1) 1-based: k0 the first matched turn both arms ended after start_s (0 = the arm start when start_s is 0),
    k1 the last matched turn both completed by end_s; None when the agent has no matched turn after k0 in the span."""
    sp, n3 = agent['sp'], agent['n3']
    k1 = max((k + 1 for k in range(len(sp)) if sp[k] <= end_s and n3[k] <= end_s), default=0)
    if start_s <= 0:
        return (0, k1) if k1 else None
    k0 = next((k + 1 for k in range(len(sp)) if sp[k] > start_s and n3[k] > start_s), None)
    if k0 is None or k1 <= k0:
        return None
    return k0, k1


def ratio(agents, start_s, end_s):
    ds = dn = 0.0
    used = 0
    for agent in agents:
        s = span(agent, start_s, end_s)
        if s is None:
            continue
        k0, k1 = s
        t0s = agent['sp'][k0 - 1] if k0 else 0.0
        t0n = agent['n3'][k0 - 1] if k0 else 0.0
        ds += agent['sp'][k1 - 1] - t0s
        dn += agent['n3'][k1 - 1] - t0n
        used += 1
    return dict(R=round(ds / dn, 6) if dn > 0 else None, sp_s=round(ds, 3), n3_s=round(dn, 3), agents=used)


def work(turns, cut):
    """The review's norm.work at a cut (partial='lin')."""
    rows = out = 0.0
    n = 0
    for t in turns:
        if t['t'] >= cut:
            continue
        if t['end'] <= cut:
            rows += t['need']
            out += t['completion_tokens']
            n += 1
            continue
        if t['ft'] <= cut:
            rows += t['need']
            ar = t.get('arrivals_ms') or []
            out += t['completion_tokens'] * (sum(1 for a in ar if a / 1000. <= cut) / len(ar) if ar else (cut - t['ft']) / (t['end'] - t['ft']))
        else:
            rows += t['need'] * (cut - t['t']) / (t['ft'] - t['t'])
    return dict(n=n, rows=round(rows, 1), out=round(out, 1), work=round(rows * C + out, 2))


def cuts_of(window, step):
    cuts = list(range(step, int(window) + 1, step))
    return cuts if cuts and cuts[-1] == window else cuts + [window]


def slices_of(window, size):
    edges = list(range(0, int(window), size)) + [window]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def pair_read(sp, n3, window):
    agents = agents_of(sp, n3, window)
    curves = {}
    for name, turns in (('sp', sp), ('n3', n3)):
        A = by_agent(turns)
        curves[name] = dict(arm=[dict(cut=c, **work(turns, c)) for c in cuts_of(window, STEP_S)],
                            agents={str(a): [dict(cut=c, **work(A[a], c)) for c in cuts_of(window, STEP_S)] for a in sorted(A)})
    sl = []
    for a, b in slices_of(window, SLICE_S):
        ws = {k: (work(t, a), work(t, b)) for k, t in (('sp', sp), ('n3', n3))}
        dturn = {k: v[1]['n'] - v[0]['n'] for k, v in ws.items()}
        dwork = {k: v[1]['work'] - v[0]['work'] for k, v in ws.items()}
        sl.append(dict(start_s=a, end_s=b, time_to_same_turns=ratio(agents, a, b),
                       turn_ratio=round(dturn['n3'] / dturn['sp'], 6) if dturn['sp'] else None,
                       work_ratio=round(dwork['n3'] / dwork['sp'], 6) if dwork['sp'] else None, turns=dturn))
    return dict(window_s=window, agents=agents, whole=ratio(agents, 0, window), from_180=ratio(agents, FROM_S, window),
                slices=sl, curves=curves, matched_turns=sum(a['K'] for a in agents),
                completed_turns=dict(sp=sum(1 for t in sp if t['end'] <= window), n3=sum(1 for t in n3 if t['end'] <= window)),
                diverged_agents=sum(1 for a in agents if a['diverged_at'] is not None))


def pool(reads, key):
    """sum the per-agent seconds over several pair reads (the review's sums over 16 agent streams)."""
    ds = sum(r[key]['sp_s'] for r in reads)
    dn = sum(r[key]['n3_s'] for r in reads)
    return dict(R=round(ds / dn, 6) if dn > 0 else None, sp_s=round(ds, 3), n3_s=round(dn, 3), agents=sum(r[key]['agents'] for r in reads))


def pool_slices(reads):
    out = []
    for i, (a, b) in enumerate((s['start_s'], s['end_s']) for s in reads[0]['slices']):
        parts = [r['slices'][i]['time_to_same_turns'] for r in reads if i < len(r['slices'])]
        ds, dn = sum(p['sp_s'] for p in parts), sum(p['n3_s'] for p in parts)
        out.append(dict(start_s=a, end_s=b, R=round(ds / dn, 6) if dn > 0 else None, agents=sum(p['agents'] for p in parts)))
    return out


def find_pairs(packet):
    packet = Path(packet)
    labels = sorted({p.name[4:p.name.index('.json')] for p in packet.glob('arm-*.json.turns.jsonl*')})
    pairs = []
    for index in sorted({lab.rsplit('-', 1)[1] for lab in labels if lab.split('.')[0] in PAIR_PREFIXES}):
        sp, n3 = 'sp.g2-live-' + index, 'n3.g2-live-' + index
        if sp in labels and n3 in labels:
            pairs.append((index, sp, n3))
    return pairs


def packet_read(packet):
    packet = Path(packet)
    pairs = {}
    for index, sp_label, n3_label in find_pairs(packet):
        hs, sp = load(packet / ('arm-%s.json.turns.jsonl' % sp_label))
        hn, n3 = load(packet / ('arm-%s.json.turns.jsonl' % n3_label))
        cs, cn = (hs or {}).get('replay_config') or {}, (hn or {}).get('replay_config') or {}
        assert cs.get('DURATION') == cn.get('DURATION') and cs.get('DURATION'), ('one length a pair', sp_label, cs, cn)
        assert cs.get('SEED') == cn.get('SEED'), ('one seed a pair', sp_label, cs.get('SEED'), cn.get('SEED'))
        read = pair_read(sp, n3, float(cs['DURATION']))
        read.update(seed=cs['SEED'], sp_label=sp_label, n3_label=n3_label)
        pairs[index] = read
    assert pairs, ('no sp/n3 pair in', str(packet))
    reads = list(pairs.values())
    return dict(packet=str(packet), name=packet.name, pairs=pairs, whole=pool(reads, 'whole'), from_180=pool(reads, 'from_180'),
                slices=pool_slices(reads) if len({r['window_s'] for r in reads}) == 1 else None)


def geo(values):
    values = [v for v in values if v]
    return round(math.exp(sum(math.log(v) for v in values) / len(values)), 6) if values else None


def read(packets):
    per = [packet_read(p) for p in packets]
    reads = [r for p in per for r in p['pairs'].values()]
    pooled = dict(whole=pool(reads, 'whole'), from_180=pool(reads, 'from_180'),
                  slices=pool_slices(reads) if len({r['window_s'] for r in reads}) == 1 else None,
                  agent_streams=sum(len(r['agents']) for r in reads), diverged_agents=sum(r['diverged_agents'] for r in reads),
                  matched_turns=sum(r['matched_turns'] for r in reads))
    return dict(schema='q2024p-j2-long-reader/1', C=C, step_s=STEP_S, slice_s=SLICE_S, from_s=FROM_S, packets=per,
                pooled=pooled, G=dict(whole=geo([p['whole']['R'] for p in per]), from_180=geo([p['from_180']['R'] for p in per])),
                definition='R = sum over agent streams of the stock arm time to the last matched turn / the same for n3 '
                           '(matched: same turn index, kind and reply target, both arms completed by W); > 1 = n3 faster')


def text(result):
    lines = []
    for p in result['packets']:
        lines.append('%s  whole R %.4f  from 180 s %.4f' % (p['name'], p['whole']['R'], p['from_180']['R'] or float('nan')))
        for index, r in sorted(p['pairs'].items()):
            lines.append('  pair %s seed %s W %.0f s: whole %.4f from180 %s matched %d of sp %d / n3 %d completed, diverged agents %d' % (
                index, r['seed'], r['window_s'], r['whole']['R'], r['from_180']['R'], r['matched_turns'],
                r['completed_turns']['sp'], r['completed_turns']['n3'], r['diverged_agents']))
            lines.append('    slices: ' + '  '.join('(%d,%d] %s/turn %s/work %s' % (
                s['start_s'], s['end_s'], s['time_to_same_turns']['R'], s['turn_ratio'], s['work_ratio']) for s in r['slices']))
    q = result['pooled']
    lines.append('pooled: whole %s  from 180 s %s  agent streams %d  diverged %d  matched turns %d' % (
        q['whole']['R'], q['from_180']['R'], q['agent_streams'], q['diverged_agents'], q['matched_turns']))
    if q['slices']:
        lines.append('pooled slices: ' + '  '.join('(%d,%d] %s' % (s['start_s'], s['end_s'], s['R']) for s in q['slices']))
    lines.append('G over packets: whole %s  from 180 s %s' % (result['G']['whole'], result['G']['from_180']))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--packet', action='append', required=True, type=Path)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    result = read(args.packet)
    if args.json_out:
        with args.json_out.open('x') as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write('\n')
    sys.stdout.write(text(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
