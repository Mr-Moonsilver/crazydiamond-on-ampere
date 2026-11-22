#!/usr/bin/env python3
"""Final-engine serve: the per-rank VRAM budget gate and the 40 MiB floor (nvidia-smi physical MiB; stdlib only).

  serve_vram.py gate --arm engine|stock --params serve-params.env --phase <name> --out <json>
      every GPU 0..3 present; free = total - used >= VRAM_FLOOR_MIB on every GPU (both arms, every phase);
      phase 'startup' and arm engine: used <= VRAM_ENGINE_REF_MIB[i] + VRAM_NET_LIMIT_MIB on every GPU
        (the cxfx42 registered startup reading + the peer-vram template's 64 MiB/rank physical net-device limit);
      arm stock: no registered startup reference exists (VRAM_STOCK_REF_MIB=none): floor only, readings recorded.
      Last line: SERVE_VRAM PASS|FAIL|HOLD <phase> per-GPU used/free/budget.  rc 0 PASS, 1 FAIL, 2 HOLD (a GPU missing or
      unreadable: never a pass).
The recorder imports sample() and floor_check() for its 5 s floor sampling.  This is a sampled host-side gate on physical
residency: it cannot see a transient peak between samples, and it is not the engine-side StartupBudget of
codex/n3-peer-vram-budget-template 93b5a379 (that one brackets the decode-layer-stamp banks inside the workers, a feature
cxfx42 does not carry); see READY.
"""
import argparse
import json
import subprocess
import time

GPUS = ('0', '1', '2', '3')



def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

def params(path):
    out = {}
    for line in open(path).read().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def table(value):
    if value in (None, '', 'none'):
        return None
    vals = [int(x) for x in value.split(',')]
    if len(vals) != 4:
        raise ValueError('per-GPU table needs 4 values')
    return dict(zip(GPUS, vals))


def sample():
    """{gpu: (used, total)} from nvidia-smi, or {'error': ...}."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20)
        rows = {}
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(',')]
            rows[parts[0]] = (int(parts[1]), int(parts[2]))
        return rows if out.returncode == 0 else {'error': 'rc%d' % out.returncode}
    except Exception as exc:  # noqa: BLE001 - unreadable is HOLD, never PASS
        _packet_error(exc)
        return {'error': type(exc).__name__}


def floor_check(rows, floor):
    """(status, low) for one sample: PASS / FAIL (a GPU below the floor) / HOLD (missing or unreadable)."""
    if 'error' in rows or set(GPUS) - set(rows):
        return 'HOLD', sorted(set(GPUS) - set(rows)) if 'error' not in rows else [rows['error']]
    low = ['%s:%d' % (g, rows[g][1] - rows[g][0]) for g in GPUS if rows[g][1] - rows[g][0] < floor]
    return ('FAIL' if low else 'PASS'), low


def gate(arm, p, phase, rows):
    floor = int(p['VRAM_FLOOR_MIB'])
    status, low = floor_check(rows, floor)
    ref = table(p['VRAM_ENGINE_REF_MIB'] if arm == 'engine' else p.get('VRAM_STOCK_REF_MIB'))
    net = int(p['VRAM_NET_LIMIT_MIB'])
    over, per = [], {}
    if status != 'HOLD':
        for g in GPUS:
            used, total = rows[g]
            budget = ref[g] + net if (ref and phase == 'startup') else None
            per[g] = dict(used=used, total=total, free=total - used, budget=budget, ref=ref[g] if ref else None)
            if budget is not None and used > budget:
                over.append('%s:%d>%d' % (g, used, budget))
    verdict = 'HOLD' if status == 'HOLD' else 'FAIL' if (status == 'FAIL' or over) else 'PASS'
    return dict(schema='serve-vram/1', arm=arm, phase=phase, verdict=verdict, floor_mib=floor, below_floor=low, over_budget=over,
                per_gpu=per, budget_basis=('cxfx42 registered startup + %d MiB/rank' % net) if (ref and phase == 'startup') else
                'floor only' + ('' if ref else ' (no registered startup reference for this arm)'),
                scope='sampled nvidia-smi physical used/total; not a peak guarantee', t=time.time())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate')
    g.add_argument('--arm', choices=('engine', 'stock'), required=True)
    g.add_argument('--params', required=True)
    g.add_argument('--phase', required=True)
    g.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    p = params(a.params)
    rec = gate(a.arm, p, a.phase, sample())
    with open(a.out, 'x') as handle:
        json.dump(rec, handle, sort_keys=True, indent=1)
    cells = ' '.join('gpu%s %d/%d free %d budget %s' % (g, v['used'], v['total'], v['free'], v['budget']) for g, v in sorted(rec['per_gpu'].items()))
    print('SERVE_VRAM %s %s %s%s%s' % (rec['verdict'], a.phase, cells, (' below-floor ' + ','.join(rec['below_floor'])) if rec['below_floor'] else '',
                                       (' over-budget ' + ','.join(rec['over_budget'])) if rec['over_budget'] else ''))
    return {'PASS': 0, 'FAIL': 1, 'HOLD': 2}[rec['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
