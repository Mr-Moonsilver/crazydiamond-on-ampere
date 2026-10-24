#!/usr/bin/env python3
"""Correlated critical-path breakdown of one decoder's token interval during prefill, across all PP x TP ranks.

Input: the torch-profiler traces of one capture (one file per rank, names containing ppX_tpY). Timestamps of
processes on one host share the host clock, so ranks are aligned directly.
A step is a GPU-side "execute_context..." range (duplicates per stream merged); "context_1" ranges carry prefill, the
others are decode-only. The decoder path: on the reference rank (pp0_tp0), each pair of consecutive decode steps
whose gap contains a prefill step is one interval a decoder waits between two tokens (it passes stage 0 in the first
decode step and next enters it in the second). For every rank the interval is attributed by kernel time (union per
category, so overlapping streams are not double counted) to: host_kv (QSA UVA gathers/copies), gdn, moe, gemm, attn,
tp_allreduce, pp_p2p, other, and idle (no kernel on any stream). Also reports each rank's step durations and the
longest single prefill step (the non-yielding segment). Kernels not matched by a category are listed by time.
usage: critical_path.py TRACE_DIR [pp0_tp0,pp1_tp1]   (optional rank filter)
"""
import bisect, collections, glob, gzip, json, os, re, sys

CATS = [('pp_bcast_wait', r'ncclDevKernel_Broadcast'), ('pp_p2p', r'nccl.*(send|recv)|SendRecv'), ('tp_allreduce', r'nccl.*AllReduce|cross_device_reduce|all_reduce|allreduce'),
        ('host_kv', r'qsa|index_select|gather|kvo|uva|Memcpy|memcpy'), ('gdn', r'gdn|chunk_|fla|fused_recurrent|causal_conv|delta'),
        ('moe', r'marlin|moe|expert|topk'), ('attn', r'flash|attn|fmha|sdpa'), ('gemm', r'gemm|cutlass|sm80|ampere|cublas|matmul')]


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name, re.I):
            return c
    return 'other'


LAUNCH = {}


def load(path):
    ev = json.load(gzip.open(path) if path.endswith('.gz') else open(path))['traceEvents']
    ann = sorted([e for e in ev if e.get('cat') == 'gpu_user_annotation' and e.get('name', '').startswith('execute_context')],
                 key=lambda e: e['ts'])
    steps = []
    for g in ann:
        if steps and g['name'] == steps[-1][0] and g['ts'] <= steps[-1][2]:
            steps[-1][2] = max(steps[-1][2], g['ts'] + g['dur']); continue
        steps.append([g['name'], g['ts'], g['ts'] + g['dur']])
    kern = sorted([(e['ts'], e['ts'] + e['dur'], e['name'], (e.get('args') or {}).get('correlation'))
                   for e in ev if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')])
    launch = {(e.get('args') or {}).get('correlation'): e['ts'] for e in ev
              if e.get('cat') in ('cuda_runtime', 'cuda_driver') and (e.get('args') or {}).get('correlation') is not None}
    LAUNCH[path] = launch
    cpu = sorted([(e['ts'], e['ts'] + e['dur'], e['name'], e.get('cat'), e.get('tid')) for e in ev
                  if e.get('cat') in ('cuda_runtime', 'cuda_driver', 'cpu_op', 'python_function') and e.get('dur')])
    return steps, kern, cpu


def union(intervals):
    tot, cur_a, cur_b = 0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return tot + ((cur_b - cur_a) if cur_b is not None else 0)


def breakdown(kern, t0, t1, unknown, names):
    # idle = no kernel at all except the sampled-token broadcast, which is posted early and spins until the last stage
    # delivers (a waiting GPU, not work)
    by = collections.defaultdict(list); allk = []
    for a, b, n, _corr in kern:
        if b <= t0 or a >= t1:
            continue
        a, b = max(a, t0), min(b, t1)
        c = cat_of(n); by[c].append((a, b))
        if c != 'pp_bcast_wait':
            allk.append((a, b))
        short = re.sub(r'[<(].*', '', n)[:60]
        names[c][short] += b - a
        if c == 'other':
            unknown[short] += b - a
    out = {c: union(v) / 1000 for c, v in by.items()}
    out['idle'] = (t1 - t0 - union(allk)) / 1000
    return out


traces = sorted(glob.glob(os.path.join(sys.argv[1], '*.json*')))
ranks = {}
only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for f in traces:
    if only and not any(o in f for o in only):
        continue
    m = re.search(r'pp(\d+)_tp(\d+)', os.path.basename(f))
    if m:
        ranks[f'pp{m.group(1)}_tp{m.group(2)}'] = load(f)
ref = 'pp0_tp0' if 'pp0_tp0' in ranks else sorted(ranks)[0]
steps = ranks[ref][0]
print('ranks', {r: (len(s), len(k)) for r, (s, k, _) in ranks.items()}, '(steps, kernels); reference', ref)
dec = [s for s in steps if 'context_1' not in s[0]]
pre = [s for s in steps if 'context_1' in s[0]]
intervals = [(a, b) for a, b in zip(dec, dec[1:]) if any(a[2] <= p[1] < b[1] for p in pre)]
print(f'{ref}: {len(dec)} decode steps, {len(pre)} prefill steps, {len(intervals)} decode->decode intervals spanning prefill')
if not intervals:
    sys.exit('no decoder interval spanning a prefill step in this capture')
unknown = collections.Counter()
names = collections.defaultdict(collections.Counter)
tot = {r: collections.Counter() for r in ranks}
wall = []
for a, b in intervals:
    t0, t1 = a[2], b[1]      # end of this decoder's stage-0 decode step -> start of its next one
    wall.append((t1 - t0) / 1000)
    for r, (_, kern, _c) in ranks.items():
        tot[r].update(breakdown(kern, t0, t1, unknown, names))
n = len(intervals)
print(f'decoder wait on stage 0 between its two decode steps: mean {sum(wall)/n:.1f} ms, max {max(wall):.1f} ms (n={n})')
cols = ['host_kv', 'gdn', 'moe', 'gemm', 'attn', 'tp_allreduce', 'pp_p2p', 'other', 'idle', 'pp_bcast_wait']
print('per rank, mean ms per interval:  ' + '  '.join(f'{c:>12}' for c in cols))
for r in sorted(tot):
    print(f'  {r:8}                        ' + '  '.join(f'{tot[r][c] / n:12.1f}' for c in cols))
for r, (st, _, _c) in sorted(ranks.items()):
    d = [(b - a) / 1000 for nm, a, b in st if 'context_1' not in nm]
    p = [(b - a) / 1000 for nm, a, b in st if 'context_1' in nm]
    print(f'  {r}: decode step mean {sum(d)/max(len(d),1):.1f} ms (n={len(d)}), prefill step mean {sum(p)/max(len(p),1):.1f} ms, '
          f'longest prefill step {max(p, default=0):.1f} ms (n={len(p)})')
print('top kernels per category (ms per interval, summed over ranks):')
for c in cols:
    if c in names:
        print(f'  {c}: ' + '; '.join(f'{nm} {us / n / 1000:.1f}' for nm, us in names[c].most_common(4)))


# --- what the reference rank's CPU does while its GPU is idle (gaps > 0.2 ms, broadcast spin excluded) ---
SYNC = re.compile(r'Synchronize|cudaMemcpy$|cudaMemcpy\b(?!Async)|StreamWaitEvent|cuStreamSynchronize|EventQuery', re.I)
_, kern, cpu = ranks[ref]
work = sorted((a, b) for a, b, nm, _c in kern if cat_of(nm) != 'pp_bcast_wait')
# every gap between consecutive work kernels (any size): CPU-late if the next kernel's launch call started after the
# previous kernel had finished (dispatch on the critical path), else GPU-side latency with work already queued
launch = LAUNCH[[f for f in traces if ref.replace('_', '_') in f.replace('dp0_', '')][0]]
wk = [(a, b, c) for a, b, nm, c in kern if cat_of(nm) != 'pp_bcast_wait']
late = early = 0; nlate = nearly = 0; small_hist = collections.Counter()
late_by_cat = collections.Counter(); late_by_op = collections.Counter()
ops = sorted((c[0], c[1], c[2]) for c in cpu if c[3] == 'cpu_op')
op_starts = [o[0] for o in ops]
kname = {c: nm for a, b, nm, c in kern}


def enclosing_op(t):
    j = bisect.bisect_right(op_starts, t)
    best = None
    for o in ops[max(0, j - 400):j]:
        if o[0] <= t <= o[1] and (best is None or o[1] - o[0] < best[1] - best[0]):
            best = o
    return best[2] if best else '?'
for (iv_a, iv_b) in intervals:
    t0, t1 = iv_a[2], iv_b[1]
    seg = [k for k in wk if t0 <= k[0] < t1]
    end = t0
    for ka, kb, corr in seg:
        g = ka - end
        if g > 0:
            lt = launch.get(corr)
            if lt is not None and lt > end:
                late += g; nlate += 1
                late_by_cat[cat_of(kname.get(corr, ''))] += g
                late_by_op[enclosing_op(lt)] += g
            else:
                early += g; nearly += 1
            small_hist[min(int(g // 10) * 10, 200)] += 1
        end = max(end, kb)
graph_launches = [c for c in cpu if c[3] in ('cuda_runtime', 'cuda_driver') and 'GraphLaunch' in c[2]]
for kind, sel in (('prefill', lambda nm: 'context_1' in nm), ('decode', lambda nm: 'context_1' not in nm)):
    st = [x for x in steps if sel(x[0])]
    g = sum(1 for c in graph_launches for x in st if x[1] - 200000 <= c[0] <= x[2])  # launches up to 0.2 s before GPU start
    k = sum(1 for a, b, nm, _c in kern for x in st if x[1] <= a < x[2])
    print(f'{ref} {kind} steps: {len(st)}, cudaGraphLaunch calls near them {g}, kernels per step {k / max(len(st), 1):.0f}')
print(f'{ref} all inter-kernel gaps per interval: CPU-late {late / n / 1000:.1f} ms ({nlate / n:.0f} gaps), '
      f'GPU-side with work queued {early / n / 1000:.1f} ms ({nearly / n:.0f} gaps)')
print('  gap size histogram (us bucket: gaps per interval):', {k: round(v / n) for k, v in sorted(small_hist.items()) if v / n >= 1})
print('  CPU-late gap time by the category of the kernel launched late (ms per interval):',
      {k: round(v / n / 1000, 1) for k, v in late_by_cat.most_common()})
print('  ... by the CPU op that launched it:')
for k, us in late_by_op.most_common(12):
    print(f'    {us / n / 1000:5.1f} ms  {k[:100]}')
attr = collections.Counter(); gap_total = 0
import bisect
starts = [c[0] for c in cpu]
for a, b in intervals:
    t0, t1 = a[2], b[1]
    i = bisect.bisect_left([w[1] for w in work], t0)
    cur = t0
    gaps = []
    for ka, kb in work[i:]:
        if ka >= t1:
            break
        if ka - cur > 200:
            gaps.append((cur, ka))
        cur = max(cur, kb)
    if t1 - cur > 200:
        gaps.append((cur, t1))
    for ga, gb in gaps:
        gap_total += gb - ga
        j = bisect.bisect_right(starts, gb)
        best_sync, best_op = None, None
        for c in cpu[max(0, j - 3000):j]:
            ca, cb, nm, cat, _tid = c
            ov = min(cb, gb) - max(ca, ga)
            if ov <= 0:
                continue
            if cat in ('cuda_runtime', 'cuda_driver') and SYNC.search(nm):
                if best_sync is None or ov > best_sync[0]:
                    best_sync = (ov, nm)
            elif cat == 'cpu_op' and (best_op is None or cb - ca < best_op[1]):
                best_op = (ov, cb - ca, nm)
        key = f'blocked in {best_sync[1]}' if best_sync else (f'cpu op {best_op[2]}' if best_op else 'no CPU event (unaccounted)')
        attr[key] += gb - ga
print(f'{ref} GPU idle in gaps > 0.2 ms: {gap_total / n / 1000:.1f} ms per interval; what its CPU was doing:')
for k, us in attr.most_common(12):
    print(f'  {us / n / 1000:6.1f} ms  {k[:110]}')
