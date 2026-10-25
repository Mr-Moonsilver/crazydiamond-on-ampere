"""Exact whole-trajectory oracle; payloads remain in memory, never in reports.

CPU-only comparisons. CUDA snapshots are copied out only in the untimed solo
precheck. ULP/absolute differences are diagnostics, never acceptance tolerances.
"""
from dataclasses import dataclass
from itertools import combinations
import math


@dataclass(frozen=True)
class TensorBytes:
    name: str
    dtype: str
    shape: tuple
    data: bytes


def capture(named_tensors):
    import torch
    return [TensorBytes(name, str(t.dtype).removeprefix('torch.'), tuple(t.shape),
                        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            for name, t in named_tensors]


def arrays(t):
    import numpy as np
    types = {'bfloat16': '<u2', 'float16': '<f2', 'float32': '<f4',
             'float64': '<f8', 'uint8': 'u1', 'int8': 'i1', 'int32': '<i4',
             'int64': '<i8', 'bool': '?'}
    a = np.frombuffer(t.data, dtype=types[t.dtype])
    if a.size != math.prod(t.shape):
        raise ValueError('snapshot byte count/shape mismatch')
    floating = t.dtype.startswith('float') or t.dtype == 'bfloat16'
    values = (a.astype(np.uint32) << 16).view(np.float32) if t.dtype == 'bfloat16' else a
    bits = a.view(np.dtype('u'+str(a.dtype.itemsize))) if floating else None
    return a, values, bits


def tensor_diff(a, b):
    import numpy as np
    r = dict(name=a.name, dtype=a.dtype, shape=list(a.shape), equal=False,
             differing_elements=None, max_abs=None, max_ulp=None)
    if (a.name, a.dtype, a.shape) != (b.name, b.dtype, b.shape):
        return dict(r, structural_mismatch=True)
    raw_a, av, ab = arrays(a); raw_b, bv, bb = arrays(b)
    # Raw equality includes signed zero and all state bits. Floating NaNs never
    # pass even if their bit patterns happen to be identical.
    equal = a.data == b.data
    finite = bool(np.isfinite(av).all() and np.isfinite(bv).all())
    byte_a = raw_a.view(np.uint8).reshape(-1, raw_a.dtype.itemsize)
    byte_b = raw_b.view(np.uint8).reshape(-1, raw_b.dtype.itemsize)
    count = 0 if equal else int(np.count_nonzero(np.any(byte_a != byte_b, axis=1)))
    r.update(equal=equal, finite=finite, elements=raw_a.size, differing_elements=count)
    if finite:
        r['max_abs'] = float(np.max(np.abs(av.astype(np.float64)-bv.astype(np.float64)), initial=0)) if not equal else 0.
        if ab is not None:
            # Ordered IEEE representation distance, including signed-zero bits.
            width = raw_a.dtype.itemsize * 8; sign = np.uint64(1 << (width-1))
            mask = np.uint64((1 << width)-1)
            def ordered(bits):
                u = bits.astype(np.uint64)
                return np.where(u & sign != 0, mask-u, u+sign)
            aa, ba = ordered(ab), ordered(bb)
            r['max_ulp'] = int(np.max(np.maximum(aa, ba)-np.minimum(aa, ba), initial=0)) if not equal else 0
    return r


def compare_sequences(a, b):
    if not a or len(a) != len(b):
        raise ValueError('empty/mismatched chunk sequences')
    chunks=[]
    for i, (aa, bb) in enumerate(zip(a, b)):
        if not aa or len(aa) != len(bb) or aa[0].name != 'output' or bb[0].name != 'output':
            raise ValueError('missing output/state snapshots')
        if len({x.name for x in aa}) != len(aa) or len({x.name for x in bb}) != len(bb):
            raise ValueError('duplicate tensor names')
        diffs=[tensor_diff(x, y) for x, y in zip(aa, bb)]
        chunks.append(dict(chunk=i, tensors=diffs))
    return dict(output_equal=all(c['tensors'][0]['equal'] for c in chunks),
                states_equal=all(x['equal'] for c in chunks for x in c['tensors'][1:]),
                finite=all(x.get('finite') is True for c in chunks for x in c['tensors']),
                chunks=chunks)


def assess(host, resident):
    if len(host) != 3 or len(resident) != 3 or any(len(s) != 3 for s in host+resident):
        raise ValueError('requires 3 matched solo trials per path, each with 3 chunks')
    named=[(f'host-{i}', s) for i, s in enumerate(host)]+[(f'resident-{i}', s) for i, s in enumerate(resident)]
    pairs=[]; matches={f'resident-{i}':[] for i in range(3)}
    for (an, a), (bn, b) in combinations(named, 2):
        p=dict(left=an, right=bn, **compare_sequences(a, b)); pairs.append(p)
        if an.startswith('host-') and bn.startswith('resident-') and p['output_equal']:
            matches[bn].append(an)
    stable=all(p['states_equal'] for p in pairs)
    finite=all(p['finite'] for p in pairs)
    # A different host oracle for each chunk would invent a never-observed
    # trajectory. Each complete candidate must match one complete host trial.
    return dict(passed=stable and finite and all(matches.values()),
                host_runs=3, resident_runs=3, chunks_per_run=3,
                all_states_and_ple_bitwise=stable, finite=finite,
                resident_matches_host_sequences=matches, comparisons=pairs,
                policy='whole observed host sequence; all recurrent/conv/PLE bits exact; no tolerance',
                ulp_definition='distance in ordered IEEE bit representation; diagnostics only')


def run(pre, transport, original_kv, resident_kv, prefetch, stream, progress):
    import torch
    from s5_terminal_mode import NullTrace
    samples={'host':[], 'resident':[]}
    # Interleave matched controls to avoid assigning all drift to one path.
    for trial in range(3):
        for name in ('host', 'resident'):
            progress.update('terminal-resident-substitution', solo_path=name, solo_trial=trial)
            transport.drain(); pre.kv=resident_kv if name == 'resident' else original_kv
            transport.prefetch=name == 'resident' and prefetch
            pre.reset(); torch.cuda.synchronize(); sequence=[]
            for _ in range(3):
                with torch.cuda.stream(stream):
                    pre.forward(NullTrace())
                stream.synchronize()
                named=[('output', pre.last_output)]
                named += [(f'state-{i}', t) for i, t in enumerate(pre.state)]
                named += [(f'conv-{i}', t) for i, t in enumerate(pre.conv)]
                named += [('ple_raw', transport.raw), ('ple_output', pre.ple_output)]
                sequence.append(capture(named))
            samples[name].append(sequence)
    progress.update('terminal-resident-substitution-comparison')
    return assess(samples['host'], samples['resident'])
