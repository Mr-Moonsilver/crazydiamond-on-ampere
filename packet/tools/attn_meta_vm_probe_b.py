#!/usr/bin/env python3
"""attn-meta probe b: supersedes tools/attn_meta_vm_probe.py's host-sync count (that file and its record stay as run).

attn-meta-vm-a (results/vm<vm-id>/attn-meta-vm-a) FAILed on ``opcount:stage*:host-sync {'cudaDeviceSynchronize': 1}``
on every side. The probe profiled ``fn(); self.sync()`` and subtracted its own flush, but torch.profiler's stop
(torch/autograd/profiler.py ``profile.__exit__``: ``device_module.synchronize()`` while the profiler is still on)
adds one more ``cudaDeviceSynchronize`` record. The count covered the whole profiler session, not the call.

Probe b changes only the opcount phase:
  - the call runs inside ``torch.profiler.record_function(WINDOW)``; the probe's flush follows it outside the window
    (still inside the profiler, so the device records are complete);
  - runtime launch / copy / sync records are split by time into inside and outside the window by a pure function,
    ``window_counts``, over plain event rows (name, start_us, end_us); a host sync INSIDE the window is a FAIL;
  - the window must be found exactly once and every launch / copy runtime record must fall inside it, else the
    phase is INCONCLUSIVE (the window would not cover the call);
  - positive control ``sync-mutant``: the same profiled call with a real ``torch.cuda.synchronize()`` inside the
    window; the counter must see one ``cudaDeviceSynchronize`` inside, else FAIL (the count is blind).
Device record counts (kernel / memcpy / memset) are counted over the whole profiler session exactly as probe a did.
Every other phase is probe a's code, imported, unchanged.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_vm_probe as A   # noqa: E402  (probe a, unchanged; also sets sys.path for the patches)

SCHEMA = 'attn-meta-vm-probe-b/1'
WINDOW = 'attn_meta_step'
RUNTIME_KINDS = ('LaunchKernel', 'Memcpy', 'Memset', 'Synchronize', 'GraphLaunch')


def is_runtime(name):
    """Probe a's runtime-record filter, unchanged."""
    return name.startswith(('cuda', 'cu')) and any(s in name for s in RUNTIME_KINDS)


def is_sync(name):
    return is_runtime(name) and 'Synchronize' in name


def window_counts(rows, window=WINDOW):
    """rows: iterable of dict(name, start_us, end_us) for host-side (CPU / runtime) profiler events.

    Returns dict(windows=[(start, end)...], inside=Counter, outside=Counter) of runtime records (is_runtime), a record
    being inside when it lies within one window interval (start >= w0 and end <= w1)."""
    rows = list(rows)
    wins = sorted((r['start_us'], r['end_us']) for r in rows if r['name'] == window)
    inside, outside = collections.Counter(), collections.Counter()
    for r in rows:
        if r['name'] == window or not is_runtime(r['name']):
            continue
        hit = any(r['start_us'] >= w0 and r['end_us'] <= w1 for w0, w1 in wins)
        (inside if hit else outside)[r['name']] += 1
    return dict(windows=wins, inside=inside, outside=outside)


def judge(counts):
    """failures, missing for one profiled call's window_counts."""
    fails, missing = [], []
    if len(counts['windows']) != 1:
        missing.append('window-count:%d' % len(counts['windows']))
        return fails, missing
    syncs = {k: v for k, v in counts['inside'].items() if is_sync(k)}
    if syncs:
        fails.append('host-sync %s' % dict(sorted(syncs.items())))
    stray = {k: v for k, v in counts['outside'].items() if not is_sync(k)}
    if stray:
        missing.append('launch-outside-window %s' % dict(sorted(stray.items())))
    return fails, missing


def host_rows(prof, torch):
    """Plain rows from torch.profiler events: host-side events only (device records are not runtime calls)."""
    out = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            continue
        tr = e.time_range
        out.append(dict(name=e.name, start_us=tr.start, end_us=tr.end))
    return out


def device_counts(prof, torch):
    dev = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            nm = e.name
            if nm == WINDOW:                            # the window's own GPU annotation, not a device op
                dev['annotation'] += 1
                continue
            dev['memcpy' if 'Memcpy' in nm or 'memcpy' in nm else 'memset' if 'Memset' in nm or 'memset' in nm
                else 'kernel'] += 1
    return dev


class ProbeB(A.Probe):
    def __init__(self, args):
        super().__init__(args)
        self.rec['schema'] = SCHEMA
        self.rec['supersedes'] = dict(probe='tools/attn_meta_vm_probe.py', record='results/vm<vm-id>/attn-meta-vm-a',
                                      change='host-sync count inside the step window only; sync-mutant control')

    def profiled(self, fn, inject_sync=False):
        torch = self.torch
        from torch.profiler import profile, record_function, ProfilerActivity
        fn()
        self.sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function(WINDOW):
                fn()
                if inject_sync:
                    torch.cuda.synchronize()
            self.sync()                                 # the probe's flush: outside the window
        counts = window_counts(host_rows(prof, torch))
        fails, missing = judge(counts)
        return dict(device=dict(device_counts(prof, torch)),
                    runtime={k: v for k, v in counts['inside'].items() if v},
                    runtime_outside_window={k: v for k, v in counts['outside'].items() if v},
                    windows=len(counts['windows']), failures=fails, missing=missing)

    def opcount_cuda(self, fn):
        return self.profiled(fn)

    def phase_opcount(self):
        c, AMB = self.c, self.AMB
        if self.args.device != 'cuda':
            return super().phase_opcount()
        res, fails, missing = {}, [], []
        for stage in (0, 1):
            row = {}
            for side in ('stock', 'overlay'):
                fx = c.fixture(stage, self.C.HISTS['h65k'])
                if side == 'overlay':
                    c.install(AMB, fx)
                row[side] = self.profiled(lambda: c.call(fx))
                fails += ['opcount:stage%d:%s:%s' % (stage, side, f) for f in row[side]['failures']]
                missing += ['opcount:stage%d:%s:%s' % (stage, side, m) for m in row[side]['missing']]
            fx = c.fixture(stage, self.C.HISTS['h65k'])
            c.install(AMB, fx)
            mut = self.profiled(lambda: c.call(fx), inject_sync=True)
            row['sync_mutant'] = mut
            caught = mut['runtime'].get('cudaDeviceSynchronize', 0) == 1 and any('host-sync' in f
                                                                                for f in mut['failures'])
            row['sync_mutant_caught'] = caught
            if not caught:
                fails.append('opcount:stage%d:sync-mutant-survived' % stage)
            res['stage%d' % stage] = row
            ks, ko = row['stock']['device'].get('kernel', 0), row['overlay']['device'].get('kernel', 0)
            if ks == 0:
                missing.append('opcount:stage%d:no-kernel-records' % stage)
            elif ko >= ks:
                fails.append('opcount:stage%d:overlay %d >= stock %d' % (stage, ko, ks))
            row['syncs_inside_call'] = {k: v for side in ('stock', 'overlay')
                                        for k, v in row[side]['runtime'].items() if is_sync(k)}
        return res, fails, missing


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-config', required=True)
    p.add_argument('--overlay-qsa', required=True)
    p.add_argument('--iters', type=int, default=None)
    a = p.parse_args()
    if a.iters is None:
        a.iters = 1000 if a.device == 'cuda' else 300
    return ProbeB(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
