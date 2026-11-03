#!/usr/bin/env python3
"""attn-meta Phase 2 probe: the overlay (patches/s5_engine/attn_meta_batch.py) against stock, on one device.

Runs inside the pinned image. ``--device cpu`` is the hub self-run (Triton launches recorded, not run; the graph
reader is called eagerly); ``--device cuda`` is the VM <vm-id> run (real builder classes on CUDA, real Triton, a real
captured CUDA graph that reads every GDN persistent buffer, CUDA profiler counts). One process, no model, no engine,
no vLLM worker: the fixture and its leaves are tools/attn_meta_witness.py (LEAVES) + tools/attn_meta_cases.py
(LEAVES), plus this file's LEAVES.

Phases (each recorded in probe.json, which is rewritten after every phase so a stop leaves evidence):
  equality   stock vs overlay on every history case, both stages, and the padded cases: dict keys/order, every
             metadata field and tensor location, every builder buffer's bytes.
  sequence   a 48-step decode plan per stage (block crossings, admissions, finishes, padding): after every step
             the GDN buffers' bytes, stock vs overlay, and what a captured graph read from them on replay
             (CUDA: one torch.cuda.CUDAGraph per fixture, captured once over the persistent buffers); full snapshot
             every 4th step.
  opcount    device operations per prepare_attn call, stock vs overlay, per stage. CUDA: torch.profiler kernel,
             memcpy and memset records plus CUDA runtime launch / sync calls; CPU: the witness's meta-device census.
  timing     host µs per call (p50/p90), stock vs overlay, per stage and history; host µs per builder (stock);
             host µs per tiny device op (the launch premium the design modelled); device synchronised between
             calls, outside the timed region.
  native     the native lane's private clones (lane_chunk.private_attention_groups, real code) share no storage
             with the stock builders and stay byte-untouched over overlay steps with the switch on.
Verdict: FAIL on any difference or shared storage; INCONCLUSIVE if a phase could not produce its evidence
(e.g. no CUDA kernel records); PASS otherwise. rc 0 / 1 / 2.
"""
import argparse
import collections
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'tools', ROOT / 'patches/s5_engine'):
    sys.path.insert(0, str(p))

SCHEMA = 'attn-meta-vm-probe/1'
MEM_FRACTION = 0.125           # VM runbook: every GPU process, before allocating
ALLOC_LIMIT = 2 * 1024 ** 3    # this probe's own allocator bound (fails above)
LEAVES = [
    dict(leaf='graph reader', real='the model forward replayed as a FULL CUDA graph, whose GDN kernels read '
         'non_spec_state_indices_tensor / non_spec_query_start_loc at their persistent addresses',
         fixture='a CUDA graph captured once per fixture that copies every GDN persistent buffer (full length) '
         'into private outputs; on CPU the same copies run eagerly',
         why='the FULL graph reads those buffers by address; the reader proves the bytes a replay sees, not GDN '
         'numerics (no model on one 3090)'),
    dict(leaf='greedy tokens', real='token-identical greedy decode, flag off vs on',
         fixture='NOT COVERED: the model does not fit one 3090; the buffers a FULL graph reads are compared instead',
         why='declared, not claimed'),
    dict(leaf='Triton (CPU pass only)', real='real Triton JIT launch on CUDA',
         fixture='CPU: the witness launcher stub records launches; CUDA: real Triton, real build_qsa_metadata path',
         why='Triton has no CPU backend'),
]
PHASES = ('equality', 'sequence', 'opcount', 'timing', 'native')


class Probe:
    def __init__(self, args):
        self.args = args
        self.rec = dict(schema=SCHEMA, device=args.device, phases={}, failures=[], missing=[], leaves=LEAVES,
                        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        self.out = Path(args.out)

    def save(self):
        (self.out / 'probe.json').write_text(json.dumps(self.rec, indent=1, sort_keys=True, default=str) + '\n')

    # -- setup -----------------------------------------------------------------------------------------------
    def setup(self):
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        cuda = self.args.device == 'cuda'
        if cuda:
            torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
            self.device = torch.device('cuda', 0)
            torch.cuda.init()
        else:
            self.device = torch.device('cpu')
        import attn_meta_witness as W
        import attn_meta_cases as C
        import attn_meta_batch as AMB
        self.W, self.C, self.AMB = W, C, AMB
        env = dict(python=sys.version.split()[0], torch=torch.__version__, host=platform.machine(),
                   cpu=platform.processor() or None)
        if cuda:
            env.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                       cuda=torch.version.cuda)
            env['triton'] = 'real'
        else:
            env['triton'] = 'stub ' + W.install_triton_stub()
            import vllm.models.qwen4_exp.common.qsa_cache as qc
            qc.build_qsa_metadata = qc.build_qsa_metadata_triton
        env['overlay_qsa_builder'] = W.install_overlay_builder(Path(self.args.overlay_qsa))
        import tempfile
        work = tempfile.mkdtemp(prefix='attn-meta-probe-', dir='/tmp')
        cfg, notes = W.build_config(self.args.model_config, Path(work), torch)
        env['config'] = notes
        text = json.loads(Path(self.args.model_config).read_text())['text_config']
        self.c = C.Cases(W, cfg, text, torch, np, self.device)
        self.rec['env'] = env
        self.rec['leaves_cases'] = C.LEAVES
        self.rec['leaves_witness'] = W.LEAVES
        self.save()

    def sync(self):
        if self.args.device == 'cuda':
            self.torch.cuda.synchronize()

    def mem(self):
        if self.args.device != 'cuda':
            return None
        t = self.torch.cuda
        return dict(allocated=t.memory_allocated(), max_allocated=t.max_memory_allocated(),
                    reserved=t.memory_reserved(), max_reserved=t.max_memory_reserved())

    def check_mem(self, where):
        m = self.mem()
        if m and m['max_allocated'] > ALLOC_LIMIT:
            raise RuntimeError('allocator above %d bytes at %s: %d' % (ALLOC_LIMIT, where, m['max_allocated']))

    # -- phases ----------------------------------------------------------------------------------------------
    def phase_equality(self):
        c, C, AMB = self.c, self.C, self.AMB
        rows = []
        for stage in (0, 1):
            for name, hist in C.HISTS.items():
                sa, sb, batch = c.compare(AMB, stage, hist)
                rows.append(dict(stage=stage, case=name, diff=c.first_difference(sa, sb),
                                 counters=batch.counters(), layers=len(sa['keys'])))
            for n, R in C.PADDED:
                sa, sb, batch = c.compare(AMB, stage, C.HISTS['boundary'][:n], pad_to=R)
                rows.append(dict(stage=stage, case='padded-%d-%d' % (n, R), diff=c.first_difference(sa, sb),
                                 counters=batch.counters(), layers=len(sa['keys'])))
            self.sync()
            self.check_mem('equality')
        bad = [r for r in rows if r['diff'] or r['counters']['batched'] != 1]
        return dict(cases=rows, differences=len(bad)), ['equality:%s/%s:%s' % (r['stage'], r['case'], r['diff'])
                                                        for r in bad]

    def reader(self, fx):
        torch = self.torch
        srcs = []
        for gid, cls, k, v in self.c.tensors(fx):
            if cls == 'GDNAttentionMetadataBuilder' and k in ('non_spec_state_indices_tensor',
                                                               'non_spec_query_start_loc'):
                srcs.append(v)
        outs = [torch.empty_like(s) for s in srcs]

        def body():
            for o, s in zip(outs, srcs):
                o.copy_(s)
        if self.args.device != 'cuda':
            return body, outs, dict(kind='eager', buffers=len(srcs))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        self.sync()
        return graph.replay, outs, dict(kind='cuda-graph', buffers=len(srcs))

    def phase_sequence(self):
        c, C, AMB = self.c, self.C, self.AMB
        plan = C.sequence_plan()
        res, fails = {}, []
        for stage in (1, 0):
            a, b = c.fixture(stage, plan['hist']), c.fixture(stage, plan['hist'])
            batch, _ = c.install(AMB, b)
            c.poison(a)
            c.poison(b)
            ra, oa, kind = self.reader(a)
            rb, ob, _ = self.reader(b)
            seq = C.Sequence(c, stage, plan)
            sizes, diff, graph_steps = set(), None, 0
            for step in range(plan['steps']):
                n, R = seq.load(a)
                seq.load(b)
                sizes.add((n, R))
                ma, mb = c.call(a), c.call(b)
                ra()
                rb()
                self.sync()
                if list(ma) != list(mb):
                    diff = 'keys/order@%d' % step
                elif c.gdn_buffers(a) != c.gdn_buffers(b):
                    diff = 'gdn-buffers@%d' % step
                elif [o.cpu().numpy().tobytes() for o in oa] != [o.cpu().numpy().tobytes() for o in ob]:
                    diff = 'graph-read@%d' % step
                elif [o.cpu().numpy().tobytes() for o in ob] != [raw for _, _, raw in c.gdn_buffers(b)]:
                    diff = 'graph-read-stale@%d' % step
                elif step % 4 == 0 or step == plan['steps'] - 1:
                    d = c.first_difference(c.snapshot(a, ma), c.snapshot(b, mb))
                    diff = d and d + '@%d' % step
                graph_steps += 1
                if diff:
                    break
                seq.advance(step)
            cnt = batch.counters()
            res['stage%d' % stage] = dict(diff=diff, steps=graph_steps, crossings=seq.crossings,
                                          sizes=sorted(sizes), counters=cnt, reader=kind)
            if diff:
                fails.append('sequence:stage%d:%s' % (stage, diff))
            elif cnt['batched'] != plan['steps']:
                fails.append('sequence:stage%d:batched=%d' % (stage, cnt['batched']))
            self.check_mem('sequence')
        return res, fails

    def opcount_cuda(self, fn):
        torch = self.torch
        from torch.profiler import profile, ProfilerActivity
        fn()
        self.sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            self.sync()
        dev = collections.Counter()
        runtime = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                nm = e.name
                dev['memcpy' if 'Memcpy' in nm or 'memcpy' in nm else 'memset' if 'Memset' in nm or 'memset' in nm
                    else 'kernel'] += 1
            elif e.name.startswith(('cuda', 'cu')) and any(s in e.name for s in (
                    'LaunchKernel', 'Memcpy', 'Memset', 'Synchronize', 'GraphLaunch')):
                runtime[e.name] += 1
        runtime['cudaDeviceSynchronize'] -= 1          # the probe's own flush
        return dict(device=dict(dev), runtime={k: v for k, v in runtime.items() if v})

    def opcount_meta(self, stage, overlay):
        torch, W, c = self.torch, self.W, self.c
        W.TRITON_LAUNCHES.clear()
        fxm = W.Fixture(stage, torch.device('meta'), c.cfg, c.text, torch, self.np, self.C.HISTS['h65k'], {})
        if overlay:
            c.install(self.AMB, fxm)
        cen = W.op_census(fxm, torch)
        tot = collections.Counter()
        for who, r in cen['rows'].items():
            tot['device_op'] += sum(r.get('device_op', {}).values())
            tot['host_op'] += sum(r.get('host_op', {}).values())
            tot['h2d_or_d2h'] += sum(r.get('h2d_or_d2h', {}).values())
        return dict(totals=dict(tot), triton=len(cen['triton_kernels']), failures=cen['failures'])

    def phase_opcount(self):
        c, AMB = self.c, self.AMB
        res, fails, missing = {}, [], []
        for stage in (0, 1):
            row = {}
            for side in ('stock', 'overlay'):
                if self.args.device == 'cuda':
                    fx = c.fixture(stage, self.C.HISTS['h65k'])
                    if side == 'overlay':
                        c.install(AMB, fx)
                    row[side] = self.opcount_cuda(lambda: c.call(fx))
                else:
                    row[side] = self.opcount_meta(stage, side == 'overlay')
            res['stage%d' % stage] = row
            if self.args.device == 'cuda':
                ks, ko = row['stock']['device'].get('kernel', 0), row['overlay']['device'].get('kernel', 0)
                if ks == 0:
                    missing.append('opcount:stage%d:no-kernel-records' % stage)
                elif ko >= ks:
                    fails.append('opcount:stage%d:overlay %d >= stock %d' % (stage, ko, ks))
                syncs = {k: v for side in row for k, v in row[side]['runtime'].items() if 'Synchronize' in k}
                row['syncs_inside_call'] = syncs
                if syncs:
                    fails.append('opcount:stage%d:host-sync %s' % (stage, syncs))
            else:
                ds, do = row['stock']['totals']['device_op'], row['overlay']['totals']['device_op']
                if do >= ds:
                    fails.append('opcount:stage%d:overlay %d >= stock %d' % (stage, do, ds))
                if row['stock']['failures'] or row['overlay']['failures']:
                    fails.append('opcount:stage%d:census-failure' % stage)
        return res, fails, missing

    def timed(self, fn, n, warm):
        pc = time.perf_counter_ns
        for _ in range(warm):
            fn()
        self.sync()
        out = []
        for _ in range(n):
            t0 = pc()
            fn()
            out.append(pc() - t0)
            self.sync()
        return self.W.summ(out)

    def op_micro(self):
        torch, dev, pc = self.torch, self.device, time.perf_counter_ns
        x = torch.arange(8, dtype=torch.int32, device=dev)
        y = torch.empty(10, dtype=torch.int32, device=dev)
        bt = torch.arange(650, dtype=torch.int32, device=dev).reshape(10, 65)[:8]
        idx = torch.zeros(8, 1, dtype=torch.int64, device=dev)
        cases = dict(add=lambda: x + 1, floordiv=lambda: x // 4048, copy_=lambda: y[:8].copy_(x),
                     gather=lambda: torch.gather(bt, 1, idx), arange=lambda: torch.arange(1, dtype=torch.int32, device=dev),
                     slice_view=lambda: y[:8], empty_fill=lambda: y[10:].fill_(0))
        res = {}
        for k, f in cases.items():
            for _ in range(500):
                f()
            self.sync()
            t = []
            for _ in range(15):
                t0 = pc()
                for _ in range(200):
                    f()
                t.append((pc() - t0) / 200)
                self.sync()
            res[k] = round(min(t) / 1e3, 3)
        return res

    def phase_timing(self):
        c, AMB, C = self.c, self.AMB, self.C
        n, warm = (self.args.iters, 100)
        res = dict(steps={}, per_builder={}, op_micro_us=self.op_micro())
        for stage in (0, 1):
            for name in ('h65k', 'h110k'):
                fs = c.fixture(stage, C.HISTS[name])
                fo = c.fixture(stage, C.HISTS[name])
                c.install(AMB, fo)
                res['steps']['stage%d-%s' % (stage, name)] = dict(
                    stock=self.timed(lambda: c.call(fs), n, warm), overlay=self.timed(lambda: c.call(fo), n, warm))
            fs = c.fixture(stage, C.HISTS['h65k'])
            pb = self.W.per_builder(fs, n, warm)
            self.sync()
            res['per_builder']['stage%d' % stage] = pb
            self.check_mem('timing')
        return res, []

    def phase_native(self):
        c, AMB, C = self.c, self.AMB, self.C
        res, fails = {}, []
        for stage in (1, 0):
            fx = c.fixture(stage, C.HISTS['boundary'])
            groups, nbytes, ntensors = c.native_clones(fx)
            shared = c.shared_storage(fx, groups)
            batch, _ = c.install(AMB, fx)
            c.poison(fx, groups)
            before = c.buffers(fx, groups)
            stock_before = c.gdn_buffers(fx)
            for hist in (C.HISTS['boundary'], C.HISTS['h110k'], C.HISTS['boundary'][:5]):
                fx2 = c.fixture(stage, hist, pad_to=8 if len(hist) < 8 else None)
                fx.batch, fx.tables, fx.slots = fx2.batch, fx2.tables, fx2.slots
                c.call(fx)
            self.sync()
            untouched = c.buffers(fx, groups) == before
            res['stage%d' % stage] = dict(clone_tensors=ntensors, clone_bytes=nbytes, shared=shared[:5],
                                          clones_untouched=untouched, stock_written=c.gdn_buffers(fx) != stock_before,
                                          batched=batch.counters()['batched'])
            if shared:
                fails.append('native:stage%d:shared-storage' % stage)
            if not untouched:
                fails.append('native:stage%d:clone-bytes-changed' % stage)
            if batch.counters()['batched'] != 3 or c.gdn_buffers(fx) == stock_before:
                fails.append('native:stage%d:not-engaged' % stage)
        return res, fails

    # -- run -------------------------------------------------------------------------------------------------
    def run(self):
        try:
            self.setup()
        except Exception as exc:
            self.rec['missing'].append('setup:' + type(exc).__name__)
            self.rec['setup_error'] = traceback.format_exc()[-3000:]
            return self.finish()
        for name in PHASES:
            t0 = time.monotonic()
            try:
                out = getattr(self, 'phase_' + name)()
                body, fails = out[0], out[1]
                missing = out[2] if len(out) > 2 else []
                self.rec['phases'][name] = dict(result=body, failures=fails, missing=missing,
                                                wall_s=round(time.monotonic() - t0, 1), memory=self.mem())
                self.rec['failures'] += fails
                self.rec['missing'] += missing
            except Exception as exc:
                self.rec['phases'][name] = dict(error=traceback.format_exc()[-3000:],
                                                wall_s=round(time.monotonic() - t0, 1), memory=self.mem())
                self.rec['missing'].append('%s:%s' % (name, type(exc).__name__))
            print('phase', name, json.dumps({k: v for k, v in self.rec['phases'][name].items() if k != 'result'},
                                            default=str)[:600], flush=True)
            self.save()
        return self.finish()

    def finish(self):
        v = 'FAIL' if self.rec['failures'] else 'INCONCLUSIVE' if self.rec['missing'] else 'PASS'
        self.rec.update(verdict=v, rc={'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[v],
                        finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), memory=self.mem() if hasattr(self, 'torch') else None,
                        box_action=False, engine_constructed=False, gpu_execution=self.args.device == 'cuda')
        self.save()
        print(json.dumps(dict(verdict=v, failures=self.rec['failures'][:10], missing=self.rec['missing'][:10])),
              flush=True)
        return self.rec['rc']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-config', required=True)
    p.add_argument('--overlay-qsa', required=True)
    p.add_argument('--iters', type=int, default=None)
    a = p.parse_args()
    if a.iters is None:
        a.iters = 1000 if a.device == 'cuda' else 300
    return Probe(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
