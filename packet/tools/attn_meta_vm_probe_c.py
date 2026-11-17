#!/usr/bin/env python3
"""attn-meta probe c: probe b (tools/attn_meta_vm_probe_b.py, unchanged, which runs probe a's phases) plus one phase,
``selector``, for the per-arm Selector (patches/s5_engine/attn_meta_select.py; lead decision A, condition 5).

selector, per stage (1 then 0), on the device:
  - two fixtures on the same 48-step decode plan (block crossings, admissions, finishes, padding): stock, and one with
    attn_meta_select.install (attn_meta_batch.install unchanged, then the Selector, not selected);
  - the Selector is switched between steps, after a device sync (the paused point the engine driver admits):
    off at step 0, on at step 16, off at step 32;
  - after every step: keys/order, the GDN buffers' bytes, what a captured CUDA graph reads from them (one graph per fixture,
    captured once over the persistent buffers, as probe a's sequence phase) stock vs Selector and against the current
    buffers; a full snapshot every 4th step and at each switch;
  - counters: GdnBatch batched == 16 and every fallback 0, Selector off_calls == 32 and selections == 2;
  - order facts (attn_meta_select.order): the Selector is the instance prepare_attn;
  - CUDA: one profiled call per side, stock vs Selector off: the same device record counts, no host sync inside the window
    (probe b's window rule), i.e. off costs what stock costs on the device.
Verdict rules as probe a (FAIL on any difference or counter mismatch; INCONCLUSIVE when evidence is missing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_vm_probe as A     # noqa: E402  (probe a, unchanged)
import attn_meta_vm_probe_b as B   # noqa: E402  (probe b, unchanged)

SCHEMA = 'attn-meta-vm-probe-c/1'
SWITCH_AT = {0: False, 16: True, 32: False}


class ProbeC(B.ProbeB):
    def __init__(self, args):
        super().__init__(args)
        self.rec['schema'] = SCHEMA
        self.rec['adds'] = dict(phase='selector', module='patches/s5_engine/attn_meta_select.py',
                                switch_at={str(k): v for k, v in SWITCH_AT.items()})

    def phase_selector(self):
        import types
        import attn_meta_select as S
        c, C = self.c, self.C
        plan = C.sequence_plan()
        res, fails, missing = {}, [], []
        for stage in (1, 0):
            a, b = c.fixture(stage, plan['hist']), c.fixture(stage, plan['hist'])
            runner = types.SimpleNamespace(model_state=b.mstate)
            sel = S.install(runner, env={S.ENV: '1'})
            order = S.order(runner)
            c.poison(a)
            c.poison(b)
            ra, oa, kind = self.reader(a)
            rb, ob, _ = self.reader(b)
            seq = C.Sequence(c, stage, plan)
            diff, steps, switches = None, 0, []
            for step in range(plan['steps']):
                if step in SWITCH_AT:
                    self.sync()                                  # the paused point: nothing in flight
                    switches.append(dict(step=step, reply=S.select(runner, SWITCH_AT[step])['after']))
                seq.load(a)
                seq.load(b)
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
                elif step % 4 == 0 or step in (15, 16, 31, 32, plan['steps'] - 1):
                    d = c.first_difference(c.snapshot(a, ma), c.snapshot(b, mb))
                    diff = d and d + '@%d' % step
                steps += 1
                if diff:
                    break
                seq.advance(step)
            cnt = sel.counters()
            row = dict(diff=diff, steps=steps, crossings=seq.crossings, counters=cnt, order=order, reader=kind,
                       switches=switches)
            if diff:
                fails.append('selector:stage%d:%s' % (stage, diff))
            elif (cnt['batched'], cnt['off_calls'], cnt['selections']) != (16, 32, 2) or any(cnt['fallback'].values()):
                fails.append('selector:stage%d:counters %s' % (stage, (cnt['batched'], cnt['off_calls'],
                                                                      cnt['selections'])))
            if not order['selector_outer']:
                fails.append('selector:stage%d:order %s' % (stage, order))
            if self.args.device == 'cuda':
                fs = c.fixture(stage, self.C.HISTS['h65k'])
                fo = c.fixture(stage, self.C.HISTS['h65k'])
                S.install(types.SimpleNamespace(model_state=fo.mstate), env={S.ENV: '1'})   # not selected
                ps, po = self.profiled(lambda: c.call(fs)), self.profiled(lambda: c.call(fo))
                row['opcount_off'] = dict(stock=ps, selector_off=po)
                fails += ['selector:stage%d:opcount:%s' % (stage, f) for f in ps['failures'] + po['failures']]
                missing += ['selector:stage%d:opcount:%s' % (stage, m) for m in ps['missing'] + po['missing']]
                if ps['device'] != po['device'] or ps['runtime'] != po['runtime']:
                    fails.append('selector:stage%d:off-device-ops %s != %s' % (stage, po['device'], ps['device']))
            res['stage%d' % stage] = row
            self.check_mem('selector')
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
    A.PHASES = A.PHASES + ('selector',)            # probe a's run() walks its module's PHASES; the file is unchanged
    return ProbeC(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
