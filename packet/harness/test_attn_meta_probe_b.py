"""Tests for tools/attn_meta_vm_probe_b.py (supersedes probe a's host-sync count; probe a and its record unchanged).

Hub (no torch):
  - the window counter (a pure function over plain event rows) on synthetic sessions shaped like vm-a's;
  - mutant: a real ``cudaDeviceSynchronize`` record inside the step window still FAILs (also a stream sync, and a
    sync that straddles the window end is not silently dropped: it counts outside and the launch rule still holds);
  - the committed attn-meta-vm-a record: every stage / side shows exactly one ``cudaDeviceSynchronize`` after probe a
    subtracted its own flush, the same on all four sides whatever the call launched (8..136 launches), no stream /
    event sync, device records equal to the launch records; i.e. raw 2 = probe flush + profiler stop flush, and the
    only sync probe a counted is the stop flush. probe a and the record are pinned by sha256 so this test reads what ran.
Container (pinned image, CPU): torch's ``profile.__exit__`` synchronizes the device while the profiler is still on
(the stop flush, source check on the image's torch), and the real torch.profiler path (``record_function`` window,
``host_rows``) on CPU events places an in-window record inside and the post-window flush outside.
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))

import attn_meta_vm_probe_b as B  # noqa: E402

VM_A = ROOT / 'results/vm<vm-id>/attn-meta-vm-a'


def session(call, flush_probe=True, flush_stop=True, extra_inside=(), extra_outside=()):
    """Synthetic host rows of one profiled call: window [100, 200]; launches inside; flushes after."""
    rows = [dict(name=B.WINDOW, start_us=100.0, end_us=200.0)]
    t = 101.0
    for name, n in call.items():
        for _ in range(n):
            rows.append(dict(name=name, start_us=t, end_us=t + 0.3))
            t += 0.4
    for name in extra_inside:
        rows.append(dict(name=name, start_us=t, end_us=t + 1))
        t += 1.1
    assert t < 200
    if flush_probe:
        rows.append(dict(name='cudaDeviceSynchronize', start_us=201.0, end_us=230.0))
    if flush_stop:
        rows.append(dict(name='cudaDeviceSynchronize', start_us=240.0, end_us=241.0))
    for name in extra_outside:
        rows.append(dict(name=name, start_us=250.0, end_us=251.0))
    rows.append(dict(name='aten::copy_', start_us=110.0, end_us=111.0))   # not a runtime record: ignored
    return rows


CALL_STOCK_S1 = {'cuLaunchKernelEx': 2, 'cudaLaunchKernel': 126, 'cudaMemcpyAsync': 36}
CALL_OVERLAY_S1 = {'cuLaunchKernelEx': 2, 'cudaLaunchKernel': 8}


class Window(unittest.TestCase):
    def test_flushes_outside_window_pass(self):
        for call in (CALL_STOCK_S1, CALL_OVERLAY_S1):
            c = B.window_counts(session(call))
            self.assertEqual(dict(c['inside']), call)
            self.assertEqual(dict(c['outside']), {'cudaDeviceSynchronize': 2})
            self.assertEqual(B.judge(c), ([], []))

    def test_probe_a_rule_on_same_session_fails(self):
        # probe a: whole-session count minus its own flush -> the stop flush is left, a false host-sync
        c = B.window_counts(session(CALL_OVERLAY_S1))
        whole = c['inside'] + c['outside']
        whole['cudaDeviceSynchronize'] -= 1
        self.assertEqual(whole['cudaDeviceSynchronize'], 1)

    def test_mutant_device_sync_inside_window_fails(self):
        c = B.window_counts(session(CALL_OVERLAY_S1, extra_inside=['cudaDeviceSynchronize']))
        fails, missing = B.judge(c)
        self.assertEqual(fails, ["host-sync {'cudaDeviceSynchronize': 1}"])
        self.assertEqual(missing, [])

    def test_mutant_stream_and_event_sync_inside_window_fail(self):
        for nm in ('cudaStreamSynchronize', 'cudaEventSynchronize', 'cuStreamSynchronize'):
            fails, _ = B.judge(B.window_counts(session(CALL_STOCK_S1, extra_inside=[nm])))
            self.assertEqual(fails, ['host-sync %s' % {nm: 1}], nm)

    def test_mutant_without_flushes_still_fails(self):
        c = B.window_counts(session(CALL_OVERLAY_S1, flush_probe=False, flush_stop=False,
                                    extra_inside=['cudaDeviceSynchronize']))
        self.assertTrue(B.judge(c)[0])

    def test_launch_outside_window_inconclusive(self):
        c = B.window_counts(session(CALL_OVERLAY_S1, extra_outside=['cudaLaunchKernel']))
        self.assertEqual(B.judge(c), ([], ["launch-outside-window {'cudaLaunchKernel': 1}"]))

    def test_straddling_sync_counts_outside_and_is_visible(self):
        rows = session(CALL_OVERLAY_S1)
        rows.append(dict(name='cudaDeviceSynchronize', start_us=199.0, end_us=205.0))
        c = B.window_counts(rows)
        self.assertEqual(c['outside']['cudaDeviceSynchronize'], 3)

    def test_window_missing_or_twice_inconclusive(self):
        rows = [r for r in session(CALL_OVERLAY_S1) if r['name'] != B.WINDOW]
        self.assertEqual(B.judge(B.window_counts(rows)), ([], ['window-count:0']))
        rows = session(CALL_OVERLAY_S1) + [dict(name=B.WINDOW, start_us=300.0, end_us=301.0)]
        self.assertEqual(B.judge(B.window_counts(rows)), ([], ['window-count:2']))

    def test_filter_is_probe_a_filter(self):
        src = (ROOT / 'tools/attn_meta_vm_probe.py').read_text()
        self.assertIn("e.name.startswith(('cuda', 'cu')) and any(s in e.name for s in (\n"
                      "                    'LaunchKernel', 'Memcpy', 'Memset', 'Synchronize', 'GraphLaunch'))", src)
        self.assertEqual(B.RUNTIME_KINDS, ('LaunchKernel', 'Memcpy', 'Memset', 'Synchronize', 'GraphLaunch'))
        for nm in ('cudaLaunchKernel', 'cuLaunchKernelEx', 'cudaMemcpyAsync', 'cudaDeviceSynchronize',
                   'cudaGraphLaunch', 'cudaMemsetAsync'):
            self.assertTrue(B.is_runtime(nm), nm)
        for nm in ('aten::copy_', B.WINDOW, 'cudaGetDevice', 'Memcpy HtoD (Pinned -> Device)'):
            self.assertFalse(B.is_runtime(nm), nm)


class VmARecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = json.loads((VM_A / 'probe.json').read_text())
        cls.input = json.loads((VM_A / 'input.json').read_text())

    def test_record_is_what_ran(self):
        # the probe source that ran: the committed vm-a input sums (results/hub/attn-meta) name its sha256
        sums = (ROOT / 'results/hub/attn-meta/vm-a-input-SHA256SUMS').read_text().splitlines()
        rows = {n: d for d, n in (x.split(maxsplit=1) for x in sums)}
        self.assertEqual(rows['w/tools/attn_meta_vm_probe.py'],
                         hashlib.sha256((ROOT / 'tools/attn_meta_vm_probe.py').read_bytes()).hexdigest())
        self.assertEqual(self.input['manifest_sha256'],
                         hashlib.sha256((ROOT / 'results/hub/attn-meta/vm-a-input-SHA256SUMS').read_bytes())
                         .hexdigest())
        self.assertEqual(self.probe['verdict'], 'FAIL')
        self.assertEqual(self.probe['missing'], [])

    def test_probe_a_subtracts_exactly_its_own_flush(self):
        src = (ROOT / 'tools/attn_meta_vm_probe.py').read_text()
        body = src.split('def opcount_cuda', 1)[1].split('\n    def ', 1)[0]
        self.assertEqual(body.count('self.sync()'), 2)          # one warm-up flush (before), one inside profile
        self.assertEqual(body.count("runtime['cudaDeviceSynchronize'] -= 1"), 1)
        self.assertLess(body.index('with profile('), body.rindex('self.sync()'))

    def test_only_failures_are_the_stop_flush(self):
        self.assertEqual(self.probe['failures'], ["opcount:stage0:host-sync {'cudaDeviceSynchronize': 1}",
                                                  "opcount:stage1:host-sync {'cudaDeviceSynchronize': 1}"])
        for name, ph in self.probe['phases'].items():
            if name != 'opcount':
                self.assertEqual(ph['failures'], [], name)
                self.assertEqual(ph['missing'], [], name)

    def test_one_sync_per_session_independent_of_the_call(self):
        res = self.probe['phases']['opcount']['result']
        launches = set()
        for stage in ('stage0', 'stage1'):
            for side in ('stock', 'overlay'):
                r = res[stage][side]
                syncs = {k: v for k, v in r['runtime'].items() if 'Synchronize' in k}
                self.assertEqual(syncs, {'cudaDeviceSynchronize': 1}, (stage, side))   # raw 2: probe + stop
                launch = r['runtime'].get('cudaLaunchKernel', 0) + r['runtime'].get('cuLaunchKernelEx', 0)
                copies = r['runtime'].get('cudaMemcpyAsync', 0)
                self.assertEqual(r['device'].get('kernel', 0), launch, (stage, side))
                self.assertEqual(r['device'].get('memcpy', 0), copies, (stage, side))
                self.assertNotIn('memset', r['device'])
                launches.add(launch)
            self.assertEqual(res[stage]['syncs_inside_call'], {'cudaDeviceSynchronize': 1})
        # the residual sync does not scale with the work: 10 .. 138 launches, one sync each
        self.assertEqual(sorted(launches), [10, 20, 128, 138])

    def test_probe_b_window_rule_reads_vm_a_as_no_sync_inside(self):
        # vm-a has counts only (no timestamps): rebuild each session with the residual sync where the profiler
        # puts it (after the window: stop flush) and judge with probe b; then the mutant placement fails
        res = self.probe['phases']['opcount']['result']
        for stage in ('stage0', 'stage1'):
            for side in ('stock', 'overlay'):
                call = {k: v for k, v in res[stage][side]['runtime'].items() if 'Synchronize' not in k}
                self.assertEqual(B.judge(B.window_counts(session(call))), ([], []))
                self.assertTrue(B.judge(B.window_counts(session(call, flush_stop=False,
                                                                extra_inside=['cudaDeviceSynchronize'])))[0])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'needs the pinned image (torch)')
class Profiler(unittest.TestCase):
    def test_stop_synchronizes_while_on(self):
        import inspect
        import torch.autograd.profiler as P
        src = inspect.getsource(P.profile.__exit__)
        sync = src.index('synchronize()')
        self.assertLess(sync, src.index('_disable_profiler'))     # the flush runs before the profiler stops

    def test_real_profiler_window_on_cpu_events(self):
        from torch.profiler import profile, record_function, ProfilerActivity
        x = torch.ones(8)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function(B.WINDOW):
                x.add_(1)
                with record_function('cudaLaunchKernel'):
                    x.mul_(2)
                with record_function('cudaDeviceSynchronize'):      # mutant: a sync record inside the window
                    x.sub_(1)
            with record_function('cudaDeviceSynchronize'):          # the probe's flush: outside
                x.add_(0)
        c = B.window_counts(B.host_rows(prof, torch))
        self.assertEqual(len(c['windows']), 1)
        self.assertEqual(dict(c['inside']), {'cudaLaunchKernel': 1, 'cudaDeviceSynchronize': 1})
        self.assertEqual(dict(c['outside']), {'cudaDeviceSynchronize': 1})
        self.assertEqual(B.judge(c)[0], ["host-sync {'cudaDeviceSynchronize': 1}"])
        dev = B.device_counts(prof, torch)
        self.assertEqual(dict(dev), {})


if __name__ == '__main__':
    unittest.main()
