#!/usr/bin/env python3
"""decode-step (lead 13 Nov 2026): FLASH_S5_DECODE_SAMPLE_FAST and FLASH_S5_DECODE_ANATOMY, hub CPU suite.

The sampling math itself (token identity against the full-vocab path, ties, source-text mutants) needs torch: it is
``test_decode_step_torch.py`` in the pinned image. This suite covers the parts that need no torch.
  * Off: neither switch imports ``decode_step``. The RPC wire, through the real owned ``collective_rpc``, is the
    given kwargs (plus the progress meta while recording), and the worker hooks' state is untouched.
  * The engine gate is ``auto_arm.manual_owned`` read before each step. The sp arm holds the manual lease, so nothing
    is tagged there. Only ``sample_tokens`` carries D's bit.
  * The clean anatomy window: opens only while the gate is on, no native progress records and the policy is idle;
    closes at 20 s, on any break, at the budget and at the window cap; the next RPC carries the close tag; the core's
    stamp lines stay bounded and under the line size.
  * The worker, through the real generated WorkerProc reply path and the real handler-anatomy recorder
    (test_stage1_handler_anatomy's rig): the tag is popped before dispatch (the leaf worker takes no extra kwargs).
    Tagged RPCs become anatomy rows and never progress-trace records. The close tag flushes them as handler lines.
    Untagged RPCs stay stock. A source-text mutant that lets clean rows into the trace fails.
  * The sampler gate: every ineligible step calls the stock ``_sample_random`` with the same arguments and returns its
    value, counted by reason. The wrapper is set on the instance.
  * Names against the pinned V2 sampler source: ``_sample_random``'s parameters, its call in ``sample``, the
    FlashInfer branch, ``sampling_states.top_k`` as a UVA-backed tensor with ``.np``.
  * Startup refusals and the overlay declarations.
CPU only; no box, GPU or torch.
"""
import ast
from contextlib import ExitStack
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
from types import SimpleNamespace as NS
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import async_handoff as ah  # noqa: E402
import decode_step as ds  # noqa: E402
import executor_hooks as eh  # noqa: E402
import native_measurement as nm  # noqa: E402

def _vllm_source(rel):
    """The pinned image's vllm source: the hub's reference tree, a bundle's e3_reference, else the image's installed
    package (found, never imported). Missing everywhere is a failure, not a skip."""
    import importlib.util
    roots = [ROOT / 'reference/vllm-src', ROOT / 'e3_reference', HERE / 'e3_reference']
    spec = importlib.util.find_spec('vllm')
    if spec is not None and spec.submodule_search_locations:
        roots.append(Path(list(spec.submodule_search_locations)[0]))
    for root in roots:
        if (root / rel).is_file():
            return root / rel
    return roots[0] / rel


SAMPLER_SRC = _vllm_source('v1/worker/gpu/sample/sampler.py')
STATES_SRC = _vllm_source('v1/worker/gpu/sample/states.py')


class Log:
    def __init__(self):
        self.lines = []

    def info(self, text, *args):
        self.lines.append(text % args if args else text)

    warning = info

    def of(self, prefix):
        return [json.loads(t[len(prefix):]) for t in self.lines if t.startswith(prefix)]


class Clock:
    def __init__(self, ns=10 ** 15):
        self.ns = ns

    def __call__(self):
        return self.ns

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)


def engine(manual_owned=False, *, policy=True, active=False, progress=None, queue=0, running=8, waiting=0):
    auto = NS(manual_owned=manual_owned, active=active) if policy else None
    executor = NS(_flash_s5_progress_recording=progress)
    return NS(_flash_s5_driver=NS(auto_arm=auto), model_executor=executor, batch_queue=[0] * queue,
              scheduler=NS(running=[0] * running, waiting=[0] * waiting))


# --- off ----------------------------------------------------------------------------------------------------------------
OFF_PROBE = r'''
import json, os, sys
sys.path[:0] = [sys.argv[1] + '/patches/s5_engine']
import executor_hooks as eh
from types import SimpleNamespace as NS
proc = NS(rank=0)
kwargs = {'x': 1}
first = eh.worker_rpc_begin(proc, 'execute_model', kwargs)
eng = NS(_flash_s5_driver=None, model_executor=NS())
side = eh._decode_step_engine(eng)
print(json.dumps(dict(imported='decode_step' in sys.modules, first=first, kwargs=kwargs, side=side is None,
                      state=eh._DECODE_STEP if eh._DECODE_STEP is False else 'module', attr=hasattr(eng.model_executor, '_flash_s5_decode_step'))))
'''


class Off(unittest.TestCase):
    def test_off_in_a_fresh_process_never_imports_decode_step(self):
        env = {k: v for k, v in os.environ.items() if k not in (ds.SAMPLE_FAST_ENV, ds.ANATOMY_ENV)}
        for extra in ({}, {ds.SAMPLE_FAST_ENV: '0', ds.ANATOMY_ENV: '0'}):
            out = subprocess.run([sys.executable, '-I', '-c', OFF_PROBE, str(ROOT)], capture_output=True, text=True,
                                 env=dict(env, **extra), check=True)
            got = json.loads(out.stdout)
            self.assertEqual(got, dict(imported=False, first=None, kwargs={'x': 1}, side=True, state=False, attr=False),
                             (extra, got))

    def test_on_in_a_fresh_process_imports_it(self):
        out = subprocess.run([sys.executable, '-I', '-c', OFF_PROBE, str(ROOT)], capture_output=True, text=True,
                             env=dict(os.environ, **{ds.SAMPLE_FAST_ENV: '1'}), check=True)
        got = json.loads(out.stdout)
        self.assertTrue(got['imported'])
        self.assertEqual(got['state'], 'module')
        self.assertEqual(got['kwargs'], {'x': 1})


# --- the owned collective_rpc wire ----------------------------------------------------------------------------------------
class Ring:
    """Writer side of the broadcast MessageQueue (as in test_native43_le_bounded_enqueue): records writes."""

    def __init__(self):
        from contextlib import nullcontext
        self.writes = []
        self.shutting_down = False
        self._is_writer = True
        self.n_remote_reader = 0
        self.current_idx = 0
        self.buffer = NS(max_chunks=64, max_chunk_bytes=1024 ** 2, n_reader=4,
                         get_metadata=lambda i: nullcontext(bytearray(5)))

    def enqueue(self, obj, timeout=None):
        self.writes.append(obj)


class Reader:
    def dequeue(self, timeout):
        time.sleep(min(timeout or 0.01, 0.01))
        raise TimeoutError


class Wire(unittest.TestCase):
    def setUp(self):
        self.ring = Ring()
        self.executor = NS(rpc_broadcast_mq=self.ring, response_mqs=[Reader()], is_failed=False,
                           futures_queue=collections.deque(), shutting_down=False, world_size=1, failure_callback=None,
                           workers=[])
        transport = eh.install(self.executor, config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP': '1'}),
                               success_status=object())
        self.addCleanup(transport.close, 1)

    def send(self, method, kwargs=None):
        eh.collective_rpc(self.executor, None, method, args=(None,), kwargs=kwargs, unique_reply_rank=0,
                          non_block=True, timeout=5)
        return self.ring.writes[-1]

    def test_off_the_wire_is_the_given_kwargs(self):
        with mock.patch.object(eh, '_DECODE_STEP', False):
            side = ds.EngineSide({ds.SAMPLE_FAST_ENV: '1'})
            side.native_arm = True
            self.executor._flash_s5_decode_step = side     # even with a side present: the module switch decides
            for method in ('execute_model', 'sample_tokens', 'flash_s5_status'):
                self.assertEqual(self.send(method, {'a': 1})[2], {'a': 1})
                self.assertEqual(self.send(method)[2], {})

    def test_off_with_progress_only_the_progress_meta(self):
        progress = NS(counters=collections.Counter(), last_stock_kind=None)
        self.executor._flash_s5_progress_recording = progress
        with mock.patch.object(eh, '_DECODE_STEP', False):
            wire = self.send('sample_tokens', {'a': 1})[2]
        self.assertEqual(set(wire), {'a', nm.PROGRESS_META_KEY})

    def test_on_sample_tokens_carries_the_bit_only_in_a_released_arm(self):
        side = ds.EngineSide({ds.SAMPLE_FAST_ENV: '1'})
        self.executor._flash_s5_decode_step = side
        with mock.patch.object(eh, '_DECODE_STEP', ds):
            side.step(engine(manual_owned=True))            # sp: the manual lease
            self.assertNotIn(ds.WIRE_KEY, self.send('sample_tokens')[2])
            side.step(engine(manual_owned=False))           # n3: released
            method, args, wire, rank = self.send('sample_tokens', {'a': 1})
            self.assertEqual(method, 'sample_tokens')
            self.assertEqual(wire['a'], 1)
            self.assertEqual(wire[ds.WIRE_KEY][:2], [1, ds.TAG_NONE])
            self.assertEqual(type(wire[ds.WIRE_KEY][2]), int)
            self.assertNotIn(ds.WIRE_KEY, self.send('execute_model')[2])
            self.assertNotIn(ds.WIRE_KEY, self.send('flash_s5_status')[2])
            side.step(engine(policy=False))                 # no auto-arm: off
            self.assertNotIn(ds.WIRE_KEY, self.send('sample_tokens')[2])

    def test_on_the_caller_kwargs_object_is_not_mutated(self):
        side = ds.EngineSide({ds.SAMPLE_FAST_ENV: '1'})
        side.native_arm = True
        self.executor._flash_s5_decode_step = side
        given = {'a': 1}
        with mock.patch.object(eh, '_DECODE_STEP', ds):
            wire = self.send('sample_tokens', given)[2]
        self.assertEqual(given, {'a': 1})
        self.assertIn(ds.WIRE_KEY, wire)


# --- the engine gate ------------------------------------------------------------------------------------------------------
class Gate(unittest.TestCase):
    def test_gate_follows_manual_owned_and_logs_transitions_only(self):
        log = Log()
        side = ds.EngineSide({ds.SAMPLE_FAST_ENV: '1'}, logger=log)
        seq = iter(range(1000))
        tags = []
        for owned in (True, True, False, False, False, True, False):
            side.step(engine(manual_owned=owned))
            tags.append(side.wire('sample_tokens', next(seq)))
            self.assertIsNone(side.wire('execute_model', next(seq)))
        self.assertEqual([t is not None for t in tags], [False, False, True, True, True, False, True])
        self.assertEqual([g['native_arm'] for g in log.of(ds.LINE_GATE)], [True, False, True])
        self.assertEqual(side.counters['fast_tags'], 4)

    def test_switch_off_never_tags_even_when_released(self):
        side = ds.EngineSide({})
        side.step(engine(manual_owned=False))
        self.assertTrue(side.native_arm)
        self.assertIsNone(side.wire('sample_tokens', 1))

    def test_a_failing_read_closes_the_gate(self):
        side = ds.EngineSide({ds.SAMPLE_FAST_ENV: '1'})
        side.step(engine(manual_owned=False))
        broken = NS(_flash_s5_driver=NS(auto_arm=property(lambda s: 1 / 0)))

        class Raising:
            @property
            def auto_arm(self):
                raise RuntimeError('read')
        broken._flash_s5_driver = Raising()
        side.step(broken)
        self.assertFalse(side.native_arm)
        self.assertEqual(side.counters['errors'], 1)
        self.assertIsNone(side.wire('sample_tokens', 2))

    def test_executor_hooks_step_calls_the_side_once_per_step(self):
        calls = []
        eng = engine(manual_owned=False)
        fake = NS(step=lambda e: calls.append(e))
        with mock.patch.object(eh, '_decode_step_engine', lambda e: fake), \
                mock.patch.object(eh, '_driver', lambda e: NS(advance=lambda: None, sync_decode_merge=None)), \
                mock.patch.object(eh, '_stock_step_bypass', lambda d, r: None), \
                mock.patch.object(eh, '_steps', lambda e: None):
            eh.before_step_with_batch_queue(eng)
        self.assertEqual(calls, [eng])


# --- the clean anatomy window -----------------------------------------------------------------------------------------
class Window(unittest.TestCase):
    def side(self):
        self.log, self.clock = Log(), Clock()
        return ds.EngineSide({ds.ANATOMY_ENV: '1'}, logger=self.log, clock=self.clock)

    def run_steps(self, side, n, period_s=0.02, **state):
        tags = []
        for i in range(n):
            side.step(engine(**state))
            tags.append((side.wire('execute_model', i), side.wire('sample_tokens', i), side.wire('flash_s5_submit', i)))
            self.clock.advance(period_s)
        return tags

    def test_opens_in_a_released_quiet_arm_and_closes_at_20_s(self):
        side = self.side()
        tags = self.run_steps(side, 1100, manual_owned=False)            # 22 s of 20 ms steps
        recorded = [t for t in tags if t[0] and t[0][1] == ds.TAG_RECORD]
        self.assertEqual(len(recorded), 1000)                            # 20 s
        close = [i for i, t in enumerate(tags) if t[0] and t[0][1] == ds.TAG_CLOSE]
        self.assertEqual(close, [1000])                                  # the first RPC after the close
        self.assertTrue(all(t[1] and t[1][1] == ds.TAG_RECORD for t in tags[:1000]))
        self.assertTrue(all(t[2] is None for t in tags))                 # native commands never tagged
        self.assertTrue(all(t[0] is None and t[1] is None for t in tags[1001:]))   # budget spent: no second window
        lines = self.log.of(ds.LINE_ANATOMY)
        self.assertEqual(sum(len(l['stamps']) for l in lines), 1000)
        self.assertEqual([l['part'] for l in lines], list(range(lines[0]['parts'])))
        self.assertLess(max(len(t) for t in self.log.lines if t.startswith(ds.LINE_ANATOMY)), ds.LINE_BYTES)
        self.assertEqual(lines[0]['fields'], ['t_us', 'batch_queue', 'running', 'waiting'])
        self.assertEqual(lines[0]['stamps'][1], [20000, 0, 8, 0])
        self.assertEqual(side.window.windows, 1)

    def test_not_in_the_stock_arm_nor_beside_native_progress_nor_an_active_policy(self):
        for state in (dict(manual_owned=True), dict(manual_owned=False, progress=object()),
                      dict(manual_owned=False, active=True), dict(policy=False)):
            side = self.side()
            tags = self.run_steps(side, 50, **state)
            self.assertTrue(all(t == (None, None, None) for t in tags), state)
            self.assertEqual(self.log.of(ds.LINE_ANATOMY), [], state)

    def test_breaks_close_early_and_the_budget_caps_the_run(self):
        side = self.side()
        quiet, busy = dict(manual_owned=False), dict(manual_owned=False, progress=object())
        self.run_steps(side, 250, **quiet)                                # 5 s
        self.run_steps(side, 10, **busy)                                  # native starts: close
        self.assertEqual(side.window.counters['closes'], 1)
        self.run_steps(side, 250, **quiet)                                # second window, 5 s
        self.run_steps(side, 10, **busy)
        tags = self.run_steps(side, 1000, **quiet)                        # third window: the 10 s left of the budget
        rec = [t for t in tags if t[0] and t[0][1] == ds.TAG_RECORD]
        self.assertEqual(len(rec), 500)
        self.run_steps(side, 10, **busy)
        tags = self.run_steps(side, 100, **quiet)                         # cap reached: never again
        self.assertTrue(all(t[0] is None or t[0][1] == ds.TAG_CLOSE for t in tags))
        self.assertEqual(side.window.windows, 3)
        self.assertLessEqual(side.window.used_ns, ds.WINDOW_BUDGET_S * 1e9 + 1)

    def test_stamps_bounded_and_lines_under_the_line_size(self):
        side = self.side()
        side.window.window_s, side.window.budget_ns = 10 ** 9, 10 ** 18
        self.run_steps(side, ds.MAX_STAMPS + 500, period_s=0.001, manual_owned=False, queue=2, running=10, waiting=3)
        side.step(engine(manual_owned=True))
        lines = [t for t in self.log.lines if t.startswith(ds.LINE_ANATOMY)]
        self.assertEqual(sum(len(json.loads(t[len(ds.LINE_ANATOMY):])['stamps']) for t in lines), ds.MAX_STAMPS)
        self.assertEqual(json.loads(lines[-1][len(ds.LINE_ANATOMY):])['counters']['stamps_dropped'], 500)
        self.assertLessEqual(len(lines), 26)
        self.assertLess(max(len(t) for t in lines), ds.LINE_BYTES)

    def test_both_switches_tag_one_value(self):
        self.log, self.clock = Log(), Clock()
        side = ds.EngineSide({ds.ANATOMY_ENV: '1', ds.SAMPLE_FAST_ENV: '1'}, logger=self.log, clock=self.clock)
        side.step(engine(manual_owned=False))
        self.assertEqual(side.wire('sample_tokens', 7), [1, ds.TAG_RECORD, 7])
        self.assertEqual(side.wire('execute_model', 8), [0, ds.TAG_RECORD, 8])


# --- the worker through the real generated WorkerProc ------------------------------------------------------------------
import test_stage1_handler_anatomy as sa  # noqa: E402


def mutant_hooks(old, new):
    """executor_hooks with one source replacement (asserted once), as a fresh module."""
    source = (ROOT / 'patches/s5_engine/executor_hooks.py').read_text()
    assert source.count(old) == 1, old
    module = types.ModuleType('executor_hooks_mutant')
    module.__file__ = str(ROOT / 'patches/s5_engine/executor_hooks.py')
    exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
    return module


def progress_rows(rig):
    """(seq, kind) of every worker progress-trace reply row in the rig's progress lines."""
    rows = []
    for line in rig.progress_lines():
        body = json.loads(line[len(nm.PROGRESS_LINE):])
        rows += [(r[0], r[1]) for r in body['rows'].get('r', [])]
    return rows


class Worker(unittest.TestCase):
    def rig(self, *, env, hooks=eh, rank=sa.OUTPUT_RANK, runner='v2'):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, env))
        for key in (ds.SAMPLE_FAST_ENV, ds.ANATOMY_ENV):
            if key not in env:
                os.environ.pop(key, None)
        stack.enter_context(mock.patch.object(hooks, '_DECODE_STEP', None))
        stack.enter_context(mock.patch.object(hooks, '_DECODE_STEP_WORKER', None))
        rig = sa.Rig(stack, rank, runner=runner, hooks=hooks)
        rig.tagged = lambda method, args, tag, output_rank=sa.OUTPUT_RANK: self.tagged(rig, method, args, tag,
                                                                                       output_rank)
        return rig

    @staticmethod
    def tagged(rig, method, args, tag, output_rank):
        rig.seq += 1
        kwargs = {} if tag is None else {ds.WIRE_KEY: [tag[0], tag[1], rig.seq]}
        rig.proc._execute_worker_rpc((method, args, kwargs, output_rank))
        return rig.seq

    def clean_steps(self, rig, n, code=ds.TAG_RECORD):
        seqs = []
        for _ in range(n):
            seqs.append(rig.tagged('execute_model', ('so',), (0, code)))
            seqs.append(rig.tagged('sample_tokens', (None,), (0, code)))
        return seqs

    ENV = {ds.ANATOMY_ENV: '1'}

    def test_clean_rows_flushed_at_close_never_in_the_progress_trace(self):
        rig = self.rig(env=self.ENV)
        seqs = self.clean_steps(rig, 3)
        rig.replies(6)
        self.assertEqual(rig.handler_lines(), [])                         # nothing before the close
        rig.torch.device.advance()
        rig.tagged('execute_model', ('so',), (0, ds.TAG_CLOSE))           # the close: flush, itself no row
        heads, main, _ = sa.parse(rig.handler_lines())
        self.assertEqual(sorted(r['seq'] for r in main), seqs)
        self.assertEqual({r['kind'] for r in main}, {'e', 's'})
        self.assertTrue(all(r['dev_forward_ns'] for r in main if r['kind'] == 'e'))
        meta = rig.step()                                                 # a stock step under the progress meta
        rig.replies(9)                                                    # 6 clean, the close, 2 meta
        rig.release()                                                     # the chunk release: progress lines
        rows = progress_rows(rig)
        self.assertEqual(sorted(s for s, _ in rows), sorted(meta))        # only the meta RPCs are trace rows
        self.assertFalse(set(seqs) & {s for s, _ in rows})
        self.assertEqual(rig.hooks._DECODE_STEP_WORKER.counters['clean_rows'], 6)
        self.assertEqual(rig.hooks._DECODE_STEP_WORKER.counters['flushes'], 1)

    def test_progress_rows_equal_untagged_and_tagged(self):
        def run(tag):
            rig = self.rig(env=self.ENV)
            for _ in range(3):
                rig.tagged('execute_model', ('so',), tag)
                rig.tagged('sample_tokens', (None,), tag)
                rig.step()
            rig.replies(12)
            rig.release()
            return progress_rows(rig), len(rig.proc.worker_response_mq.results)
        plain = run(None)
        clean = run((0, ds.TAG_RECORD))
        self.assertEqual(len(plain[0]), 6)
        self.assertEqual(plain, clean)

    def test_untagged_without_the_switch_is_stock(self):
        rig = self.rig(env={})
        for _ in range(3):
            rig.tagged('execute_model', ('so',), None)
            rig.tagged('sample_tokens', (None,), None)
        rig.replies(6)
        rig.tagged('execute_model', ('so',), None)
        self.assertEqual(rig.handler_lines(), [])
        self.assertIs(rig.hooks._DECODE_STEP, False)

    def test_a_tag_with_the_anatomy_switch_off_records_nothing(self):
        rig = self.rig(env={ds.SAMPLE_FAST_ENV: '1'})
        self.clean_steps(rig, 2)
        rig.tagged('execute_model', ('so',), (0, ds.TAG_CLOSE))
        self.assertEqual(rig.handler_lines(), [])

    def test_mutant_clean_rows_in_the_trace_is_caught(self):
        hooks = mutant_hooks("        if type(rpc) is not _CleanRpc:\n"
                             "            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], time.monotonic_ns(), 0, 1)",
                             "        if True:\n"
                             "            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], time.monotonic_ns(), 0, 1)")
        rig = self.rig(env=self.ENV, hooks=hooks)
        seqs = self.clean_steps(rig, 2)
        rig.step()
        rig.replies(6)
        rig.release()
        self.assertTrue(set(seqs) & {s for s, _ in progress_rows(rig)}, 'the mutant must leak clean rows')
        good = self.rig(env=self.ENV)
        seqs = self.clean_steps(good, 2)
        good.step()
        good.replies(6)
        good.release()
        self.assertFalse(set(seqs) & {s for s, _ in progress_rows(good)})

    def test_mutant_tag_not_popped_reaches_the_stock_method(self):
        hooks = mutant_hooks("    popped = _DECODE_STEP_WORKER.pop(proc, name, kwargs)\n",
                             "    popped = _DECODE_STEP_WORKER.pop(proc, name, dict(kwargs))\n")
        rig = self.rig(env=self.ENV, hooks=hooks, runner='v2')
        rig.tagged('execute_model', ('so',), (0, ds.TAG_RECORD))
        # the leaf worker's execute_model takes no extra keyword: the reply is a failure
        results = rig.replies(1)
        self.assertTrue(results and 'FAILURE' in repr(results[-1]), results)


# --- the worker side and the sampler gate (no torch) ------------------------------------------------------------------------
class FakeTorch:
    float32 = 'float32'


class Logits:
    def __init__(self, rows, vocab, dtype='float32', dims=2):
        self.shape = (rows, vocab)[:dims] if dims <= 2 else (rows, vocab, 1)
        self.dtype = dtype
        self._dims = dims

    def dim(self):
        return self._dims


class TopK:
    def __init__(self, values):
        import array
        self.np = Arr(values)


class Arr(list):
    def __getitem__(self, index):
        if isinstance(index, list):
            return Arr([list.__getitem__(self, i) for i in index])
        return list.__getitem__(self, index)

    def __eq__(self, other):
        return Bools([v == other for v in self])

    __hash__ = None


class Bools(list):
    def all(self):
        return all(self)


class Sampler:
    def __init__(self, ks):
        self.sampling_states = NS(top_k=TopK(ks))
        self.calls = []

    def _sample_random(self, *args):
        self.calls.append(args)
        return ('stock', args)


class SamplerGate(unittest.TestCase):
    def fast(self, ks=(20,) * 8):
        sampler = Sampler(list(ks))
        fast = ds.FastSampler(sampler, torch=FakeTorch(), top_p_sample=lambda *a: None, sample=lambda *a: None)
        return sampler, fast

    def test_wrapper_is_on_the_instance(self):
        before = dict(vars(Sampler))
        sampler, fast = self.fast()
        self.assertIn('_sample_random', vars(sampler))
        self.assertEqual(dict(vars(Sampler)), before)                  # the class is untouched
        self.assertEqual(vars(Sampler)['_sample_random'].__qualname__, 'Sampler._sample_random')
        self.assertEqual(sampler._sample_random, fast.sample_random)

    def test_every_ineligible_step_is_the_stock_call_unchanged(self):
        idx = list(range(8))
        cases = [
            (dict(), dict(gated=False), 'not_gated'),
            (dict(), dict(use_flashinfer=False), 'not_flashinfer'),
            (dict(), dict(top_k=None), 'no_top_k'),
            (dict(), dict(logits=Logits(9, 248320)), 'rows'),
            (dict(), dict(logits=Logits(8, 248320, dims=1)), 'rows'),
            (dict(), dict(logits=Logits(8, 248320, dtype='bfloat16')), 'dtype'),
            (dict(), dict(logits=Logits(8, 32000)), 'vocab'),
            (dict(ks=(20,) * 7 + (40,)), dict(), 'k_mixed'),
            (dict(ks=(129,) * 8), dict(), 'k_range'),
            (dict(ks=(0,) * 8), dict(), 'k_range'),
        ]
        for build, call, reason in cases:
            sampler, fast = self.fast(**build)
            fast.gated = call.pop('gated', True)
            args = dict(processed_logits=Logits(8, 248320), expanded_idx_mapping='e', idx_mapping_np=idx, pos='p',
                        top_k='k', top_p='q', use_flashinfer=True)
            if 'logits' in call:
                args['processed_logits'] = call.pop('logits')
            args.update(call)
            out = sampler._sample_random(*args.values())
            self.assertEqual(out, ('stock', tuple(args.values())), reason)
            self.assertEqual(sampler.calls, [tuple(args.values())], reason)
            self.assertEqual(fast.counters['stock_' + reason], 1, (reason, fast.counters))
            self.assertEqual(fast.counters['fast'], 0)

    def test_eligible_step_reaches_the_fast_path(self):
        sampler, fast = self.fast()
        fast.gated = True
        seen = []
        fast.sample_top = lambda logits, k, top_p: seen.append((k, top_p)) or 'tokens'
        out = sampler._sample_random(Logits(8, 248320), 'e', list(range(8)), 'p', 'k', 'q', True)
        self.assertEqual(out[0], 'tokens')
        self.assertEqual(seen, [(20, 'q')])
        self.assertEqual(sampler.calls, [])

    def test_worker_side_gates_per_rpc_and_installs_once_on_a_sampling_rank(self):
        log = Log()
        sampler = Sampler([20] * 4)
        proc = NS(worker=NS(model_runner=NS(sampler=sampler)))
        side = ds.WorkerSide({ds.SAMPLE_FAST_ENV: '1'}, logger=log)
        with mock.patch.object(ds, '_torch', lambda: FakeTorch()):
            kw = {ds.WIRE_KEY: [1, 0, 5], 'x': 1}
            self.assertEqual(side.pop(proc, 'sample_tokens', kw), (1, ds.TAG_NONE, 5))
            self.assertEqual(kw, {'x': 1})
            self.assertTrue(side.fast.gated)
            first = side.fast
            side.pop(proc, 'execute_model', {})                   # the next RPC resets the gate
            self.assertFalse(side.fast.gated)
            side.pop(proc, 'sample_tokens', {})                   # untagged sample_tokens: stock
            self.assertFalse(side.fast.gated)
            side.pop(proc, 'sample_tokens', {ds.WIRE_KEY: [1, 0, 6]})
            self.assertIs(side.fast, first)
            self.assertTrue(side.fast.gated)
            side.pop(proc, 'execute_model', {ds.WIRE_KEY: [1, 0, 7]})   # the bit only acts on sample_tokens
            self.assertFalse(side.fast.gated)

    def test_periodic_lines_carry_the_rank_and_cumulative_counts(self):
        """Compose line (13 Nov): a cumulative sample-fast line every log_every sampler steps, fast or stock, with the
        rank, beside the power-of-two lines, so the cx reader takes per-arm deltas."""
        log = Log()
        sampler = Sampler([20] * 8)
        fast = ds.FastSampler(sampler, torch=FakeTorch(), top_p_sample=lambda *a: None, sample=lambda *a: None,
                              logger=log, rank=2, log_every=4)
        fast.sample_top = lambda logits, k, top_p: 'tokens'
        args = (Logits(8, 248320), 'e', list(range(8)), 'p', 'k', 'q', True)
        rows = lambda: [json.loads(x[len(ds.LINE_SAMPLE):]) for x in log.lines]
        for _ in range(3):                                   # the stock arm: gate off
            sampler._sample_random(*args)
        self.assertEqual(log.lines, [])
        sampler._sample_random(*args)                        # step 4: periodic
        self.assertEqual([(r['rank'], r['counters']['stock'], r['counters']['stock_not_gated'], r['counters']['fast'])
                          for r in rows()], [(2, 4, 4, 0)])
        fast.gated = True
        for _ in range(8):                                   # fast 1, 2, 4, 8 (powers) and total 8, 12 (periodic)
            sampler._sample_random(*args)
        got = [(r['counters']['fast'], r['counters']['stock']) for r in rows()[1:]]
        self.assertEqual(got, [(1, 4), (2, 4), (4, 4), (8, 4)])   # total 8 is fast 4: one line; total 12 is fast 8
        self.assertTrue(all(x.startswith(ds.LINE_SAMPLE) and len(x) < ds.LINE_BYTES for x in log.lines))
        side = ds.WorkerSide({ds.SAMPLE_FAST_ENV: '1'}, logger=log)
        with mock.patch.object(ds, '_torch', lambda: FakeTorch()):
            side.pop(NS(rank=3, worker=NS(model_runner=NS(sampler=Sampler([20])))), 'sample_tokens',
                     {ds.WIRE_KEY: [1, 0, 1]})
        self.assertEqual(side.fast.rank, 3)

    def test_worker_side_without_a_sampler_stays_stock(self):
        side = ds.WorkerSide({ds.SAMPLE_FAST_ENV: '1'})
        proc = NS(worker=NS(model_runner=NS()))                   # stage 0: no sampler
        side.pop(proc, 'sample_tokens', {ds.WIRE_KEY: [1, 0, 1]})
        side.pop(proc, 'sample_tokens', {ds.WIRE_KEY: [1, 0, 2]})
        self.assertIsNone(side.fast)
        self.assertEqual(side.counters['no_sampler'], 1)

    def test_worker_side_refuses_a_sampler_that_overrides_the_draw(self):
        """Owner review (13 Nov): GPUWatermarkSampler (any watermark_config) overrides _sample_random with its own
        watermarked draw, which D's fast path would bypass. D installs only where Sampler's own _sample_random runs and
        there is no watermarking state; otherwise every step is the sampler's own call and one refusal line is logged."""
        class WatermarkSampler(Sampler):                 # the image's GPUWatermarkSampler shape
            def __init__(self, ks):
                super().__init__(ks)
                self.watermarking = NS(np=[True] * len(ks))

            def _sample_random(self, *args):
                return ('watermark', args)

        class StateOnly(Sampler):                        # keeps the stock draw but carries watermarking state
            def __init__(self, ks):
                super().__init__(ks)
                self.watermarking = NS(np=[True] * len(ks))

        wrapped = Sampler([20] * 8)
        wrapped._sample_random = lambda *args: ('other wrapper', args)
        self.assertEqual(ds.sample_random_owner(Sampler([20])), 'Sampler')
        self.assertEqual(ds.sample_random_owner(WatermarkSampler([20])), WatermarkSampler.__qualname__)
        self.assertEqual(ds.sample_random_owner(wrapped), 'instance')
        self.assertIsNone(ds.sample_random_owner(NS()))
        args = (Logits(8, 248320), 'e', list(range(8)), 'p', 'k', 'q', True)
        for sampler, owner, stock in ((WatermarkSampler([20] * 8), WatermarkSampler.__qualname__, 'watermark'),
                                      (StateOnly([20] * 8), 'Sampler', 'stock'),
                                      (wrapped, 'instance', 'other wrapper')):
            log = Log()
            side = ds.WorkerSide({ds.SAMPLE_FAST_ENV: '1'}, logger=log)
            proc = NS(rank=1, worker=NS(model_runner=NS(sampler=sampler)))
            with mock.patch.object(ds, '_torch', lambda: FakeTorch()):
                side.pop(proc, 'sample_tokens', {ds.WIRE_KEY: [1, 0, 1]})
                side.pop(proc, 'sample_tokens', {ds.WIRE_KEY: [1, 0, 2]})
            self.assertIsNone(side.fast)
            self.assertEqual(side.counters['sampler_override'], 1)
            self.assertEqual(side.counters['fast_rpcs'], 0)
            self.assertEqual(sampler._sample_random(*args)[0], stock)    # the sampler's own draw, unwrapped
            self.assertEqual(len(log.lines), 1)
            self.assertTrue(log.lines[0].startswith(ds.LINE_REFUSED))
            self.assertFalse(log.lines[0].startswith(ds.LINE_SAMPLE))    # the reader's prefix never matches it
            got = json.loads(log.lines[0][len(ds.LINE_REFUSED):])
            self.assertEqual((got['rank'], got['owner'], got['sampler']), (1, owner, type(sampler).__qualname__))
        plain = Sampler([20] * 8)                         # control: the plain sampler still installs
        side = ds.WorkerSide({ds.SAMPLE_FAST_ENV: '1'})
        with mock.patch.object(ds, '_torch', lambda: FakeTorch()):
            side.pop(NS(rank=0, worker=NS(model_runner=NS(sampler=plain))), 'sample_tokens', {ds.WIRE_KEY: [1, 0, 1]})
        self.assertIsNotNone(side.fast)
        self.assertEqual(side.counters['sampler_override'], 0)

    def test_worker_side_switch_off_ignores_the_bit(self):
        sampler = Sampler([20])
        side = ds.WorkerSide({})
        side.pop(NS(worker=NS(model_runner=NS(sampler=sampler))), 'sample_tokens', {ds.WIRE_KEY: [1, 0, 1]})
        self.assertIsNone(side.fast)
        self.assertNotIn('_sample_random', vars(sampler))


# --- the pinned V2 sampler names ---------------------------------------------------------------------------------------
class Names(unittest.TestCase):
    def test_sample_random_signature_and_call_site(self):
        tree = ast.parse(SAMPLER_SRC.read_text())
        sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sampler')
        defs = {n.name: n for n in sampler.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual([a.arg for a in defs['_sample_random'].args.args],
                         ['self', 'processed_logits', 'expanded_idx_mapping', 'idx_mapping_np', 'pos', 'top_k', 'top_p',
                          'use_flashinfer'])
        text = ast.get_source_segment(SAMPLER_SRC.read_text(), defs['sample'])
        self.assertIn('return self._sample_random(', text)
        body = ast.get_source_segment(SAMPLER_SRC.read_text(), defs['_sample_random'])
        self.assertIn('flashinfer_sample(processed_logits, top_k, top_p)', body)
        self.assertIn('return sampled, processed_logits', body)

    def test_top_k_is_a_uva_tensor_with_np(self):
        text = STATES_SRC.read_text()
        self.assertIn('self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)', text)
        self.assertIn('self.top_k.np[req_idx] = top_k', text)


# --- startup and declarations ---------------------------------------------------------------------------------------
class Startup(unittest.TestCase):
    def test_refusals(self):
        with self.assertRaises(RuntimeError):
            ds.check_startup({ds.ANATOMY_ENV: '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst'})
        with self.assertRaises(RuntimeError):
            ds.check_startup({ds.ANATOMY_ENV: '1', 'FLASH_S5_HANDLER_ANATOMY': '1'})
        with self.assertRaises(RuntimeError):
            ds.check_startup({ds.SAMPLE_FAST_ENV: '2'})
        ds.check_startup({ds.ANATOMY_ENV: '1', 'FLASH_S5_HANDLER_ANATOMY': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst'})
        ds.check_startup({ds.SAMPLE_FAST_ENV: '1'})

    def test_s5_startup_reads_it_only_under_a_switch(self):
        text = (ROOT / 'patches/s5_engine/s5_startup.py').read_text()
        self.assertIn("if env.get('FLASH_S5_DECODE_SAMPLE_FAST', '0') != '0' or env.get('FLASH_S5_DECODE_ANATOMY', '0') != '0':\n"
                      "        from decode_step import check_startup as decode_step_check\n", text)

    def test_declared_in_the_overlay_and_never_imported_at_module_level(self):
        import decode_probe
        import make_s5_overlap
        self.assertIn('decode_step', make_s5_overlap.E3_MODULES)
        self.assertIn('decode_step', make_s5_overlap.LAUNCHER_LIST_EXEMPT)
        self.assertIn('decode_step', decode_probe.GENERATED_SOURCES)
        tree = ast.parse((ROOT / 'patches/s5_engine/executor_hooks.py').read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top for a in n.names} | {getattr(n, 'module', None) for n in top}
        self.assertNotIn('decode_step', names)


if __name__ == '__main__':
    unittest.main()
