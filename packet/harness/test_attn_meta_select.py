"""Tests for patches/s5_engine/attn_meta_select.py: the per-arm Selector over attn_meta_batch's GdnBatch (lead decision A).

Container only (pinned image: real torch + vllm builders; the fixture and leaves are tools/attn_meta_cases.py over
tools/attn_meta_witness.py, as harness/test_attn_meta_overlay.py). Lead condition 1: "Selector off must equal stock":
  - every history case and every padded case, both stages: the Selector installed and NOT selected against uninstalled
    stock: same keys and order, same metadata (every field and tensor location), same builder buffer bytes; the GdnBatch is
    never entered (batched 0, every fallback 0) and off_calls counts the call;
  - selected: the same equality with batched 1 (the overlay as vm-b proved it);
  - the 48-step decode sequence (block crossings, admissions, finishes, padding) toggled off -> on -> off at step boundaries
    (16 / 16 / 16) against stock: GDN buffers and full snapshots equal at every step; batched == 16 exactly, off_calls == 32;
  - install order: a model state that already carries an instance prepare_attn (a seam installed first) refuses; handler
    anatomy's real wrapper installed after the Selector is reported by ``order`` as wrapped_by_other (ok), an unrelated
    wrapper that does not hold the Selector is reported not ok;
  - typed select: non-bool refused, not installed refused; off: ``startup`` sets nothing and imports nothing;
  - the startup order line (one per rank) names selector_outer=1 selected=0.
Mutants (each must be caught by the assertions above, run through the same check functions):
  off-calls-batch      the off path calls the GdnBatch (equal outputs: caught by the batched counter, the box rule
                       attn-meta-selection-leak);
  select-ignored       select() leaves the flag (caught by the on-arm batched counter, attn-meta-not-engaged);
  on-calls-stock       the on path calls stock (caught by the batched counter).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / 'tools', ROOT / 'patches/s5_engine'):
    sys.path.insert(0, str(p))

import attn_meta_select as S  # noqa: E402  (no torch at import)
import handler_anatomy  # noqa: E402,F401  (hub-importable: torch only at install)

try:
    import numpy as np
    import torch
    import vllm  # noqa: F401
except ImportError:  # hub: no torch/vllm
    torch = None

if torch is not None:
    import attn_meta_witness as W  # noqa: E402
    import attn_meta_cases as C  # noqa: E402

MODEL_CONFIG = os.environ.get('ATTN_META_MODEL_CONFIG', '/w/model-config.json')
OVERLAY_QSA = os.environ.get('ATTN_META_OVERLAY_QSA', '/w/overlay-qsa.py')
_S = {}


class Hub(unittest.TestCase):
    """No torch needed."""
    def test_enabled_values(self):
        self.assertFalse(S.enabled({}))
        self.assertFalse(S.enabled({S.ENV: '0'}))
        self.assertTrue(S.enabled({S.ENV: '1'}))
        for bad in ('', 'yes', '2', 'true'):
            with self.assertRaises(S.AttnMetaSelectError):
                S.enabled({S.ENV: bad})

    def test_off_startup_sets_nothing_imports_nothing(self):
        code = ('import sys; sys.path.insert(0, %r); import attn_meta_select as S; '
                'r = type("R", (), {})(); r.model_state = type("M", (), {})(); '
                'assert S.startup(r, env={}) is None; assert S.startup(r, True, env={S.ENV: "1"}) is None; '
                'assert vars(r.model_state) == {}; '
                'bad = [m for m in ("torch", "vllm", "attn_meta_batch") if m in sys.modules]; '
                'assert not bad, bad; print("ok")' % str(ROOT / 'patches/s5_engine'))
        out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env={'PATH': '/usr/bin'})
        self.assertEqual(out.stdout.strip(), 'ok', out.stderr[-2000:])

    def test_selector_dispatch_and_counters(self):
        calls = []
        sel = S.Selector('mstate', _Batch(calls), lambda m, *a, **k: calls.append(('stock', m, a, k)))
        sel(1, x=2)
        self.assertEqual(calls, [('stock', 'mstate', (1,), {'x': 2})])
        sel.selected = True
        sel(3)
        self.assertEqual(calls[-1], ('batch', (3,), {}))
        c = sel.counters()
        self.assertEqual((c['off_calls'], c['batched'], c['selected']), (1, 1, True))

    def test_select_typed(self):
        runner = types.SimpleNamespace(model_state=types.SimpleNamespace())
        with self.assertRaises(S.AttnMetaSelectError):
            S.select(runner, True)                      # not installed
        sel = S.Selector(runner.model_state, _Batch([]), lambda *a, **k: None)
        setattr(runner.model_state, S._MARK, sel)
        runner.model_state.prepare_attn = sel
        for bad in (1, 0, 'on', None):
            with self.assertRaises(S.AttnMetaSelectError):
                S.select(runner, bad)
        r = S.select(runner, True)
        self.assertEqual((r['before']['selected'], r['after']['selected'], r['after']['selections']), (False, True, 1))
        self.assertEqual(r['order'], dict(outer='Selector', selector_outer=True, wrapped_by_other=False, ok=True))
        S.select(runner, True)
        self.assertEqual(S.read(runner)['counters']['selections'], 1)

    def test_order_detects_wrappers(self):
        runner = types.SimpleNamespace(model_state=types.SimpleNamespace())
        sel = S.Selector(runner.model_state, _Batch([]), lambda *a, **k: None)
        setattr(runner.model_state, S._MARK, sel)
        inner = sel

        def seam(*a, **k):
            return inner(*a, **k)
        runner.model_state.prepare_attn = seam
        self.assertEqual(S.order(runner)['ok'], True)
        self.assertEqual(S.order(runner)['wrapped_by_other'], True)

        def other(*a, **k):
            return None
        runner.model_state.prepare_attn = other
        self.assertEqual(S.order(runner)['ok'], False)

    def test_real_anatomy_wrapper_is_seen_over_selector(self):
        import handler_anatomy as HA
        runner = types.SimpleNamespace(model_state=types.SimpleNamespace())
        sel = S.Selector(runner.model_state, _Batch([]), lambda *a, **k: None)
        setattr(runner.model_state, S._MARK, sel)
        runner.model_state.prepare_attn = sel
        # Leaf: no CUDA on the hub; the recorder's event pool only needs Event objects (never recorded here)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=lambda **k: object()))
        rec = HA.Recorder(None, fake_torch, None, None)
        runner.model_state.prepare_attn = rec._wrap(sel, HA.SEAM_ID['prep_attn_meta'], 'plain')
        o = S.order(runner)
        self.assertEqual((o['selector_outer'], o['wrapped_by_other'], o['ok']), (False, True, True))

    def test_startup_line(self):
        lines = []
        logger = types.SimpleNamespace(info=lines.append)
        runner = types.SimpleNamespace(model_state=types.SimpleNamespace())
        real = S.install

        def fake(r, env=None):
            sel = S.Selector(r.model_state, _Batch([]), lambda *a, **k: None)
            setattr(r.model_state, S._MARK, sel)
            r.model_state.prepare_attn = sel
            return sel
        S.install = fake
        try:
            S.startup(runner, env={S.ENV: '1'}, rank=2, logger=logger)
        finally:
            S.install = real
        self.assertEqual(lines, ['flash-attn-meta installed schema=attn-meta-select/1 rank=2 selector_outer=1 '
                                 'wrapped_by_other=0 selected=0'])


class _Batch:
    def __init__(self, calls):
        self.calls = calls
        self.batched = 0

    def __call__(self, *a, **k):
        self.batched += 1
        self.calls.append(('batch', a, k))

    def counters(self):
        return dict(schema='attn-meta-gdn-batch/1', batched=self.batched, fallback={})


# -- container: real builders ----------------------------------------------------------------------------------
def setUpModule():
    if torch is None:
        return
    W.install_triton_stub()
    import vllm.models.qwen4_exp.common.qsa_cache as qc
    qc.build_qsa_metadata = qc.build_qsa_metadata_triton
    if Path(OVERLAY_QSA).is_file():
        W.install_overlay_builder(Path(OVERLAY_QSA))
    import tempfile
    work = tempfile.mkdtemp(prefix='attn-meta-select-', dir='/tmp')
    cfg, _ = W.build_config(MODEL_CONFIG, Path(work), torch)
    text = json.loads(Path(MODEL_CONFIG).read_text())['text_config']
    _S['c'] = C.Cases(W, cfg, text, torch, np, torch.device('cpu'))


def _mod(selected, cls=None):
    """A module-shaped installer for Cases.compare: the Selector installed, then selected or not."""
    def install(runner, env=None):
        sel = S.install(runner, env={S.ENV: '1'})
        if cls is not None:
            sel.__class__ = cls
        S.select(runner, selected)
        return sel
    return types.SimpleNamespace(ENV=S.ENV, install=install)


def check_equal(tc, c, mod, *, selected, mutant=False):
    """Every history and padded case, both stages, equality vs stock and the counters. Returns the failures."""
    fails = []
    for stage in (0, 1):
        cases = [(name, hist, None) for name, hist in C.HISTS.items()]
        cases += [('pad-%d-%d' % (n, R), C.HISTS['boundary'][:n], R) for n, R in C.PADDED]
        for name, hist, pad in cases:
            sa, sb, sel = c.compare(mod, stage, hist, pad_to=pad)
            d = c.first_difference(sa, sb)
            cnt = sel.counters()
            want = (1, 0) if selected else (0, 1)
            if d:
                fails.append('%d:%s:%s' % (stage, name, d))
            if (cnt['batched'], cnt['off_calls']) != want or sum(cnt['fallback'].values()) != 0:
                fails.append('%d:%s:counters %s' % (stage, name, (cnt['batched'], cnt['off_calls'])))
    if not mutant:
        tc.assertEqual(fails, [])
    return fails


def check_toggle_sequence(tc, c, stage, cls=None):
    """off 16 / on 16 / off 16 steps against stock; returns failures."""
    plan = C.sequence_plan()
    a, b = c.fixture(stage, plan['hist']), c.fixture(stage, plan['hist'])
    runner = types.SimpleNamespace(model_state=b.mstate)
    sel = S.install(runner, env={S.ENV: '1'})
    if cls is not None:
        sel.__class__ = cls
    c.poison(a)
    c.poison(b)
    seq = C.Sequence(c, stage, plan)
    fails = []
    for step in range(plan['steps']):
        if step in (0, 16, 32):
            S.select(runner, step == 16)
        seq.load(a)
        seq.load(b)
        ma, mb = c.call(a), c.call(b)
        if list(ma) != list(mb) or c.gdn_buffers(a) != c.gdn_buffers(b):
            fails.append('step %d buffers/keys' % step)
            break
        if step % 4 == 0 or step in (15, 16, 31, 32, 47):
            d = c.first_difference(c.snapshot(a, ma), c.snapshot(b, mb))
            if d:
                fails.append('step %d %s' % (step, d))
                break
        seq.advance(step)
    cnt = sel.counters()
    if (cnt['batched'], cnt['off_calls'], cnt['selections']) != (16, 32, 2):
        fails.append('counters %s' % ((cnt['batched'], cnt['off_calls'], cnt['selections']),))
    return fails


class OffCallsBatch(S.Selector):
    def __call__(self, *a, **k):
        if not self.selected:
            self.off_calls += 1
        return self.batch(*a, **k)


class OnCallsStock(S.Selector):
    def __call__(self, *a, **k):
        if not self.selected:
            self.off_calls += 1
        return self.stock(self.mstate, *a, **k)


@unittest.skipIf(torch is None, 'needs the pinned image (torch + vllm)')
class Container(unittest.TestCase):
    def test_off_equals_stock_every_case(self):
        check_equal(self, _S['c'], _mod(False), selected=False)

    def test_on_equals_stock_every_case(self):
        check_equal(self, _S['c'], _mod(True), selected=True)

    def test_toggle_off_on_off_sequence(self):
        for stage in (1, 0):
            with self.subTest(stage=stage):
                self.assertEqual(check_toggle_sequence(self, _S['c'], stage), [])

    def test_capture_call_off_is_stock(self):
        c = _S['c']
        from vllm.config.compilation import CUDAGraphMode
        a, b = c.fixture(1, C.HISTS['h65k']), c.fixture(1, C.HISTS['h65k'])
        runner = types.SimpleNamespace(model_state=b.mstate)
        sel = S.install(runner, env={S.ENV: '1'})
        for fx in (a, b):
            fx.mode = CUDAGraphMode.NONE
            c.poison(fx)
        ma, mb = c.call(a, for_capture=True), c.call(b, for_capture=True)
        self.assertIsNone(c.first_difference(c.snapshot(a, ma), c.snapshot(b, mb)))
        self.assertEqual(sel.counters()['off_calls'], 1)

    def test_instance_seam_first_refuses(self):
        c = _S['c']
        fx = c.fixture(1, C.HISTS['one'])
        fx.mstate.prepare_attn = lambda *a, **k: None          # a seam installed first (e.g. handler anatomy)
        import attn_meta_batch
        with self.assertRaises(attn_meta_batch.AttnMetaBatchError):
            S.install(types.SimpleNamespace(model_state=fx.mstate), env={S.ENV: '1'})

    def test_second_install_refuses(self):
        c = _S['c']
        fx = c.fixture(1, C.HISTS['one'])
        runner = types.SimpleNamespace(model_state=fx.mstate)
        S.install(runner, env={S.ENV: '1'})
        with self.assertRaises(S.AttnMetaSelectError):
            S.install(runner, env={S.ENV: '1'})

    def test_mutants_caught(self):
        c = _S['c']
        caught = {}
        caught['off-calls-batch'] = bool(check_equal(self, c, _mod(False, OffCallsBatch), selected=False, mutant=True))
        caught['on-calls-stock'] = bool(check_equal(self, c, _mod(True, OnCallsStock), selected=True, mutant=True))

        real = S.select

        def ignored(runner, on):
            if type(on) is not bool:
                raise S.AttnMetaSelectError('attn-meta-select-not-bool')
            return dict(before=None, after=None, order=None)
        S.select = ignored
        try:
            caught['select-ignored'] = bool(check_equal(self, c, _mod(True), selected=True, mutant=True))
        finally:
            S.select = real
        print('MUTANTS', json.dumps(caught, sort_keys=True))
        self.assertEqual(caught, dict.fromkeys(caught, True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
