"""Acceptance tests for the attn-meta overlay: patches/s5_engine/attn_meta_batch.py (option A, default off).

Container only (pinned image: real torch + vllm builders); on the hub without torch the module is skipped.
Ported from harness/test_attn_meta_gdn_batch.py (w5, the prototype) to the overlay code itself, installed as the
engine installs it (``attn_meta_batch.install(runner)``, an instance attribute on the real model state), plus:
the install / default-off contract, a dynamic decode sequence (block crossings, admissions, finishes), and the
native35 rule (the native lane's private builder clones share no storage with the stock builders and stay
byte-untouched with the switch on; a mutant that writes into a clone and a mutant clone that shares a buffer are
caught). Fixture and leaves: tools/attn_meta_cases.py (LEAVES) over tools/attn_meta_witness.py (LEAVES).

Run in the image: ``tools/attn_meta_selfrun2.py --mode tests``.
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

try:
    import numpy as np
    import torch
    import vllm  # noqa: F401
except ImportError:  # hub: no torch/vllm
    raise unittest.SkipTest('needs the pinned image (torch + vllm)')

import attn_meta_witness as W  # noqa: E402
import attn_meta_cases as C  # noqa: E402
import attn_meta_batch as AMB  # noqa: E402

MODEL_CONFIG = os.environ.get('ATTN_META_MODEL_CONFIG', '/w/model-config.json')
OVERLAY_QSA = os.environ.get('ATTN_META_OVERLAY_QSA', '/w/overlay-qsa.py')
_S = {}


def setUpModule():
    W.install_triton_stub()
    import vllm.models.qwen4_exp.common.qsa_cache as qc
    qc.build_qsa_metadata = qc.build_qsa_metadata_triton
    if Path(OVERLAY_QSA).is_file():
        W.install_overlay_builder(Path(OVERLAY_QSA))
    import tempfile
    work = tempfile.mkdtemp(prefix='attn-meta-overlay-', dir='/tmp')
    cfg, _ = W.build_config(MODEL_CONFIG, Path(work), torch)
    text = json.loads(Path(MODEL_CONFIG).read_text())['text_config']
    _S['c'] = C.Cases(W, cfg, text, torch, np, torch.device('cpu'))


def cases():
    return _S['c']


def assert_same(tc, sa, sb):
    tc.assertEqual(sa['keys'], sb['keys'])
    tc.assertEqual(sa['objects'], sb['objects'])
    for k in sa['keys']:
        tc.assertEqual(sa['meta'][k], sb['meta'][k], k)
    tc.assertEqual(sa['buffers'], sb['buffers'])


def run_sequence(c, mod, stage, plan, check_every=1):
    """Stock and overlay fixtures through the same decode plan; first difference or None, plus stats."""
    a, b = c.fixture(stage, plan['hist']), c.fixture(stage, plan['hist'])
    batch, _ = c.install(mod, b)
    c.poison(a)
    c.poison(b)
    seq = C.Sequence(c, stage, plan)
    sizes = set()
    for step in range(plan['steps']):
        n, R = seq.load(a)
        seq.load(b)
        sizes.add((n, R))
        ma, mb = c.call(a), c.call(b)
        if list(ma) != list(mb):
            return 'keys/order@%d' % step, None
        if c.gdn_buffers(a) != c.gdn_buffers(b):
            return 'gdn-buffers@%d' % step, None
        if step % check_every == 0 or step == plan['steps'] - 1:
            sa, sb = c.snapshot(a, ma), c.snapshot(b, mb)
            d = c.first_difference(sa, sb)
            if d:
                return d + '@%d' % step, None
        seq.advance(step)
    return None, dict(crossings=seq.crossings, sizes=sorted(sizes), counters=batch.counters() if batch else None)


class Equality(unittest.TestCase):
    def test_equal_all_histories_both_stages(self):
        c = cases()
        for stage in (0, 1):
            for name, hist in C.HISTS.items():
                with self.subTest(stage=stage, hist=name):
                    sa, sb, batch = c.compare(AMB, stage, hist)
                    assert_same(self, sa, sb)
                    self.assertEqual(batch.counters()['batched'], 1)
                    self.assertEqual(sum(batch.counters()['fallback'].values()), 0)

    def test_equal_under_full_graph_padding(self):
        c = cases()
        for n, R in C.PADDED:
            for stage in (0, 1):
                with self.subTest(n=n, R=R, stage=stage):
                    sa, sb, batch = c.compare(AMB, stage, C.HISTS['boundary'][:n], pad_to=R)
                    assert_same(self, sa, sb)
                    self.assertEqual(batch.counters()['batched'], 1)

    def test_decode_sequence_crossings_admissions_finishes(self):
        c = cases()
        plan = C.sequence_plan()
        for stage in (1, 0):
            with self.subTest(stage=stage):
                diff, stats = run_sequence(c, AMB, stage, plan, check_every=4)
                self.assertIsNone(diff)
                self.assertGreaterEqual(stats['crossings'], 7)
                self.assertGreater(len({n for n, _ in stats['sizes']}), 2)   # admissions and finishes happened
                self.assertTrue(any(R > n for n, R in stats['sizes']))       # padded steps happened
                self.assertEqual(stats['counters']['batched'], plan['steps'])

    def test_batched_path_really_engaged(self):
        c = cases()
        for stage in (0, 1):
            fx = c.fixture(stage, C.HISTS['h65k'])
            ids, reason = AMB.eligible_groups(fx.mstate, fx.batch, fx.mode, fx.attn, False)
            self.assertEqual((len(ids), reason), (18, None))
        calls = []
        real = torch.stack
        fx = c.fixture(1, C.HISTS['h65k'])
        c.install(AMB, fx)
        try:
            torch.stack = lambda *a, **k: calls.append(1) or real(*a, **k)
            c.call(fx)
        finally:
            torch.stack = real
        self.assertEqual(len(calls), 1)

    def test_gdn_metadata_points_into_builder_buffers(self):
        c = cases()
        fx = c.fixture(1, C.HISTS['h110k'])
        c.install(AMB, fx)
        md = c.call(fx)
        n = 0
        for gid, b in c.builders(fx):
            if type(b).__name__ == 'GDNAttentionMetadataBuilder':
                m = md[fx.attn[gid][0].layer_names[0]]
                self.assertEqual(m.non_spec_state_indices_tensor.data_ptr(), b.non_spec_state_indices_tensor.data_ptr())
                self.assertEqual(m.non_spec_query_start_loc.data_ptr(), b.non_spec_query_start_loc.data_ptr())
                n += 1
        self.assertEqual(n, 18)


class Fallbacks(unittest.TestCase):
    """Outside the eligible case the installed overlay is the stock call, exactly, and says why."""

    def stock_path(self, stage, hist, reason, **kw):
        sa, sb, batch = cases().compare(AMB, stage, hist, **kw)
        assert_same(self, sa, sb)
        cnt = batch.counters()
        self.assertEqual(cnt['batched'], 0)
        self.assertEqual({k: v for k, v in cnt['fallback'].items() if v}, {reason: 1})

    def test_cudagraph_none_and_piecewise(self):
        from vllm.config.compilation import CUDAGraphMode
        for mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            with self.subTest(mode=str(mode)):
                self.stock_path(1, C.HISTS['h65k'], 'mode', mode=mode)

    def test_prefill_in_batch(self):
        def mutate(fx):
            import dataclasses
            nst = fx.batch.num_scheduled_tokens.copy()
            nst[-1] = 2
            fx.batch = dataclasses.replace(fx.batch, num_scheduled_tokens=nst)
        self.stock_path(1, C.HISTS['h65k'], 'not-all-decode', mutate=mutate)

    def test_is_prefilling_row(self):
        def mutate(fx):
            import dataclasses
            p = fx.batch.is_prefilling_np.copy()
            p[0] = True
            fx.batch = dataclasses.replace(fx.batch, is_prefilling_np=p)
        self.stock_path(0, C.HISTS['h65k'], 'not-all-decode', mutate=mutate)

    def test_spec_decode_builder(self):
        def mutate(fx):
            for _, b in cases().builders(fx):
                if type(b).__name__ == 'GDNAttentionMetadataBuilder':
                    b.use_spec_decode = True
                    break
        self.stock_path(1, C.HISTS['h65k'], 'builder', mutate=mutate)

    def test_not_align_mode(self):
        c = cases()
        old = c.cfg.cache_config.mamba_cache_mode
        try:
            c.cfg.cache_config.mamba_cache_mode = 'none'
            fx = c.fixture(1, C.HISTS['h65k'])
            self.assertEqual(AMB.eligible_groups(fx.mstate, fx.batch, fx.mode, fx.attn, False), (None, 'cache-mode'))
        finally:
            c.cfg.cache_config.mamba_cache_mode = old

    def test_capture_calls_are_stock(self):
        """cudagraph_utils' capture call (mode NONE, for_capture) and a FULL for_capture call both run stock."""
        from vllm.config.compilation import CUDAGraphMode
        c = cases()
        for mode, reason in ((CUDAGraphMode.NONE, 'mode'), (CUDAGraphMode.FULL, 'capture')):
            with self.subTest(mode=str(mode)):
                a, b = c.fixture(1, C.HISTS['h65k']), c.fixture(1, C.HISTS['h65k'])
                batch, _ = c.install(AMB, b)
                c.poison(a)
                c.poison(b)
                ma = a.mstate.prepare_attn(a.batch, mode, a.tables, a.slots, a.attn, a.kv_cfg, for_capture=True)
                mb = b.mstate.prepare_attn(b.batch, mode, b.tables, b.slots, b.attn, b.kv_cfg, for_capture=True)
                assert_same(self, c.snapshot(a, ma), c.snapshot(b, mb))
                self.assertEqual({k: v for k, v in batch.counters()['fallback'].items() if v}, {reason: 1})


class Install(unittest.TestCase):
    def test_off_sets_nothing(self):
        c = cases()
        for env in ({}, {AMB.ENV: '0'}):
            with self.subTest(env=env):
                fx = c.fixture(1, C.HISTS['one'])
                batch, runner = c.install(AMB, fx, env=env)
                self.assertIsNone(batch)
                self.assertNotIn('prepare_attn', vars(fx.mstate))
                self.assertIsNone(AMB.installed(runner))

    def test_unset_process_env_is_off(self):
        os.environ.pop(AMB.ENV, None)
        fx = cases().fixture(1, C.HISTS['one'])
        self.assertIsNone(AMB.install(types.SimpleNamespace(model_state=fx.mstate)))
        self.assertNotIn('prepare_attn', vars(fx.mstate))

    def test_invalid_values_refuse(self):
        for v in ('', 'true', 'yes', 'on', ' 1', '1 ', '2', 'TRUE'):
            with self.subTest(v=v):
                with self.assertRaises(AMB.AttnMetaBatchError):
                    AMB.enabled({AMB.ENV: v})

    def test_off_imports_nothing(self):
        code = ('import sys; sys.path.insert(0, %r); import attn_meta_batch as m; '
                'r = m.install(object(), env={}); '
                'print(r is None, any(k == "torch" or k.startswith("vllm") for k in sys.modules))'
                % str(ROOT / 'patches/s5_engine'))
        out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual((out.returncode, out.stdout.split()), (0, ['True', 'False']), out.stderr[-500:])

    def test_instance_only_and_refusals(self):
        c = cases()
        fx = c.fixture(1, C.HISTS['one'])
        cls = type(fx.mstate)
        before = cls.__dict__.get('prepare_attn')
        batch, runner = c.install(AMB, fx)
        self.assertIs(vars(fx.mstate)['prepare_attn'], batch)
        self.assertIs(cls.__dict__.get('prepare_attn'), before)
        with self.assertRaises(AMB.AttnMetaBatchError):           # twice
            AMB.install(runner, env={AMB.ENV: '1'})
        fx2 = c.fixture(1, C.HISTS['one'])                         # a wrapper already on the instance
        fx2.mstate.prepare_attn = lambda *a, **k: None
        with self.assertRaises(AMB.AttnMetaBatchError):
            AMB.install(types.SimpleNamespace(model_state=fx2.mstate), env={AMB.ENV: '1'})
        with self.assertRaises(AMB.AttnMetaBatchError):           # not a mamba-hybrid model state
            AMB.install(types.SimpleNamespace(model_state=object()), env={AMB.ENV: '1'})
        fx3 = c.fixture(1, C.HISTS['one'])                         # a class that overrides prepare_attn

        class Over(cls):
            def prepare_attn(self, *a, **k):
                return cls.prepare_attn(self, *a, **k)
        fx3.mstate.__class__ = Over
        with self.assertRaises(AMB.AttnMetaBatchError):
            AMB.install(types.SimpleNamespace(model_state=fx3.mstate), env={AMB.ENV: '1'})

    def test_seam_wrapper_installed_after_composes(self):
        """handler_anatomy and engine_offers wrap whatever the instance holds: both run, the batch still engages."""
        c = cases()
        fx = c.fixture(1, C.HISTS['h65k'])
        batch, _ = c.install(AMB, fx)
        inner = fx.mstate.prepare_attn
        seen = []

        def seam(*a, **k):
            seen.append(1)
            return inner(*a, **k)
        fx.mstate.prepare_attn = seam
        c.call(fx)
        self.assertEqual((seen, batch.counters()['batched']), ([1], 1))


class NativeIsolation(unittest.TestCase):
    """native35: no buffer shared between the native lane's private builders and the stock builders, ever."""

    def check(self, mod, lane_chunk_mod=None, stages=(1, 0)):
        c = cases()
        for stage in stages:
            fx = c.fixture(stage, C.HISTS['boundary'])
            if lane_chunk_mod is not None:
                groups = lane_chunk_mod.private_attention_groups(fx.attn, torch_module=torch)[0]
            else:
                groups = c.native_clones(fx)[0]
            shared = c.shared_storage(fx, groups)
            if shared:
                return 'shared-storage', shared[:3]
            mod.__dict__['_native_view'] = lambda attn_groups, groups=groups: groups
            batch, _ = c.install(mod, fx)
            c.poison(fx, groups)
            before = c.buffers(fx, groups)
            stock_before = c.gdn_buffers(fx)
            for hist in (C.HISTS['boundary'], C.HISTS['h110k'], C.HISTS['boundary'][:5]):
                fx2 = c.fixture(stage, hist, pad_to=8 if len(hist) < 8 else None)
                fx.batch, fx.tables, fx.slots = fx2.batch, fx2.tables, fx2.slots
                c.call(fx)
            if c.buffers(fx, groups) != before:
                return 'clone-bytes-changed', stage
            if batch is None or batch.counters()['batched'] != 3:
                return 'not-engaged', None
            if c.gdn_buffers(fx) == stock_before:
                return 'stock-not-written', None
        return None, None

    def test_clones_untouched_with_switch_on(self):
        self.assertEqual(self.check(AMB), (None, None))

    def test_native_mutants_caught(self):
        res = {}
        mod = mutant_module(AMB, 'writes-native-clone', NATIVE_MUTANTS['writes-native-clone'])
        res['writes-native-clone'] = self.check(mod, stages=(1,))[0]
        import lane_chunk
        lc = mutant_module(lane_chunk, 'clone-shares-gdn-buffer', NATIVE_MUTANTS['clone-shares-gdn-buffer'])
        res['clone-shares-gdn-buffer'] = self.check(AMB, lane_chunk_mod=lc, stages=(1,))[0]
        print('NATIVE_MUTANTS', json.dumps(res, sort_keys=True), flush=True)
        self.assertEqual([n for n, r in res.items() if r is None], [])


NATIVE_MUTANTS = {
    'writes-native-clone': ('    builders = [attn_groups[i][0].metadata_builders[0] for i in ids]\n',
                            '    builders = [_native_view(attn_groups)[i][0].metadata_builders[0] for i in ids]\n'),
    'clone-shares-gdn-buffer': ("                    cloned = (value if field_class.startswith('shared-read-only') else",
                                "                    cloned = (value if field_class.startswith('shared-read-only') "
                                "or name.startswith('non_spec_') else"),
}

MUTANTS = {
    'start-without-minus-one': ('start = (seq_lens - 1) // bs', 'start = seq_lens // bs'),
    'no-clamp': ('    start.clamp_(min=0)\n', '    pass\n'),
    'rotated-destinations': ('dst_states = [b.non_spec_state_indices_tensor[:R] for b in builders]',
                             'dst_states = [b.non_spec_state_indices_tensor[:R] for b in builders[1:] + builders[:1]]'),
    'qsl-not-written': ('    torch._foreach_copy_(dst_qsl, [qsl] * len(builders), non_blocking=True)\n', '    pass\n'),
    'fresh-tensor-in-metadata': ('non_spec_state_indices_tensor=st,', 'non_spec_state_indices_tensor=st.clone(),'),
    'num-decode-tokens-unpadded': ('num_decodes=R, num_decode_tokens=T,',
                                   'num_decodes=R, num_decode_tokens=input_batch.num_tokens,'),
    'dict-order-rest-first': ('                out[name] = gdn[name] if i in idset else rest[name]\n    return out',
                              '                out[name] = gdn[name] if i in idset else rest[name]\n'
                              '    return {**rest, **out}'),
    'unpadded-rows': ('    R = input_batch.num_reqs_after_padding          # FULL', '    R = input_batch.num_reqs  # FULL'),
    'flag-default-on': ("value = env.get(ENV, '0')", "value = env.get(ENV, '1')"),
    'flag-loose': ("    if value not in ('0', '1'):\n        raise AttnMetaBatchError", "    if False:\n        raise AttnMetaBatchError"),
    'ignores-capture-gate': ("    if for_capture:\n        return None, 'capture'\n", ''),
    'ignores-prefill': ("    if int(input_batch.num_scheduled_tokens.max()) != 1 or bool(input_batch.is_prefilling_np[:n].any()):",
                        "    if False:"),
    'class-install': ('    mstate.prepare_attn = batch\n', '    type(mstate).prepare_attn = batch\n'),
    'no-prewrapped-refusal': ("    if 'prepare_attn' in vars(mstate):", "    if False:"),
}


def mutant_module(base, name, rule):
    src = Path(base.__file__).read_text()
    a, b = rule
    assert src.count(a) == 1, (name, src.count(a))
    mod = types.ModuleType(base.__name__ + '_mutant_' + name.replace('-', '_'))
    mod.__file__ = base.__file__
    exec(compile(src.replace(a, b), base.__file__ + ':' + name, 'exec'), mod.__dict__)
    return mod


class Mutants(unittest.TestCase):
    """Each overlay mutant must be caught by one of the checks above (the first one that catches it is named)."""

    def caught(self, mod):
        c = cases()
        fx = None
        try:
            if mod.enabled({}) is not False:
                return 'default-off'
            try:
                mod.enabled({mod.ENV: 'true'})
                return 'invalid-value'
            except mod.AttnMetaBatchError:
                pass
            fx = c.fixture(1, C.HISTS['one'])
            batch, runner = c.install(mod, fx)
            if 'prepare_attn' not in vars(fx.mstate):
                return 'instance-only'
            fx2 = c.fixture(1, C.HISTS['one'])
            fx2.mstate.prepare_attn = lambda *a, **k: None
            try:
                mod.install(types.SimpleNamespace(model_state=fx2.mstate), env={mod.ENV: '1'})
                return 'prewrapped-refusal'
            except mod.AttnMetaBatchError:
                pass
            for stage, hist, pad in ((0, C.HISTS['boundary'], None), (1, C.HISTS['boundary'][:7], 8),
                                     (1, C.HISTS['h110k'], None)):
                sa, sb, _ = c.compare(mod, stage, hist, pad_to=pad)
                d = c.first_difference(sa, sb)
                if d:
                    return d
            for mutate in (prefill_row, prefill_tokens):
                # Defence in depth: FULL_DECODE_ONLY never replays a mixed batch in FULL, and these fixture rows
                # differ from a decode row only in the fields the gate reads (stock then still sees decodes), so
                # the gate's own verdict and its fallback counter are the check, as in Fallbacks.stock_path.
                sa, sb, b2 = c.compare(mod, 1, C.HISTS['h65k'], mutate=mutate)
                d = c.first_difference(sa, sb)
                if d:
                    return 'fallback-' + d
                if b2.counters()['fallback']['not-all-decode'] != 1:
                    return 'prefill-gate'
            from vllm.config.compilation import CUDAGraphMode
            f3 = c.fixture(1, C.HISTS['h65k'])
            b3, _ = c.install(mod, f3)
            f3.mstate.prepare_attn(f3.batch, CUDAGraphMode.FULL, f3.tables, f3.slots, f3.attn, f3.kv_cfg,
                                   for_capture=True)
            if b3.counters()['fallback']['capture'] != 1:
                return 'capture-gate'
            diff, _ = run_sequence(c, mod, 1, dict(C.sequence_plan(), steps=16), check_every=8)
            if diff:
                return 'sequence-' + diff
        except Exception as exc:  # a crash is a catch too, but say so
            return 'exception:' + type(exc).__name__
        finally:
            restore_classes(fx)
        return None

    def test_every_mutant_caught(self):
        res = {name: self.caught(mutant_module(AMB, name, rule)) for name, rule in MUTANTS.items()}
        print('MUTANTS', json.dumps(res, sort_keys=True), flush=True)
        self.assertEqual([n for n, r in res.items() if r is None], [])


def prefill_row(fx):
    import dataclasses
    p = fx.batch.is_prefilling_np.copy()
    p[0] = True
    fx.batch = dataclasses.replace(fx.batch, is_prefilling_np=p)


def prefill_tokens(fx):
    import dataclasses
    nst = fx.batch.num_scheduled_tokens.copy()
    nst[-1] = 2
    fx.batch = dataclasses.replace(fx.batch, num_scheduled_tokens=nst)


def restore_classes(fx):
    """The class-install mutant writes the class: put the stock function back so later cases stay stock."""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    import vllm.models.qwen4_exp.nvidia.model_state as ms
    for cls in (ms.Qwen4ExpModelState,):
        if 'prepare_attn' in cls.__dict__:
            delattr(cls, 'prepare_attn')
    assert ms.Qwen4ExpModelState.prepare_attn is MambaHybridModelState.prepare_attn


if __name__ == '__main__':
    unittest.main(verbosity=2)
