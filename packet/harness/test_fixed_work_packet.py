"""Fixed work and the seed order on the j2 packet (brief 12, 14 Nov 2026): harness/fixed_work_packet and the builder's
patch of the bundle's mechanism rule (tools/n3j2_junction_rule.py). Hub and bundle, no skip.

* seed order: plain is j2's seeds, swap moves them (sp0/n30 seed 2, n31/sp1 seed 1), unknown values refuse; the real
  n3j2_packet.configure wrapped in fresh pythons per mode (arm order and every other replay key unchanged);
* the rule's answers: unset = the parent rule's literals (480 s, 1 + index); fixed work parses as the driver does;
  replay_window_ok accepts only a summary that reached this setting's targets without the cap, inside cap + finish;
* the REAL parent rule, patched as the builder patches it, over its own fixture: unset reads GO as the parent rule
  does; swap refuses j2-seeded arms and reads GO on swapped ones; fixed work reads GO on an arm that sent past 480 s
  and stopped before it, and HOLD on a cap hit, other targets, a send past the cap; the unpatched rule refuses the
  fixed-work arm (the patch is what lets it through);
* mutants: each patch line dropped is caught.
CPU only, synthetic records, no engine.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compose4_replay as replay  # noqa: E402
import fixed_work_packet as fwp  # noqa: E402

TOOLS = next(p for p in (HERE.parent / 'tools', HERE / 'tools') if (p / 'build_q2024p_j2_cx.py').is_file())
IN_BUNDLE = (HERE / 'n3j2_packet.py').is_file() and (HERE / 'tools' / 'n3j2_junction_rule.py').is_file()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cx = _load('build_q2024p_j2_cx_under_fw', TOOLS / 'build_q2024p_j2_cx.py')
PACKET_ROOT = HERE if IN_BUNDLE else Path(cx.hgb.PARENT)
RULE_PATH = PACKET_ROOT / 'tools' / 'n3j2_junction_rule.py'
SPEC, CAP = cx.FIXED_WORK['n480']
FIXED_ENV = {replay.REPLY_TEXT_ENV: 'seeded', replay.FIXED_WORK_ENV: SPEC, replay.FIXED_CAP_ENV: str(CAP)}
CLEAR = (fwp.ORDER_ENV, replay.FIXED_WORK_ENV, replay.FIXED_CAP_ENV, replay.REPLY_TEXT_ENV)


def parent_rule_text():
    text = RULE_PATH.read_text()
    if '_fwp.' in text:                       # a bundle built with fixed work or a seed order: already patched
        for old, new in cx.RULE_PATCH:
            assert text.count(new) == 1 and old not in text, new
            text = text.replace(new, old)
    return text


def rule_module(text, name='n3j2_junction_rule_fw'):
    module = types.ModuleType(name)
    module.__file__ = str(RULE_PATH)
    with mock.patch.object(sys, 'argv', ['rule']):
        exec(compile(text, str(RULE_PATH), 'exec'), module.__dict__)
    return module


def fixture_module(rule):
    saved = sys.modules.get('n3j2_junction_rule')
    sys.modules['n3j2_junction_rule'] = rule
    try:
        return _load('test_n3j2_junction_rule_fw', RULE_PATH.parent / 'test_n3j2_junction_rule.py')
    finally:
        if saved is None:
            sys.modules.pop('n3j2_junction_rule', None)
        else:
            sys.modules['n3j2_junction_rule'] = saved


PARENT_TEXT = parent_rule_text()
PATCHED = rule_module(cx.rule_patch(PARENT_TEXT))
CHECKS = fixture_module(PATCHED)


def env(**extra):
    out = {k: v for k, v in os.environ.items() if k not in CLEAR}
    out.update(extra)
    return out


class Order(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(fwp.SEEDS, {'plain': {0: 1, 1: 2}, 'swap': {0: 2, 1: 1}})
        self.assertEqual(fwp.seed_order({}), 'plain')
        self.assertEqual(fwp.seed_order({fwp.ORDER_ENV: ''}), 'plain')
        self.assertEqual([fwp.expected_seed(l, {}) for l in ('sp.g2-live-0', 'n3.g2-live-1')], [1, 2])
        self.assertEqual([fwp.expected_seed(l, {fwp.ORDER_ENV: 'swap'}) for l in ('sp.g2-live-0', 'n3.g2-live-1')], [2, 1])
        with self.assertRaises(ValueError):
            fwp.seed_order({fwp.ORDER_ENV: 'alternate'})      # a cohort word, never a block's export

    def test_wrap_on_a_table(self):
        table = types.SimpleNamespace(REPLAY_SEEDS={0: 1, 1: 2})
        self.assertIs(fwp.wrap_env(lambda: table, {})(), table)
        self.assertEqual(table.REPLAY_SEEDS, {0: 1, 1: 2})
        fwp.wrap_env(lambda: table, {fwp.ORDER_ENV: 'swap'})()
        self.assertEqual(table.REPLAY_SEEDS, {0: 2, 1: 1})
        with self.assertRaises(AssertionError):       # plain over a table left swapped is refused, not silently kept
            fwp.wrap_env(lambda: table, {})()

    def test_real_configure_in_fresh_pythons(self):
        code = ('import importlib.util, json, sys\n'
                'sys.path.insert(0, %r)\n'
                'spec = importlib.util.spec_from_file_location("n3j2_packet", %r)\n'
                'm = importlib.util.module_from_spec(spec); sys.modules["n3j2_packet"] = m; spec.loader.exec_module(m)\n'
                'import fixed_work_packet as f\n'
                't = f.wrap_env(m.configure)()\n'
                'rows = [r for r in t.arms() if r["kind"] == "replay"]\n'
                'print(json.dumps([[r["label"], {k: v for k, v in t.replay_config(r).items()}] for r in rows]))\n'
                % (str(HERE), str(PACKET_ROOT / 'n3j2_packet.py')))
        got = {}
        for order in ('plain', 'swap'):
            done = subprocess.run([sys.executable, '-c', code], env=env(**{fwp.ORDER_ENV: order}), capture_output=True,
                                  text=True, timeout=120, cwd=str(HERE))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            got[order] = json.loads(done.stdout.strip().splitlines()[-1])
        labels = [l for l, _ in got['plain']]
        self.assertEqual(labels, ['sp.g2-live-0', 'n3.g2-live-0', 'n3.g2-live-1', 'sp.g2-live-1'])
        self.assertEqual(labels, [l for l, _ in got['swap']])
        self.assertEqual([c['SEED'] for _, c in got['plain']], [1, 1, 2, 2])
        self.assertEqual([c['SEED'] for _, c in got['swap']], [2, 2, 1, 1])
        for (_, a), (_, b) in zip(got['plain'], got['swap']):
            self.assertEqual({k: v for k, v in a.items() if k != 'SEED'}, {k: v for k, v in b.items() if k != 'SEED'})


class Answers(unittest.TestCase):
    def test_unset_is_the_parent_literals(self):
        self.assertEqual(fwp.send_limit_s({}), 480)
        self.assertIsNone(fwp.fixed(1, {}))
        arm = dict(replay_start_s=10.0, replay_end_s=490.0, replay_config=dict(SEED=1))
        self.assertTrue(fwp.replay_window_ok(arm, {}))
        self.assertFalse(fwp.replay_window_ok(dict(arm, replay_end_s=489.0), {}))

    def test_fixed_parses_as_the_driver(self):
        for seed in (1, 2):
            cfg = dict(replay.T.REPLAY, SEED=seed, DURATION=480.0, AGENTS=8)
            self.assertEqual(fwp.fixed(seed, FIXED_ENV), replay.fixed_work_targets(cfg, environ=FIXED_ENV, reply_text='seeded'))
        self.assertEqual(fwp.send_limit_s(FIXED_ENV), float(CAP))
        with self.assertRaises(ValueError):
            fwp.fixed(1, dict(FIXED_ENV, **{replay.REPLY_TEXT_ENV: ''}))

    def summary(self, seed=1, **kw):
        targets = fwp.fixed(seed, FIXED_ENV)[0]
        reached = [100.0 + i for i in range(8)]
        out = dict(targets=targets, cap_s=float(CAP), completed=list(targets), reached_s=reached, all_reached=True,
                   end_s=max(reached), cap_hit=False)
        out.update(kw)
        return out

    def test_window(self):
        base = dict(replay_start_s=10.0, replay_end_s=130.0, replay_config=dict(SEED=1))
        ok = dict(base, replay_summary=dict(fixed_work=self.summary()))
        self.assertTrue(fwp.replay_window_ok(ok, FIXED_ENV))
        self.assertTrue(fwp.replay_window_ok(dict(ok, replay_end_s=10.0 + CAP + replay.FINISH_BOUND_S), FIXED_ENV))
        bad = [dict(base),                                                             # no fixed_work in the summary
               dict(base, replay_summary=dict(fixed_work=self.summary(cap_hit=True))),
               dict(base, replay_summary=dict(fixed_work=self.summary(all_reached=False))),
               dict(base, replay_summary=dict(fixed_work=self.summary(targets=[1] * 8))),
               dict(base, replay_summary=dict(fixed_work=self.summary(seed=2))),          # seed 2's targets on seed 1
               dict(base, replay_summary=dict(fixed_work=self.summary(cap_s=700.0))),
               dict(base, replay_summary=dict(fixed_work=self.summary(end_s=99.0))),      # not the last agent's time
               dict(ok, replay_end_s=100.0),                                              # ended before the stop
               dict(ok, replay_end_s=10.0 + CAP + replay.FINISH_BOUND_S + 1)]
        for arm in bad:
            with self.subTest(arm=arm):
                self.assertFalse(fwp.replay_window_ok(arm, FIXED_ENV))


class Rule(unittest.TestCase):
    """The parent's mechanism rule, patched as the builder patches it, over the parent's own fixture."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = CHECKS.Fixture(self.root)

    def read(self, module=PATCHED, **extra):
        self.fixture.write()
        with mock.patch.dict(os.environ, env(**extra), clear=True):
            return module.read(self.root)

    def reseed(self, order):
        for label in PATCHED.REPLAYS:
            arm = self.fixture.arms[label]
            arm['replay_config'] = dict(arm['replay_config'], SEED=fwp.SEEDS[order][int(label[-1])])
        self.fixture.refresh_grade()

    def make_fixed(self, **summary):
        """Every replay arm: fixed work reached at 470 s (before 480), one turn sent at 600 s (past 480, inside the cap)."""
        for label in PATCHED.REPLAYS:
            arm = self.fixture.arms[label]
            seed = arm['replay_config']['SEED']
            targets = fwp.fixed(seed, FIXED_ENV)[0]
            reached = [400.0 + 10 * i for i in range(7)] + [470.0]
            fw = dict(targets=targets, cap_s=float(CAP), completed=list(targets), reached_s=reached, all_reached=True,
                      end_s=470.0, cap_hit=False)
            fw.update(summary)
            arm['replay_summary'] = dict(arm['replay_summary'], fixed_work=fw, reply_text='seeded')
            arm['replay_end_s'] = arm['replay_start_s'] + 475.0
            arm['wall_end_s'] = arm['replay_end_s'] + 1
            row = self.fixture.turns[label][-1]
            row['t'], row['arrivals_ms'] = 600, [(600 + row['ttft']) * 1000, (600 + row['ttft'] + 5) * 1000]
            if label.startswith('n3.'):          # the fixture's in-arm memory receipt sits at wall + 500: one inside
                self.fixture.memory.extend((arm['wall_start_s'] + 400, rank, 0) for rank in range(4))
        self.fixture.refresh_grade()

    def assert_go(self, result):
        self.assertEqual((result['verdict'], result.get('missing'), result.get('failures')), ('GO', [], []), result.get('missing'))

    def missing(self, result):
        return ' '.join(result.get('missing') or [])

    def test_unset_reads_as_the_parent_rule(self):
        parent = rule_module(PARENT_TEXT, 'n3j2_junction_rule_parent')
        a, b = self.read(), self.read(parent)
        self.assert_go(a)
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_swap(self):
        self.assertIn('replay-scope:sp.g2-live-0', self.missing(self.read(**{fwp.ORDER_ENV: 'swap'})))
        self.reseed('swap')
        self.assert_go(self.read(**{fwp.ORDER_ENV: 'swap'}))
        self.assertIn('replay-scope:sp.g2-live-0', self.missing(self.read()))           # and swapped arms fail plain

    def test_fixed_work(self):
        self.make_fixed()
        self.assert_go(self.read(**FIXED_ENV))
        parent = rule_module(PARENT_TEXT, 'n3j2_junction_rule_parent')
        self.assertNotEqual(self.read(parent, **FIXED_ENV)['verdict'], 'GO')            # the patch is what lets it through
        self.assertNotEqual(self.read()['verdict'], 'GO')                                # and only under the setting
        self.reseed('swap')
        self.make_fixed()
        self.assert_go(self.read(**dict(FIXED_ENV, **{fwp.ORDER_ENV: 'swap'})))

    def test_fixed_work_refusals(self):
        for change, code in ((dict(cap_hit=True), 'replay-window:'), (dict(targets=[1] * 8), 'replay-window:'),
                             (dict(all_reached=False), 'replay-window:')):
            with self.subTest(change=change):
                self.setUp()
                self.make_fixed(**change)
                self.assertIn(code + 'sp.g2-live-0', self.missing(self.read(**FIXED_ENV)))
        self.setUp()
        self.make_fixed()
        row = self.fixture.turns['n3.g2-live-1'][-1]
        row['t'], row['arrivals_ms'] = CAP + 1, [(CAP + 1 + row['ttft']) * 1000]
        self.assertIn('journal-turn-times:n3.g2-live-1', self.missing(self.read(**FIXED_ENV)))


class Mutants(unittest.TestCase):
    def test_each_patch_line_dropped_is_caught(self):
        for i in range(1, len(cx.RULE_PATCH)):
            with self.subTest(dropped=cx.RULE_PATCH[i][0][:40]):
                text = PARENT_TEXT
                for j, (old, new) in enumerate(cx.RULE_PATCH):
                    if j != i:
                        text = text.replace(old, new)
                mutant = rule_module(text, 'n3j2_junction_rule_mutant')
                case = Rule('test_swap' if 'SEED' in cx.RULE_PATCH[i][0] else 'test_fixed_work')
                case.setUp()
                with mock.patch.object(Rule, 'read', lambda self, module=mutant, **extra: Rule.__dict__['_read'](self, module, **extra)):
                    with self.assertRaises(AssertionError):
                        getattr(case, case._testMethodName)()

    def test_builder_patch_texts_are_once_in_the_parent_rule(self):
        for old, new in cx.RULE_PATCH:
            self.assertEqual(PARENT_TEXT.count(old), 1, old)
            self.assertNotIn(new, PARENT_TEXT)


Rule._read = Rule.read


if __name__ == '__main__':
    unittest.main()
