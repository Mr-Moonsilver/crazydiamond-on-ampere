"""alpha_next_stop_race_h.py: the stop-race class over the health-scoped scan.  A startup read is not an "other death"; a stall after
the first health 200 still is.  Every stop-race test of the frozen module's own file also runs against this module (InheritedStopRace).
Fixture: harness/fixtures/native43_engine_death_h.  Leaves: native43_engine_death_h_fixture_leaves.json.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alpha_next_stop_race as FROZEN  # noqa: E402
import alpha_next_stop_race_h as sr  # noqa: E402

FROZEN_SHA256 = '94a7aa4ea3426eed5ec4387f0a0714a1ed11a0830dcebb89ac00ac44eb5a91be'
_spec = importlib.util.spec_from_file_location('test_alpha_next_queue_for_h', HERE / 'test_alpha_next_queue.py')
_old = importlib.util.module_from_spec(_spec)  # a private copy: the frozen module's own test run keeps its globals
_spec.loader.exec_module(_old)
_old.sr = sr
EXCERPT = (HERE / 'fixtures/native43_engine_death_h/startup-stall-engine-stream.log').read_text().splitlines()
STALL = next(x for x in EXCERPT if 'No available shared memory' in x)
HEALTH = next(x for x in EXCERPT if 'GET /health' in x)


def at(ns, line):
    return sr.death.iso(ns) + line[line.index('Z ') + 1:]


class InheritedStopRace(_old.StopRace):
    """Every frozen stop-race test over this module (only the record's schema name differs), then the startup read: the real stop
    lines of q18a-step-b-e with the real startup stall and health lines placed before them."""

    INTENT = _old.STOP_INTENTS['step-b-e']

    def test_cli_record_and_rcs(self):
        logs, intent, idle = self.files(_old.NATIVE26_STOP, _old.STOP_INTENTS['native26'])
        out = self.dir / 'stop-race.json'
        argv = ['--log', logs[0], '--log', logs[1], '--stop-intent', intent, '--end-full-idle', idle, '--out', str(out)]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(sr.main(argv), 0)
        self.assertEqual(printed.getvalue().strip(), 'ALPHA_NEXT_STOP_RACE STOP_RACE')
        self.assertEqual(json.loads(out.read_text())['schema'], 'alpha-next-stop-race-h/1')
        (self.dir / 'host-stop-intent.json').write_text(json.dumps(dict(schema=1, wall_ns=_old.STOP_INTENTS['native26'] + 10 ** 12)))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(sr.main(argv), 1)
        self.assertTrue(printed.getvalue().startswith('ALPHA_NEXT_STOP_RACE CRASH '))

    def test_a_startup_read_is_not_an_other_death(self):
        lines = [at(self.INTENT - 7200 * 10 ** 9, STALL), at(self.INTENT - 3600 * 10 ** 9, HEALTH)] + list(_old.STEP_B_STOP)
        record, rc = self.classify(lines, self.INTENT)
        self.assertEqual((rc, record['stop_race'], record['engine_crashed'], record['reasons']), (0, True, False, []))
        self.assertEqual(record['death_counts'], {'engine_dead': 1})
        self.assertEqual(record['startup_broadcast_stall']['count'], 1)
        self.assertIn('GET /health', record['first_health_200']['text'])

    def test_a_post_health_stall_is_an_other_death(self):
        lines = [at(self.INTENT - 7200 * 10 ** 9, HEALTH), at(self.INTENT - 3600 * 10 ** 9, STALL)] + list(_old.STEP_B_STOP)
        record, rc = self.classify(lines, self.INTENT)
        self.assertEqual((rc, record['stop_race'], record['engine_crashed']), (1, False, True))
        self.assertIn('deaths-other-than-one-engine-dead:{"broadcast_block_stall": 1, "engine_dead": 1}', record['reasons'])
        self.assertEqual(record['startup_broadcast_stall']['count'], 0)

    def test_a_stall_with_no_health_line_is_an_other_death(self):
        lines = [at(self.INTENT - 7200 * 10 ** 9, STALL)] + list(_old.STEP_B_STOP)
        record, rc = self.classify(lines, self.INTENT)
        self.assertEqual(rc, 1)
        self.assertIn('deaths-other-than-one-engine-dead:{"broadcast_block_stall": 1, "engine_dead": 1}', record['reasons'])

    def test_the_excerpt_alone_is_not_dead(self):
        record, rc = self.classify(EXCERPT, self.INTENT)
        self.assertEqual((rc, record['scan_verdict'], record['engine_crashed'], record['reasons']), (2, 'ALIVE', False, ['not-dead']))
        self.assertEqual(record['startup_broadcast_stall']['count'], 1)


class Delta(unittest.TestCase):
    @staticmethod
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

    def test_imports_only_the_h_module(self):
        tree = ast.parse((HERE / 'alpha_next_stop_race_h.py').read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual({n for n in names if n and n.startswith(('native43', 'alpha_next'))}, {'native43_engine_death_h'})

    def test_only_classify_differs_and_the_rule_constants_are_the_frozen_ones(self):
        old, new = self.functions(HERE / 'alpha_next_stop_race.py'), self.functions(HERE / 'alpha_next_stop_race_h.py')
        self.assertEqual(set(old), set(new))
        self.assertEqual({k for k in old if old[k] != new[k]}, {'classify'})
        for name in ('WINDOW_NS', 'HANDLER_SPAN_NS', 'WORKER_FIELDS'):
            self.assertEqual(getattr(sr, name), getattr(FROZEN, name))
        for name in ('TRIGGER', 'HANDLER', 'WORKER_FAILURE', 'PROCESS'):
            self.assertEqual(getattr(sr, name).pattern, getattr(FROZEN, name).pattern)

    def test_the_frozen_module_is_byte_identical(self):
        self.assertEqual(hashlib.sha256((HERE / 'alpha_next_stop_race.py').read_bytes()).hexdigest(), FROZEN_SHA256)


if __name__ == '__main__':
    unittest.main()
