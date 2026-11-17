"""Health-scoped engine-death scan and watch (native43_engine_death_h.py) over the verbatim excerpt of the voided q18a-append-yield.

The one rule change: a broadcast-block stall before the first APIServer health 200 is a startup read.  Must be ALIVE: the excerpt.
Must be DEAD: the stall moved after the health line, no health line, a pre-health teardown, the l-e fixture.  Every test of the
frozen module's own file also runs against this module (Inherited*).  Fixture: harness/fixtures/native43_engine_death_h.
Leaves: native43_engine_death_h_fixture_leaves.json.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native43_engine_death as frozen  # noqa: E402
import native43_engine_death_h as death  # noqa: E402

FIX = HERE / 'fixtures/native43_engine_death_h'
TOOL = HERE / 'native43_engine_death_h.py'
FROZEN_SHA256 = 'f44373419825530def90b0ca0aea8f14545b5be0a0b62a317279d49820c73a15'
STALL_WALL = '2026-11-03T15:06:22.537733245Z'
HEALTH_WALL = '2026-11-03T15:06:54.259323166Z'
TEARDOWN = b'(Worker_PP0_TP0_EP0 pid=113) INFO 11-03 15:06:30 [multiproc.py:1] [shutdown] Executor: shutdown start'
_spec = importlib.util.spec_from_file_location('test_native43_engine_death_for_h', HERE / 'test_native43_engine_death.py')
_old = importlib.util.module_from_spec(_spec)  # a private copy: the frozen module's own test run keeps its globals
_spec.loader.exec_module(_old)
_old.death, _old.TOOL = death, TOOL


def excerpt():
    return (FIX / 'startup-stall-engine-stream.log').read_bytes().split(b'\n')[:-1]


def find(lines, needle):
    return next(i for i, line in enumerate(lines) if needle in line)


def stamped(ns, body):
    return death.iso(ns).encode() + b' ' + body


def body(line):
    return line.split(b' ', 1)[1]


def stall_after_health():
    """The excerpt with the stall line moved to 1 s after the first health 200 (stamp and place)."""
    lines = excerpt()
    stall = lines.pop(find(lines, b'No available shared memory'))
    at = find(lines, b'GET /health')
    lines.insert(at + 1, stamped(death.wall_ns(lines[at]) + 10 ** 9, body(stall)))
    return lines


def teardown_before_health(lines=None):
    lines = excerpt() if lines is None else lines
    at = find(lines, b'No available shared memory')
    lines.insert(at + 1, stamped(death.wall_ns(lines[at]) + 10 ** 9, TEARDOWN))
    return lines


class Fixture(unittest.TestCase):
    def test_excerpt_matches_its_provenance(self):
        prov = json.loads((FIX / 'PROVENANCE.json').read_text())
        self.assertEqual(prov['schema'], 'native43-engine-death-h-fixtures/1')
        self.assertEqual(sorted(p.name for p in FIX.iterdir() if p.name != 'PROVENANCE.json'),
                         sorted(row['fixture'] for row in prov['fixtures']))
        for row in prov['fixtures']:
            self.assertEqual(hashlib.sha256((FIX / row['fixture']).read_bytes()).hexdigest(), row['fixture_sha256'])
            source = Path(prov['root']) / row['source']
            if source.is_file():  # the hub keeps the fetch; a copy elsewhere still checks the excerpt hash
                data = source.read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), row['source_sha256'])
                lines = data.split(b'\n')
                piece = b''.join(b'\n'.join(lines[a - 1:b]) + b'\n' for a, b in row['segments'])
                self.assertEqual(piece, (FIX / row['fixture']).read_bytes())
                self.assertIn(b'No available shared memory', lines[row['stall_source_line'] - 1])
                self.assertEqual([i for i, x in enumerate(lines, 1) if death.HEALTH_200.match(x)][0],
                                 row['first_health_200_source_line'])

    def test_excerpt_holds_engine_and_api_lines_only(self):
        for line in excerpt():
            self.assertRegex(line, rb'^\S+ \((APIServer|EngineCore|Worker_PP\d_TP\d_EP\d) pid=\d+\) INFO')


class Delta(unittest.TestCase):
    """Self-contained, and the frozen module's functions stay as they were except the named ones."""

    @staticmethod
    def functions(path):
        tree = ast.parse(path.read_text())
        rows = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                rows.update({'%s.%s' % (node.name, m.name): ast.dump(m) for m in node.body if isinstance(m, ast.FunctionDef)})
        return rows

    def test_no_import_of_the_frozen_module(self):
        tree = ast.parse(TOOL.read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse({n for n in names if n and n.startswith('native43')}, names)

    def test_only_the_named_functions_differ(self):
        old, new = self.functions(HERE / 'native43_engine_death.py'), self.functions(TOOL)
        self.assertEqual(set(old), set(new))
        self.assertEqual({k for k in old if old[k] != new[k]},
                         {'Scanner.__init__', 'Scanner._line', 'Scanner.decide', 'scan', 'watch'})

    def test_the_death_tables_are_the_frozen_tables(self):
        for name in ('HARD', 'UNPLANNED', 'ORDERLY'):
            self.assertEqual([(k, r.pattern) for k, r in getattr(death, name)], [(k, r.pattern) for k, r in getattr(frozen, name)])

    def test_the_frozen_module_is_byte_identical(self):
        self.assertEqual(hashlib.sha256((HERE / 'native43_engine_death.py').read_bytes()).hexdigest(), FROZEN_SHA256)


class Scan(_old.Root):
    def verdict(self, lines, intent=None):
        return self.scan([self.write('engine.log', lines)], intent)

    def assert_dead(self, lines, kind, count=1, startup=None):
        rc, rec, out = self.verdict(lines)
        self.assertEqual((rc, rec['verdict'], rec['death_counts'].get(kind)), (1, 'DEAD', count), out)
        if startup is not None:
            self.assertEqual(rec['startup_broadcast_stall']['count'], startup)
        return rec

    def test_todays_excerpt_is_alive_with_one_startup_read(self):
        rc, rec, out = self.verdict(excerpt())
        self.assertEqual((rc, rec['verdict'], rec['death_counts'], rec['first_death']), (0, 'ALIVE', {}, None))
        self.assertEqual(rec['schema'], 'native43-engine-death-scan-h/1')
        read = rec['startup_broadcast_stall']
        self.assertEqual((read['count'], read['first']['wall'], read['first']['line'], read['first']['kind']),
                         (1, STALL_WALL, 4, 'broadcast_block_stall'))
        self.assertIn('No available shared memory', read['first']['text'])
        self.assertEqual((rec['first_health_200']['wall'], rec['first_health_200']['line']), (HEALTH_WALL, 12))
        self.assertIn('GET /health HTTP/1.1" 200', rec['first_health_200']['text'])
        self.assertEqual(rec['rules']['startup_read'], death.STARTUP_RULE)
        self.assertEqual(out.strip(), 'ENGINE_DEATH_SCAN ALIVE startup_broadcast_stall=1')

    def test_the_frozen_module_calls_the_same_excerpt_dead(self):
        scanner = frozen.Scanner([self.write('engine.log', excerpt())])
        scanner.poll(final=True)
        self.assertEqual(scanner.decide()['death_counts'], {'broadcast_block_stall': 1})

    def test_the_whole_box_record_when_the_hub_holds_it(self):
        prov = json.loads((FIX / 'PROVENANCE.json').read_text())
        row = prov['fixtures'][0]
        log = Path(prov['root']) / row['source']
        if not log.is_file():  # a copy elsewhere still has every excerpt case
            return
        rc, rec, _ = self.scan([log, log.with_name('engine-stream.err')], log.with_name('host-stop-intent.json'))
        self.assertEqual((rc, rec['verdict'], rec['startup_broadcast_stall']['count']), (0, 'ALIVE', 1))
        self.assertEqual(rec['startup_broadcast_stall']['first']['line'], row['stall_source_line'])
        self.assertEqual(rec['first_health_200']['line'], row['first_health_200_source_line'])

    def test_the_stall_moved_after_the_first_health_200_is_dead(self):
        rec = self.assert_dead(stall_after_health(), 'broadcast_block_stall', startup=0)
        self.assertEqual(rec['first_death']['why'], 'no orderly shutdown trigger in the log and no stop intent')
        self.assertEqual(rec['first_health_200']['wall'], HEALTH_WALL)

    def test_a_stall_at_the_very_stamp_of_the_health_line_is_dead(self):
        lines = excerpt()
        at = find(lines, b'No available shared memory')
        lines[at] = stamped(death.wall_ns(lines[find(lines, b'GET /health')]), body(lines[at]))
        self.assert_dead(lines, 'broadcast_block_stall', startup=0)

    def test_a_pre_health_stall_with_no_health_line_anywhere_is_dead(self):
        lines = [x for x in excerpt() if b'GET /health' not in x]
        rec = self.assert_dead(lines, 'broadcast_block_stall', startup=0)
        self.assertIsNone(rec['first_health_200'])

    def test_a_pre_health_executor_teardown_is_dead(self):
        rec = self.assert_dead(teardown_before_health(), 'executor_teardown', startup=1)
        self.assertEqual(rec['death_counts'], {'executor_teardown': 1})

    def test_the_other_teardowns_and_a_hard_marker_before_health_are_dead(self):
        for kind, text in (('worker_parent_exited', b'(Worker_PP0_TP0_EP0 pid=113) INFO Parent process exited, terminating worker'),
                           ('worker_shutdown', b'(Worker_PP0_TP0_EP0 pid=113) INFO [shutdown] WorkerProc: start'),
                           ('nccl_dist_backend', b'(Worker_PP0_TP0_EP0 pid=113) ERROR DistBackendError: x')):
            with self.subTest(kind=kind):
                lines = excerpt()
                at = find(lines, b'No available shared memory')
                lines.insert(at, stamped(death.wall_ns(lines[at]) - 10 ** 9, text))
                self.assertEqual(self.assert_dead(lines, kind)['startup_broadcast_stall']['count'], 1)

    def test_the_le_fixture_unchanged_is_dead_as_the_frozen_module_says(self):
        logs = [_old.FIX / 'le-engine-stream.log', _old.FIX / 'le-engine-stream.err']
        intent = _old.FIX / 'le-host-stop-intent.json'
        rc, rec, _ = self.scan(logs, intent)
        scanner = frozen.Scanner([str(p) for p in logs], str(intent))
        scanner.poll(final=True)
        old = scanner.decide()
        self.assertEqual((rc, rec['verdict']), (1, 'DEAD'))
        self.assertEqual(rec['death_counts'], old['death_counts'])
        self.assertEqual(rec['first_death']['wall'], old['first_death']['wall'])
        self.assertEqual(rec['first_death']['wall'], _old.LE_SHUTDOWN)
        self.assertEqual(rec['startup_broadcast_stall']['count'], 0)

    def test_health_lines_that_do_not_count(self):
        health = excerpt()[find(excerpt(), b'GET /health')]
        for name, line in (('503', health.replace(b'" 200 OK', b'" 503 Service Unavailable')),
                           ('not-the-api-server', health.replace(b'APIServer', b'EngineCore')),
                           ('inside-other-text', health.replace(b'INFO:     127.0.0.1:59692 - ', b'INFO 11-03 [x.py:1] said ')),
                           ('other-path', health.replace(b'GET /health', b'GET /healthz')),
                           ('unstamped', body(health))):
            with self.subTest(line=name):
                lines = excerpt()
                lines[find(lines, b'GET /health')] = line
                self.assert_dead(lines, 'broadcast_block_stall', startup=0)

    def test_an_unstamped_stall_stays_a_death(self):
        lines = excerpt()
        at = find(lines, b'No available shared memory')
        lines[at] = body(lines[at])
        self.assert_dead(lines, 'broadcast_block_stall', startup=0)

    def test_a_restart_inside_the_logs_stays_dead(self):
        # A second API server's startup stall sits after the first incarnation's health 200, even before its own.
        lines = excerpt()
        health = lines[find(lines, b'GET /health')]
        stall = lines[find(lines, b'No available shared memory')]
        t = death.wall_ns(health)
        lines += [stamped(t + 60 * 10 ** 9, body(stall).replace(b'pid=103', b'pid=203')),
                  stamped(t + 90 * 10 ** 9, body(health).replace(b'pid=51', b'pid=61'))]
        self.assert_dead(lines, 'broadcast_block_stall', startup=1)

    def test_the_stall_and_the_health_line_may_sit_in_different_streams(self):
        lines = excerpt()
        at = find(lines, b'No available shared memory')
        stall = lines.pop(at)
        rc, rec, _ = self.scan([self.write('engine.log', lines), self.write('engine.err', [stall])])
        self.assertEqual((rc, rec['startup_broadcast_stall']['count']), (0, 1))
        # Past the health line across streams it is a death, whichever file holds it.
        late = stamped(death.wall_ns(lines[find(lines, b'GET /health')]) + 1, body(stall))
        rc, rec, _ = self.scan([self.write('engine.log', lines), self.write('engine.err', [late])])
        self.assertEqual((rc, rec['death_counts']), (1, {'broadcast_block_stall': 1}))

    def test_several_startup_reads_are_counted_with_the_first(self):
        lines = excerpt()
        at = find(lines, b'No available shared memory')
        t = death.wall_ns(lines[at])
        lines.insert(at + 1, stamped(t + 60 * 10 ** 9 // 2, body(lines[at])))
        rc, rec, _ = self.verdict(lines)
        self.assertEqual((rc, rec['startup_broadcast_stall']['count'], rec['startup_broadcast_stall']['first']['wall']), (0, 2, STALL_WALL))

    def test_a_stop_intent_does_not_change_the_read(self):
        intent = self.root / 'intent.json'
        intent.write_text(json.dumps(dict(wall_ns=death.wall_ns(excerpt()[-1]) + 10 ** 9)))
        self.assertEqual(self.verdict(excerpt(), intent)[0], 0)
        self.assertEqual(self.verdict(stall_after_health(), intent)[0], 1)

    def test_incremental_reads_decide_like_one_read(self):
        data = b''.join(x + b'\n' for x in excerpt())
        path = self.root / 'engine.log'
        scanner = death.Scanner([path])
        path.write_bytes(b'')
        for i in range(0, len(data), 97):
            with path.open('ab') as out:
                out.write(data[i:i + 97])
            scanner.poll()
        scanner.poll(final=True)
        rec = scanner.decide()
        self.assertEqual((rec['verdict'], rec['startup_broadcast_stall']['count'], rec['first_health_200']['wall']), ('ALIVE', 1, HEALTH_WALL))


class WatchPair(_old.Watch):
    """The watch over the real excerpt: a pre-health stall leaves the client alone, a post-health stall stops it.  The inherited
    frozen tests run here too (over this module)."""

    def test_a_pre_health_stall_does_not_stop_the_client(self):
        log = self.write('engine.log', excerpt())
        rc, rec, _, _, _ = self.run_watch(
            [sys.executable, '-c', 'import sys,time; time.sleep(1.5); sys.exit(7)'], logs=[log])
        self.assertEqual((rc, rec['verdict'], rec['client_rc']), (7, 'CLIENT_EXIT', 7))
        self.assertGreater(rec['polls'], 1)
        self.assertEqual(rec['schema'], 'native43-engine-death-watch-h/1')
        self.assertEqual(rec['startup_broadcast_stall']['count'], 1)
        self.assertEqual(rec['first_health_200']['wall'], HEALTH_WALL)
        self.assertEqual(rec['startup_read'], death.STARTUP_RULE)

    def test_a_post_health_stall_stops_the_client_within_seconds(self):
        log = self.root / 'engine.log'
        lines = excerpt()
        stall = lines.pop(find(lines, b'No available shared memory'))
        log.write_bytes(b''.join(x + b'\n' for x in lines))
        marker = self.root / 'client-started'
        late = stamped(death.wall_ns(lines[-1]) + 10 ** 9, body(stall))

        def feed():
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(1.0)
            with log.open('ab') as handle:
                handle.write(late + b'\n')
        child = [sys.executable, '-c', 'import pathlib,time,sys; pathlib.Path(sys.argv[1]).touch(); time.sleep(600)', str(marker)]
        rc, rec, _, err, wall = self.run_watch(child, logs=[log], feeder=feed)
        self.assertEqual((rc, rec['verdict'], rec['dead']['reason']), (1, 'DEAD', 'engine-log'))
        self.assertEqual(rec['dead']['death_counts'], {'broadcast_block_stall': 1})
        self.assertLess(wall, 30)
        self.assertIn('ENGINE_DEATH_WATCH DEAD reason=engine-log', err)

    def test_a_stall_with_no_health_line_yet_stops_the_client(self):
        # Fail-closed.  The queue calls the watch only after its startup health wait, so it does not meet this case on a good start.
        log = self.write('engine.log', [x for x in excerpt() if b'GET /health' not in x])
        rc, rec, _, _, _ = self.run_watch([sys.executable, '-c', 'import time; time.sleep(600)'], logs=[log])
        self.assertEqual((rc, rec['verdict'], rec['dead']['death_counts']), (1, 'DEAD', {'broadcast_block_stall': 1}))


class InheritedScan(_old.Scan):
    """Every frozen scan test over this module."""


if __name__ == '__main__':
    unittest.main()
