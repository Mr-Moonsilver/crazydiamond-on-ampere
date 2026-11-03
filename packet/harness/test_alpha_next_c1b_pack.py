#!/usr/bin/env python3
"""alpha-next-c1b packaging tools on the hub, offline: the parent pin, the VM carry-over gate, the swapped build_native43 assertion
(both ways), the engine comparison against c1's frozen record, the negative judge and pin writer, the re-bound queue pins writer,
the CPU probe derivation, the overlay tamper control and the pre-queue spec.

Hub only (reads alpha-next-c1's delivered build-b under tmp/alpha-next-c1 and the committed VM records; not registered in selftest.py;
run by the c1b gate chain's composition-test step). Every scratch copy is in a temporary directory; c1's records are only read.
CPU only, no network, no container, no box action.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_c1b_parent_pin as parent_pin  # noqa: E402

C1_OUT = Path('<hub-dir>/agents/flash-next/tmp/alpha-next-c1/build-b')
C1_BUNDLE = C1_OUT / 'q18a-alpha-next-c1-bundle-0c33ca841620'
C1_ENGINE = C1_OUT / 'engine-selfrun'
TOOLS = ROOT / 'tools'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    return Path(path)


def writable(root):
    for path in [root, *Path(root).rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def no_dev_env():
    env = mock.patch.dict(os.environ, {}, clear=False)
    env.start()
    os.environ.pop(parent_pin.DEV_ENV, None)
    return env


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='alpha-next-c1b-pack-'))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.addCleanup(no_dev_env().stop)

    def overlay_copy(self, name='overlay'):
        place = self.tmp / name
        shutil.copytree(C1_BUNDLE / 'native30_overlay', place)
        writable(place)
        return place


class ParentPin(Scratch):
    def test_the_committed_pin_is_c1_build_b_and_verifies(self):
        pins = parent_pin.load()
        self.assertFalse(pins.get('dev'))
        self.assertEqual(pins['manifest'][:12], '0c33ca841620')
        self.assertEqual(pins['build'], str(C1_OUT / 'build.json'))
        self.assertEqual(sha(pins['build']), pins['build_sha256'])
        self.assertEqual(sha(Path(pins['engine_selfrun']) / 'result.json'), pins['engine_selfrun_sha256'])
        for rel, digest in pins['vm_probes'].items():
            self.assertEqual(sha(ROOT / rel), digest)
        got = subprocess.run([sys.executable, '-B', str(TOOLS / 'alpha_next_c1b_parent_pin.py'), '--out', str(C1_OUT), '--steps',
                              pins['steps'], '--ready', pins['ready'], '--delivery', pins['delivery'], '--check'],
                             cwd=ROOT, capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                           if k != parent_pin.DEV_ENV})
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])

    def test_placeholder_and_misplaced_dev_pin_refuse_rc_3(self):
        placeholder = dump(self.tmp / 'p.json', dict(schema=parent_pin.SCHEMA, status='NOT-PINNED: x'))
        with self.assertRaises(SystemExit) as caught:
            parent_pin.load(placeholder)
        self.assertEqual(caught.exception.code, 3)
        dev = dump(self.tmp / 'd.json', dict(parent_pin.load(), dev=True))
        with mock.patch.object(parent_pin, 'TARGET', dev), self.assertRaises(SystemExit) as caught:
            parent_pin.load()
        self.assertEqual(caught.exception.code, 3)

    def test_steps_must_hold_rebind_and_the_stop_line(self):
        text = Path(parent_pin.load()['steps']).read_text()
        self.assertEqual(parent_pin.steps_ok(text), [])
        self.assertIn('rebind-default-off', parent_pin.steps_ok(text.replace('rebind-default-off rc=0', 'rebind-default-off rc=1')))
        self.assertIn('stop-line', parent_pin.steps_ok(text.replace('stopped-before-delivery: VM run pending', '')))


class VmCarry(Scratch):
    def setUp(self):
        super().setUp()
        import alpha_next_c1b_vm_carry as carry
        self.carry = carry

    def bundle(self):
        place = self.tmp / 'bundle'
        place.mkdir()
        shutil.copyfile(C1_BUNDLE / 'SHA256SUMS', place / 'SHA256SUMS')
        shutil.copytree(C1_BUNDLE / 'native30_overlay', place / 'native30_overlay')
        writable(place)
        return place

    def run_tool(self, bundle, *extra):
        out = self.tmp / ('carry-%d.json' % len(list(self.tmp.glob('carry-*.json'))))
        rc = self.carry.main(['--bundle', str(bundle), '--out', str(out), *extra])
        return rc, (json.loads(out.read_text()) if out.exists() else None)

    def test_c1s_own_bundle_passes_72_of_72_and_no_row_is_metadata(self):
        rc, rec = self.run_tool(C1_BUNDLE)
        self.assertEqual((rc, rec['verdict'], rec['same'], rec['rows']), (0, 'PASS', 72, 72))
        import build_native43
        paths = {r['overlay_path'] for r in rec['table']}
        self.assertFalse(paths & build_native43.METADATA)
        self.assertTrue(all(p.endswith('.py') for p in paths))
        self.assertEqual(sum(1 for r in rec['table'] if r['mount'] == 'directory'), 59)

    def test_one_tampered_module_fails_rc_1_with_the_row(self):
        bundle = self.bundle()
        target = bundle / 'native30_overlay/e3/stage_overlap.py'
        target.write_bytes(target.read_bytes() + b'\n# tamper\n')
        rc, rec = self.run_tool(bundle)
        self.assertEqual((rc, rec['verdict'], rec['same']), (1, 'FAIL', 71))
        self.assertEqual([(p['verdict'], p['overlay_path']) for p in rec['problems']], [('DIFFERS', 'e3/stage_overlap.py')])

    def test_a_sums_row_that_disagrees_fails_even_with_equal_bytes(self):
        bundle = self.bundle()
        sums = bundle / 'native30_overlay/SHA256SUMS'
        text = sums.read_text()
        row = [l for l in text.splitlines() if l.endswith('  qsa.py')][0]
        sums.write_text(text.replace(row, '0' * 64 + '  qsa.py'))
        rc, rec = self.run_tool(bundle)
        self.assertEqual((rc, [p['overlay_path'] for p in rec['problems']]), (1, ['qsa.py']))

    def test_a_missing_file_and_an_unmapped_origin_fail(self):
        bundle = self.bundle()
        (bundle / 'native30_overlay/e2_sources/block_table.py').unlink()
        launcher = bundle / 'native30_overlay/run-flash-s5-test.sh'
        launcher.write_text(launcher.read_text().replace('-v "$OVERLAY/ops_qsa.py:$D/ops/qsa.py:ro"', ''))
        rc, rec = self.run_tool(bundle)
        self.assertEqual(rc, 1)
        self.assertEqual(sorted((p['verdict'], p['origin'].rsplit('/', 2)[-1]) for p in rec['problems']),
                         [('MISSING', 'block_table.py'), ('UNMAPPED', 'qsa.py')])

    def test_probe_lists_that_differ_or_are_not_72_are_bad_input_rc_2(self):
        pins = parent_pin.load()
        plain, stage = [ROOT / rel for rel in sorted(pins['vm_probes'])]
        record = json.loads(plain.read_text())
        record['sections']['local_overlap']['mounted_origins'] = record['sections']['local_overlap']['mounted_origins'][:71]
        short = dump(self.tmp / 'short.json', record)
        rc, rec = self.run_tool(C1_BUNDLE, '--probe', str(short), '--probe', str(short))
        self.assertEqual((rc, rec), (2, None))
        rc, rec = self.run_tool(C1_BUNDLE, '--probe', str(plain), '--probe', str(short))
        self.assertEqual((rc, rec), (2, None))
        with mock.patch.dict(pins['vm_probes'], {}), mock.patch.object(self.carry, 'pinned_probes',
                                                                         lambda: [(plain, '0' * 64), (stage, None)]):
            rc, rec = self.run_tool(C1_BUNDLE)
        self.assertEqual(rc, 2)


class BuildSwap(Scratch):
    @classmethod
    def setUpClass(cls):
        cls.c1b = importlib.import_module('build_alpha_next_c1b')
        cls.b = cls.c1b.b

    def test_the_swap_is_applied_exactly_once_and_declared(self):
        old, new = self.c1b.SUPERSEDED_CHECK
        import inspect
        text = (TOOLS / 'build_native43.py').read_text()
        self.assertEqual(text.count(old), 1)
        self.assertEqual(self.c1b.BUILD_TEXT.count(new), 1)
        self.assertNotIn(old, self.c1b.BUILD_TEXT)
        self.assertEqual(self.b.build.__code__.co_filename, 'build_alpha_next_c1b:build_native43.build(swapped)')
        rec = self.c1b.superseded_record()
        self.assertEqual((rec['old'], rec['new'], rec['applied_exactly_once']), (old, new, True))
        fake = types.SimpleNamespace(build=lambda: None)
        with mock.patch('inspect.getsource', return_value='def build():\n    pass\n'), self.assertRaises(AssertionError):
            self.c1b.derive_build(fake)

    def edited(self, edit):
        place = self.overlay_copy('cand-%d' % len(list(self.tmp.glob('cand-*'))))
        edit(place)
        self.b.manifest(place)
        return self.c1b.metadata_review_only(C1_BUNDLE / 'native30_overlay', place)

    def review_edit(self, place):
        """What a re-prepared review does: closure review + its digest, and the review-bound digests downstream."""
        closure_path = place / 'e3/native-implementation-closure.json'
        closure = json.loads(closure_path.read_text())
        closure['review'] = dict(closure['review'], alpha_next_c1b=dict(test=True))
        closure['review_sha256'] = '1' * 64
        closure_path.write_text(json.dumps(closure, sort_keys=True, indent=2) + '\n')
        for rel in ('S5-OVERLAP-GENERATION.json', 'e3/native-overlap-readiness.json'):
            data = json.loads((place / rel).read_text())
            data['package_sha256'] = '2' * 64
            (place / rel).write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')

    def test_metadata_only_review_movement_is_accepted(self):
        self.assertEqual(self.edited(self.review_edit), [])
        self.assertEqual(self.edited(lambda place: None), [])

    def test_engine_row_so_launcher_added_file_and_non_review_key_are_refused(self):
        def append(rel, data=b'\n# x\n'):
            return lambda place: (self.review_edit(place), (place / rel).write_bytes((place / rel).read_bytes() + data))

        def closure_key(place):
            self.review_edit(place)
            path = place / 'e3/native-implementation-closure.json'
            data = json.loads(path.read_text())
            data['generated_source_shas']['engine_driver'] = '3' * 64
            path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')

        cases = {'e3/engine_driver.py': append('e3/engine_driver.py'),
                 'e3/native/flash_s5_green.so': append('e3/native/flash_s5_green.so', b'\0'),
                 'run-flash-s5-test.sh': append('run-flash-s5-test.sh'),
                 'core/SHA256SUMS': append('core/SHA256SUMS', b'\n'),
                 'added': lambda place: (place / 'e3/c1b_extra.py').write_text('X = 1\n'),
                 'closure-key': closure_key}
        for name, edit in cases.items():
            with self.subTest(name):
                self.assertTrue(self.edited(edit), name)

    def test_overlay_level_refuses_an_engine_row_and_accepts_metadata(self):
        spec = dict(engine_changed=[], overlay_definitions={})
        parent_rows = self.b.verify(C1_BUNDLE / 'native30_overlay')
        meta = self.overlay_copy('meta')
        self.review_edit(meta)
        self.b.manifest(meta)
        proof = self.c1b.overlay_level(spec, meta, parent_rows)
        self.assertEqual(proof['engine_changed'], [])
        self.assertEqual(proof['moved'], ['S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
                                          'e3/native-overlap-readiness.json'])
        eng = self.overlay_copy('eng')
        (eng / 'e3/qsa_operation.py').write_text((eng / 'e3/qsa_operation.py').read_text() + '\nX = 1\n')
        self.b.manifest(eng)
        self.assertEqual(self.b.refused(lambda: self.c1b.overlay_level(spec, eng, parent_rows)), 'REFUSED')

    def test_tamper_tool_over_a_metadata_only_candidate(self):
        import alpha_next_c1b_overlay_tamper as tamper
        cand = self.overlay_copy('cand')
        self.review_edit(cand)
        self.b.manifest(cand)
        result = tamper.tamper(C1_BUNDLE / 'native30_overlay', cand)
        self.assertEqual(result['verdict'], 'PASS', {k: v['got'] for k, v in result['cases'].items()})
        self.assertEqual(result['cases']['untouched']['got'], 'ACCEPTED')

    def test_load_spec_refuses_an_engine_change(self):
        spec = json.loads((TOOLS / 'native43_spec.alpha-next-c1b.json').read_text())
        spec['expected_tests'] = spec['expected_tests'] or 1
        for change in (dict(engine_changed=['e3/engine_driver.py']), dict(overlay_definitions={'e3/x.py': []}),
                       dict(candidate='q18a-alpha-next-c1')):
            path = dump(self.tmp / 'spec.json', dict(spec, **change))
            with self.subTest(change), self.assertRaises(AssertionError):
                self.c1b.load_spec(path)
        self.assertEqual(self.c1b.load_spec(dump(self.tmp / 'ok.json', spec))['candidate'], 'q18a-alpha-next-c1b')


class EngineComparison(Scratch):
    @classmethod
    def setUpClass(cls):
        cls.selfrun = importlib.import_module('alpha_next_c1b_engine_selfrun')

    def test_rebinding(self):
        s = self.selfrun
        self.assertEqual(s.run.__globals__['SCHEMA'], 'alpha-next-c1b-selfrun/1')
        self.assertEqual(s.NS['run_overlap'].__globals__['OVERLAP_SUITES'][-1], 'test_alpha_next_c1b_queue')
        self.assertIs(s.negative.__globals__['judge'], s.judge)
        self.assertIs(s.negative.__globals__['frozen'], s.frozen)
        self.assertEqual(s.frozen(), C1_ENGINE)
        self.assertEqual(len(s.expected_cases()), 65)

    def test_c1s_own_record_compares_equal(self):
        out = self.tmp / 'cmp.json'
        self.assertEqual(self.selfrun.compare(C1_ENGINE, out), 0)
        rec = json.loads(out.read_text())
        self.assertEqual((rec['compared'], rec['verdict_and_checks'], rec['differs']), (61, 4, []))

    def copy_engine(self):
        place = self.tmp / 'engine'
        shutil.copytree(C1_ENGINE, place, ignore=lambda d, names: [n for n in names if stat.S_ISFIFO(os.lstat(os.path.join(d, n)).st_mode)])
        writable(place)
        return place

    def test_one_changed_trace_fails(self):
        place = self.copy_engine()
        path = place / 'overlap/alpha-c1-overlap-burst-slow1.json'
        case = json.loads(path.read_text())
        case['request_tokens'] = (case.get('request_tokens') or 0) + 1 if not isinstance(case.get('request_tokens'), list) \
            else case['request_tokens'] + [1]
        path.write_text(json.dumps(case))
        self.assertEqual(self.selfrun.compare(place, self.tmp / 'cmp.json'), 1)
        self.assertEqual(json.loads((self.tmp / 'cmp.json').read_text())['differs'], ['alpha-c1-overlap-burst-slow1'])

    def test_a_receipts_case_with_another_check_count_or_a_missing_case_fails(self):
        place = self.copy_engine()
        record = json.loads((place / 'result.json').read_text())
        record['cases']['receipts-off']['checks'] += 1
        del record['cases']['tp-skew']
        (place / 'result.json').write_text(json.dumps(record))
        self.assertEqual(self.selfrun.compare(place, self.tmp / 'cmp.json'), 1)
        rec = json.loads((self.tmp / 'cmp.json').read_text())
        self.assertIn('receipts-off', rec['differs'])
        self.assertIn('case-set-differs', rec['differs'])

    def test_judge_lifts_suites_not_run_only_for_an_explicit_empty_pin_with_outcomes(self):
        s = self.selfrun
        cases = {m: json.loads((C1_ENGINE / s.part_of(m) / (m + '.json')).read_text()) for m in s.CASES}
        pins = json.loads(s.PINS.read_text())
        base = dict(pins, tests={})
        outcomes = {'t.a': ['ok', '']}
        parent = set(s.PARENT_PASS)

        def reasons(p, o, c=cases):   # the parent modes are not in this case set: their reasons are not under test here
            return [r for r in s.judge(p, o, c, C1_ENGINE) if not r.startswith('parent-case-not-pass:')
                    and not (r.startswith('negative-trace-differs-from-frozen:') and r.split(':', 1)[1] in parent)]

        self.assertEqual(reasons(base, outcomes), ['negative-suites-not-run'])
        self.assertEqual(reasons(dict(base, no_failing_tests=True), outcomes), [])
        self.assertEqual(reasons(dict(base, no_failing_tests=True), {}), ['negative-suites-not-run'])
        mode = list(s.CASES)[0]
        bent = dict(cases, **{mode: dict(cases[mode], trace=['moved'])})
        self.assertEqual(reasons(dict(base, no_failing_tests=True), outcomes, bent), ['negative-trace-differs-from-frozen:' + mode])


class NegativePins(Scratch):
    def setUp(self):
        super().setUp()
        import alpha_next_c1b_negative_pins_write as writer
        import alpha_next_c1b_engine_selfrun as selfrun
        self.writer, self.selfrun = writer, selfrun

    def test_the_stub(self):
        stub = json.loads(self.writer.OUT.read_text())
        self.assertTrue(stub['status'].startswith('AWAITING-OBSERVATION'))
        self.assertEqual(stub['tests'], {})
        self.assertEqual(stub['cases'], {m: ['PASS', None, None] for m in self.selfrun.CASES})
        self.assertEqual(stub['frozen_record_sha256'], parent_pin.load()['engine_selfrun_sha256'])

    def observation(self, failing_case=None, observed=None):
        place = self.tmp / 'obs'
        place.mkdir()
        cases = {}
        for mode in list(self.selfrun.PARENT_PASS) + list(self.selfrun.CASES):
            raw = dict(verdict='PASS', error_type=None, error=None)
            if mode == failing_case:
                raw = dict(verdict='FAIL', error_type='ImportError', error='x')
            dump(place / (mode + '.json'), raw)
            cases[mode] = dict(verdict=raw['verdict'])
        rec = dict(schema='alpha-next-c1b-negative/1', verdict='OBSERVED', negative_control=True, pins_sha256=None, pinned_tests=0,
                   suites=dict(skips=0, tests=100), suite_names=list(self.selfrun.NEGATIVE_SUITES), reasons=[],
                   cases=cases, observed=observed or {}, frozen_record_sha256=parent_pin.load()['engine_selfrun_sha256'],
                   overlay_manifest='o', manifest='m', container='c')
        return dump(place / 'result.json', rec), dump(self.tmp / 'log.txt', 'log')

    def test_zero_non_passes_pin_no_failing_tests(self):
        obs, log = self.observation()
        got = self.writer.pins(json.loads(self.writer.OUT.read_text()), obs, log, None, 'test')
        self.assertEqual((got['status'], got['tests'], got['no_failing_tests']), ('OBSERVED', {}, True))
        self.assertEqual(got['cases'], {m: ['PASS', None, None] for m in self.selfrun.CASES})

    def test_a_failing_probe_case_or_an_unexplained_non_pass_is_refused(self):
        obs, log = self.observation(failing_case=list(self.selfrun.CASES)[0])
        with self.assertRaises(AssertionError):
            self.writer.pins(json.loads(self.writer.OUT.read_text()), obs, log, None, 'test')
        shutil.rmtree(self.tmp / 'obs')
        obs, log = self.observation(observed={'t.x': ['error', 'boom']})
        with self.assertRaises(AssertionError):
            self.writer.pins(json.loads(self.writer.OUT.read_text()), obs, log, None, 'test')
        rows = dump(self.tmp / 'rows.json', [['t.', 'boom', 'c1b harness reads its own review', 'HARNESS']])
        got = self.writer.pins(json.loads(self.writer.OUT.read_text()), obs, log, rows, 'test')
        self.assertEqual((got['tests'], got['no_failing_tests']), ({'t.x': ['error', 'boom']}, False))


class Rebound(Scratch):
    def test_queue_pins_writer_is_c1s_rebound_and_refuses_c1s_out(self):
        import alpha_next_c1b_queue_pins_write as w
        import alpha_next_c1_queue_pins_write as c1
        self.assertEqual(w.from_out.__globals__['PINS'], TOOLS / 'alpha_next_c1b_queue_pins.json')
        self.assertEqual(c1.PINS, TOOLS / 'alpha_next_c1_queue_pins.json')
        self.assertTrue(w.placeholders(json.loads((TOOLS / 'alpha_next_c1b_queue_pins.json').read_text())))
        with self.assertRaises(AssertionError) as caught:
            w.from_out(C1_OUT)
        self.assertIn('alpha-next-c1b-freeze.json', str(caught.exception))

    def test_precheck_conf_and_clock_scan_rebinding(self):
        import alpha_next_c1b_precheck_conf as conf
        import alpha_next_c1b_clock_scan as scan
        self.assertEqual(conf.generate.__globals__['STEM'], 'q18a_alpha_next_c1b')
        self.assertEqual(conf.generate.__globals__['ADDED'], set())
        self.assertEqual(conf.PARENT_BUILD, C1_OUT / 'build.json')
        self.assertEqual(scan.ARM_MODULES[:2], ('e3_alpha_next_c1b_box', 'e3_alpha_next_c1_box'))
        self.assertEqual(scan.step_a.TEXTS, scan.TEXTS)

    def test_review_pins_add_nothing(self):
        import alpha_next_c1b_review_pins as pins
        before, after = pins.refreshed()
        self.assertEqual(set(before['source_shas']), set(after['source_shas']))
        self.assertEqual(pins.base.ADDED, {'stage_overlap.py'})      # c1's module is left as it was

    def test_cpu_probe_derivation_with_stub_counts(self):
        import alpha_next_c1b_tools_derive as derive
        self.assertEqual(derive.parent_tests(), 519)
        text = derive.generate(parent=519, ours=7)['alpha_next_c1b_cpu_probe.py']
        compile(text, 'probe', 'exec')
        self.assertIn('EXPECTED_TESTS=526  # the 519 alpha-next-c1 probe cases plus the 7 alpha-next-c1b queue cases', text)
        for needle in ("owner.QNAME=='q18a-alpha-next-c1b'", 'alpha_next_c1b_gate.ACTIVE_S+alpha_next_c1b_gate.CLEANUP_S',
                       "pattern='test_alpha_next_c1b_queue.py'", 'q18a_alpha_next_c1b_settings.sh', "alpha_next_c1b_queue='PASS'",
                       "alpha_next_c1_queue='PASS'", 'assert c1b_slots==[]'):
            self.assertEqual(text.count(needle), 1, needle)
        if (ROOT / 'harness/test_alpha_next_c1b_queue.py').is_file() and (TOOLS / 'alpha_next_c1b_cpu_probe.py').is_file():
            self.assertEqual((TOOLS / 'alpha_next_c1b_cpu_probe.py').read_text(), derive.generate()['alpha_next_c1b_cpu_probe.py'])


class Spec(unittest.TestCase):
    def test_committed_spec_is_harness_only_and_its_runner_swaps_are_c1s_rows(self):
        spec = json.loads((TOOLS / 'native43_spec.alpha-next-c1b.json').read_text())
        self.assertEqual((spec['candidate'], spec['label'], spec['review_key'], spec['review']),
                         ('q18a-alpha-next-c1b', 'alpha-next-c1b', 'alpha_next_c1b', 'prepare'))
        self.assertEqual((spec['engine_changed'], spec['overlay_definitions']), ([], {}))
        self.assertEqual(spec['queue'], ['q18a_alpha_next_c1b.draft.sh', 'q18a_alpha_next_c1b_declare.sh',
                                         'q18a_alpha_next_c1b_settings.sh'])
        self.assertFalse([r for r in spec['source_paths'] if r.split('\t', 1)[1].startswith('patches/s5_engine/')
                          and r != 'M\tpatches/s5_engine/native_candidate_review.json'])
        c1 = json.loads((TOOLS / 'native43_spec.alpha-next-c1.json').read_text())
        olds = [row[0] for row in spec['runner_swaps']]
        self.assertEqual(olds, [c1['runner_swaps'][0][1], c1['runner_swaps'][1][1]])
        self.assertEqual(spec['runner_swaps'][0][1], "engine.get('schema')=='alpha-next-c1b-selfrun/1'")
        self.assertTrue(spec['runner_swaps'][1][1].endswith(" and result.get('alpha_next_c1_queue')=='PASS' and "
                                                            "result.get('alpha_next_c1b_queue')=='PASS'"))
        self.assertNotIn('tools/alpha_next_c1b_queue_pins.json', ['tools/' + t for t in spec['tools']])


if __name__ == '__main__':
    unittest.main()
