#!/usr/bin/env python3
"""alpha-next-c1 packaging tools on the hub: parent pin refusals, the re-pointed freeze, the composition proof and the delivery tool end
to end on a dev build record shape (C1's lesson: its delivery raised KeyError on the nested freeze record only on its first real run).

Hub only, like test_alpha_next_composition (git and the hub's alpha-next dev build; not registered in selftest.py, run by the gate
chain's composition-test step). The development parent is alpha-next's build-g1 (tmp/alpha-next/build-g1, pre-VM, no engine self-run),
pinned with alpha_next_c1_parent_pin.pins(dev=True) into a temporary file read through ALPHA_NEXT_C1_PARENT; the committed parent file
is never written. CPU only, no network, no box action.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_c1_parent_pin as parent_pin  # noqa: E402

DEV_OUT = Path('<hub-dir>/agents/flash-next/tmp/alpha-next/build-g1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    return Path(path)


class DevParent:
    """A development pin in a temporary directory, read through ALPHA_NEXT_C1_PARENT for the duration of a test class."""

    @classmethod
    def setUpClass(cls):
        assert (DEV_OUT / 'build.json').is_file(), 'the hub dev parent (alpha-next build-g1) is missing'
        cls.tmp = Path(tempfile.mkdtemp(prefix='alpha-next-c1-pack-'))
        cls.dev_pin = dump(cls.tmp / 'dev-parent.json', parent_pin.pins(DEV_OUT, dev=True))
        cls.env = mock.patch.dict(os.environ, {parent_pin.DEV_ENV: str(cls.dev_pin)})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp)

    @staticmethod
    def fresh(name):
        """Import a tool fresh, so it reads the dev pin set for this class."""
        sys.modules.pop(name, None)
        return importlib.import_module(name)


class ParentPin(unittest.TestCase):
    def test_the_committed_placeholder_refuses_with_rc_3(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(parent_pin.DEV_ENV, None)
            if json.loads(parent_pin.TARGET.read_text()).get('status', '').startswith(parent_pin.PLACEHOLDER):
                with self.assertRaises(SystemExit) as caught:
                    parent_pin.load()
                self.assertEqual(caught.exception.code, 3)
            else:
                self.assertFalse(parent_pin.load().get('dev'), 'a development pin is committed')

    def test_a_dev_pin_is_refused_outside_the_dev_route(self):
        with tempfile.TemporaryDirectory() as held:
            path = dump(Path(held) / 'p.json', dict(schema=parent_pin.SCHEMA, status='DEV: x', dev=True))
            with mock.patch.object(parent_pin, 'TARGET', path), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(parent_pin.DEV_ENV, None)
                with self.assertRaises(SystemExit) as caught:
                    parent_pin.load()
                self.assertEqual(caught.exception.code, 3)
            with mock.patch.dict(os.environ, {parent_pin.DEV_ENV: str(path)}):
                self.assertTrue(parent_pin.load()['dev'])

    def test_steps_need_every_step_and_the_stop_line(self):
        rows = ['10:00:00Z %s rc=0' % s for s in parent_pin.STEPS_REQUIRED + ('strict', 'focused-2', 'engine-selfrun')]
        full = '\n'.join(rows + [parent_pin.STOPPED]) + '\n'
        self.assertEqual(parent_pin.steps_ok(full), [])
        self.assertEqual(parent_pin.steps_ok(full.replace(parent_pin.STOPPED, '')), ['stop-line'])
        self.assertIn('precheck', parent_pin.steps_ok(full.replace(' precheck rc=0', ' precheck rc=1')))
        self.assertIn('focused', parent_pin.steps_ok(full.replace(' focused-2 rc=0', ' focused-2 rc=1')))


class Freeze(DevParent, unittest.TestCase):
    def test_the_composed_rows_are_c1s_rows_rebased_once(self):
        freeze = self.fresh('alpha_next_c1_freeze')
        import stage_default_off_c1 as off
        self.assertEqual(set(freeze.COMPOSED_PROJECTIONS), set(off.PROJECTIONS))
        for name, rows in off.PROJECTIONS.items():
            if name not in freeze.COMPOSED_SWAPS:
                self.assertEqual([tuple(r) for r in rows], freeze.COMPOSED_PROJECTIONS[name])
        self.assertIn("'handler_anatomy', 'stage_overlap')\n", freeze.COMPOSED_PROJECTIONS['decode_probe.py'][0][0])
        with self.assertRaisesRegex(AssertionError, 'exactly once'):
            freeze.composed_projections({'decode_probe.py': [('x', 'y')]})

    def test_the_tree_proves_against_the_dev_parent_and_an_undeclared_byte_refuses(self):
        freeze = self.fresh('alpha_next_c1_freeze')
        with tempfile.TemporaryDirectory() as held:
            receipt = freeze.prove(ROOT / freeze.ENGINE, Path(held) / 'work')
            self.assertEqual(receipt['verdict'], 'PASS', receipt['faults'])
            self.assertTrue(receipt['both_off_reaches_alpha_next_freeze']['projected_byte_identical'])
            engine = Path(held) / 'engine'
            shutil.copytree(ROOT / freeze.ENGINE, engine)
            with (engine / 'engine_driver.py').open('a') as handle:
                handle.write('\n# undeclared\n')
            bad = freeze.prove(engine, Path(held) / 'work-bad')
            self.assertEqual(bad['verdict'], 'FAIL')
            self.assertTrue(any('engine_driver.py' in f for f in bad['faults']), bad['faults'])

    def test_alpha_next_freeze_proving_another_commit_refuses(self):
        freeze = self.fresh('alpha_next_c1_freeze')
        with mock.patch.dict(freeze.PIN, freeze_sha256='0' * 64):
            with tempfile.TemporaryDirectory() as held:
                bad = freeze.prove(ROOT / freeze.ENGINE, Path(held) / 'work')
        self.assertIn('both-off: alpha-next freeze record changed', bad['faults'])


class Composition(DevParent, unittest.TestCase):
    def test_the_chain_proves_with_the_committed_declaration(self):
        comp = self.fresh('alpha_next_c1_composition')
        record = comp.prove(parent_pin.load()['source'])
        self.assertEqual(record['verdict'], 'PASS')
        self.assertIn(record['c1_merge'], record['merges'])

    def test_the_committed_declaration_is_the_draft_for_its_merges(self):
        comp = self.fresh('alpha_next_c1_composition')
        committed = json.loads(comp.DECLARATION.read_text())
        self.assertEqual(committed['blocks'], [b for b in comp.draft()['blocks'] if b['merge'] in {x['merge'] for x in committed['blocks']}])

    def test_a_changed_resolution_refuses(self):
        comp = self.fresh('alpha_next_c1_composition')
        bad = json.loads(comp.DECLARATION.read_text())
        bad['blocks'][0]['text'] += '# extra\n'
        with tempfile.TemporaryDirectory() as held:
            path = dump(Path(held) / 'd.json', bad)
            with mock.patch.object(comp, 'DECLARATION', path), self.assertRaisesRegex(AssertionError, 'undeclared resolution'):
                comp.prove()


class Delivery(DevParent, unittest.TestCase):
    """deliver() end to end over a synthetic OUT in the shapes the chain's tools write; the freeze source is the real prove() output."""

    def out_dir(self, held):
        freeze_tool = self.fresh('alpha_next_c1_freeze')
        pin = parent_pin.load()
        pin = dict(pin, engine_selfrun=str(held / 'alpha-next-engine'), engine_selfrun_sha256=None)
        out = held / 'out'
        bundle = out / 'q18a-alpha-next-c1-bundle-aaaaaaaaaaaa'
        dump(bundle / 'x.json', {})
        (bundle / 'SHA256SUMS').write_text('0 x.json\n')
        manifest = sha(bundle / 'SHA256SUMS')
        runner = out / ('q18a-alpha-next-c1-owner-run-%s.py' % manifest[:12])
        runner.write_text('# runner\n')
        proof = dump(out / 'freeze-proof.json', dict(verdict='PASS'))
        overlay_manifest = 'b' * 64
        build = dump(out / 'build.json', dict(
            schema='native43-build/1', name='q18a-alpha-next-c1', parent_manifest=pin['manifest'], manifest=manifest,
            overlay_manifest=overlay_manifest, bundle=str(bundle), runner=str(runner), runner_sha256=sha(runner), runner_size=9,
            review_sha256='c' * 64, source_commit='d' * 40, source='e' * 40, parent_bundle='/p', freeze_proof_sha256=sha(proof),
            engine_changed=['e3/s5_startup.py'], regression={}, focused={}, expected_selfrun_tests=460))
        source = freeze_tool.prove(ROOT / freeze_tool.ENGINE, held / 'work')
        freeze = dump(out / 'alpha-next-c1-freeze.json', dict(
            schema='alpha-next-c1-freeze/1', verdict='PASS', source=source,
            negative_controls={'undeclared-module': 'REFUSED'}, tools_replaced=dict(declared=[], replaced=[]),
            overlay={}, reference=dict(engine_commit=pin['source'], overlay_byte_identical_to_parent=True),
            candidate_manifest=manifest, candidate_overlay_manifest=overlay_manifest, parent_manifest=pin['manifest'],
            parent_pins_sha256='f' * 64, dev_parent=True, gpu_execution=False, box_action=False))
        engine_dir = out / 'engine-selfrun'
        log = engine_dir / 'dense/engine/suites.log'
        log.parent.mkdir(parents=True)
        log.write_text('Ran 1 test\n')
        dump(engine_dir / 'dense/result.json', dict(suites=dict(log_sha256=sha(log))))
        case = dict(verdict='PASS', checks=3)
        engine = dump(engine_dir / 'result.json', dict(
            schema='alpha-next-c1-selfrun/1', verdict='PASS', negative_control=False, manifest=manifest, overlay_manifest=overlay_manifest,
            suites=dict(ok=True), skips=0, tests=10, cases={'alpha-c1-overlap-burst-slow1': case}, connected_checks=3,
            pairs={'alpha-c1-overlap-burst': dict(same=True)}, c1_differs=[], early_open=dict(ok=True, early_opens=2),
            dense_left_out=dict(ok=True, suites_log_sha256=sha(log)), composed_left_out={'x': 'y'}, overlap_suites=['s'],
            composed_suites=['c'], clock_split=dict(verdict='PASS'), scheduler_tick_probe=dict(verdict='PASS'),
            probe39=dict(verdict='PASS'), probe37=dict(verdict='PASS'), declared_limit='limit'))
        frozen = dump(held / 'alpha-next-engine/result.json', dict(verdict='PASS'))
        pin['engine_selfrun_sha256'] = sha(frozen)
        dump(self.dev_pin, pin)
        compare = dump(out / 'engine-comparison.json', dict(verdict='PASS', differs=[], compared=61, candidate_record_sha256=sha(engine),
                                                            parent_record=str(frozen), parent_record_sha256=sha(frozen)))
        observed = dump(held / 'observed.json', {})
        pins = dump(held / 'pins.json', dict(schema='alpha-next-c1-negative-pins/1', tests={'t': ['fail', 'x']}, cases={},
                                              frozen_record_sha256=sha(frozen), discovery='d',
                                              observation=dict(observed_json=str(observed), observed_sha256=sha(observed))))
        negative = dump(out / 'engine-negative/result.json', dict(verdict='PASS', negative_control=True, manifest=manifest, reasons=[],
                                                                 pinned_tests=1, suites=dict(tests=5), container_rc=0,
                                                                 overlay_manifest='o'))
        tamper = dump(out / 'overlay-tamper.json', dict(schema='alpha-next-c1-overlay-tamper/1', verdict='PASS',
                                                         candidate_manifest=overlay_manifest, cases=[dict(ok=True)]))
        import step_b_clock_bound as clock_bound
        queue = dump(out / 'selfrun/result.json', dict(
            verdict='PASS', bundle_manifest=manifest, skips=0, frozen_runner_sha256=sha(runner), tests_run=460,
            alpha_next_c1_queue='PASS', alpha_next_queue='PASS', owner_prelaunch=dict(verdict='PASS'),
            clock_bound_frozen_excluded=clock_bound.IN_PROBE, rc=0, container='c', gpu_device_requests=0))
        clock = dump(out / 'clock-scan.json', dict(schema='alpha-next-c1-clock-scan/1', verdict='PASS', scan=dict(problems=[]),
                                                    bundle_manifest=manifest, runner=dict(sha256=sha(runner)),
                                                    negative_controls={'a': True}))
        compile_proof = dump(out / 'compile-proof.json', dict(verdict='PASS', files=3))
        precheck = out / 'precheck-t.log'
        precheck.write_text('SUMMARY checks=15 pass=15 fail=0\n')
        conf = out / 'native-precheck.conf'
        conf.write_text('# conf\n')
        staging = dump(out / 'staging-check.json', dict(
            schema='alpha-next-c1-staging-check/1', verdict='PASS', negative_controls_refused={'a': True}, runner_sha256=sha(runner),
            selfrun_sha256=sha(queue), engine_selfrun_sha256=sha(engine), requires={'r': dict(holds=True)}))
        note = out / 'note.md'
        note.write_text('note\n')
        vm = dump(out / 'vm-compare.json', dict(schema='alpha-next-c1-vm-compare/1', verdict='PASS', rc=0, scope={}, selection={}))
        args = types.SimpleNamespace(build=build, freeze=freeze, engine=engine, compare=compare, negative=negative, tamper=tamper,
                                     queue=queue, clock_scan=clock, compile_proof=compile_proof, precheck=precheck, conf=conf,
                                     staging_check=staging, note=note, out=out / 'delivery.json', vm=vm)
        return args, pins, manifest, overlay_manifest

    def run_delivery(self, mutate=None):
        delivery = self.fresh('alpha_next_c1_delivery')
        held = Path(tempfile.mkdtemp(prefix='delivery-', dir=self.tmp))
        args, pins, manifest, overlay_manifest = self.out_dir(held)
        if mutate:
            mutate(args)
        gate = types.SimpleNamespace(EXPECTED_RUNTIME_S=1, BOX_BUDGET_S=9000, ACTIVE_S=8100, CLEANUP_S=900, LAST_ARM_ENTRY_S=4000)
        present = tuple(n for n in delivery.LEAVES if (ROOT / 'harness' / n).is_file())
        with mock.patch.dict(sys.modules, alpha_next_c1_gate=gate), mock.patch.object(delivery, 'PINS', pins), \
                mock.patch.object(delivery, 'LEAVES', present), \
                mock.patch.object(delivery, 'queue_pins_check', lambda out: dict(
                    verdict='PASS', harness_bundle_sha256=manifest, overlay_manifest_sha256=overlay_manifest)):
            argv = ['--' + k.replace('_', '-') + '=' + str(getattr(args, k)) for k in delivery.ARGS] + ['--vm', str(args.vm)]
            rc = delivery.main(argv)
            if mutate is None:
                with self.assertRaises(FileExistsError):    # the delivery record is never overwritten
                    delivery.main(argv)
            return delivery, rc, args

    def test_the_declared_leaves_name_c1s_and_the_composition_fixs(self):
        delivery = self.fresh('alpha_next_c1_delivery')
        for name in ('alpha_next_c1_engine_fixture_leaves.json', 'stage_overlap_fixture_leaves.json',
                     'alpha_next_c1_queue_fixture_leaves.json', 'alpha_next_queue_fixture_leaves.json'):
            self.assertIn(name, delivery.LEAVES)

    def test_end_to_end_on_the_dev_build_record_shape(self):
        delivery, rc, args = self.run_delivery()
        self.assertEqual(rc, 0)
        row = json.loads(args.out.read_text())
        self.assertEqual(row['verdict'], 'PASS')
        self.assertEqual(row['alpha_next_c1_freeze']['alpha_next_freeze']['proved_commit'], parent_pin.load()['source'])
        self.assertTrue(row['alpha_next_base']['dev'])
        self.assertEqual(row['vm']['verdict'], 'PASS')
        self.assertTrue(row['engine_sources_vs_alpha_next']['paths_differing'])

    def test_a_freeze_record_without_its_nested_link_refuses(self):
        def mutate(args):
            record = json.loads(args.freeze.read_text())
            del record['source']['both_off_reaches_alpha_next_freeze']
            dump(args.freeze, record)
        with self.assertRaises(KeyError):
            self.run_delivery(mutate)

    def test_an_engine_record_without_early_open_refuses(self):
        def mutate(args):
            record = json.loads(args.engine.read_text())
            record['early_open'] = dict(ok=False, early_opens=0)
            dump(args.engine, record)
            staging = json.loads(args.staging_check.read_text())
            staging['engine_selfrun_sha256'] = sha(args.engine)
            dump(args.staging_check, staging)
            cmp_ = json.loads(args.compare.read_text())
            cmp_['candidate_record_sha256'] = sha(args.engine)
            dump(args.compare, cmp_)
        with self.assertRaises(AssertionError):
            self.run_delivery(mutate)


if __name__ == '__main__':
    unittest.main()
