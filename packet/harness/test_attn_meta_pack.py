#!/usr/bin/env python3
"""q18a-attn-meta chain tools (CPU only, git; no container, no box):

  * parent pin: the committed pin verifies against c1's delivered build b and its delivery record; the placeholder exits 3
    ('parent not pinned'); a dev pin is refused when read as the committed file;
  * freeze: attn-meta off projects onto c1's delivery and is c1's build-source engine byte for byte, c1's freeze record pinned;
    five mutated engine copies refused;
  * engine self-run: c1's run text with exactly the three swaps (an eighth part attn/, its suites recorded), c1's negative and
    comparison with attn-meta's schema names, the parent cases and the comparison rules are c1's cases, the frozen record is
    c1's pinned one; no swap leaves a c1 schema name in the derived texts;
  * probe derivation: the committed attn_meta_cpu_probe.py is exactly the derivation (530 = c1's 519 + the queue suite);
  * build, spec, review pins, tamper, clock scan, precheck conf, staging check, queue pins: their declared names (two added
    overlay modules, schema names, the queue stem, the seam list resolving to harness files, the queue pins placeholder);
    the clock scan PASSes over this tree with the queue's setcfg reference.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))


def tool(*args, env=None):
    return subprocess.run([sys.executable, '-B', *args], cwd=ROOT, capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, **(env or {})))


class ParentPin(unittest.TestCase):
    def test_pinned_and_verified_against_c1s_delivery(self):
        import attn_meta_parent_pin as pin
        p = pin.load()
        self.assertEqual(p['schema'], 'attn-meta-parent/1')
        self.assertEqual(p['manifest'][:12], '0c33ca841620')
        self.assertEqual(p['delivery_commit'], 'fbb9123e62ad0364470293fc2f4744279b3667ee')
        got = tool('tools/attn_meta_parent_pin.py', '--out', str(Path(p['build']).parent), '--check')
        self.assertEqual(got.returncode, 0, got.stderr[-800:])
        self.assertIn('attn-meta parent verified', got.stdout)

    def test_placeholder_and_dev_rules(self):
        import attn_meta_parent_pin as pin
        with tempfile.TemporaryDirectory() as tmp:
            ph = Path(tmp) / 'p.json'
            ph.write_text(json.dumps(dict(schema=pin.SCHEMA, status='NOT-PINNED: x')))
            with self.assertRaises(SystemExit) as e:
                pin.load(ph)
            self.assertEqual(e.exception.code, 3)
            dev = Path(tmp) / 'd.json'
            dev.write_text(json.dumps(dict(pin.load(), dev=True)))
            with self.assertRaises(SystemExit) as e:          # a dev pin read without ATTN_META_PARENT: refused
                pin.load(dev)
            self.assertEqual(e.exception.code, 3)


class ParentPinB(unittest.TestCase):
    """The c1b READY pin of this tree; the steps rule is c1b's chain's names (tools/alpha_next_c1b_gates_b.sh)."""

    def test_the_pin_is_c1b_b2s_ready_build(self):
        import attn_meta_parent_pin_b as pin
        p = pin.load()
        self.assertEqual(p['schema'], pin.SCHEMA)
        self.assertEqual(json.loads(Path(p['build']).read_text())['name'], 'q18a-alpha-next-c1b')
        self.assertEqual(json.loads(Path(p['build']).read_text())['manifest'], p['manifest'])
        self.assertIn('engine_selfrun_sha256', p)
        self.assertIn('ready_commit', p)
        self.assertNotIn('dev', p)

    def test_placeholder_and_dev_rules(self):
        import attn_meta_parent_pin_b as pin
        with tempfile.TemporaryDirectory() as tmp:
            ph = Path(tmp) / 'p.json'
            ph.write_text(json.dumps(dict(schema=pin.SCHEMA, status='NOT-PINNED: x')))
            with self.assertRaises(SystemExit) as e:
                pin.load(ph)
            self.assertEqual(e.exception.code, 3)
            dev = Path(tmp) / 'd.json'
            dev.write_text(json.dumps(dict(schema=pin.SCHEMA, status='DEV: x', dev=True)))
            with self.assertRaises(SystemExit) as e:          # a dev pin read without the environment: refused
                pin.load(dev)
            self.assertEqual(e.exception.code, 3)

    def test_steps_rule(self):
        import attn_meta_parent_pin_b as pin
        rows = pin.STEPS_REQUIRED + ('strict-2', 'focused', 'engine-selfrun')
        ok = ''.join('13:00:00Z %s rc=0\n' % s for s in rows) + '13:00:00Z strict-1 rc=1\n' + pin.STOPPED + '\n'
        self.assertEqual(pin.steps_ok(ok), [])
        self.assertEqual(pin.steps_ok(ok.replace(pin.STOPPED, 'x')), ['stop-line'])
        self.assertEqual(pin.steps_ok(ok.replace('vm-carry rc=0', 'vm-carry rc=1')), ['vm-carry'])
        self.assertEqual(pin.steps_ok(ok.replace('strict-2 rc=0', 'strict-2 rc=1')), ['strict'])
        for name in ('parent-pin', 'composition', 'rebind-default-off', 'vm-carry', 'staging-check'):
            self.assertIn(name, pin.STEPS_REQUIRED)


class Freeze(unittest.TestCase):
    def test_off_is_c1_and_the_mutants_are_refused(self):
        import attn_meta_freeze as freeze
        with tempfile.TemporaryDirectory() as tmp:
            r = freeze.prove(ROOT, Path(tmp) / 'work')
        self.assertEqual(r['verdict'], 'PASS', r['faults'])
        self.assertTrue(r['both_off_reaches_c1_freeze']['projected_byte_identical'])
        self.assertEqual(sorted(r['added_modules_sha256']), ['attn_meta_batch.py', 'attn_meta_select.py'])
        self.assertEqual(freeze.negative_controls(), {k: 'REFUSED' for k in (
            'undeclared_byte', 'byte_outside_rows', 'row_text_gone', 'third_added_module', 'removed_base_file')})
        self.assertEqual(freeze.NEW, {'e3/attn_meta_batch.py', 'e3/attn_meta_select.py'})


class EngineSelfrun(unittest.TestCase):
    def test_derived_from_c1_by_exact_swaps(self):
        import attn_meta_engine_selfrun as m
        self.assertEqual(m.PARTS[-1], 'attn')
        self.assertEqual(m.PARTS[:-1], m.base.PARTS)
        self.assertIn("run_attn(bundle, out / 'attn', name + '-attn')]", m.RUN_TEXT)
        self.assertIn('    p, d, f, g, s, n, o, t = parts\n', m.RUN_TEXT)
        self.assertIn("schema='attn-meta-negative/1'", m.NEGATIVE_TEXT)
        self.assertIn("negative_kind='c1-parent'", m.NEGATIVE_TEXT)
        self.assertIn("schema='attn-meta-default-off-comparison/1'", m.COMPARE_TEXT)
        for text in (m.NEGATIVE_TEXT, m.COMPARE_TEXT):
            self.assertNotIn('alpha-next-c1-', text)
        self.assertEqual(m._run_ns['SCHEMA'], 'attn-meta-selfrun/1')
        self.assertEqual(m.expected_cases(), m.base.expected_cases())
        self.assertEqual([r[1] for r in m.comparison_rules()], m.base.expected_cases())
        self.assertEqual(m._ns['PARENT_PASS'], list(m.base.PARENT_PASS) + list(m.base.CASES))
        self.assertEqual(m._ns['CASES'], ())
        self.assertEqual(str(m.frozen()), m._pin()['engine_selfrun'])
        self.assertEqual(json.loads(m.PINS.read_text())['status'].split(':')[0], 'AWAITING-OBSERVATION')

    def test_a_changed_c1_run_text_refuses(self):
        import attn_meta_engine_selfrun as m
        with self.assertRaisesRegex(AssertionError, 'c1 run\\(\\) changed'):
            m.derive(m.RUN_TEXT, m.SWAPS[:1], 'c1 run()')        # applied twice: the old text is gone


class NegativePins(unittest.TestCase):
    def test_stub_and_writer_names(self):
        import attn_meta_negative_pins_write as w
        stub = json.loads((TOOLS / 'attn_meta_negative_pins.json').read_text())
        base, expected = w.base_of(stub)
        self.assertEqual((base['frozen_record_sha256'], expected), (None, {}))
        self.assertEqual((w.SCHEMA, w.OBSERVED_SCHEMA), ('attn-meta-negative-pins/1', 'attn-meta-negative/1'))
        self.assertEqual(w.OUT, TOOLS / 'attn_meta_negative_pins.json')
        with self.assertRaises(AssertionError):                 # an observed-looking stub is refused
            w.base_of(dict(stub, tests={'x': ['fail', 'y']}))


class CrossRun(unittest.TestCase):
    def test_pinned_record_verifies_against_fn_main(self):
        fn_main = ROOT.parent / 'fn-main'
        record = ROOT / 'results/hub/attn-meta/cross-run-c1.json'
        got = tool('tools/attn_meta_cross_run.py', '--fn-main', str(fn_main), '--out', str(record), '--check')
        self.assertEqual(got.returncode, 0, got.stderr[-800:])
        r = json.loads(record.read_text())
        self.assertEqual((r['supplied'], r['arms']), (True, 4))
        self.assertIn('never a verdict input', r['label'])
        self.assertEqual(sorted(r['c1_off_band1_decode_only_tok_s']),
                         ['c1off.g2-live-0', 'c1off.g2-live-1', 'c1off.g2-solo-0', 'c1off.g2-solo-1'])


class PostRead(unittest.TestCase):
    """tools/attn_meta_post_read.py over harness/fixtures/attn_meta_post_read: 23 real handler lines (4 whole flushes, ranks
    0-3; numbers only) cut from c1's box record, and two decode-only arm windows split at the middle line's stamp."""
    FIX = ROOT / 'harness/fixtures/attn_meta_post_read'

    def test_flushes_by_stamp_and_the_per_step_numbers(self):
        import statistics
        import attn_meta_post_read as post
        anat = post.anat
        r = post.read(str(self.FIX))
        self.assertEqual((r['schema'], r['read_only'], r['is_verdict_input']), ('attn-meta-post-read/1', True, False))
        self.assertEqual((r['torn_handler_lines'], r['incomplete_flushes'], r['schema_errors']), (0, 0, 0))
        records, _, _ = anat.read_handler_lines(anat.sar.engine_logs(str(self.FIX)))
        flushes, _ = anat.assemble(records)
        spans = {name: (json.loads((self.FIX / ('arm-%s.json' % name)).read_text())['wall_start_s'],
                        json.loads((self.FIX / ('arm-%s.json' % name)).read_text())['wall_end_s']) for name in ('decode-am0', 'decode-am1')}
        want, unassigned = {}, 0
        for f in flushes:
            arm = next((n for n, (s, e) in spans.items() if s <= f['stamp'] <= e), None)
            if arm is None:
                unassigned += 1
                continue
            for row in f['main']:
                if anat.KIND_NAMES.get(row['kind']) == 'execute_model' and not row['flags'] & anat.FLAG_INSTALL:
                    want.setdefault(arm, {}).setdefault(str(f['rank']), []).append(row)
        self.assertEqual(r['unassigned_flushes'], unassigned)
        self.assertEqual(r['flushes'], len(flushes))
        self.assertTrue(want, 'the fixture assigns no execute_model row')
        self.assertEqual(sorted(r['arms']), sorted(want))
        for arm, ranks in want.items():
            for rank, rows in ranks.items():
                got = r['arms'][arm]['ranks'][rank]
                self.assertEqual(list(got), ['outside_chunk_span'])          # decode-only: no chunk span
                cell = got['outside_chunk_span']
                self.assertEqual(cell['execute_model_rows'], len(rows))
                meta = [x['seams']['prep_attn_meta']['ns'] for x in rows if 'prep_attn_meta' in x['seams']]
                self.assertEqual(cell['prep_attn_meta']['n'], len(meta))
                if meta:
                    self.assertAlmostEqual(cell['prep_attn_meta']['median_ms'], statistics.median(meta) / 1e6)
                cpu = [x['cpu_ns'] for x in rows if type(x['cpu_ns']) is int]
                if cpu:
                    self.assertAlmostEqual(cell['cpu_per_execute_model']['median_ms'], statistics.median(cpu) / 1e6)
            s1 = [x['seams']['prep_attn_meta']['ns'] for rk in ('2', '3') for x in ranks.get(rk, []) if 'prep_attn_meta' in x['seams']]
            self.assertEqual(r['arms'][arm]['stage1_decode_step']['prep_attn_meta']['n'], len(s1))

    def test_output_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'post.json'
            self.assertEqual(tool('tools/attn_meta_post_read.py', str(self.FIX), str(out)).returncode, 0)
            self.assertNotEqual(tool('tools/attn_meta_post_read.py', str(self.FIX), str(out)).returncode, 0)


class Probe(unittest.TestCase):
    def test_committed_probe_is_the_derivation(self):
        got = tool('tools/attn_meta_tools_derive_b.py', '--check')
        self.assertEqual(got.returncode, 0, got.stderr[-800:])
        text = (TOOLS / 'attn_meta_cpu_probe_b.py').read_text()
        self.assertIn("owner.QNAME=='q18a-attn-meta'", text)
        self.assertRegex(text, r"(?m)^EXPECTED_TESTS=579  # the 563 alpha-next-c1b probe cases plus the 16 attn-meta queue cases")
        self.assertIn('import alpha_next_c1b_gate\nimport attn_meta_gate\n', text)
        self.assertIn("record=dict(attn_meta_queue='PASS',", text)
        self.assertIn("alpha_next_c1b_queue='PASS',", text)


class Family(unittest.TestCase):
    def test_every_derived_tool_is_its_declared_swaps(self):
        got = tool('tools/attn_meta_b_derive.py', '--check')
        self.assertEqual(got.returncode, 0, got.stderr[-800:])

    def test_a_changed_source_text_refuses(self):
        import attn_meta_b_derive as d
        with self.assertRaises(AssertionError):
            d.swap('no such text', (d.PIN,), 'x')

    def test_engine_selfrun_b_binds_c1bs_record_and_compare(self):
        import attn_meta_engine_selfrun_b as m
        self.assertEqual(m.ATTN_SUITES[-1], 'test_attn_meta_box_b')
        self.assertIn('test_alpha_next_c1b_queue', m._run_ns['OVERLAP_SUITES'])
        self.assertIs(m.negative.__globals__['frozen'], m.frozen)
        self.assertIs(m.compare.__globals__['frozen'], m.frozen)
        self.assertEqual(m.expected_cases(), m.base.expected_cases())
        self.assertEqual(json.loads(m.PINS.read_text())['schema'], 'attn-meta-negative-pins/1')
        import attn_meta_parent_pin_b as pin
        self.assertEqual(str(m.frozen()), pin.load()['engine_selfrun'])

    def test_freeze_b_and_build_b_read_the_c1b_pin(self):
        import attn_meta_freeze_b as freeze
        import build_attn_meta_b as build
        import attn_meta_parent_pin_b as pin
        p = pin.load()
        self.assertEqual(freeze.PARENT_SOURCE, p['source'])
        self.assertEqual((build.CANDIDATE, build.b.PARENT_CANDIDATE, build.b.PARENT_STEM),
                         ('q18a-attn-meta', 'q18a-alpha-next-c1b', 'q18a_alpha_next_c1b'))
        with tempfile.TemporaryDirectory() as tmp:
            r = freeze.prove(ROOT, Path(tmp) / 'work')
        self.assertEqual(r['verdict'], 'PASS', r['faults'])
        self.assertTrue(r['both_off_reaches_c1_freeze']['projected_byte_identical'])
        self.assertEqual(r['both_off_reaches_c1_freeze']['proved_commit'], p['source'])

    def test_precheck_conf_b_seams_exist_and_name_the_h_modules(self):
        import attn_meta_precheck_conf_b as conf
        seams = conf.SEAMS.split()
        self.assertEqual(len(seams), len(set(seams)))
        self.assertTrue(all((ROOT / 'harness' / s).is_file() for s in seams), [s for s in seams if not (ROOT / 'harness' / s).is_file()])
        for name in ('native43_engine_death_h.py', 'alpha_next_stop_race_h.py', 'e3_attn_meta_box_b.py', 'e3_c1_open_rows.py'):
            self.assertIn(name, seams)

    def test_gates_b_calls_only_the_b_family(self):
        text = (TOOLS / 'attn_meta_gates_b.sh').read_text()
        for old in ('build_attn_meta.py', 'attn_meta_parent_pin.py', 'attn_meta_engine_selfrun.py', 'attn_meta_overlay_tamper.py',
                    'attn_meta_clock_scan.py', 'attn_meta_precheck_conf.py'):
            self.assertNotIn('tools/' + old, text.replace('tools/attn_meta_gates.sh', ''), old)
        self.assertEqual(subprocess.run(['bash', '-n', str(TOOLS / 'attn_meta_gates_b.sh')]).returncode, 0)
        self.assertEqual(subprocess.run(['bash', '-n', str(TOOLS / 'attn_meta_gates_b_cont.sh')]).returncode, 0)
        for suite in ('test_attn_meta_box_b', 'test_native43_engine_death_h', 'test_alpha_next_stop_race_h', 'test_alpha_next_c1b_queue'):
            self.assertIn(suite, text)
        self.assertIn('--cap 1800', text)

    def test_tamper_b_falls_back_only_for_a_module_without_a_comment(self):
        import attn_meta_overlay_tamper_b as tamper
        with_comment = b'"""Doc."""\n# a comment\nx = 1\n'
        self.assertEqual(with_comment[tamper.c0.comment_offset(with_comment):][:1], b'a')
        without = b'"""Doc."""\nx = 1\n'
        self.assertEqual(without[tamper.c0.comment_offset(without):][:1], b'D')
        with self.assertRaises((AssertionError, ValueError)):
            tamper.c0.comment_offset(b'x = 1\n')

    def test_negative_pins_are_hub_side(self):
        spec = json.loads((TOOLS / 'native43_spec.attn-meta-b.json').read_text())
        self.assertNotIn('attn_meta_negative_pins.json', spec['tools'])
        self.assertNotIn('attn_meta_queue_pins.json', spec['tools'])


class Continuation(unittest.TestCase):
    """tools/attn_meta_gates_b_cont.sh refuses anywhere but the observation stop, with the pins filled, clean, the bundle unmoved."""
    SCRIPT = TOOLS / 'attn_meta_gates_b_cont.sh'

    def refused(self, steps, status='FILLED', moved=False):
        script = self.SCRIPT.read_text()
        cut = script.index('wait_clear engine-negative\n')
        with tempfile.TemporaryDirectory() as tmp:
            wt, out, evid = Path(tmp) / 'wt', Path(tmp) / 'out', Path(tmp) / 'evid'
            (wt / 'tools').mkdir(parents=True)
            out.mkdir()
            evid.mkdir()
            subprocess.run(['git', 'init', '-q', str(wt)], check=True)
            (wt / 'tools/attn_meta_negative_pins.json').write_text(json.dumps(dict(status=status)))
            subprocess.run(['git', '-C', str(wt), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(wt), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x'], check=True)
            digest = hashlib.sha256(b'x\n').hexdigest()
            bundle = out / ('q18a-attn-meta-bundle-' + digest[:12])
            bundle.mkdir()
            (bundle / 'SHA256SUMS').write_text('moved\n' if moved else 'x\n')
            (out / 'build.json').write_text(json.dumps(dict(manifest=digest)))
            stepfile = evid / 'gates-t.steps'
            stepfile.write_text(steps)
            # Only the refusal block runs (everything before the chain's own tail): the tail is the chain's text, unchanged.
            body = script[:cut] + 'echo through\nexit 0\n'
            runner = Path(tmp) / 'cont.sh'
            runner.write_text(body)
            got = subprocess.run(['bash', str(runner), str(wt), 'spec', str(out), str(evid), 't', 'note', 'clear'],
                                 capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
            return got.returncode, got.stdout, stepfile.read_text()

    STOP = '12:00:00Z engine-negative-observe rc=0\npins awaiting review\n12:00:01Z engine-negative-awaiting-pins rc=5\n'

    def test_through_at_the_stop_with_filled_pins(self):
        rc, out, steps = self.refused(self.STOP)
        self.assertEqual((rc, out.strip()), (0, 'through'), steps)
        self.assertIn('continuation head ', steps)

    def test_refused_elsewhere(self):
        rc, out, steps = self.refused('12:00:00Z engine-comparison rc=0\n')
        self.assertEqual(rc, 3)
        self.assertIn('not at the observation stop', steps)

    def test_refused_with_the_pins_awaiting(self):
        rc, out, steps = self.refused(self.STOP, status='AWAITING-OBSERVATION: x')
        self.assertEqual(rc, 3)
        self.assertIn('pins not filled', steps)

    def test_refused_when_the_bundle_moved(self):
        rc, out, steps = self.refused(self.STOP, moved=True)
        self.assertEqual(rc, 3)
        self.assertIn('the bundle moved', steps)

    def test_the_tail_is_the_chains_text_and_ends_at_the_stop_line(self):
        import attn_meta_b_derive as d
        gates, cont = d.gates(), self.SCRIPT.read_text()
        tail = gates[gates.index('wait_clear engine-negative\n'):]
        self.assertTrue(cont.endswith(tail))
        self.assertIn("echo 'stopped-before-delivery: VM run pending'", tail)


class Names(unittest.TestCase):
    def test_declared_names(self):
        import build_attn_meta as build
        import attn_meta_precheck_conf as conf
        import attn_meta_staging_check as staging
        import attn_meta_clock_scan as scan
        import attn_meta_review_pins as review
        self.assertEqual(build.NEW_OVERLAY, ('e3/attn_meta_batch.py', 'e3/attn_meta_select.py'))
        self.assertEqual((build.CANDIDATE, build.b.PARENT_CANDIDATE, build.b.PARENT_STEM),
                         ('q18a-attn-meta', 'q18a-alpha-next-c1', 'q18a_alpha_next_c1'))
        self.assertEqual(review.ADDED, {'attn_meta_batch.py', 'attn_meta_select.py'})
        self.assertEqual(conf.ADDED, {'e3/attn_meta_batch.py', 'e3/attn_meta_select.py'})
        self.assertEqual(conf.STEM, 'q18a_attn_meta')
        seams = conf.SEAMS.split()
        self.assertEqual(len(seams), len(set(seams)))
        self.assertTrue(all((ROOT / 'harness' / s).is_file() for s in seams), [s for s in seams if not (ROOT / 'harness' / s).is_file()])
        for container_only in ('test_attn_meta_select.py', 'test_attn_meta_probe_b.py', 'test_attn_meta_overlay.py'):
            self.assertNotIn(container_only, seams)
        self.assertEqual(staging.SCHEMA, 'attn-meta-staging-check/1')
        self.assertEqual(scan.ARM_MODULES[:3], ('e3_attn_meta_box', 'attn_meta_arms', 'e3_alpha_next_c1_box'))
        self.assertEqual(scan.TEXTS, ('q18a_attn_meta.draft.sh', 'q18a_attn_meta_declare.sh', 'q18a_attn_meta_settings.sh'))
        pins = json.loads((TOOLS / 'attn_meta_queue_pins.json').read_text())
        self.assertEqual(pins['schema'], 'attn-meta-queue-pins/1')
        self.assertTrue(pins['status'].startswith('PLACEHOLDER'))
        tamper = (TOOLS / 'attn_meta_overlay_tamper.py').read_text()
        self.assertIn('import attn_meta_freeze as freeze', tamper)
        self.assertIn("r['path'] in freeze.NEW", tamper)

    def test_spec_names_and_runner_swaps(self):
        spec = json.loads((TOOLS / 'native43_spec.attn-meta-b.json').read_text())
        self.assertEqual((spec['candidate'], spec['label'], spec['review_key']), ('q18a-attn-meta', 'attn-meta', 'attn_meta'))
        self.assertEqual(spec['queue'], ['q18a_attn_meta.draft.sh', 'q18a_attn_meta_declare.sh', 'q18a_attn_meta_settings.sh'])
        self.assertEqual(spec['bundle_selfrun']['probe'], 'attn_meta_cpu_probe_b.py')
        self.assertEqual(spec['details']['attn_meta_settings']['FLASH_ATTN_META_GDN_BATCH'], '1')
        self.assertEqual(len(spec['runner_swaps']), 2)
        self.assertEqual(spec['runner_swaps'][0][1], "engine.get('schema')=='attn-meta-selfrun/1'")
        self.assertTrue(spec['runner_swaps'][1][1].endswith(" and result.get('attn_meta_queue')=='PASS'"))
        self.assertNotIn('attn_meta_queue_pins.json', spec['tools'])

    def test_clock_scan_passes_over_this_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'scan.json'
            got = tool('tools/attn_meta_clock_scan_b.py', '--tree', str(ROOT), '--setcfg-reference', str(ROOT / 'deploy/setcfg.py'),
                       '--out', str(out))
            self.assertEqual(got.returncode, 0, got.stdout[-1500:] + got.stderr[-800:])
            self.assertEqual(json.loads(out.read_text())['verdict'], 'PASS')


if __name__ == '__main__':
    unittest.main()
