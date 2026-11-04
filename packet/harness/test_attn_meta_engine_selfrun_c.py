#!/usr/bin/env python3
"""tools/attn_meta_engine_selfrun_c.py and chain c (CPU only, no container, no box).

Chain b's negative raised TypeError at `modes = PARENT_PASS + CASES` (a list plus a tuple) and no test called negative(): the old
EngineSelfrun test even pinned CASES == (). So these tests call it, with the real argument types and c1b's real frozen record:
  * Assembly: the real guarded_container and the real step_a.command_lines over the modes negative() assembles; only the docker call is
    replaced. The same drive over _b's negative() is the mutant: it must raise the TypeError chain b hit.
  * Phases: observe -> the _c pin writer -> pinned, the whole two-phase path with fabricated suite outcomes and the frozen traces, so
    every line after the assembly runs. Mutants: a changed trace, an undeclared non-pass, a pinned test that passes, a missing record.
  * Chain c's texts: the derivation, the name wait with no clearance file, the carry check in both directions, the continuation's refusals.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / 'harness'))
import attn_meta_engine_selfrun_b as old  # noqa: E402
import attn_meta_engine_selfrun_c as new  # noqa: E402
import test_attn_meta_pack as pack  # noqa: E402

STUB = dict(schema='attn-meta-negative-pins/1', status='AWAITING-OBSERVATION: test stub', cases={}, tests={}, frozen_record_sha256=None,
            cases_from='none: test stub')
BITES = 'test_attn_meta_control.Control.test_needs_the_candidate'
BITES_TEXT = "ModuleNotFoundError: No module named 'attn_meta_select'"
ROWS = [['test_attn_meta_control.', 'attn_meta_select', 'needs what only the candidate has', 'HARNESS']]


def outcomes(extra=None):
    got = {suite + '.Row.test_row': ['ok', ''] for suite in new.NEGATIVE_SUITES}
    got[BITES] = ['error', BITES_TEXT]
    got.update(extra or {})
    return got


class Tree:
    """A temporary bundle, overlay, sources and pins file; only their SHA256SUMS are read when docker is replaced."""

    def __init__(self, tmp, module=new):
        self.root = Path(tmp)
        self.bundle, self.overlay, self.sources = self.root / 'bundle', self.root / 'overlay', self.root / 'sources'
        for place in (self.bundle, self.overlay, self.sources):
            place.mkdir()
            (place / 'SHA256SUMS').write_text('')
        self.pins = self.root / 'pins.json'
        self.pins.write_text(json.dumps(STUB))
        self.module = module

    def container(self, bundle, overlay, sources, out, name, modes, suites, hide_candidate=False):
        """What step_a.container returns: suites.json's content and each case's frozen trace, as a healthy parent gives them."""
        out.mkdir()
        (out / 'inner.sh').write_text('\n'.join(new.base.step_a.command_lines(modes, suites)) + '\n')
        place = new.frozen()
        cases = {}
        for mode in modes:
            data = json.loads((place / new.part_of(mode) / (mode + '.json')).read_text())
            (out / (mode + '.json')).write_text(json.dumps(data))
            cases[mode] = data
        self.modes, self.suites, self.hide = modes, suites, hide_candidate
        result = dict(outcomes=self.outcomes, skips=0, tests=len(self.outcomes), ok=False, receipt_writer_resets=0)
        return 0, result, cases, ['docker', 'replaced'], 0.5

    outcomes = outcomes()

    def guarded(self, *args, **kwargs):
        return self.container(*args, **kwargs), dict(rc=0, wall_s=0.5, limit_s=1800, timed_out=False), None

    def drive(self, out, observe, pins=None, guarded=None):
        ns = self.module._ns
        with mock.patch.dict(ns, dict(PINS=pins or self.pins, guarded_container=guarded or self.guarded)):
            return self.module.negative(self.bundle, self.overlay, self.sources, self.root / out, 'test-negative', observe)

    def observe(self):
        rc = self.drive('observed', True)
        return rc, json.loads((self.root / 'observed/result.json').read_text())

    def write_pins(self):
        """The _c writer over the observed record, as the chain's review step calls it."""
        import attn_meta_negative_pins_write_c as w
        (self.root / 'observed.log').write_text('observed\n')
        (self.root / 'classify.json').write_text(json.dumps(ROWS))
        written = w.pins(dict(STUB), self.root / 'observed/result.json', self.root / 'observed.log', self.root / 'classify.json', 'test')
        place = self.root / 'written.json'
        place.write_text(json.dumps(written))
        return place, written


class Assembly(unittest.TestCase):
    def real_container(self, tree, module):
        """negative() through the real guarded_container: step_a.container runs up to its docker call, which is replaced."""
        step_a = new.base.step_a
        seen = {}

        def container(bundle, overlay, sources, out, name, modes, suites, hide_candidate=False):
            seen.update(modes=modes, suites=suites, hide=hide_candidate, lines=step_a.command_lines(modes, suites))
            raise RuntimeError('stopped at the docker call')
        with mock.patch.object(step_a, 'container', container), mock.patch.dict(module._ns, dict(PINS=tree.pins)):
            rc = module.negative(tree.bundle, tree.overlay, tree.sources, tree.root / 'out', 'test-negative', True)
        return rc, seen

    def test_the_fixed_assembly_reaches_the_container_with_c1s_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            (tree.root / 'out').mkdir()      # the replaced container never made it; negative() writes its INC record here
            rc, seen = self.real_container(tree, new)
            self.assertEqual(rc, 4)          # no guard and an unreadable record: INC, never an observation
            record = json.loads((tree.root / 'out/result.json').read_text())
            self.assertEqual(record['verdict'], 'INC')
            self.assertEqual(record['pinned_tests'], 0)
        self.assertIs(type(seen['modes']), tuple)
        self.assertEqual(list(seen['modes']), list(new.base.PARENT_PASS) + list(new.base.CASES))
        self.assertEqual(seen['suites'], new.NEGATIVE_SUITES)
        self.assertTrue(seen['hide'])
        self.assertTrue(seen['lines'])
        for mode in seen['modes']:
            self.assertTrue(any(mode in line for line in seen['lines']), mode)

    def test_mutant_the_old_b_assembly_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp, old)
            with self.assertRaisesRegex(TypeError, 'can only concatenate list'):
                self.real_container(tree, old)

    def test_every_sequence_name_has_c1s_type(self):
        """The family of the defect: a name negative() and judge() read, of another type than c1's text was written for."""
        names = ('PARENT_PASS', 'CASES', 'NEGATIVE_SUITES')
        self.assertEqual({n: type(new._ns[n]) for n in names}, {n: type(new.base.__dict__[n]) for n in names})
        self.assertNotEqual(type(old._ns['PARENT_PASS']), type(new.base.PARENT_PASS))      # the mutant: _b's list
        self.assertEqual(new._ns['CASES'], ())
        self.assertEqual(list(new.PARENT_PASS), list(old.PARENT_PASS))
        self.assertEqual(new.negative.__globals__['PARENT_PASS'], new.PARENT_PASS)
        self.assertIs(new.judge.__globals__, new.negative.__globals__)

    def test_only_the_negative_moved(self):
        self.assertIs(new.run, old.run)
        self.assertIs(new.compare, old.compare)
        self.assertIs(new.frozen, old.frozen)
        self.assertEqual((new.SCHEMA, new.PARTS, new.ATTN_SUITES), (old.SCHEMA, old.PARTS, old.ATTN_SUITES))
        self.assertEqual(new.expected_cases(), old.expected_cases())


class Phases(unittest.TestCase):
    def test_observe_then_pin_then_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            rc, record = tree.observe()
            self.assertEqual(rc, 1)          # --observe exits 1 by design; the chain's step reads the verdict
            self.assertEqual(record['verdict'], 'OBSERVED', record['reasons'] + record['inc_reasons'])
            self.assertEqual(record['schema'], 'attn-meta-negative/1')
            self.assertEqual(record['observed'], {BITES: ['error', BITES_TEXT]})
            self.assertEqual(record['frozen_record_sha256'], new.sha(new.frozen() / 'result.json'))
            self.assertEqual(sorted(record['cases']), sorted(new.PARENT_PASS))
            self.assertEqual(record['suite_names'], list(new.NEGATIVE_SUITES))
            place, written = tree.write_pins()
            self.assertEqual(written['tests'], {BITES: ['error', BITES_TEXT]})
            self.assertEqual(written['cases'], {})
            self.assertEqual(written['classes'], {BITES: 'HARNESS'})
            rc = tree.drive('pinned', False, pins=place)
            record = json.loads((tree.root / 'pinned/result.json').read_text())
            self.assertEqual((rc, record['verdict'], record['reasons']), (0, 'PASS', []))
            self.assertEqual(record['pinned_tests'], 1)

    def pinned(self, tree):
        tree.observe()
        return tree.write_pins()[0]

    def test_mutant_a_changed_trace_fails_the_pinned_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            place = self.pinned(tree)
            mode = new.PARENT_PASS[0]

            def guarded(*args, **kwargs):
                got, guard, error = tree.guarded(*args, **kwargs)
                rc, result, cases, c, wall = got
                self.assertEqual(cases[mode]['verdict'], 'PASS')
                return (rc, result, dict(cases, **{mode: dict(cases[mode], verdict='FAIL')}), c, wall), guard, error
            self.assertEqual(tree.drive('pinned', False, pins=place, guarded=guarded), 1)
            record = json.loads((tree.root / 'pinned/result.json').read_text())
            self.assertEqual(record['verdict'], 'FAIL')
            self.assertIn('parent-case-not-pass:' + mode, record['reasons'])

    def test_mutant_an_undeclared_non_pass_fails_the_pinned_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            place = self.pinned(tree)
            tree.outcomes = outcomes({'test_attn_meta_box.Row.test_other': ['fail', 'x']})
            self.assertEqual(tree.drive('pinned', False, pins=place), 1)
            reasons = json.loads((tree.root / 'pinned/result.json').read_text())['reasons']
            self.assertTrue(any(r.startswith('undeclared-negative-outcome:test_attn_meta_box.') for r in reasons), reasons)

    def test_mutant_a_pinned_test_that_passes_fails_the_pinned_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            place = self.pinned(tree)
            tree.outcomes = outcomes({BITES: ['ok', '']})
            self.assertEqual(tree.drive('pinned', False, pins=place), 1)
            reasons = json.loads((tree.root / 'pinned/result.json').read_text())['reasons']
            self.assertTrue(any(r.startswith('negative-test-differs:' + BITES) for r in reasons), reasons)

    def test_an_unreported_suite_is_inc_and_carries_no_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            tree.outcomes = {BITES: ['error', BITES_TEXT]}
            self.assertEqual(tree.drive('observed', True), 4)
            record = json.loads((tree.root / 'observed/result.json').read_text())
            self.assertEqual(record['verdict'], 'INC')
            self.assertNotIn('observed', record)

    def test_the_writer_refuses_an_engine_row_and_an_unexplained_non_pass(self):
        import attn_meta_negative_pins_write_c as w
        self.assertIs(w.__dict__.get('selfrun'), None)        # imported inside pins(), as _b's
        with tempfile.TemporaryDirectory() as tmp:
            tree = Tree(tmp)
            tree.observe()
            (tree.root / 'observed.log').write_text('observed\n')
            for rows, refused in ((ROWS[:0] + [['test_attn_meta_box.', 'x', 'y', 'HARNESS']], 'no reviewed row'),
                                  ([ROWS[0][:3] + ['ENGINE']], 'ENGINE non-passes')):
                (tree.root / 'classify.json').write_text(json.dumps(rows))
                with self.assertRaisesRegex(AssertionError, refused):
                    w.pins(dict(STUB), tree.root / 'observed/result.json', tree.root / 'observed.log', tree.root / 'classify.json', 't')


def block(text, start, stop):
    return text[text.index(start):text.index(stop)]


class ChainC(unittest.TestCase):
    GATES = TOOLS / 'attn_meta_gates_c.sh'

    def test_every_derived_tool_is_its_declared_swaps(self):
        got = subprocess.run([sys.executable, '-B', 'tools/attn_meta_c_derive.py', '--check'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr[-800:])

    def test_the_chain_is_b_with_the_fixed_tool_and_nothing_else_named(self):
        text = self.GATES.read_text()
        for script in (self.GATES, TOOLS / 'attn_meta_gates_c_cont.sh'):
            self.assertEqual(subprocess.run(['bash', '-n', str(script)]).returncode, 0, script)
            self.assertNotIn('attn_meta_engine_selfrun_b.py', script.read_text())
            self.assertNotIn('$CLEAR', script.read_text())
        self.assertEqual(text.count('python3 -B tools/attn_meta_engine_selfrun_c.py'), 4)
        for step in ('parent-pin', 'base-proof', 'default-off', 'review-pins', 'strict', 'focused', 'composition-test', 'prepare-review',
                     'build', 'bundle-selfrun', 'engine-selfrun', 'engine-comparison', 'engine-negative-observe', 'engine-negative',
                     'overlay-tamper', 'clock-scan', 'compile-proof', 'precheck-conf', 'precheck', 'staging-check'):
            self.assertRegex(text, r'step (%s\b|"?%s)' % (step, step), step)
        order = [text.index(m) for m in ('step default-off', 'step review-pins', 'RC=$(strict', 'RC=$(focused', 'step composition-test',
                                         'step prepare-review', 'step build', '# >>> carry_check', 'step bundle-selfrun',
                                         'queue-pins-write $?', 'step engine-comparison', 'step engine-negative-observe')]
        self.assertEqual(order, sorted(order))

    def wait(self, script, listing):
        text = script.read_text()
        fn = block(text, '# >>> wait_clear', '# <<< wait_clear')
        with tempfile.TemporaryDirectory() as tmp:
            steps = Path(tmp) / 'steps'
            calls = Path(tmp) / 'calls'
            body = 'set -u\nSTEPS=%s\nCALLS=%s\nCLEAR_POLL_S=0\n%s\n' % (steps, calls, fn)
            body += ('clear_ps() { n=$(cat "$CALLS" 2>/dev/null || echo 0); echo $((n+1)) >"$CALLS";'
                     ' (( n == 0 )) && printf %%s "%s"; return 0; }\nwait_clear bundle-selfrun\n' % listing)
            got = subprocess.run(['bash', '-c', body], capture_output=True, text=True, timeout=30)
            self.assertEqual(got.returncode, 0, got.stderr)
            return steps.read_text()

    def test_the_name_wait_alone_holds_and_is_logged(self):
        for script in (self.GATES, TOOLS / 'attn_meta_gates_c_cont.sh'):
            clear = self.wait(script, 'forgejo')
            self.assertIn('clear-before bundle-selfrun waited', clear)
            self.assertNotIn('waiting-before', clear)
            for name in ('alpha-next-c1-x', 'append-yield-h-hub', 'returning-append-y'):
                held = self.wait(script, name)
                self.assertIn('waiting-before bundle-selfrun: ' + name, held)
                self.assertIn('clear-before bundle-selfrun waited', held)

    def carry(self, manifest):
        carry_manifest = 'a' * 64
        text = self.GATES.read_text()
        code = block(text, '# >>> carry_check', '# <<< carry_check')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, chain_b, held = root / 'out', root / 'chain-b', root / 'held'
            for place in (out, chain_b / 'selfrun', chain_b / 'engine-selfrun', held):
                place.mkdir(parents=True)
            for name in ('selfrun/result.json', 'engine-selfrun/result.json', 'engine-comparison.json'):
                (chain_b / name).write_text(name)
            (held / 'SHA256SUMS').write_text('x\n')
            shim = root / 'bin'
            shim.mkdir()
            (shim / 'sha256sum').write_text('#!/bin/bash\n[[ $1 == *SHA256SUMS ]] && echo "%s  $1" || /usr/bin/env -i %s "$1"\n'
                                            % (manifest, subprocess.check_output(['bash', '-c', 'command -v sha256sum'], text=True).strip()))
            (shim / 'sha256sum').chmod(0o755)
            body = ('set -u\nPATH=%s:$PATH\nOUT=%s\nHELD=%s\nSTEPS=%s\nPIN=%s\nCARRY_MANIFEST=%s\nCARRY_OUT=%s\n%s\n'
                    'printf "<%%s><%%s><%%s>" "$CARRY" "$SELFRUN" "$ENGINE"\n'
                    % (shim, out, held, root / 'steps', manifest, carry_manifest, chain_b, code))
            got = subprocess.run(['bash', '-c', body], capture_output=True, text=True, timeout=30)
            self.assertEqual(got.returncode, 0, got.stderr)
            return got.stdout.replace(str(root), ''), (root / 'steps').read_text().replace(str(root), '')

    def test_carry_only_on_the_equal_manifest(self):
        got, steps = self.carry('a' * 64)
        self.assertEqual(got, '<1></chain-b/selfrun></chain-b/engine-selfrun>')
        self.assertIn('carry-by-manifest-identity manifest ' + 'a' * 64 + ' selfrun ', steps)
        self.assertRegex(steps, r'selfrun [0-9a-f]{64} engine [0-9a-f]{64} comparison [0-9a-f]{64}')
        got, steps = self.carry('a' * 63 + 'b')
        self.assertEqual(got, '<></out/selfrun></out/engine-selfrun>')
        self.assertIn('carry-check differs', steps)
        self.assertNotIn('carry-by-manifest-identity', steps)

    def test_the_carry_guards_exactly_the_three_steps(self):
        import attn_meta_c_derive as d
        text = self.GATES.read_text()
        self.assertIn('CARRY_MANIFEST=' + d.CARRY_MANIFEST + '\n', text)
        self.assertEqual(text.count('if [[ -z $CARRY ]]; then\n'), 2)
        self.assertIn('if [[ -z $CARRY ]]; then\nwait_clear bundle-selfrun\n', text)
        self.assertIn('if [[ -z $CARRY ]]; then\nENGINE=$OUT/engine-selfrun\n', text)
        guarded = text[text.index('if [[ -z $CARRY ]]; then\nENGINE='):text.index('wait_clear engine-negative-observe')]
        self.assertLess(guarded.index('step engine-comparison'), guarded.index('\nfi\n'))
        observe = text.index('wait_clear engine-negative-observe')
        self.assertEqual(text.count('$OUT/selfrun/result.json'), 0)
        self.assertEqual(text.count('$SELFRUN/result.json'), 3)      # the carry record, the precheck conf, the staging check
        self.assertLess(text.index('step engine-comparison'), observe)

    def test_negative_pins_stay_hub_side_in_the_c_spec_generator(self):
        text = (TOOLS / 'attn_meta_spec_c.py').read_text()
        self.assertIn("HUB_SIDE = ('tools/attn_meta_queue_pins.json', 'tools/attn_meta_negative_pins.json')", text)
        self.assertIn("NAME = 'native43_spec.attn-meta-c.json'", text)
        self.assertIn('import attn_meta_engine_selfrun_c as selfrun', text)


class Seams(unittest.TestCase):
    """The owner precheck's own seam scan over the harness tree: chain b's dry run failed it (12 undeclared) and no test held it."""

    def scan(self, bundle):
        import attn_meta_precheck_conf_b as conf
        script = (TOOLS / 'owner-precheck/native-precheck.sh').read_text()
        code = script[script.index("PY_SEAMS=$(cat <<'PYEOF'\n") + len("PY_SEAMS=$(cat <<'PYEOF'\n"):]
        code = code[:code.index('\nPYEOF\n')]
        got = subprocess.run([sys.executable, '-B', '-c', code, str(bundle), r'fixture[-_]leaves.*\.json$', *conf.SEAMS.split()],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(got.returncode, 0, got.stderr[-800:])
        return [line for line in got.stdout.splitlines() if line.startswith('  UNDECLARED')], got.stdout

    def test_every_mocked_seam_is_declared(self):
        undeclared, out = self.scan(ROOT / 'harness')
        self.assertEqual(undeclared, [])
        self.assertIn('undeclared=0', out)

    def test_mutant_without_the_c_leaves_the_twelve_are_undeclared(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in (ROOT / 'harness').iterdir():
                if path.name != 'attn_meta_fixture_leaves_c.json':
                    (Path(tmp) / path.name).symlink_to(path)
            undeclared, _ = self.scan(tmp)
        self.assertEqual(len(undeclared), 12, undeclared)


class ContinuationC(pack.Continuation):
    """The chain b refusals over chain c's continuation; and it reads a carried chain's records."""
    SCRIPT = TOOLS / 'attn_meta_gates_c_cont.sh'

    def test_the_tail_is_the_chains_text_and_ends_at_the_stop_line(self):
        import attn_meta_c_derive as d
        gates, cont = d.gates(), self.SCRIPT.read_text()
        tail = gates[gates.index('wait_clear engine-negative\n'):]
        self.assertTrue(cont.endswith(tail))
        self.assertIn("echo 'stopped-before-delivery: VM run pending'", tail)
        self.assertIn('$SELFRUN/result.json', tail)
        self.assertIn('SELFRUN=' + d.CARRY_OUT + '/selfrun; ENGINE=' + d.CARRY_OUT + '/engine-selfrun;', cont)


if __name__ == '__main__':
    unittest.main()
