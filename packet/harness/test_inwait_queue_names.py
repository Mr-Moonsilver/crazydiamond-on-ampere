#!/usr/bin/env python3
"""QueueNames for q18a-inwait (lead, 3 Nov 2026 13:08 CET; returning-append's QueueNames pattern, d8099ba7): the runs/
name rules executed with the QNAME the owner runner will carry. That QNAME is tools/build_inwait.CANDIDATE, which
build_native43.derive_runner writes into the runner's QNAME='...' line; the c1b queue texts are tools/inwait_queue.py's.
test_inwait_queue imports QueueNames, so the bundle self-run (it discovers that suite) and chain b's focused step run it.
CPU only: bash =~, and the declaration executed to its last name rule in a temporary dir (tools/inwait_queue_names.py).
"""
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools')]
import inwait_queue_names as names  # noqa: E402

RUNS = HERE / 'runs'
QLIB = HERE / 'qlib.sh'
DECLARE, DRAFT, SETTINGS = (RUNS / n for n in ('q18a_inwait_declare.sh', 'q18a_inwait.draft.sh',
                                                 'q18a_inwait_settings.sh'))


def runner_qname():
    """The runner's QNAME (the build's CANDIDATE literal) and the two runs/ texts it calls."""
    (candidate,) = re.findall(r"^CANDIDATE = '([^']+)'$", (ROOT / 'tools/build_inwait.py').read_text(), re.M)
    stem = candidate.replace('-', '_')
    return candidate, [stem + '.draft.sh', stem + '_declare.sh']


def runner_text(qname, phases):
    """The runner's QNAME line and its two phase calls, as the built runner carries them (owner-run lines 24, 284, 300)."""
    draft, declare = sorted(phases)
    return ("QNAME='%s'\nBUILD_LOGS=S0/(QNAME+'-build-logs')\n"
            "        p=S0/(QNAME+suffix);require(not p.exists() and not p.is_symlink(),'fresh-output-path-required')\n"
            "        rc=phase('%s',S0/(QNAME+'-declare-chain.log'),env)\n"
            "    rc=phase('%s',S0/(QNAME+'.log'),env)\n" % (qname, declare, draft))


class QueueNames(unittest.TestCase):
    def test_the_runner_qname_passes_every_name_rule_and_the_declaration_preflight(self):
        qname, phases = runner_qname()
        self.assertEqual(qname, 'q18a-inwait')
        self.assertEqual(names.check_names(qname, phases, RUNS, QLIB), [])
        declare, draft, settings = (p.read_text() for p in (DECLARE, DRAFT, SETTINGS))
        self.assertEqual(names.name_rules(declare), ['^q18a[-_a-zA-Z0-9]*$'])     # the rule that refused q18b-append-yield
        self.assertEqual((names.name_rules(draft), names.name_rules(settings)), ([], []))
        import inwait_queue as queue
        self.assertIn("QNAME=${QNAME:-%s}" % queue.QNAME_DEFAULT, declare)         # the declaration's default is the launch's, the runner's QNAME overrides it
        rc, out = names.run_declare_names(DECLARE, qname, QLIB)
        self.assertEqual((rc, out.strip().splitlines()[-1]), (0, names.PASSED), out[-400:])

    def test_the_queue_texts_are_the_runner_stem(self):
        import inwait_queue as queue
        qname, phases = runner_qname()
        stem = qname.replace('-', '_')
        self.assertEqual(sorted(Path(n).name for n in queue.derive()),
                         sorted(phases + [stem + '_settings.sh']))

    def test_mutant_a_q18b_name_fails(self):
        problems = names.check_names('q18b-inwait', ['q18a_inwait.draft.sh', 'q18a_inwait_declare.sh'],
                                     RUNS, QLIB)
        self.assertTrue(any('fails ^q18a' in x for x in problems), problems)
        self.assertTrue(any('executed to its name rules: rc 1' in x for x in problems), problems)
        self.assertTrue(any(x.startswith('runner phases ') for x in problems), problems)
        renamed = names.check_names('q18b-inwait', ['q18b_inwait.draft.sh', 'q18b_inwait_declare.sh'],
                                    RUNS, QLIB)                                   # no q18b texts exist in this tree
        self.assertEqual(sorted(renamed), ['missing q18b_inwait%s' % s for s in ('.draft.sh', '_declare.sh', '_settings.sh')])
        rc, out = names.run_declare_names(DECLARE, 'q18b-inwait', QLIB)
        self.assertEqual(rc, 1)
        self.assertIn('STEP preflight FAIL invalid QNAME', out)
        self.assertNotIn(names.PASSED, out)

    def test_mutant_a_name_with_a_dot_or_slash_fails_the_declaration(self):
        for qname in ('q18a-alpha-next.c1b', 'q18a/inwait'):
            with self.subTest(qname=qname):
                rc, out = names.run_declare_names(DECLARE, qname, QLIB)
                self.assertEqual(rc, 1, out)
                self.assertIn('STEP preflight FAIL invalid QNAME', out)

    def test_a_runner_text_is_read_and_checked(self):
        qname, phases = runner_qname()
        got = names.runner_names(runner_text(qname, phases))
        self.assertEqual(got, (qname, sorted(phases), []))
        self.assertEqual(names.check_names(*got[:2], RUNS, QLIB, got[2]), [])
        bad = names.runner_names(runner_text('q18b-inwait', phases))
        self.assertTrue(names.check_names(*bad[:2], RUNS, QLIB, bad[2]))
        guarded = names.runner_names(runner_text(qname, phases) + "require(re.fullmatch('q18b.*',QNAME),'x')\n")
        self.assertTrue(guarded[2])
        self.assertTrue(names.check_names(*guarded[:2], RUNS, QLIB, guarded[2]))

    def test_the_builder_candidate_rule_is_the_swapped_one(self):
        """build_native43.load_spec's q18a-native rule would refuse this QNAME; build_inwait replaces load_spec (the
        accepted build-assert swap) and applies no pattern to the candidate other than equality with CANDIDATE."""
        qname, _ = runner_qname()
        native = (ROOT / 'tools/build_native43.py').read_text()
        self.assertEqual(re.findall(r"re\.fullmatch\(r'([^']+)', spec\['candidate'\]\)", native), ['q18a-native[0-9a-z-]+'])
        self.assertIsNone(re.fullmatch(r'q18a-native[0-9a-z-]+', qname))
        build = (ROOT / 'tools/build_inwait.py').read_text()
        self.assertIn('\nb.load_spec = load_spec\n', build)
        self.assertEqual(re.findall(r"re\.\w+\([^)]*spec\['candidate'\]", build), [])
        self.assertIn("assert spec['candidate'] == CANDIDATE and", build)

    def test_the_cli_over_a_runner_and_a_bundle_shaped_dir(self):
        import tempfile
        import shutil
        qname, phases = runner_qname()
        with tempfile.TemporaryDirectory(prefix='c1b-names-cli-') as tmp:
            tmp = Path(tmp)
            (tmp / 'runs').mkdir()
            for p in (DECLARE, DRAFT, SETTINGS):
                shutil.copy(p, tmp / 'runs' / p.name)
            shutil.copy(QLIB, tmp / 'qlib.sh')
            (tmp / 'runner.py').write_text(runner_text(qname, phases))
            tool = ROOT / 'tools/inwait_queue_names.py'
            done = subprocess.run([sys.executable, '-B', str(tool), '--names', str(tmp / 'runner.py'), '--bundle', str(tmp)],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-400:])
            self.assertTrue(done.stdout.startswith('queue names PASS: QNAME q18a-inwait'), done.stdout)
            (tmp / 'runs' / SETTINGS.name).unlink()
            done = subprocess.run([sys.executable, '-B', str(tool), '--names', str(tmp / 'runner.py'), '--bundle', str(tmp)],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 1)
            self.assertIn('missing q18a_inwait_settings.sh', done.stdout)


if __name__ == '__main__':
    unittest.main()
