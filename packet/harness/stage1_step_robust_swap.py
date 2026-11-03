#!/usr/bin/env python3
"""stage1-step's SUPERSEDED map (attn-meta's, by name: harness/attn_meta_robust_swap.py on claude/attn-meta-c1b; alpha-next's mechanism, 699bada8), beside
alpha_next_robust_swap.py (unchanged: chains ran it).

Four c1 test ids assert that the working tree is c1's own (the rebind default-off proof's "no other engine file differs", and c1's pack freeze and delivery proofs
over the same tree). The stage1-step tree moves gated rows in ten engine files, so they fail on it by construction. They are left out by id, each with its
reason, and their claim is carried by two proofs that must hold in the same run:
  1. the base-tree receipt (tools/stage1_step_base_proof.py, a chain step): c1's two modules at c1's delivery fbb9123e in a detached checkout, rc 0 and the expected
     test count (22);
  2. tools/stage1_step_default_off.py, run here: this tree projects byte-identically onto fbb9123e (declared rows, each with why it holds with
     FLASH_S5_DECODE_STEP_UNUSED unset) and the switch-unset refusal tests pass; and its test module test_stage1_step_default_off, added to the same run when the
     modules given do not name it (its failure fails the run).
A missing receipt or a default-off proof that cannot run is INC (exit 2), never a silent skip; a receipt that does not prove, or a default-off FAIL, is FAIL (exit 1).
The map must name exactly the declared ids (DECLARED; a fifth id is refused) and each must be found exactly once in the loaded modules. alpha-next's ROBUST_SWAP and
SUPERSEDED and step_b_clock_bound's ids apply as in alpha_next_robust_swap.

usage: stage1_step_robust_swap.py --base-receipt FILE MODULE [MODULE ...]
"""
import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import unittest

import alpha_next_robust_swap as an_swap
import step_b_clock_bound

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_SCHEMA = 'stage1-step-base-proof/1'
BASE = 'fbb9123e62ad0364470293fc2f4744279b3667ee'
EXPECTED_BASE_TESTS = 22
_TREE = ('asserts the working tree is c1\'s own (fbb9123e); the stage1-step tree moves gated '
         'rows in the engine files and adds the stage1-step rows, so it fails by construction. Carried by the base-tree receipt (this test passes at '
         'fbb9123e) and tools/stage1_step_default_off.py (this tree projects byte-identically onto fbb9123e), both in the same run')
SUPERSEDED = {
    'test_alpha_next_c1_rebind_default_off.Holds.test_pass_on_this_tree_and_receipt_written_once':
        'c1 rebind default-off proof on this tree (other_engine_files_differing == []) ' + _TREE,
    'test_alpha_next_c1_pack.Freeze.test_the_tree_proves_against_the_dev_parent_and_an_undeclared_byte_refuses':
        'c1 freeze proof of this tree against its dev parent (an undeclared byte refuses) ' + _TREE,
    'test_alpha_next_c1_pack.Delivery.test_a_rebind_proof_that_does_not_hold_refuses':
        'c1 delivery: a failing rebind proof refuses with exactly its own problems, over this tree ' + _TREE,
    'test_alpha_next_c1_pack.Delivery.test_end_to_end_on_the_dev_build_record_shape':
        'c1 delivery end to end on the dev build record shape, over this tree ' + _TREE,
}
DECLARED = frozenset(SUPERSEDED)          # pinned below by literal in check_map (attn-meta's four, the lead's)
_LEAD_FOUR = ('test_alpha_next_c1_rebind_default_off.Holds.test_pass_on_this_tree_and_receipt_written_once',
              'test_alpha_next_c1_pack.Freeze.test_the_tree_proves_against_the_dev_parent_and_an_undeclared_byte_refuses',
              'test_alpha_next_c1_pack.Delivery.test_a_rebind_proof_that_does_not_hold_refuses',
              'test_alpha_next_c1_pack.Delivery.test_end_to_end_on_the_dev_build_record_shape')
MODULES = sorted({name.split('.', 1)[0] for name in SUPERSEDED})
DEFAULT_OFF_TESTS = 'test_stage1_step_default_off'


class Refused(RuntimeError):
    pass


_SELFTEST_RECEIPT = []


def selftest_base_receipt():
    """The base-tree receipt for harness/selftest.py's rows: the chain's own (STAGE1_STEP_BASE_RECEIPT, the base-proof step's record, not read here), else one proof
    per process, in a temporary directory. Imported by selftest.py, so tools/strict_parallel.py's plan, evaluating selftest's command expression in the same process as
    the exec'd selftest, gets the one path. With the variable set no proof runs (the plan check names a path and reads nothing)."""
    chain = os.environ.get('STAGE1_STEP_BASE_RECEIPT')
    if chain:
        return chain
    if not _SELFTEST_RECEIPT:
        where = tempfile.mkdtemp(prefix='stage1-step-base-proof-')
        atexit.register(shutil.rmtree, where, True)
        path = os.path.join(where, 'base-proof.json')
        proof = subprocess.run([sys.executable, '-B', str(ROOT / 'tools' / 'stage1_step_base_proof.py'), '--out', path], capture_output=True, text=True)
        print('selftest stage1-step base proof: rc %d' % proof.returncode)
        _SELFTEST_RECEIPT.append(path)
    return _SELFTEST_RECEIPT[0]


def check_map(superseded=None):
    names = set(SUPERSEDED if superseded is None else superseded)
    if names != set(_LEAD_FOUR):
        raise Refused('superseded map names other ids than the declared four: extra %s, missing %s'
                      % (sorted(names - set(_LEAD_FOUR)), sorted(set(_LEAD_FOUR) - names)))
    if any(not (SUPERSEDED if superseded is None else superseded)[n] for n in names):
        raise Refused('superseded id without a reason')


def filtered(suite, superseded=None):
    superseded = SUPERSEDED if superseded is None else superseded
    tests = list(an_swap._flatten(suite))
    ids = [t.id() for t in tests]
    modules = {i.split('.', 1)[0] for i in ids}
    for name in superseded:
        if name.split('.', 1)[0] in modules and ids.count(name) != 1:
            raise Refused('superseded id not found exactly once: %s (%d)' % (name, ids.count(name)))
    removed = sorted(i for i in ids if i in superseded)
    return unittest.TestSuite(t for t in tests if t.id() not in superseded), removed


def base_receipt(path):
    """(status, facts): 'ok', 'missing' (INC) or 'fail'."""
    if path is None or not Path(path).is_file():
        return 'missing', dict(path=None if path is None else str(path))
    try:
        r = json.loads(Path(path).read_text())
    except ValueError:
        return 'fail', dict(path=str(path), problem='unreadable')
    facts = {k: r.get(k) for k in ('schema', 'base', 'rc', 'tests_run', 'expected_tests', 'skipped', 'passed', 'checkout_kind')}
    ok = (r.get('schema') == BASE_SCHEMA and r.get('base') == BASE and r.get('rc') == 0 and r.get('skipped') == 0
          and r.get('tests_run') == EXPECTED_BASE_TESTS and r.get('expected_tests') == EXPECTED_BASE_TESTS and r.get('passed') is True)
    return ('ok' if ok else 'fail'), dict(facts, path=str(path))


def default_off():
    """(status, facts) of tools/stage1_step_default_off.py run now (no receipt file)."""
    tool = ROOT / 'tools' / 'stage1_step_default_off.py'
    if not tool.is_file():
        return 'missing', dict(tool=str(tool))
    env = {k: v for k, v in os.environ.items() if k not in ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE')}
    proc = subprocess.run([sys.executable, '-B', str(tool)], cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
    try:
        r = json.loads(proc.stdout)
    except ValueError:
        return 'missing', dict(rc=proc.returncode, problem='no receipt printed')
    ok = proc.returncode == 0 and r.get('verdict') == 'PASS' and r.get('passed') is True
    return ('ok' if ok else 'fail'), dict(rc=proc.returncode, verdict=r.get('verdict'), problems=r.get('problems'))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--base-receipt')
    p.add_argument('modules', nargs='+')
    a = p.parse_args(argv)
    modules = [m[:-3] if m.endswith('.py') else m for m in a.modules]
    try:
        check_map()
        loader = unittest.defaultTestLoader
        suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
        suite, clock_removed = step_b_clock_bound.filtered(suite)
        assert clock_removed == step_b_clock_bound.expected_removed(modules), clock_removed
        suite, an_removed = an_swap.filtered(suite)
        assert an_removed == an_swap.expected_removed(modules), an_removed
        kept, removed = filtered(suite)
        if removed and DEFAULT_OFF_TESTS not in modules:
            # Attn-meta's condition 2 (lead, 3 Nov): test_stage1_step_default_off PASS in the same run as the superseded ids' removal.
            kept.addTests(loader.loadTestsFromName(DEFAULT_OFF_TESTS))
            print('stage1-step superseded: %s added to this run (the default-off proof\'s tests, same run)' % DEFAULT_OFF_TESTS)
    except Refused as exc:
        print('stage1-step superseded map refused: %s' % exc)
        return 1
    if removed:
        evidence = dict(base_receipt=base_receipt(a.base_receipt), default_off=default_off())
        print('stage1-step superseded evidence: ' + json.dumps(evidence, sort_keys=True))
        states = {s for s, _ in evidence.values()}
        if 'fail' in states:
            print('stage1-step superseded: FAIL (a carrying proof does not hold)')
            return 1
        if 'missing' in states:
            print('stage1-step superseded: INC (a carrying proof is missing; nothing is skipped silently)')
            return 2
    for name in removed:
        print('pinned-tree frozen, superseded (declared, stage1-step): %s: %s' % (name, SUPERSEDED[name]))
    result = unittest.TextTestRunner(verbosity=1, resultclass=an_swap._WriterResetResult).run(kept)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
