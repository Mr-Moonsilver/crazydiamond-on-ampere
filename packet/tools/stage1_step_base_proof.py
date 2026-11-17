#!/usr/bin/env python3
"""stage1-step base-tree proof (attn-meta's condition 2, by name): c1's tree-identity test modules run at c1's delivery commit.

The stage1-step tree is not c1's tree by design (the engine moves), so four c1 test ids that assert "this tree is c1's" are superseded on it by name
(harness/stage1_step_robust_swap.py). Their claim is carried by composition: this proof (c1's modules pass at the base commit, in a detached checkout of it)
and tools/stage1_step_default_off.py (this tree projects byte-identically onto the base).

The checkout is a ``git clone --shared --no-checkout`` of this repository in a fresh temporary directory, then ``checkout --detach BASE`` (the repository's own
worktree list is not touched; the temporary clone is this tool's own and goes with its directory), or ``--checkout DIR``: an existing clean checkout whose HEAD is
BASE. The untracked ``reference`` link the trees use is linked in when this tree has one. The modules run with FLASH_S5_LAZY_RECV and FLASH_S5_DECODE_MERGE removed from the environment.

Receipt (``--out``, written once): base, checkout kind and head, modules, rc, tests run, the expected count, passed (rc 0 and the count matched). Exit 0 only when
passed.
usage: stage1_step_base_proof.py --out FILE [--base REV] [--checkout DIR]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'stage1-step-base-proof/1'
BASE = 'fbb9123e'
MODULES = ('test_alpha_next_c1_rebind_default_off', 'test_alpha_next_c1_pack')
EXPECTED_TESTS = 22          # both modules at fbb9123e (3 Nov 13:29 CET, detached base worktree: 22 OK in 30.8 s)
TIMEOUT_S = 900


def git(*args, cwd=ROOT):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def run_modules(checkout):
    env = {k: v for k, v in os.environ.items() if k not in ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE')}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', *MODULES], cwd=Path(checkout) / 'harness', env=env,
                          capture_output=True, text=True, timeout=TIMEOUT_S)
    out = proc.stdout + proc.stderr
    ran = re.findall(r'^Ran (\d+) tests? in ', out, re.MULTILINE)
    skipped = sum(int(x) for x in re.findall(r'\bskipped=(\d+)\b', out))
    return proc.returncode, int(ran[-1]) if ran else None, skipped, out[-1500:]


def prove(base, checkout=None):
    full = git('rev-parse', base + '^{commit}')
    record = dict(schema=SCHEMA, base=full, modules=list(MODULES), expected_tests=EXPECTED_TESTS,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source=git('rev-parse', 'HEAD'))
    if checkout is not None:
        head = git('rev-parse', 'HEAD', cwd=checkout)
        dirty = [l for l in git('status', '--porcelain', '--untracked-files=no', cwd=checkout).splitlines() if l]
        record.update(checkout_kind='given', checkout_head=head, checkout_clean=not dirty)
        if head != full or dirty:
            return dict(record, rc=None, tests_run=None, skipped=None, passed=False,
                        problems=['checkout-not-a-clean-checkout-of-the-base'])
        rc, ran, skipped, tail = run_modules(checkout)
    else:
        with tempfile.TemporaryDirectory(prefix='stage1-step-base-') as tmp:
            clone = Path(tmp) / 'base'
            subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(ROOT), str(clone)], check=True,
                           capture_output=True)
            git('checkout', '--quiet', '--detach', full, cwd=clone)
            if (ROOT / 'reference').exists() and not (clone / 'reference').exists():
                (clone / 'reference').symlink_to(os.path.realpath(ROOT / 'reference'))
            record.update(checkout_kind='shared-clone-detached', checkout_head=git('rev-parse', 'HEAD', cwd=clone))
            rc, ran, skipped, tail = run_modules(clone)
    passed = rc == 0 and ran == EXPECTED_TESTS and not skipped
    problems = [] if passed else ['rc %s, tests %s (expected %d), skipped %s' % (rc, ran, EXPECTED_TESTS, skipped)]
    return dict(record, rc=rc, tests_run=ran, skipped=skipped, passed=passed, problems=problems, tail='' if passed else tail)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base', default=BASE)
    p.add_argument('--checkout', type=Path)
    a = p.parse_args(argv)
    if a.out.exists():
        print('receipt exists: ' + str(a.out), file=sys.stderr)
        return 2
    receipt = prove(a.base, a.checkout)
    text = json.dumps(receipt, indent=1, sort_keys=True) + '\n'
    with a.out.open('x') as handle:
        handle.write(text)
    print(text, end='')
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
