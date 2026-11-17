#!/usr/bin/env python3
"""Queue-name preflight for q18a-stage1-step: every name rule a QNAME meets on the box, executed on this host first.

Why (lead, 3 Nov 2026 13:08 CET): a box slot was lost today to an inherited q18a rule (returning-append's q18b-append-yield launch
was refused at staging, runs/..._declare.sh:41, QNAME =~ ^q18a[-_a-zA-Z0-9]*$). This is the same check as returning-append's
(tools/q18a_append_yield_queue.py, d8099ba7: name_rules / rule_holds / runner_names / declare_prefix / run_declare_names /
check_names), copied here so c1b carries its own; the c1b names and the marker are the only changes.

What it checks, for one QNAME:
  * every bash rule [[ $QNAME =~ RE ]] in the runs/ texts the runner calls (draft, declaration) and in the settings, run in bash's
    own =~ (no Python stand-in);
  * the runner's phase texts are exactly <stem>.draft.sh and <stem>_declare.sh, stem = QNAME with '-' -> '_';
  * the declaration itself executed on this host, unedited, up to and including its last QNAME rule (its own harness checksum,
    qlib.sh and offline inputs; no box path, no engine, no request dir made: the cut is before the fresh-R mkdir);
  * no regex applied to QNAME in the runner text.
--names RUNNER --bundle BUNDLE runs it over a built owner runner and its bundle (runs/ and qlib.sh). CPU only.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

NAME_RULE = re.compile(r'\[\[\s*\$\{?QNAME\}?\s*=~\s*(\S+)\s*\]\]')
PASSED = 'STAGE1-STEP-NAME-CHECKS-PASSED'


def name_rules(text):
    return NAME_RULE.findall(text)


def rule_holds(rule, qname):
    """bash's own =~ for one rule (no Python regex stand-in)."""
    done = subprocess.run(['bash', '-c', '[[ $1 =~ %s ]]' % rule, 'rule', qname], capture_output=True, timeout=30)
    return done.returncode == 0


def runner_names(text):
    """(QNAME, phase scripts, lines applying a pattern to QNAME) of an owner runner text."""
    (qname,) = re.findall(r"^QNAME='([^']+)'$", text, re.M)
    phases = sorted(set(re.findall(r"phase\('([a-z0-9_]+(?:\.draft|_declare)\.sh)'", text)))
    regex_on_qname = [line for line in text.splitlines() if 'QNAME' in line and re.search(r'\bre\.|fullmatch|=~', line)]
    return qname, phases, regex_on_qname


def declare_prefix(text):
    """The declaration's lines up to and including its last QNAME name rule, then the marker and exit 0."""
    lines = text.splitlines()
    rows = [i for i, line in enumerate(lines) if NAME_RULE.search(line)]
    assert rows, 'the declaration has no QNAME name rule'
    return '\n'.join(lines[:rows[-1] + 1] + ['echo %s' % PASSED, 'exit 0']) + '\n'


def run_declare_names(declare, qname, qlib):
    """Execute declare_prefix(declaration) with QNAME: offline task inputs, a harness dir holding qlib.sh and its SHA256SUMS, all in
    a temporary dir. Returns (rc, stdout)."""
    with tempfile.TemporaryDirectory(prefix='c1b-names-') as tmp:
        tmp = Path(tmp)
        h = tmp / 'h'
        h.mkdir()
        shutil.copy(qlib, h / 'qlib.sh')
        (h / 'SHA256SUMS').write_text('%s  qlib.sh\n' % hashlib.sha256((h / 'qlib.sh').read_bytes()).hexdigest())
        for d in ('s0', 'task', 'overlay'):
            (tmp / d).mkdir()
        for f in ('setcfg.py', 'image.txt'):
            (tmp / f).write_text('')
        script = tmp / 'declare-names.sh'
        script.write_text(declare_prefix(Path(declare).read_text()))
        env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=str(tmp), HARNESS_DIR=str(h), S0=str(tmp / 's0'),
                   QNAME=qname, Q17_TEST_OFFLINE='1', Q17_TEST_TASK_BASE=str(tmp / 'task'), Q17_TEST_SETCFG=str(tmp / 'setcfg.py'),
                   Q17_TEST_IMAGE_FILE=str(tmp / 'image.txt'), Q17_TEST_SERVICE='vllm-flash-test', S5_OVERLAY=str(tmp / 'overlay'))
        done = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=env, timeout=60)
        made = sorted(p.name for p in (tmp / 's0').iterdir())
        assert made == [], ('the cut declaration made request paths', made)
        return done.returncode, done.stdout


def check_names(qname, phases, runs, qlib, regex_on_qname=()):
    """Every refusal a QNAME meets in the runner and in the runs/ texts it calls; [] when it passes them all."""
    problems = []
    if regex_on_qname:
        problems.append('runner applies a pattern to QNAME: %r' % list(regex_on_qname))
    stem = qname.replace('-', '_')
    want = sorted([stem + '.draft.sh', stem + '_declare.sh'])
    if sorted(phases) != want:
        problems.append('runner phases %r, want %r' % (phases, want))
    texts = {}
    for name in sorted(set(phases) | {stem + '_settings.sh'}):
        path = Path(runs) / name
        if not path.is_file():
            problems.append('missing ' + name)
            continue
        texts[name] = path.read_text()
        for rule in name_rules(texts[name]):
            if not rule_holds(rule, qname):
                problems.append('%s: QNAME %s fails %s' % (name, qname, rule))
    for declare in [n for n in phases if n.endswith('_declare.sh') and n in texts]:
        rc, out = run_declare_names(Path(runs) / declare, qname, qlib)
        if rc != 0 or PASSED not in out:
            problems.append('%s executed to its name rules: rc %d, %s' % (declare, rc, out.strip().splitlines()[-1:] or ''))
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description='queue-name check of a built owner runner over its bundle')
    p.add_argument('--names', type=Path, required=True, help='the built owner runner')
    p.add_argument('--bundle', type=Path, required=True, help='the held bundle (runs/ and qlib.sh)')
    a = p.parse_args(argv)
    qname, phases, regex = runner_names(a.names.read_text())
    problems = check_names(qname, phases, a.bundle / 'runs', a.bundle / 'qlib.sh', regex)
    print('queue names %s: QNAME %s, phases %s%s' % ('PASS' if not problems else 'FAIL', qname, phases,
                                                    '' if not problems else ', ' + '; '.join(problems)))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
