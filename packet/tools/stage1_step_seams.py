#!/usr/bin/env python3
"""The owner precheck's seams check by itself: every mocked seam in the declared test files must be named in a fixture-leaves JSON (the lead's standing order, 3 Nov: c1d lost an hour to it after a 53-minute self-run).

The scan is tools/owner-precheck/native-precheck.sh's own PY_SEAMS text, taken from that file (not a copy), with its FIXTURE_LEAVES_REGEX default, over either
  a tree   --root TREE (default: this repository): TREE/harness stands for the bundle, which is what build_native43.stage copies there; run this before launch;
  a bundle --bundle DIR: the built bundle, exactly as the chain's static precheck sees it.
The test files are the conf's: c1's template SEAM_TESTS, tools/stage1_step_precheck_conf.SEAMS, the engine suites of TREE/tools/stage1_step_engine_suites.json, every harness/test_*fifo*.py of the tree (so a new test file is never missed), and --tests NAME.py....
One stricter rule than the precheck: a listed test file with 'stage1_step' in its name that is absent is a problem here (the precheck only notes it). Prints the precheck's lines, UNDECLARED rows first, then one summary line. rc 0 only with no undeclared seam and no absent fifo file. CPU only; reads files, writes nothing.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
PRECHECK = TOOLS / 'owner-precheck/native-precheck.sh'
TEMPLATE = TOOLS / 'owner-precheck/native-precheck.conf.native35-p-c'


def scan_text():
    return re.search(r"PY_SEAMS=\$\(cat <<'PYEOF'\n(.*?)\nPYEOF\n", PRECHECK.read_text(), re.S).group(1)


def leaf_regex():
    return re.search(r"FIXTURE_LEAVES_REGEX='([^']+)'", PRECHECK.read_text()).group(1)


def template_tests():
    body = re.search(r'SEAM_TESTS=\((.*?)\)', TEMPLATE.read_text(), re.S).group(1)
    return [word for word in re.sub(r'#.*', '', body).split() if word.endswith('.py')]


def listed(root, extra=()):
    import stage1_step_precheck_conf as conf
    names = template_tests() + conf.SEAMS.split()
    suites = Path(root) / 'tools/stage1_step_engine_suites.json'
    if suites.is_file():
        names += [n + '.py' for n in json.loads(suites.read_text())['engine_suites']]
    names += sorted(p.name for p in (Path(root) / 'harness').glob('test_*stage1_step*.py'))
    return list(dict.fromkeys(list(names) + list(extra)))


def run(place, tests):
    got = subprocess.run([sys.executable, '-I', '-B', '-c', scan_text(), str(place), leaf_regex(), *tests], capture_output=True, text=True)
    return got.returncode, got.stdout, got.stderr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', type=Path, default=TOOLS.parent)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--tests', nargs='*', default=[])
    a = p.parse_args(argv)
    place = a.bundle if a.bundle else a.root / 'harness'
    tests = listed(a.root, a.tests)
    rc, out, err = run(place, tests)
    lines = out.splitlines()
    undeclared = [l for l in lines if 'UNDECLARED' in l]
    absent = [l for l in lines if l.endswith('absent from bundle') and 'stage1_step' in l]
    for line in undeclared + absent + [l for l in lines if l not in undeclared and l not in absent]:
        print(line)
    if rc or not re.search(r'^total=\d+$', out, re.M):
        print('seam scan failed: rc %d %s' % (rc, err[-400:]))
        return 2
    print('SEAMS %s place=%s undeclared=%d absent_step_files=%d' % ('FAIL' if undeclared or absent else 'PASS', place, len(undeclared), len(absent)))
    return 1 if undeclared or absent else 0


if __name__ == '__main__':
    raise SystemExit(main())
