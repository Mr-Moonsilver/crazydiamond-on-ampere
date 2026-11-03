#!/usr/bin/env python3
"""Owner precheck configuration for q18a-stage1-step against the exact c1b parent (tools/stage1_step_parent.json).

tools/attn_meta_precheck_conf_b.py's generation by name over this build: the bundle/runner/source/review pins, the self-run container, the work root, the queue glob and pins, the
exact c1b -> stage1-step runner diff, the declared seam tests (c1b's, read from its text, plus this queue's CPU suites and arm modules) and synthetic auth fixtures. VM_OVERLAY is the
source comparison against the parent bundle, so the exclusions are the generated metadata plus every overlay file whose bytes differ from the parent's, recomputed from both SHA256SUMS. Exactly
e3/native_fifo.py is added and nothing removed (any other set refuses). The parent is read through stage1_step_parent_pin.load() (rc 3 while a placeholder stands).
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import stage1_step_parent_pin as parent_pin  # noqa: E402

PARENT_BUILD = Path(parent_pin.load()['build'])
STEM = 'q18a_stage1_step'
ADDED = set()
METADATA = ('S5-OVERLAP-GENERATION.json', 'e3/native-overlap-readiness.json', 'e3/native-implementation-closure.json',
            'e3/decode-probe-evidence.json')


def _c1_seams():
    """c1's SEAMS (its module, imported unchanged: it reads c1's own pinned parent)."""
    import alpha_next_c1b_precheck_conf as c1
    return c1.SEAMS


# stage1-step's CPU suites and arm modules, then c1b's (which end with a space; its list includes alpha-next's). Their leaves
# (harness/stage1_step_queue_fixture_leaves.json) are harness files, so build_native43.stage bundles them and FIXTURE_LEAVES_REGEX
# ('fixture[-_]leaves.*\.json$') matches them.
SEAMS = ('native43_engine_death_h.py alpha_next_stop_race_h.py test_native43_engine_death_h.py test_alpha_next_stop_race_h.py '
         'e3_stage1_step_box.py stage1_step_arms.py stage1_step_gate.py stage1_step_read.py test_stage1_step_box.py test_test_stage1_step_read.py test_stage1_step_queue.py '
         'test_stage1_step_robust_swap.py stage1_step_robust_swap.py ' + _c1_seams())
# The engine suites are the engine builder's (tools/stage1_step_engine_suites.json): their mocked seams count too.
SEAMS = ' '.join(n + '.py' for n in json.loads((ROOT / 'tools/stage1_step_engine_suites.json').read_text())['engine_suites']) + ' ' + SEAMS if json.loads(
    (ROOT / 'tools/stage1_step_engine_suites.json').read_text())['engine_suites'] else SEAMS
assert SEAMS.endswith(' ') and len(SEAMS.split()) == len(set(SEAMS.split())), 'seam list'



def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        pin, name = line.split(None, 1)
        out[name.lstrip('*')] = pin
    return out


def generate(build, out, selfrun=None):
    b, prev = json.loads(Path(build).read_text()), json.loads(PARENT_BUILD.read_text())
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-stage1-step' and prev['name'] == 'q18a-alpha-next-c1b'
    assert prev['manifest'] == b['parent_manifest'] and prev['runner_sha256'] == b['parent_runner_sha256']
    container = 'stage1-step-hub-cpu-' + b['manifest'][:12]
    if selfrun is not None:
        record = json.loads(Path(selfrun).read_text())
        assert record['container'] == container and record['bundle_manifest'] == b['manifest'], record['container']
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    text = text[text.index('FT='):]
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=prev['bundle'], RUNNER=b['runner'], PREV_RUNNER=prev['runner'],
                  CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'], VM_OVERLAY=prev['bundle'],
                  VM_OVERLAY_SUBDIR='native30_overlay', EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
                  EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
                  SELFRUN_CONTAINER=container, WORK_ROOT='<hub-dir>/agents/flash-next/tmp/stage1-step/pack/precheck-work',
                  QUEUE_NAME_GLOB=STEM + '[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + '=.*$', lambda _: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    diffs = difflib.ndiff(Path(prev['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allow = sorted({'^' + re.escape(line[2:]) + '$' for line in diffs if line.startswith(('- ', '+ '))})
    block = "RUNNER_DIFF_ALLOW=$(cat <<'STAGE1_STEP_EXACT_DIFF'\n" + '\n'.join(allow) + '\nSTAGE1_STEP_EXACT_DIFF\n)'
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda _: block, text, flags=re.S)
    assert n == 1
    names = [STEM + s for s in ('.draft.sh', '_declare.sh', '_settings.sh')]
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join('  ' + name + ':' + hashlib.sha256(
        (Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n' for name in names) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda _: pins, text, flags=re.S)
    assert n == 1
    ours, parent = rows(Path(b['overlay']) / 'SHA256SUMS'), rows(Path(prev['overlay']) / 'SHA256SUMS')
    assert not set(parent) - set(ours) and set(ours) - set(parent) == ADDED, sorted(set(ours) ^ set(parent))
    moved = sorted(k for k in ours if ours[k] != parent.get(k))
    excluded = sorted(set(METADATA) | set(moved))
    text, n = re.subn(r'^VM_OVERLAY_EXCLUSIONS=.*$', lambda _: 'VM_OVERLAY_EXCLUSIONS=(' + ' '.join(excluded) + ')', text, flags=re.M)
    assert n == 1
    fixtures = ['test_native39_trunk_queue.py'] + ['test_q18a_native%d_owner_run.py' % i for i in range(5, 18)]
    text = text.replace('FIXTURE_FILES=(\n', 'FIXTURE_FILES=(\n' + ''.join('  ' + x + ' # declared synthetic unit-test credential literal only\n' for x in fixtures))
    assert text.count('SEAM_TESTS=(') == 1
    text = text.replace('SEAM_TESTS=(', 'SEAM_TESTS=(' + SEAMS)
    with Path(out).open('x') as handle:
        handle.write('# stage1-step on the composed engine. PREV = exact alpha-next-c1b (%s); VM_OVERLAY is source comparison only.\n%s' % (prev['manifest'][:12], text))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path)
    a = p.parse_args()
    generate(a.build, a.out, a.selfrun)
