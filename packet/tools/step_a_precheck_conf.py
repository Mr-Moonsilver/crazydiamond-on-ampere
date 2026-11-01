#!/usr/bin/env python3
"""Owner precheck configuration for q18a-step-a against the exact alpha-perf build-d parent.

tools/alpha_perf_precheck_conf.py's generation over this build: the bundle/runner/source/review pins, the self-run
container, the work root, the queue glob and pins, the exact alpha-perf -> step-a runner diff, the declared seam tests and
synthetic auth fixtures. VM_OVERLAY is the source comparison against the parent bundle, so the exclusions are the generated
metadata plus every overlay file whose bytes differ from the parent's, recomputed from both SHA256SUMS.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
PARENT_BUILD = Path('<hub-dir>/agents/flash-next/tmp/alpha-perf/build-d/build.json')
STEM = 'q18a_step_a'
METADATA = ('S5-OVERLAP-GENERATION.json', 'e3/native-overlap-readiness.json', 'e3/native-implementation-closure.json',
            'e3/decode-probe-evidence.json')
SEAMS = ('test_step_a_queue.py test_step_a_report.py test_e3_native_qsa_fused.py e3_step_a_box.py e3_step_a_probe.py '
         'e3_native_qsa_fused_device_probe.py test_alpha_perf_queue.py test_alpha_m3_receipts.py e3_alpha_perf_box.py test_native41_progress_queue.py '
         'test_e3_native_reply_hol.py e3_native41_progress_probe.py e3_native41_progress_box.py test_native41_box_queue.py '
         'test_native41_autoarm.py e3_native41_auto_probe.py e3_native41_receipts_probe.py e3_native41_autoarm_box.py '
         'test_native41_unarmed_box.py e3_native41_unarmed_box.py test_native40_box_queue.py test_native40_unarmed_hold.py '
         'e3_native40_unarmed_probe.py test_native43_dense_boundary.py test_native43_le_fix_queue_deltas.py '
         'test_native43_engine_death.py test_native43_le_fatal_log.py test_native43_le_bounded_enqueue.py '
         'test_native39_trunk_queue.py ')


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        pin, name = line.split(None, 1)
        out[name.lstrip('*')] = pin
    return out


def generate(build, out, selfrun=None):
    b, prev = json.loads(Path(build).read_text()), json.loads(PARENT_BUILD.read_text())
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-step-a'
    assert prev['manifest'] == b['parent_manifest'] and prev['runner_sha256'] == b['parent_runner_sha256']
    container = 'step-a-hub-cpu-' + b['manifest'][:12]
    if selfrun is not None:
        record = json.loads(Path(selfrun).read_text())
        assert record['container'] == container and record['bundle_manifest'] == b['manifest'], record['container']
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    text = text[text.index('FT='):]
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=prev['bundle'], RUNNER=b['runner'], PREV_RUNNER=prev['runner'],
                  CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'], VM_OVERLAY=prev['bundle'],
                  VM_OVERLAY_SUBDIR='native30_overlay', EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
                  EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
                  SELFRUN_CONTAINER=container, WORK_ROOT='<hub-dir>/agents/flash-next/tmp/step-a/pack/precheck-work',
                  QUEUE_NAME_GLOB=STEM + '[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + '=.*$', lambda _: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    diffs = difflib.ndiff(Path(prev['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allow = sorted({'^' + re.escape(line[2:]) + '$' for line in diffs if line.startswith(('- ', '+ '))})
    block = "RUNNER_DIFF_ALLOW=$(cat <<'STEP_A_EXACT_DIFF'\n" + '\n'.join(allow) + '\nSTEP_A_EXACT_DIFF\n)'
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda _: block, text, flags=re.S)
    assert n == 1
    names = [STEM + s for s in ('.draft.sh', '_declare.sh', '_settings.sh')]
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join('  ' + name + ':' + hashlib.sha256(
        (Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n' for name in names) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda _: pins, text, flags=re.S)
    assert n == 1
    ours, parent = rows(Path(b['overlay']) / 'SHA256SUMS'), rows(Path(prev['overlay']) / 'SHA256SUMS')
    assert set(ours) == set(parent), sorted(set(ours) ^ set(parent))
    moved = sorted(k for k in ours if ours[k] != parent[k])
    excluded = sorted(set(METADATA) | set(moved))
    text, n = re.subn(r'^VM_OVERLAY_EXCLUSIONS=.*$', lambda _: 'VM_OVERLAY_EXCLUSIONS=(' + ' '.join(excluded) + ')', text, flags=re.M)
    assert n == 1
    fixtures = ['test_native39_trunk_queue.py'] + ['test_q18a_native%d_owner_run.py' % i for i in range(5, 18)]
    text = text.replace('FIXTURE_FILES=(\n', 'FIXTURE_FILES=(\n' + ''.join('  ' + x + ' # declared synthetic unit-test credential literal only\n' for x in fixtures))
    assert text.count('SEAM_TESTS=(') == 1
    text = text.replace('SEAM_TESTS=(', 'SEAM_TESTS=(' + SEAMS)
    with Path(out).open('x') as handle:
        handle.write('# Step A on the composed engine. PREV = exact alpha-perf build-d (d6b0f69ead0b); VM_OVERLAY is source comparison only.\n' + text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path)
    a = p.parse_args()
    generate(a.build, a.out, a.selfrun)
