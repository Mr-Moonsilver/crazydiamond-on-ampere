#!/usr/bin/env python3
"""Owner precheck configuration for q18a-alpha-next against the exact Step B parent (build e, 4b1eccb869db).

tools/step_b_precheck_conf.py's generation over this build: the bundle/runner/source/review pins, the self-run container, the work
root, the queue glob and pins, the exact step-b -> alpha-next runner diff, the declared seam tests and synthetic auth fixtures.
VM_OVERLAY is the source comparison against the parent bundle, so the exclusions are the generated metadata plus every overlay file
whose bytes differ from the parent's, recomputed from both SHA256SUMS. C0's rule for added overlay modules
(tools/alpha_c0_precheck_conf.py): the three modules the composition adds (C0's e3/stage_progress.py and e3/stage_wire.py, stage-1's
e3/handler_anatomy.py) have no parent bytes, so they count as moved; any other added or removed overlay file refuses.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
PARENT_BUILD = Path(json.loads((ROOT / 'tools/alpha_next_parent.json').read_text())['build'])
STEM = 'q18a_alpha_next'
ADDED = {'e3/handler_anatomy.py', 'e3/stage_progress.py', 'e3/stage_wire.py'}
METADATA = ('S5-OVERLAP-GENERATION.json', 'e3/native-overlap-readiness.json', 'e3/native-implementation-closure.json',
            'e3/decode-probe-evidence.json')
SEAMS = ('test_alpha_next_queue.py e3_alpha_next_box.py test_alpha_next_stopfix_matrix.py test_alpha_next_c0_ring.py '
         'test_alpha_next_peak_receipt.py test_e3_stopfix_partial_segment.py test_e3_stopfix_refused_open.py '
         'test_alpha_next_receipts.py test_alpha_next_receipts_mutants.py test_alpha_next_stage1_slow_rank.py '
         'test_alpha_next_path_qualification.py '
         'test_e3_w1_decode_driver_ordered.py test_native43_le_bounded_enqueue_stepped.py '
         'test_stage1_handler_anatomy.py test_stage1_anatomy_report.py test_e3_native_anatomy.py test_e3_native_lean_replies.py '
         'e3_native41_held_g_probe.py '
         'test_alpha_c0_queue.py test_alpha_c0_report.py test_e3_stage_local.py test_e3_stage_local_control.py '
         'test_e3_stage_local_qsa.py stage_net.py e3_alpha_c0_box.py e3_alpha_c0_probe.py '
         'test_step_b_queue.py test_e3_native_layer_groups.py e3_step_b_box.py e3_step_b_probe.py '
         'test_step_a_queue.py test_step_a_report.py test_step_a_torn_lines.py test_e3_native_qsa_fused.py e3_step_a_box.py e3_step_a_probe.py '
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
    assert b['schema'] == 'native43-build/1' and b['name'] == 'q18a-alpha-next'
    assert prev['manifest'] == b['parent_manifest'] and prev['runner_sha256'] == b['parent_runner_sha256']
    container = 'alpha-next-hub-cpu-' + b['manifest'][:12]
    if selfrun is not None:
        record = json.loads(Path(selfrun).read_text())
        assert record['container'] == container and record['bundle_manifest'] == b['manifest'], record['container']
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    text = text[text.index('FT='):]
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=prev['bundle'], RUNNER=b['runner'], PREV_RUNNER=prev['runner'],
                  CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'], VM_OVERLAY=prev['bundle'],
                  VM_OVERLAY_SUBDIR='native30_overlay', EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
                  EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
                  SELFRUN_CONTAINER=container, WORK_ROOT='<hub-dir>/agents/flash-next/tmp/alpha-next/pack/precheck-work',
                  QUEUE_NAME_GLOB=STEM + '[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + '=.*$', lambda _: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    diffs = difflib.ndiff(Path(prev['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allow = sorted({'^' + re.escape(line[2:]) + '$' for line in diffs if line.startswith(('- ', '+ '))})
    block = "RUNNER_DIFF_ALLOW=$(cat <<'ALPHA_NEXT_EXACT_DIFF'\n" + '\n'.join(allow) + '\nALPHA_NEXT_EXACT_DIFF\n)'
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda _: block, text, flags=re.S)
    assert n == 1
    names = [STEM + s for s in ('.draft.sh', '_declare.sh', '_settings.sh')]
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join('  ' + name + ':' + hashlib.sha256(
        (Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n' for name in names) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda _: pins, text, flags=re.S)
    assert n == 1
    ours, parent = rows(Path(b['overlay']) / 'SHA256SUMS'), rows(Path(prev['overlay']) / 'SHA256SUMS')
    # The three added overlay modules have no parent bytes to compare with, so they count as moved (C0's rule).
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
        handle.write('# alpha-next on the composed engine. PREV = exact Step B (%s); VM_OVERLAY is source comparison only.\n%s' % (prev['manifest'][:12], text))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path)
    a = p.parse_args()
    generate(a.build, a.out, a.selfrun)
