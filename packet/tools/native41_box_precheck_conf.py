#!/usr/bin/env python3
"""Owner precheck configuration for native41 against the exact delivered native40 parent (rebuild-b)."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools')]


def generate(build, parent, out, container):
    b, prev = json.loads(build.read_text()), json.loads(parent.read_text())
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    text = text[text.index('FT='):]
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=prev['bundle'], RUNNER=b['runner'], PREV_RUNNER=prev['runner'],
        CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'], VM_OVERLAY=prev['bundle'],
        VM_OVERLAY_SUBDIR='native30_overlay', EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
        EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
        SELFRUN_CONTAINER=container, WORK_ROOT='<hub-dir>/agents/flash-next/tmp/native41-box-precheck-work',
        QUEUE_NAME_GLOB='q18a_native41_autoarm[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + '=.*$', lambda _: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    diffs = difflib.ndiff(Path(prev['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allow = sorted({'^' + re.escape(line[2:]) + '$' for line in diffs if line.startswith(('- ', '+ '))})
    block = "RUNNER_DIFF_ALLOW=$(cat <<'NATIVE41_EXACT_DIFF'\n" + '\n'.join(allow) + '\nNATIVE41_EXACT_DIFF\n)'
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda _: block, text, flags=re.S)
    assert n == 1
    names = ['q18a_native41_autoarm' + s for s in ('.draft.sh', '_declare.sh', '_settings.sh')]
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join('  ' + name + ':' + hashlib.sha256(
        (Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n' for name in names) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda _: pins, text, flags=re.S)
    assert n == 1
    # Native40's generated metadata plus native41's runtime sources (native41_autoarm_freeze.RUNTIME);
    # e3/native_auto_arm.py is new, so the pin check reports it as generated-only.
    from native41_autoarm_freeze import RUNTIME
    excluded = ['S5-OVERLAP-GENERATION.json', 'e3/native-overlap-readiness.json', 'e3/native-implementation-closure.json',
                'e3/decode-probe-evidence.json'] + sorted(RUNTIME)
    text, n = re.subn(r'^VM_OVERLAY_EXCLUSIONS=.*$', lambda _: 'VM_OVERLAY_EXCLUSIONS=(' + ' '.join(excluded) + ')', text, flags=re.M)
    assert n == 1
    fixtures = ['test_native39_trunk_queue.py'] + ['native35_tools/test_q18a_native%d_owner_run.py' % n for n in range(5, 18)]
    text = text.replace('FIXTURE_FILES=(\n', 'FIXTURE_FILES=(\n' + ''.join('  ' + n + ' # declared synthetic unit-test credential literal only\n' for n in fixtures))
    assert text.count('SEAM_TESTS=(') == 1
    text = text.replace('SEAM_TESTS=(', 'SEAM_TESTS=(test_native41_box_queue.py test_native41_autoarm.py e3_native41_auto_probe.py '
                        'e3_native41_window_probe.py e3_native41_autoarm_box.py test_native40_box_queue.py '
                        'test_native40_unarmed_hold.py e3_native40_unarmed_probe.py test_native39_trunk_queue.py ')
    with out.open('x') as handle:
        handle.write('# Native41 auto-arm. PREV = exact delivered native40 (rebuild-b); VM_OVERLAY is source comparison only.\n' + text)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('build', 'parent', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--container', required=True)
    a = p.parse_args()
    generate(a.build, a.parent, a.out, a.container)
