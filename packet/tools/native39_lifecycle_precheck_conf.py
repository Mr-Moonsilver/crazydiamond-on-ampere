#!/usr/bin/env python3
"""Native39 lifecycle owner precheck conf; PREV is the frozen native39 trunk held-b bundle and runner.

Derived from the native35 P-c conf (the newest committed conf; the trunk has none):
the bundle/runner/source/review pins, the self-run container, the work root, the
queue glob/pins, the exact trunk -> lifecycle runner diff, the lifecycle and trunk
seam tests, the trunk queue suite as a declared synthetic auth fixture, and the
VM-overlay comparison exclusions extended by every overlay file whose bytes differ
between the P-c overlay and this bundle's overlay (the trunk engine delta since
P-c; the lifecycle adds none, its overlay is the trunk's byte for byte).
SELFRUN_CONTAINER is derived from the candidate name exactly as the bundle self-run
names its container (<name without q18a->-hub-cpu-<manifest12>) and checked.
Source maps, the bundle-only allowance, credential patterns and key files are P-c's.
"""
import argparse, difflib, hashlib, json, re, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('<hub-dir>/agents/flash-next/tmp')
PARENT_BUILD = BASE / 'native39-trunk-held-b/build.json'
PC_OVERLAY = BASE / 'native35-p-held-c/q18a-native35-p-c-bundle-fe194172a6c8/native30_overlay'
PC_EXCLUSIONS = ('VM_OVERLAY_EXCLUSIONS=(S5-OVERLAP-GENERATION.json e3/native-overlap-readiness.json '
                 'e3/native-implementation-closure.json e3/decode-probe-evidence.json e3/lane_chunk.py e3/lane_dispatch.py '
                 'e3/native_startup.py e3/s5_startup.py)')
# Overlay files whose bytes differ between P-c (fe194172a6c8) and the trunk held-b overlay (b8f34594...),
# beyond P-c's own exclusions; recomputed from both SHA256SUMS below and asserted equal.
TRUNK_DELTA = ('core/SHA256SUMS', 'core/scheduler.py', 'e3/engine_driver.py', 'e3/executor_hooks.py',
               'e3/native_overlap_session.py', 'e3/probe_control.py', 'e3/reply_transport.py', 'e3/scheduler_hooks.py',
               'e3_sources/core.py', 'e3_sources/executor.py')
SEAMS = 'test_native39_lifecycle_queue.py test_native39_trunk_queue.py test_native39_trunk.py '
FIXTURE_ROW = '  test_native39_trunk_queue.py # synthetic CPU auth fixture only (VLLM_API_KEY/Bearer fixture-bearer-value)\n'


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        pin, name = line.split(None, 1)
        out[name.lstrip('*')] = pin
    return out


def selfrun_container(build):
    assert build['name'].startswith('q18a-')
    return build['name'][len('q18a-'):] + '-hub-cpu-' + build['manifest'][:12]


def generate(build, out, selfrun=None):
    b = json.loads(build.read_text())
    assert b['arm'] == 'LIFECYCLE' and b['name'] == 'q18a-native39-lifecycle'
    stem = b['name'].replace('-', '_')
    previous = json.loads(PARENT_BUILD.read_text())
    assert previous['manifest'] == b['parent_manifest'] and previous['runner_sha256'] == b['parent_runner_sha256']
    assert b['overlay_manifest'] == previous['overlay_manifest'] and b['review_sha256'] == previous['review_sha256']
    assert b['engine_changed'] == [] and b['engine_env_delta'] == {'FLASH_POOL_STATS': '200'}
    container = selfrun_container(b)
    assert container == 'native39-lifecycle-hub-cpu-' + b['manifest'][:12]
    if selfrun is not None:
        record = json.loads(Path(selfrun).read_text())
        assert record['container'] == container and record['bundle_manifest'] == b['manifest'], \
            (record['container'], container)
    pc, ours = rows(PC_OVERLAY / 'SHA256SUMS'), rows(Path(b['overlay']) / 'SHA256SUMS')
    assert set(pc) == set(ours)
    exclusions = re.findall(r'[^ ()]+', PC_EXCLUSIONS.split('=', 1)[1])
    delta = sorted(k for k in ours if pc[k] != ours[k] and k not in exclusions)
    assert delta == sorted(TRUNK_DELTA), delta
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=previous['bundle'], RUNNER=b['runner'], PREV_RUNNER=previous['runner'],
                  CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'],
                  EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
                  EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
                  SELFRUN_CONTAINER=container, WORK_ROOT=str(BASE / 'native39-lifecycle-owner-precheck-work'),
                  QUEUE_NAME_GLOB=stem + '[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + r'=.*$', lambda m: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    changed = difflib.ndiff(Path(previous['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allowed = sorted({'^' + re.escape(line[2:]) + '$' for line in changed if line.startswith(('- ', '+ '))})
    block = ("RUNNER_DIFF_ALLOW=$(cat <<'NATIVE39_LIFECYCLE_EXACT_DIFF'\n" + '\n'.join(allowed)
             + "\nNATIVE39_LIFECYCLE_EXACT_DIFF\n)")
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda m: block, text, flags=re.S)
    assert n == 1
    names = [stem + s for s in ('.draft.sh', '_declare.sh', '_settings.sh')]
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join(
        '  ' + name + ':' + hashlib.sha256((Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n'
        for name in names) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda m: pins, text, flags=re.S)
    assert n == 1
    text, n = re.subn(r'SEAM_TESTS=\(', 'SEAM_TESTS=(' + SEAMS, text)
    assert n == 1
    assert text.count('FIXTURE_FILES=(\n') == 1
    text = text.replace('FIXTURE_FILES=(\n', 'FIXTURE_FILES=(\n' + FIXTURE_ROW)
    assert text.count(PC_EXCLUSIONS) == 1
    text = text.replace(PC_EXCLUSIONS, PC_EXCLUSIONS[:-1] + ' ' + ' '.join(TRUNK_DELTA) + ')')
    out.write_text(
        '# Native39 lifecycle (trunk engine byte for byte, lifecycle/leak queue); PREV is the frozen native39 trunk held-b bundle and runner.\n'
        '# Engine delta vs PREV: none in code (overlay, engine sources, review and compiled libraries byte-identical); one declared\n'
        '# telemetry env delta in the settings, FLASH_POOL_STATS=200 (pool counters for the pressure-lifetime oracle; no speed claim). VM-overlay exclusions\n'
        '# add the trunk engine delta since P-c (%s).\n'
        '# The lifecycle adds its queue under fresh names, the lifecycle window, suite, fixture leaves and tools.\n'
        % ' '.join(TRUNK_DELTA) + text)
    print(json.dumps(dict(out=str(out), selfrun_container=container, sha256=hashlib.sha256(out.read_bytes()).hexdigest()),
                     sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path, help='the bundle self-run record; its container must equal SELFRUN_CONTAINER')
    a = p.parse_args()
    generate(a.build, a.out, a.selfrun)
