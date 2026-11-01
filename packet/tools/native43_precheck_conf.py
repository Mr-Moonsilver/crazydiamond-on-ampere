#!/usr/bin/env python3
"""Native43 owner precheck conf; PREV is the frozen native39 trunk held-b bundle and runner.

Derived from the native35 P-c conf (as the native39 lifecycle conf was): the bundle/runner/
source/review pins, the self-run container, the work root, the queue glob/pins, the exact
held-b -> candidate runner diff, the spec's seam tests and declared synthetic auth fixtures,
and the VM-overlay comparison exclusions extended by every overlay file whose bytes differ
between the P-c overlay and this bundle's overlay: the trunk engine delta since P-c plus
the candidate's moved overlay files (none in the held-b dry run), recomputed from both
SHA256SUMS and asserted equal.  SELFRUN_CONTAINER is derived from the candidate name exactly
as the bundle self-run names its container (<name without q18a->-hub-cpu-<manifest12>) and
checked against the self-run record.  Source maps, the bundle-only allowance, credential
patterns and key files are P-c's.
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
TRUNK_DELTA = ('core/SHA256SUMS', 'core/scheduler.py', 'e3/engine_driver.py', 'e3/executor_hooks.py',
               'e3/native_overlap_session.py', 'e3/probe_control.py', 'e3/reply_transport.py', 'e3/scheduler_hooks.py',
               'e3_sources/core.py', 'e3_sources/executor.py')


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        pin, name = line.split(None, 1)
        out[name.lstrip('*')] = pin
    return out


def selfrun_container(build):
    assert build['name'].startswith('q18a-')
    return build['name'][len('q18a-'):] + '-hub-cpu-' + build['manifest'][:12]


def generate(spec_path, build, out, selfrun=None):
    spec = json.loads(Path(spec_path).read_text())
    b = json.loads(build.read_text())
    assert b['schema'] == 'native43-build/1' and b['name'] == spec['candidate'] and b['arm'] == spec['arm']
    assert b['spec_sha256'] == hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(), 'build used another spec'
    stem = b['name'].replace('-', '_')
    previous = json.loads(PARENT_BUILD.read_text())
    assert previous['manifest'] == b['parent_manifest'] and previous['runner_sha256'] == b['parent_runner_sha256']
    assert sorted(b['engine_changed']) == sorted(spec['engine_changed'])
    container = selfrun_container(b)
    if selfrun is not None:
        record = json.loads(Path(selfrun).read_text())
        assert record['container'] == container and record['bundle_manifest'] == b['manifest'], \
            (record['container'], container)
    pc, ours, heldb = (rows(PC_OVERLAY / 'SHA256SUMS'), rows(Path(b['overlay']) / 'SHA256SUMS'),
                       rows(Path(previous['overlay']) / 'SHA256SUMS'))
    assert set(pc) == set(ours) == set(heldb)
    exclusions = re.findall(r'[^ ()]+', PC_EXCLUSIONS.split('=', 1)[1])
    moved = sorted(k for k in ours if ours[k] != heldb[k])
    extra = sorted(set(moved) - set(exclusions) - set(TRUNK_DELTA))
    delta = sorted(k for k in ours if pc[k] != ours[k] and k not in exclusions)
    assert delta == sorted(set(TRUNK_DELTA) | set(extra)), (delta, extra)
    text = (ROOT / 'tools/owner-precheck/native-precheck.conf.native35-p-c').read_text()
    fields = dict(BUNDLE=b['bundle'], PREV_BUNDLE=previous['bundle'], RUNNER=b['runner'], PREV_RUNNER=previous['runner'],
                  CODEX_WT=str(ROOT), CODEX_COMMIT=b['source'], GENERATION=b['overlay'],
                  EXPECTED_MANIFEST=b['manifest'], EXPECTED_RUNNER_SHA=b['runner_sha256'],
                  EXPECTED_RUNNER_SIZE=str(b['runner_size']), EXPECTED_REVIEW_PIN=b['review_sha256'],
                  SELFRUN_CONTAINER=container, WORK_ROOT=str(BASE / ('native43-%s-owner-precheck-work' % spec['label'])),
                  QUEUE_NAME_GLOB=stem + '[._]*')
    for key, value in fields.items():
        text, n = re.subn(r'(?m)^' + key + r'=.*$', lambda m: key + '=' + shlex.quote(value), text)
        assert n == 1, key
    changed = difflib.ndiff(Path(previous['runner']).read_text().splitlines(), Path(b['runner']).read_text().splitlines())
    allowed = sorted({'^' + re.escape(line[2:]) + '$' for line in changed if line.startswith(('- ', '+ '))})
    block = ("RUNNER_DIFF_ALLOW=$(cat <<'NATIVE43_EXACT_DIFF'\n" + '\n'.join(allowed) + "\nNATIVE43_EXACT_DIFF\n)")
    text, n = re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)", lambda m: block, text, flags=re.S)
    assert n == 1
    pins = 'EXPECTED_QUEUE_SHAS=(\n' + ''.join(
        '  ' + name + ':' + hashlib.sha256((Path(b['bundle']) / 'runs' / name).read_bytes()).hexdigest() + '\n'
        for name in spec['queue']) + ')'
    text, n = re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)', lambda m: pins, text, flags=re.S)
    assert n == 1
    text, n = re.subn(r'SEAM_TESTS=\(', 'SEAM_TESTS=(' + spec['precheck']['seams'], text)
    assert n == 1
    assert text.count('FIXTURE_FILES=(\n') == 1
    text = text.replace('FIXTURE_FILES=(\n', 'FIXTURE_FILES=(\n' + ''.join(spec['precheck']['fixture_rows']))
    assert text.count(PC_EXCLUSIONS) == 1
    text = text.replace(PC_EXCLUSIONS, PC_EXCLUSIONS[:-1] + ' ' + ' '.join(TRUNK_DELTA + tuple(extra)) + ')')
    header = ''.join('# ' + line + '\n' for line in spec['precheck']['header'])
    header += '# VM-overlay exclusions add the trunk engine delta since P-c (%s)%s.\n' % (
        ' '.join(TRUNK_DELTA), (' and this candidate\'s moved overlay files (%s)' % ' '.join(extra)) if extra else '')
    out.write_text(header + text)
    print(json.dumps(dict(out=str(out), selfrun_container=container, extra_exclusions=extra,
                          sha256=hashlib.sha256(out.read_bytes()).hexdigest()), sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path, help='the bundle self-run record; its container must equal SELFRUN_CONTAINER')
    a = p.parse_args()
    generate(a.spec, a.build, a.out, a.selfrun)
