#!/usr/bin/env python3
"""Rewindow the pinned native18 delivery in fresh local output; never launch it.

Copies the immutable, owner-checked bundle instead of editing the worktree.
Every output remains HOLD. --dry-run labels a placeholder window; it still
performs the complete CPU build/finalizer/owner verification. No box access,
GPU flags, auth/key reads, cleanup, engine control or implicit GO.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HELD_SHA = 'b0ef77bd16fa16f896cf64d8d49784db87ab452457374299e8edaca20f2b348b'
HELD = Path('<hub-dir>/agents/flash-next/tmp/q18a-native18-bundle-' + HELD_SHA[:12])
RUNNER_SHA = '05dafe559a427c5aa6d8ec6d5a897146d0296a415347d2703d457c6dc787a3bf'
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
QUEUE = 'runs/q18a_native18.draft.sh'
REVIEW = 'e3_builder/native_candidate_review.json'
BUILD_CLOSURE = 'e3_build_closure.json'
OLD_DATE = 'datetime.datetime(2026,10,29,7,0,tzinfo=datetime.timezone.utc)'
PIN_NAMES = {'H', 'BUNDLE', 'REVIEW', 'EXPECTED_SOURCES'}
TIMEOUT_S, KILL_MARGIN_S = 14400, 900


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_end(value):
    require(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', value)),
            'hard-end must be explicit UTC YYYY-MM-DDTHH:MM:SSZ')
    return dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)


def resolve_revision(root, tool_commit=None):
    """Resolve provenance before output/builds; archives use an explicit assertion."""
    if tool_commit is not None:
        require(bool(re.fullmatch('[0-9a-f]{40}', tool_commit)),
                '--tool-commit must be a full lowercase 40-hex exported commit')
        return {'commit': tool_commit, 'basis': 'caller-declared export commit; tool bytes hashed separately'}
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                            capture_output=True, text=True)
    require(result.returncode == 0 and Path(result.stdout.strip()).resolve() == root.resolve(),
            'export requires --tool-commit <full-40-hex-commit>; no output or build created')
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    require(result.returncode == 0 and bool(re.fullmatch('[0-9a-f]{40}', result.stdout.strip())),
            'cannot resolve checkout commit; no output or build created')
    return {'commit': result.stdout.strip(), 'basis': 'checkout HEAD; tool bytes hashed separately'}


def date_expression(end):
    return ('datetime.datetime(' + ','.join(str(x) for x in
            (end.year, end.month, end.day, end.hour, end.minute, end.second))
            + ',tzinfo=datetime.timezone.utc)')


def window_text(text, end, *, runner=False):
    require(text.count(OLD_DATE) == 1, 'expected exactly one held hard-end constant')
    require('14400+900' in text, 'held timeout arithmetic missing')
    text = text.replace(OLD_DATE, date_expression(end))
    if runner:
        old = 'Operator order28Oct night requires queue hard end by29Oct08:00CET, owner restores by08:30.'
        require(text.count(old) == 1, 'held runner window description drift')
        text = text.replace(old, 'Prepared hard end ' + end.strftime('%Y-%m-%dT%H:%M:%SZ')
                            + '; HOLD until operator authorizes this window. Owner restores production.')
    else:
        old = ('# Operator order 28 Oct night: production serving by 29 Oct 08:30 CET.\n'
               '# Worst-case owned queue + kill margin must end by 08:00 CET; the owner keeps\n'
               '# the remaining 30 minutes for production restoration. No workload/gate changes.')
        require(text.count(old) == 1, 'held queue window description drift')
        text = text.replace(old, '# Prepared hard end ' + end.strftime('%Y-%m-%dT%H:%M:%SZ')
                            + '; this file grants no operator window.\n'
                            '# Queue timeout 14400 s + kill margin 900 s unchanged.\n'
                            '# Production restoration and its reserved margin remain owner-controlled.')
        text = text.replace('queue worst case exceeds 08:00 CET',
                            'queue worst case exceeds ' + end.strftime('%Y-%m-%dT%H:%M:%SZ'))
    return text


def replace_pins(text, pins):
    lines = text.splitlines(keepends=True)
    seen = set()
    for node in reversed(ast.parse(text).body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in pins:
                require(name not in seen, 'duplicate runner pin')
                seen.add(name)
                lines[node.lineno - 1:node.end_lineno] = [name + '=' + pins[name] + '\n']
    require(seen == set(pins), 'runner pin missing')
    result = ''.join(lines)
    compile(result, 'prepared-owner-runner', 'exec')
    return result


def literal_pin(text, name):
    rows = [n.value for n in ast.parse(text).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(rows) == 1, 'runner literal pin missing')
    return ast.literal_eval(rows[0])


def inventory(root, digest=None):
    require(root.is_dir() and not root.is_symlink(), 'regular bundle directory required')
    if digest:
        require(sha(root / 'SHA256SUMS') == digest, 'held manifest pin')
    entries = list(root.rglob('*'))
    require(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in entries),
            'bundle symlink or special entry')
    rows = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split(maxsplit=1)
        require(re.fullmatch('[0-9a-f]{64}', value) and not Path(name).is_absolute()
                and '..' not in Path(name).parts and name not in rows, 'unsafe manifest entry')
        rows[name] = value
    require({str(p.relative_to(root)) for p in entries if p.is_file()} == set(rows) | {'SHA256SUMS'},
            'bundle inventory differs')
    require(all(sha(root / n) == v for n, v in rows.items()), 'bundle content checksum differs')
    require(not any(p.suffix == '.pyc' or '__pycache__' in p.parts for p in entries), 'pyc forbidden')
    return rows


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write('\n')


def build_env(bundle):
    env = {k: v for k, v in os.environ.items() if k not in
           ('VLLM_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ARM_AUTH_ENV_FILE',
            'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD')
           and not k.startswith(('Q17_TEST_', 'Q18A_', 'FLASH_S5_', 'S5_'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(map(str,
               (bundle / 'e3_builder', bundle / 'e2_builder', bundle))))
    return env


def guard_boundary_check(queue, runner, end):
    """Execute only actual extracted guard expressions at the three time edges."""
    q = queue.split("<<'PY_SLOT'", 1)[1].split('\n', 1)[1].split('\nPY_SLOT', 1)[0]
    # Queue's heredoc opener includes its shell failure branch, excluded above.
    qtree = ast.parse(q)
    rt = ast.parse(runner)
    fn = next(n for n in rt.body if isinstance(n, ast.FunctionDef) and n.name == 'check_slot')
    require('14400+900' in ast.get_source_segment(runner, fn), 'runner budget changed')
    outcomes = []
    from types import SimpleNamespace
    for delta, expected in ((-1, True), (0, True), (1, False)):
        now = end.timestamp() - TIMEOUT_S - KILL_MARGIN_S + delta
        ns = {'datetime': dt, 'time': SimpleNamespace(time=lambda: now), 'require': require}
        body = ast.Module(body=[n for n in qtree.body if not isinstance(n, ast.Import)], type_ignores=[])
        try:
            exec(compile(body, 'queue-guard', 'exec'), ns)
            queue_ok = True
        except AssertionError:
            queue_ok = False
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'runner-guard', 'exec'), ns)
        try:
            ns['check_slot']()
            runner_ok = True
        except ValueError:
            runner_ok = False
        require(queue_ok == runner_ok == expected, 'queue/runner deadline boundary differs')
        outcomes.append({'seconds_from_latest_entry': delta, 'queue_accepts': queue_ok,
                         'runner_accepts': runner_ok})
    return outcomes


def prepare(args):
    started = time.monotonic()
    end = utc_end(args.hard_end)
    revision = resolve_revision(ROOT, args.tool_commit)
    out = args.out.absolute()
    require(not out.exists() and not out.is_symlink(), 'fresh output directory required')
    require(not str(out).startswith(('/data/vllm/', '/home/<service-user>/s0/')), 'local hub output only')
    source = args.held.absolute()
    old_rows = inventory(source, HELD_SHA)
    template = ROOT / 'tools/q18a_native18_owner_run.py'
    require(sha(template) == RUNNER_SHA, 'held runner changed; new explicit baseline required')
    require(sha(args.base / 'SHA256SUMS') == BASE_SHA, 'base manifest pin')
    require(args.dry_run or end.timestamp() >= time.time() + TIMEOUT_S + KILL_MARGIN_S,
            'window already too short; use --dry-run for a placeholder')
    out.mkdir(mode=0o700, parents=True)
    commands = []

    def run(cmd, label, env=None):
        before = time.monotonic()
        with (out / (label + '.log')).open('xb') as handle:
            p = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=900)
        row = {'command': list(map(str, cmd)), 'rc': p.returncode,
               'wall_s': time.monotonic() - before, 'log': label + '.log'}
        commands.append(row)
        write_json(out / (label + '.command.json'), row)
        require(p.returncode == 0, 'self-run failed at ' + label + '; retained log/command record')
        print(label + ' PASS', flush=True)

    bundle = out / 'bundle-build'
    shutil.copytree(source, bundle, copy_function=shutil.copyfile)
    for p in [bundle, *bundle.rglob('*')]:
        p.chmod(0o700 if p.is_dir() else 0o600)
    (bundle / QUEUE).write_text(window_text((bundle / QUEUE).read_text(), end))
    review = json.loads((bundle / REVIEW).read_text())
    require(review['harness_source_shas'][QUEUE] == old_rows[QUEUE], 'queue review pin drift')
    review['harness_source_shas'][QUEUE] = sha(bundle / QUEUE)
    (bundle / REVIEW).write_text(json.dumps(review, sort_keys=True, indent=2) + '\n')
    closure = json.loads((bundle / BUILD_CLOSURE).read_text())
    require(closure['builder_sources']['native_candidate_review.json'] == old_rows[REVIEW],
            'builder review pin drift')
    closure['builder_sources']['native_candidate_review.json'] = sha(bundle / REVIEW)
    (bundle / BUILD_CLOSURE).write_text(json.dumps(closure, sort_keys=True, indent=2) + '\n')
    new_rows = {n: sha(bundle / n) for n in old_rows}
    changed = {n for n in old_rows if old_rows[n] != new_rows[n]}
    require(changed == {QUEUE, REVIEW, BUILD_CLOSURE}, 'unexpected bundle delta')
    (bundle / 'SHA256SUMS').write_text(''.join(v + '  ' + n + '\n' for n, v in sorted(new_rows.items())))
    digest = sha(bundle / 'SHA256SUMS')
    target = out / ('q18a-native18-bundle-' + digest[:12])
    bundle.rename(target)
    bundle = target
    inventory(bundle, digest)
    for p in [bundle, *bundle.rglob('*')]:
        p.chmod(0o500 if p.is_dir() else 0o400)
    env = build_env(bundle)
    run(['bash', '-n', str(bundle / QUEUE)], 'queue-syntax', env)
    raw, final = out / 'raw', out / 'final'
    run(['python3', '-B', str(bundle / 'e3_builder/make_s5_overlap.py'), str(args.base), str(raw),
         '--reference', str(bundle / 'e3_reference'), '--base-manifest-sha256', BASE_SHA,
         '--native-candidate'], 'generation', env)
    generation = json.loads((raw / 'S5-OVERLAP-GENERATION.json').read_text())
    require(generation['armable'] is False, 'raw must remain unarmed')
    expected = {str(p.relative_to(raw)): sha(p) for p in raw.rglob('*') if p.is_file()
                and str(p.relative_to(raw)) not in ('SHA256SUMS', 'S5-OVERLAP-GENERATION.json')}
    old_expected = literal_pin(template.read_text(), 'EXPECTED_SOURCES')
    require(set(expected) == set(old_expected), 'generated source inventory changed')
    source_changes = {n for n in expected if expected[n] != old_expected[n]}
    require(source_changes <= {'e3/native-implementation-closure.json', 'e3/decode-probe-evidence.json'},
            'runtime source changed in time-only revision')
    runner_text = window_text(template.read_text(), end, runner=True)
    runner_text = replace_pins(runner_text, {
        'H': "Path('/home/<service-user>/harness/" + digest[:12] + "')", 'BUNDLE': repr(digest),
        'REVIEW': repr(generation['implementation_review_sha256']),
        'EXPECTED_SOURCES': json.dumps(expected, sort_keys=True, indent=4)})
    runner = out / ('q18a-native18-owner-run-' + digest[:12] + '.py')
    runner.write_text(runner_text)
    runner.chmod(0o500)
    boundaries = guard_boundary_check((bundle / QUEUE).read_text(), runner_text, end)
    # Helper script does not select a runtime: refuse unless the daemon uses runc.
    info = subprocess.check_output(['sudo', '-n', 'docker', 'info', '--format', '{{.DefaultRuntime}}'],
                                   text=True).strip()
    require(info == 'runc', 'CPU helper requires runc default runtime')
    run(['sudo', '-n', 'bash', str(bundle / 'e3_builder/native/build_q17_helpers.sh'), IMAGE,
         str(raw / 'e3/native')], 'cpu-helper-build', env)
    log = (out / 'cpu-helper-build.log').read_text()
    names = re.findall(r'Q17_NATIVE_BUILD PASS retained-container=(\S+)', log)
    require(len(names) == 1, 'CPU build receipt missing')
    inspection = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', names[0]], text=True))[0]
    require(inspection['State']['ExitCode'] == 0 and not inspection['State']['Running']
            and inspection['HostConfig']['Runtime'] == 'runc'
            and not inspection['HostConfig'].get('DeviceRequests')
            and not inspection['HostConfig'].get('Devices'), 'CPU venue changed')
    run(['python3', '-B', str(bundle / 'e3_finalize_package.py'), str(raw), str(final)], 'finalize', env)
    # Execute real owner validators only, never its preflight/main/live guards.
    verify = out / 'verify_owner.py'
    verify.write_text('''import importlib.util,json,sys
from pathlib import Path
runner,raw,final,output=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('prepared_owner',runner)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
owner.verify_sources(raw,built=True)
owner.OVERLAY=final
green,manifest,receipt=owner.verify_final_build()
assert receipt['hardware_qualified'] is False and receipt['release_qualified'] is False
with output.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2)
print('OWNER_VERIFY PASS '+manifest)
''')
    run(['python3', '-B', str(verify), str(runner), str(raw), str(final), str(out / 'owner-verifier.json')],
        'owner-verifier', env)
    require(inventory(source, HELD_SHA) == old_rows and sha(template) == RUNNER_SHA,
            'held inputs changed')
    helpers = json.loads((final / 'e3/native/q17-helper-build.json').read_text())
    require(helpers['gpu_work'] is False and helpers['image'] == IMAGE, 'CPU helper identity')
    provenance = {
        'schema': 'native18-rewindow/1', 'verdict': 'PASS', 'delivery_state': 'HOLD', 'box_go': False,
        'dry_run': args.dry_run, 'hard_end_utc': args.hard_end,
        'latest_entry_utc': (end - dt.timedelta(seconds=TIMEOUT_S + KILL_MARGIN_S)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'timeout_s': TIMEOUT_S, 'kill_margin_s': KILL_MARGIN_S,
        'tool_sha256': sha(Path(__file__)), 'source_commit': revision['commit'],
        'source_commit_basis': revision['basis'],
        'held_bundle_sha256': HELD_SHA, 'held_runner_sha256': RUNNER_SHA, 'held_unchanged': True,
        'bundle': str(bundle), 'manifest_sha256': digest, 'files': len(new_rows),
        'runner': str(runner), 'runner_sha256': sha(runner), 'runner_bytes': runner.stat().st_size,
        'review_sha256': generation['implementation_review_sha256'],
        'launcher_sha256': expected['run-flash-s5-test.sh'], 'source_pins': expected,
        'bundle_delta': {n: {'old': old_rows[n], 'new': new_rows[n]} for n in sorted(changed)},
        'generated_metadata_delta': sorted(source_changes), 'runtime_code_equal': True,
        'guard_boundaries': boundaries, 'commands': commands,
        'cpu_container': names[0], 'cpu_runtime': inspection['HostConfig']['Runtime'],
        'device_requests': inspection['HostConfig'].get('DeviceRequests'),
        'network_mode': inspection['HostConfig']['NetworkMode'],
        'owner_verifier': json.loads((out / 'owner-verifier.json').read_text()),
        'wall_s': time.monotonic() - started, 'gpu_execution': False, 'box_action': False,
        'hardware_qualified': False, 'release_qualified': False,
    }
    write_json(out / 'provenance.json', provenance)
    (out / 'delivery-hold.md').write_text(
        '# Native18 window revision — HOLD\n\n'
        + ('**PLACEHOLDER DRY RUN. Not an operator window.**\n\n' if args.dry_run else '')
        + f"Hard end `{args.hard_end}`; latest queue entry `{provenance['latest_entry_utc']}`. "
        'Timeout 14400 s + kill margin 900 s unchanged. Owner must reserve restoration time outside this bound.\n\n'
        + f"Bundle `{bundle}`; {len(new_rows)} + SUMS; manifest `{digest}`.\n\n"
        + f"Runner `{runner}`; SHA `{sha(runner)}`; {runner.stat().st_size} bytes.\n\n"
        + f"Review `{generation['implementation_review_sha256']}`; launcher `{expected['run-flash-s5-test.sh']}`.\n\n"
        'Only queue hard-end/text and its review/build-closure hashes changed in the bundle. '
        'Runner changes: matching hard-end/text, H/BUNDLE/REVIEW/EXPECTED_SOURCES. '
        'Full source pins, command/rc/timing records and exact metadata delta are in provenance.json. '
        'Actual CPU helper compile, finalizer and owner verifier PASS; no GPU or box execution.\n\n'
        'Held b0ef77bd16fa and runner 6f3bbd1 remain untouched. Fresh local bundle and runner filenames; '
        'logical native18 run/overlay/key names remain reserved and must pass owner absence checks. '
        'Existing VM proof and strict regression apply to byte-identical runtime sources. '
        'No qualification or ARM is granted by this tool. Production stays owner-controlled.\n')
    print('REWINDOW PASS HOLD ' + digest, flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hard_end', help='UTC hard end including queue kill margin; excludes owner restore time')
    parser.add_argument('--out', required=True, type=Path, help='fresh hub-local scratch/delivery output')
    parser.add_argument('--dry-run', action='store_true', help='placeholder window, full CPU checks, no GO')
    parser.add_argument('--tool-commit', help='full exported commit for git-archive trees; recorded as caller-declared')
    parser.add_argument('--held', type=Path, default=HELD, help='same pinned immutable bundle at another local path')
    parser.add_argument('--base', type=Path, default=BASE, help='local frozen E2 base with pinned manifest')
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as exc:
        # Public diagnostics only; subprocess contents live in retained logs.
        print('REWINDOW FAIL ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
