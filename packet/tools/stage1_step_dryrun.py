#!/usr/bin/env python3
"""The preconditions tools/stage1_step_gates.sh --dry-run checks that need python: one read-only check per call, exit 0 when it holds. No container, no build, no write.
  spec FILE    the spec's source commit is an ancestor of HEAD and only tools/ moved since (the spec's own rule), its candidate and queue are this family's
  parent       every record the parent pin names is there with its sha256; the bundle and its overlay manifests match the pin
  tools GATES  every tools/X named by the chain's executable lines exists; the shared ones are the declared ones
  modules M..  each test module exists under harness/
  swap         the superseded map is the declared four ids
  pins PS      the queue pins are the placeholder (or PS, the PINS_SUPERSEDE reason, is given); the negative pins carry a status
  image        the docker image the self-runs use is present (docker image inspect: read-only)
  disk OUT     at least DISK_MIN_GIB free on OUT's filesystem
"""
import json
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
DISK_MIN_GIB = 40
SHARED = {'alpha_c1_flaky.py', 'native43_bundle_selfrun.py', 'native35_compile_proof.py', 'dryrun-record.sh', 'strict_parallel.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def check_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec['candidate'] == 'q18a-stage1-step' and spec['label'] == 'stage1-step', spec['candidate']
    assert spec['queue'] == ['q18a_stage1_step.draft.sh', 'q18a_stage1_step_declare.sh', 'q18a_stage1_step_settings.sh'], spec['queue']
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', spec['source_commit'], 'HEAD'], cwd=ROOT)
    moved = [n for n in git('diff', '--name-only', spec['source_commit'], 'HEAD').split() if not n.startswith('tools/')]
    assert not moved, ('moved after the spec outside tools/', moved)
    assert spec['expected_tests'], 'a pre-queue spec cannot build'


def check_parent():
    import stage1_step_parent_pin as pin
    p = pin.load()
    for key, value in (('build', p['build_sha256']), ('freeze', p['freeze_sha256']), ('steps', p['steps_sha256']), ('ready', p['ready_sha256'])):
        assert sha(p[key]) == value, key + ' changed'
    assert sha(Path(p['engine_selfrun']) / 'result.json') == p['engine_selfrun_sha256'], 'engine self-run changed'
    build = json.loads(Path(p['build']).read_text())
    assert sha(Path(build['bundle']) / 'SHA256SUMS') == p['manifest'], 'parent bundle manifest'
    assert sha(Path(build['bundle']) / 'native30_overlay/SHA256SUMS') == p['overlay_manifest'], 'parent overlay manifest'
    assert sha(build['runner']) == p['runner_sha256'], 'parent runner'


def check_tools(gates):
    lines = [l for l in Path(gates).read_text().splitlines() if not l.lstrip().startswith('#')]
    called = set(re.findall(r'tools/([A-Za-z0-9_.-]+\.(?:py|sh))', '\n'.join(lines)))
    other = sorted(n for n in called if not n.startswith(('stage1_step_', 'build_stage1_step')) and n not in SHARED and n != 'hub-slot.sh')
    assert not other or other == ['native-precheck.sh'], ('undeclared chain tool', other)
    for name in called - {'native-precheck.sh', 'hub-slot.sh'}:
        assert (ROOT / 'tools' / name).is_file(), name
    assert (ROOT / 'tools/owner-precheck/native-precheck.sh').is_file() and (ROOT / 'deploy/setcfg.py').is_file()


def check_modules(names):
    missing = [n for n in names if not (ROOT / 'harness' / (n + '.py')).is_file()]
    assert not missing, missing


def check_swap():
    import stage1_step_robust_swap as swap
    swap.check_map()


def check_pins(supersede):
    pins = json.loads((ROOT / 'tools/stage1_step_queue_pins.json').read_text())
    assert supersede or str(pins['status']).startswith('PLACEHOLDER'), 'queue pins filled and PINS_SUPERSEDE not given'
    negative = json.loads((ROOT / 'tools/stage1_step_negative_pins.json').read_text())
    assert str(negative['status']).startswith(('AWAITING-OBSERVATION', 'OBSERVED')), negative['status']


def check_image():
    import stage1_step_engine_selfrun as selfrun
    subprocess.check_call(['sudo', '-n', 'docker', 'image', 'inspect', '--format', '{{.Id}}', selfrun.base.an.IMAGE], stdout=subprocess.DEVNULL)


def check_disk(out):
    place = Path(out)
    while not place.exists():
        place = place.parent
    free = shutil.disk_usage(place).free / 2 ** 30
    assert free >= DISK_MIN_GIB, ('free GiB', round(free, 1), DISK_MIN_GIB)


CHECKS = dict(spec=lambda a: check_spec(a[0]), parent=lambda a: check_parent(), tools=lambda a: check_tools(a[0]), modules=check_modules,
              swap=lambda a: check_swap(), pins=lambda a: check_pins(a[0] if a else ''), image=lambda a: check_image(), disk=lambda a: check_disk(a[0]))

if __name__ == '__main__':
    CHECKS[sys.argv[1]](sys.argv[2:])
