#!/usr/bin/env python3
"""Build a fresh, local, immutable native28 host-operations HOLD candidate."""

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('<hub-dir>/agents/flash-next/tmp/native27-held-a/q18a-native27-bundle-cbb17da8e41e')
OLD_SHA = 'cbb17da8e41e97993b467b107e2505ba2ca27e3d90c1302b0c3ccb3569366d34'
OLD_FINAL = Path('<hub-dir>/agents/flash-next/tmp/native27-held-a/final')
BASE = Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA = 'da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
RUNNER = ROOT / 'tools/q18a_native18_owner_run.py'
PLACEHOLDER = 'datetime.datetime(2030,1,2,7,0,tzinfo=datetime.timezone.utc)'
DEFAULT_HARD_END = '2030-01-02T07:00:00Z'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


def dump(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


def inventory(root):
    rows = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        need(re.fullmatch('[0-9a-f]{64}', digest) and name not in rows, 'manifest row')
        rows[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual == set(rows) | {'SHA256SUMS'}, 'bundle inventory')
    need(all(not p.is_symlink() for p in root.rglob('*')), 'bundle symlink')
    need(all(sha(root / name) == digest for name, digest in rows.items()), 'bundle hashes')
    return rows


def run(cmd, record, *, env=None):
    start = time.monotonic()
    with (record.parent / (record.stem + '.log')).open('xb') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, timeout=900)
    dump(record, {'command': list(map(str, cmd)), 'rc': result.returncode,
                  'wall_s': time.monotonic() - start,
                  'log': record.stem + '.log'})
    need(result.returncode == 0, 'CPU-only step failed: ' + record.stem)


def replace_assignments(source, values):
    lines = source.splitlines(keepends=True)
    found = set()
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in values:
                need(name not in found, 'duplicate runner pin')
                found.add(name)
                lines[node.lineno - 1:node.end_lineno] = [name + '=' + values[name] + '\n']
    need(found == set(values), 'runner pin missing')
    result = ''.join(lines)
    compile(result, 'native20-runner', 'exec')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--regression-log', required=True, type=Path)
    p.add_argument('--regression-record', required=True, type=Path,
                   help='strict native28 record pinned to the committed host and runtime source')
    p.add_argument('--coordinator-result', required=True, type=Path)
    p.add_argument('--coordinator-input', required=True, type=Path)
    p.add_argument('--vm-result', required=True, type=Path)
    p.add_argument('--vm-input', required=True, type=Path)
    p.add_argument('--audited-input', required=True, type=Path)
    p.add_argument('--audited-selfrun', required=True, type=Path)
    p.add_argument('--graph-vm-result', required=True, type=Path)
    p.add_argument('--graph-vm-input', required=True, type=Path)
    p.add_argument('--graph-hub-result', required=True, type=Path)
    p.add_argument('--graph-hub-input', required=True, type=Path)
    p.add_argument('--census-vm-result', required=True, type=Path)
    p.add_argument('--census-vm-input', required=True, type=Path)
    p.add_argument('--census-hub-result', required=True, type=Path)
    p.add_argument('--census-hub-input', required=True, type=Path)
    p.add_argument('--host-selfrun-record', required=True, type=Path)
    p.add_argument('--profile-vm-result', required=True, type=Path)
    p.add_argument('--profile-vm-input', required=True, type=Path)
    p.add_argument('--profile-hub-result', required=True, type=Path)
    p.add_argument('--profile-hub-input', required=True, type=Path)
    p.add_argument('--image-test-log', type=Path)
    p.add_argument('--image-test-container')
    p.add_argument('--disable-audit', action='store_true',
                   help='build the frozen native19 runtime with no selection capture')
    p.add_argument('--hard-end', default=DEFAULT_HARD_END,
                   help='UTC queue/runner hard end; default is a nonlaunch placeholder')
    a = p.parse_args()
    a.disable_audit = True  # Native21 keeps the frozen audit-free native19 route.
    variant = 'native28'
    need(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', a.hard_end),
         'hard end must be an exact UTC second')
    end = datetime.datetime.fromisoformat(a.hard_end.replace('Z', '+00:00'))
    end_expr = (f'datetime.datetime({end.year},{end.month},{end.day},'
                f'{end.hour},{end.minute},{end.second},tzinfo=datetime.timezone.utc)')
    is_placeholder = a.hard_end == DEFAULT_HARD_END
    out = a.out.absolute()
    need(not out.exists() and not out.is_symlink(), 'fresh output required')
    need(sha(OLD / 'SHA256SUMS') == OLD_SHA and inventory(OLD), 'frozen source bundle')
    need(OLD_FINAL.is_dir(), 'frozen native25 finalized runtime absent')
    need(sha(BASE / 'SHA256SUMS') == BASE_SHA, 'frozen E2 base')
    vm = json.loads(a.vm_result.read_text())
    vm_probe = a.vm_result.parent / 'probe.json'
    vm_rows = inventory(a.vm_input)
    audit_rows = inventory(a.audited_input)
    audit = json.loads(a.audited_selfrun.read_text())
    audit_sha = '0353e33b95a0d5ff89efa63ad0b5bdcb402a00ac7793dcb734d0f704f380459f'
    need(sha(a.audited_input / 'SHA256SUMS') == audit_sha
         and audit.get('manifest_sha256') == audit_sha
         and audit.get('verdict') == 'PASS' and audit.get('rc') == 0
         and audit.get('gpu_device_requests') == 0
         and audit.get('paused_predicate_audit', {}).get('verdict') == 'PASS'
         and sha(a.audited_selfrun.parent / 'probe.json') == audit.get('probe_sha256'),
         'exact audited hub CPU self-run absent')
    # E and F have the same mounted overlay and GPU probe. F's only VM wrapper
    # change is the fresh container name; its CPU self-run adds the requested
    # paused-predicate audit. A successful E GPU run can therefore be paired
    # with F's exact-input audit without repeating identical device work.
    compared = sorted(name for name in audit_rows
                      if name.startswith(('overlay/', 'proof/'))
                      and name != 'proof/import-graph.json')
    need(all(vm_rows.get(name) == audit_rows[name] for name in compared)
         and {name for name in vm_rows if name.startswith(('overlay/', 'proof/'))
              and name != 'proof/import-graph.json'} == set(compared),
         'VM and audited mounted payloads differ')
    vm_graph = json.loads((a.vm_input / 'proof/import-graph.json').read_text())
    audit_graph = json.loads((a.audited_input / 'proof/import-graph.json').read_text())
    vm_graph.pop('overlay', None)
    audit_graph.pop('overlay', None)
    need(vm_graph == audit_graph, 'VM and audited import graphs differ')
    vm_wrapper = (a.vm_input / 'run_gpu.py').read_text()
    audit_wrapper = (a.audited_input / 'run_gpu.py').read_text()
    need(vm_wrapper.replace('flashnext-native23-row4-pause-e',
                            'flashnext-native23-row4-pause-f') == audit_wrapper,
         'VM and audited GPU wrappers differ beyond container name')
    need(vm.get('verdict') == 'PASS' and vm.get('rc') == 0
         and vm.get('process_rc') == 0 and vm.get('gpu_execution') is True
         and vm.get('cleanup_ok') is True and vm.get('container_retained') is True
         and vm.get('failures') == [] and vm.get('missing') == []
         and vm.get('manifest_sha256') == sha(a.vm_input / 'SHA256SUMS')
         and vm.get('manifest_sha256') in (
             '22046b2347e73f6ae7d9db5e41a50e1677ad9748c22d903ac3b2fc612b32d18d',
             audit_sha)
         and vm_probe.is_file() and sha(vm_probe) == vm.get('probe_sha256')
         and json.loads(vm_probe.read_text()).get('verdict') == 'PASS',
         'native23 pause/resume GPU proof absent')
    graph_rows = inventory(a.graph_vm_input)
    graph_hub_rows = inventory(a.graph_hub_input)
    graph_input_sha = '963a7ef3c34403a0ef20b3782169d952a7b592cbe159163b88b32cde2083a59a'
    need(sha(a.graph_vm_input / 'SHA256SUMS') == graph_input_sha
         and graph_rows == graph_hub_rows
         and sha(a.graph_hub_input / 'SHA256SUMS') == graph_input_sha
         and graph_rows['proof/native24_graph_fixture_leaves.json']
             == sha(ROOT / 'harness/native24_graph_fixture_leaves.json'),
         'native24 graph proof input or fixture leaves differ')
    graph_runtime_delta = {name for name in graph_rows
                           if name.startswith('overlay/')
                           and graph_rows.get(name) != vm_rows.get(name)}
    need(graph_runtime_delta == {'overlay/e3/worker_hooks.py'}
         and {name for name in vm_rows if name.startswith('overlay/')}
             == {name for name in graph_rows if name.startswith('overlay/')},
         'native24 VM overlay differs from native23 beyond worker_hooks')
    graph_hub = json.loads(a.graph_hub_result.read_text())
    graph_hub_probe_path = a.graph_hub_result.parent / 'probe.json'
    need(graph_hub.get('verdict') == 'PASS' and graph_hub.get('rc') == 0
         and graph_hub.get('manifest_sha256') == graph_input_sha
         and graph_hub.get('gpu_device_requests') == 0
         and graph_hub.get('gpu_stream_scope_audit', {}).get('verdict') == 'PASS'
         and graph_hub_probe_path.is_file()
         and sha(graph_hub_probe_path) == graph_hub.get('probe_sha256')
         and json.loads(graph_hub_probe_path.read_text())['sections']['cpu_shape']['verdict'] == 'PASS',
         'exact native24 hub CPU-shaped self-run absent')
    graph_vm = json.loads(a.graph_vm_result.read_text())
    graph_vm_probe_path = a.graph_vm_result.parent / 'probe.json'
    need(graph_vm.get('verdict') == 'PASS' and graph_vm.get('rc') == 0
         and graph_vm.get('process_rc') == 0
         and graph_vm.get('manifest_sha256') == graph_input_sha
         and graph_vm.get('gpu_execution') is True
         and graph_vm.get('cleanup_ok') is True
         and graph_vm.get('container_retained') is True
         and graph_vm.get('failures') == [] and graph_vm.get('missing') == []
         and graph_vm_probe_path.is_file()
         and sha(graph_vm_probe_path) == graph_vm.get('probe_sha256'),
         'native24 actual CUDA graph proof absent')
    graph_probe = json.loads(graph_vm_probe_path.read_text())
    need(graph_probe.get('verdict') == 'PASS'
         and graph_probe.get('sections', {}).get('gpu_graph_bind', {}).get('real_observer_bound') is True
         and graph_probe['sections']['gpu_graph_bind'].get('raw_handle_stable') is True
         and graph_probe['sections']['gpu_graph_bind'].get('capture_stream_checks') == 1,
         'native24 real replay observer bind unproved')
    census_rows = inventory(a.census_vm_input)
    census_hub_rows = inventory(a.census_hub_input)
    census_input_sha = '1d06581ff544882397a4036cdce66b7018c844935755acb40587e38947c5df2a'
    need(sha(a.census_vm_input / 'SHA256SUMS') == census_input_sha
         and census_rows == census_hub_rows
         and sha(a.census_hub_input / 'SHA256SUMS') == census_input_sha
         and census_rows['proof/native25_census_fixture_leaves.json']
             == sha(ROOT / 'harness/native25_census_fixture_leaves.json'),
         'native25 census proof input or fixture leaves differ')
    census_delta = {name for name in census_rows
                    if name.startswith('overlay/')
                    and census_rows.get(name) != graph_rows.get(name)}
    need(census_delta == {'overlay/e3/native_qualification.py'}
         and {name for name in graph_rows if name.startswith('overlay/')}
             == {name for name in census_rows if name.startswith('overlay/')},
         'native25 VM overlay differs from native24 beyond native_qualification')
    census_hub = json.loads(a.census_hub_result.read_text())
    census_hub_probe_path = a.census_hub_result.parent / 'probe.json'
    need(census_hub.get('verdict') == 'PASS' and census_hub.get('rc') == 0
         and census_hub.get('manifest_sha256') == census_input_sha
         and census_hub.get('gpu_device_requests') == 0
         and census_hub.get('gpu_stream_scope_audit', {}).get('verdict') == 'PASS'
         and census_hub_probe_path.is_file()
         and sha(census_hub_probe_path) == census_hub.get('probe_sha256')
         and json.loads(census_hub_probe_path.read_text())['sections']['census_cpu_shape']['verdict'] == 'PASS',
         'exact native25 hub CPU-shaped self-run absent')
    census_vm = json.loads(a.census_vm_result.read_text())
    census_vm_probe_path = a.census_vm_result.parent / 'probe.json'
    need(census_vm.get('verdict') == 'PASS' and census_vm.get('rc') == 0
         and census_vm.get('process_rc') == 0
         and census_vm.get('manifest_sha256') == census_input_sha
         and census_vm.get('gpu_execution') is True
         and census_vm.get('cleanup_ok') is True
         and census_vm.get('container_retained') is True
         and census_vm.get('failures') == [] and census_vm.get('missing') == []
         and census_vm_probe_path.is_file()
         and sha(census_vm_probe_path) == census_vm.get('probe_sha256'),
         'native25 actual CUDA census proof absent')
    census_probe = json.loads(census_vm_probe_path.read_text())
    census_gpu = census_probe.get('sections', {}).get('gpu_graph_bind', {})
    need(census_probe.get('verdict') == 'PASS'
         and census_gpu.get('real_observer_bound') is True
         and census_gpu.get('real_bank_replay') is True
         and census_gpu.get('census_start') == {'total':0,'graphs':{'graph:0':0}}
         and census_gpu.get('census_after_replay') == {'total':1,'graphs':{'graph:0':1}}
         and census_gpu.get('malformed_refused') is True,
         'native25 real split-bank replay census unproved')
    record = json.loads(a.regression_record.read_text())
    regression = a.regression_log.read_text()
    counts = re.findall(r'selftest: (\d+)/(\d+) cases as required', regression)
    suites = [json.JSONDecoder().raw_decode(line.split('selftest suite: ', 1)[1])[0]
              for line in regression.splitlines() if line.startswith('selftest suite: ')]
    coverage = re.findall(r'selftest coverage: (\{[^\n]+\})', regression)
    source_commit = record.get('source_commit')
    changed_since = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'diff', '--name-only',
         str(source_commit) + '..HEAD', '--', 'patches/s5_engine', 'harness'],
        text=True).splitlines())
    # Strict runtime/harness coverage precedes the separate one-card receipt proof.
    need(record.get('schema') == 'native28-strict-regression/1'
         and type(source_commit) is str and len(source_commit) == 40
         and changed_since <= {'patches/s5_engine/native_candidate_review.json'}
         and record.get('exit_code') == 0 and record.get('skipped') == 0
         and record.get('log_sha256') == sha(a.regression_log)
         and len(coverage) == 1
         and json.loads(coverage[0]) == {'skipped': 0, 'suites': len(suites)},
         'strict native28 regression does not cover frozen source')
    regression_sha = sha(a.regression_log)
    regression_record_sha = sha(a.regression_record)
    regression_basis = variant + ' strict zero-skip regression on current committed source'
    need(len(counts) == 1 and counts[0][0] == counts[0][1]
         and suites and all(row['returncode'] == 0 and row['skipped'] == 0 for row in suites),
         'strict zero-skip regression absent')
    host_selfrun = json.loads(a.host_selfrun_record.read_text())
    host_log = a.host_selfrun_record.parent / host_selfrun.get('log', '')
    need(host_selfrun.get('schema') == 'native27-profile-selfrun/1'
         and host_selfrun.get('verdict') == 'PASS'
         and host_selfrun.get('rc') == 0
         and host_selfrun.get('tests') == 4
         and host_selfrun.get('skipped') == 0
         and host_selfrun.get('source_commit') == '334f0cd05c61456a20282d4210703c6506aa2406'
         and host_selfrun.get('image') == IMAGE
         and host_selfrun.get('gpu_device_requests') == 0
         and host_log.is_file()
         and sha(host_log) == host_selfrun.get('log_sha256')
         and 'Ran 4 tests' in host_log.read_text()
         and host_log.read_text().rstrip().endswith('OK'),
         'pinned-image profile self-run absent')
    host_container = json.loads(subprocess.check_output(
        ['docker', 'inspect', host_selfrun['container']], text=True))[0]
    need(host_container['State']['ExitCode'] == 0
         and host_container['Config']['Image'] == IMAGE
         and not host_container['HostConfig'].get('DeviceRequests')
         and not host_container['HostConfig'].get('Devices'),
         'host self-run image, exit or CPU-only venue differs')
    profile_rows = inventory(a.profile_vm_input)
    profile_hub_rows = inventory(a.profile_hub_input)
    profile_manifest = '4e96ea483ce48e89cef96ee19f71e2999a40b35dd3a3b1612f334c0467fcce54'
    need(sha(a.profile_vm_input / 'SHA256SUMS') == profile_manifest
         and profile_rows == profile_hub_rows
         and sha(a.profile_hub_input / 'SHA256SUMS') == profile_manifest
         and profile_rows['proof/native27_profile_fixture_leaves.json']
             == sha(ROOT / 'harness/native27_profile_fixture_leaves.json'),
         'native27 profile VM input or fixture leaves differ')
    profile_hub = json.loads(a.profile_hub_result.read_text())
    profile_hub_probe = a.profile_hub_result.parent / 'probe.json'
    need(profile_hub.get('verdict') == 'PASS' and profile_hub.get('rc') == 0
         and profile_hub.get('manifest_sha256') == profile_manifest
         and profile_hub.get('gpu_device_requests') == 0
         and profile_hub_probe.is_file()
         and sha(profile_hub_probe) == profile_hub.get('probe_sha256')
         and json.loads(profile_hub_probe.read_text())['sections']['receipt_path']['verdict'] == 'PASS',
         'native27 exact-input hub receipt proof absent')
    profile_vm = json.loads(a.profile_vm_result.read_text())
    profile_vm_probe = a.profile_vm_result.parent / 'probe.json'
    need(profile_vm.get('verdict') == 'PASS' and profile_vm.get('rc') == 0
         and profile_vm.get('process_rc') == 0
         and profile_vm.get('manifest_sha256') == profile_manifest
         and profile_vm.get('gpu_execution') is True
         and profile_vm.get('cleanup_ok') is True
         and profile_vm.get('failures') == [] and profile_vm.get('missing') == []
         and profile_vm_probe.is_file()
         and sha(profile_vm_probe) == profile_vm.get('probe_sha256'),
         'native27 one-card receipt proof absent')
    receipt_probe = json.loads(profile_vm_probe.read_text())
    receipt_section = receipt_probe.get('sections', {}).get('receipt_path', {})
    need(receipt_section.get('verdict') == 'PASS'
         and receipt_section.get('journal_refusal_evicted') is True
         and receipt_section.get('phase_receipt_exact') is True
         and receipt_section.get('queue_grade_exact') is True
         and receipt_section.get('status_peer_selected_counts') == [4097, 97153]
         and receipt_section.get('status_peer_bucket_waves') == [[17], [380, 1]],
         'native27 VM proof did not retain both peer profiles end to end')
    coordinator = json.loads(a.coordinator_result.read_text())
    need(coordinator.get('rc') == 0 and coordinator.get('verdict') == 'PASS'
         and coordinator.get('mode') == 'cpu-full', '12-scenario CPU-full proof absent')
    probe_path = a.coordinator_result.parent / 'probe.json'
    probe = json.loads(probe_path.read_text())
    need(sha(probe_path) == coordinator.get('probe_sha256')
         and len(probe.get('sections', {})) == 12
         and all(section.get('verdict') == 'PASS'
                 for section in probe['sections'].values()),
         'all twelve connected CPU scenarios must pass')
    coordinator_rows = inventory(a.coordinator_input)
    need(sha(a.coordinator_input / 'SHA256SUMS') == coordinator['manifest_sha256']
         and 'proof/native21_coordinator_fixture_leaves.json' in coordinator_rows
         and sha(a.coordinator_input / ('proof/native21_coordinator_fixture_leaves.json'))
             == sha(ROOT / 'harness/native21_coordinator_fixture_leaves.json'),
         'coordinator input or leaf declaration differs')
    if not a.disable_audit:
        need(a.image_test_log is not None and a.image_test_container,
             'capture self-run arguments absent')
        image_test = a.image_test_log.read_text()
        need('Ran 4 tests' in image_test and image_test.rstrip().endswith('OK'),
             'pinned-image CPU capture test absent')
        image_container = json.loads(subprocess.check_output(
            ['docker', 'inspect', a.image_test_container], text=True))[0]
        need(image_container['State']['ExitCode'] == 0
             and image_container['Config']['Image'] == IMAGE
             and not image_container['HostConfig'].get('DeviceRequests'),
             'capture self-run image, exit, or CPU-only venue differs')
    out.mkdir(mode=0o700, parents=True)
    bundle = out / 'bundle-build'
    shutil.copytree(OLD, bundle, copy_function=shutil.copyfile)
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o700 if path.is_dir() else 0o600)
    for path in (ROOT / 'patches/s5_engine').rglob('*'):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            target = bundle / 'e3_builder' / path.relative_to(ROOT / 'patches/s5_engine')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    prior_builder_files = [p for p in (OLD / 'e3_builder').rglob('*')
                           if p.is_file() and p.name not in
                           ('native_candidate_review.json', 'make_s5_overlap.py')]
    need(all(sha(p) == sha(bundle / 'e3_builder' / p.relative_to(OLD / 'e3_builder'))
             for p in prior_builder_files),
         'native28 engine builder differs from frozen native27 beyond launcher generator')
    # The frozen base bundle also contains VM qualification source copies.
    # Refresh them from the same committed inputs as e3_builder: otherwise a
    # new engine overlay can be correct while vmid_sources still carries the
    # previous candidate's QSA operation. This does not affect the mounted
    # engine path, but a stale proof bundle misstates its own source closure.
    vm_sources = bundle / 'vmid_sources'
    vm_map = {
        'ops_qsa.py': ROOT / 'patches/retain/ops_qsa.py.v8.1',
        'sparse_stage/kernels.py': ROOT / 'patches/sparse_stage/kernels.py',
    }
    vm_map.update({
        's5_engine/' + name + '.py': ROOT / 'patches/s5_engine' / (name + '.py')
        for name in ('e3_types', 'sparse_history_dma', 'qsa_operation',
                     'operation_source', 'arena_attention', 'resident_attention',
                     'native_source', 'row1_oracle')
    })
    need({p.relative_to(vm_sources).as_posix() for p in vm_sources.rglob('*')
          if p.is_file()} == set(vm_map), 'VM source inventory differs')
    need(sha(vm_map['ops_qsa.py']) ==
         '5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b',
         'VM QSA reference source pin differs')
    for name, source in vm_map.items():
        need(source.is_file() and not source.is_symlink(),
             'VM source is missing or linked: ' + name)
        target = vm_sources / name
        shutil.copyfile(source, target)
        need(sha(target) == sha(source), 'VM source copy differs: ' + name)
    indexer_name = 'models/qwen4_exp/nvidia/ops/qsa_indexer.py'
    indexer_source = ROOT / 'reference/vllm-src' / indexer_name
    indexer_target = bundle / 'e3_reference' / indexer_name
    need(indexer_source.is_file() and not indexer_source.is_symlink(),
         'pinned QSA indexer reference source absent')
    indexer_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(indexer_source, indexer_target)
    need(sha(indexer_target) ==
         '2c34883a6396bac0238f6308196854b653eadab8bbebbe7dd2d558e30d065cbb',
         'QSA indexer source pin differs')
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    review = json.loads(review_path.read_text())
    source_names = set(review['source_shas'])
    if not a.disable_audit:
        source_names |= {'native_selection_audit.py', 'native_selection_source.py'}
    review['source_shas'] = {name: sha(bundle / 'e3_builder' / name)
                             for name in sorted(source_names)}
    tracked = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z', '--', 'harness'])
    harness_names = {Path(name.decode()).relative_to('harness').as_posix()
                     for name in tracked.split(b'\0') if name}
    need(set(review['harness_source_shas']) <= harness_names,
         'prior harness pin is not tracked at the delivery commit')
    need({'e3_native_speculation_gate.py', 'runs/q18a_' + variant + '.draft.sh',
          'runs/q18a_' + variant + '_declare.sh', 'runs/q18a_' + variant + '_settings.sh',
          'selftest.py'} <= harness_names,
         'required harness source is untracked')
    for name in sorted(harness_names):
        source = ROOT / 'harness' / name
        need(source.is_file() and not source.is_symlink(), 'harness source absent: ' + name)
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    ops_selfrun_container = 'flashnext-native28-ops-' + str(time.time_ns())
    ops_selfrun_cmd = [
        'docker', 'run', '--name', ops_selfrun_container, '--network', 'none',
        '--read-only', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1024m',
        '-v', str(bundle) + ':/bundle:ro', '-w', '/bundle',
        '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-e', 'PYTHONPATH=/bundle:/bundle/e3_builder:/bundle/e2_builder',
        '--entrypoint', 'python3', IMAGE, '-B', '-m', 'unittest', '-v',
        'test_e3_native28_host_guard', 'test_e3_native28_rule4_continuation',
    ]
    run(ops_selfrun_cmd, out / 'ops-selfrun-command.json')
    ops_log = (out / 'ops-selfrun-command.log').read_text()
    need('Ran 5 tests' in ops_log and ops_log.rstrip().endswith('OK'),
         'native28 host self-run did not exercise five tests')
    ops_container = json.loads(subprocess.check_output(
        ['docker', 'inspect', ops_selfrun_container], text=True))[0]
    need(ops_container['State']['ExitCode'] == 0
         and ops_container['Config']['Image'] == IMAGE
         and not ops_container['HostConfig'].get('DeviceRequests')
         and not ops_container['HostConfig'].get('Devices'),
         'native28 host self-run used device or failed')
    dump(out / 'ops-selfrun.json', {
        'schema': 'native28-ops-selfrun/1', 'verdict': 'PASS',
        'container': ops_selfrun_container, 'image': IMAGE,
        'tests': 5, 'skipped': 0, 'log_sha256': sha(out / 'ops-selfrun-command.log'),
        'source_shas': {name: sha(bundle / name) for name in (
            'e3_native28_host_observer.py', 'e3_native28_rule4_continuation.py',
            'e3_native_performance.py', 'test_e3_native28_host_guard.py',
            'test_e3_native28_rule4_continuation.py')},
    })
    queue_path = bundle / ('runs/q18a_' + variant + '.draft.sh')
    queue_source = queue_path.read_text()
    queue_anchor = ('datetime.datetime(2026,10,29,7,0,tzinfo=datetime.timezone.utc)'
                    if a.disable_audit else PLACEHOLDER)
    need(queue_source.count('end=' + queue_anchor) == 1,
         variant + ' queue hard-end anchor differs')
    queue_path.write_text(queue_source.replace('end=' + queue_anchor,
                                              'end=' + end_expr))
    review['harness_source_shas'] = {name: sha(bundle / name) for name in sorted(harness_names)}
    review['cpu_regression'] = {
        'basis': regression_basis,
        'cases': int(counts[0][0]), 'required': int(counts[0][1]),
        'exit_code': 0, 'skips': 0, 'suites': len(suites),
        'reported_tests': sum(row['tests_run'] or 0 for row in suites),
        'suites_without_test_count': sum(row['tests_run'] is None for row in suites),
        'log_sha256': regression_sha,
        'evidence_record_sha256': regression_record_sha,
    }
    if a.disable_audit:
        need('native20_selection_observation' not in review
             and 'native_selection_audit.py' not in review['source_shas'],
             'native19 review unexpectedly contains audit')
        need('native21_rpc_stream' in review, 'native21 baseline evidence missing')
        need('native23_row4_pause' in review,
             'native23 row-4 evidence missing from frozen base')
        need('native24_graph_capture' in review,
             'native24 graph proof missing from frozen base')
        need('native25_replay_census' in review,
             'native25 replay-census proof pin absent from frozen review')
        review['native28_host_operations'] = {
            'scope': 'Docker json-file cap, host disk guard, 1200 s client first-advance read bound, bounded rule-4 control continuation',
            'launcher_source_sha256': sha(bundle / 'e3_builder/make_s5_overlap.py'),
            'queue_source_sha256': sha(queue_path),
            'host_observer_sha256': sha(bundle / 'e3_native28_host_observer.py'),
            'rule4_continuation_sha256': sha(bundle / 'e3_native28_rule4_continuation.py'),
            'w2_client_sha256': sha(bundle / 'e3_native_performance.py'),
            'settings_sha256': sha(bundle / 'runs/q18a_native28_settings.sh'),
            'host_selfrun_record_sha256': sha(out / 'ops-selfrun.json'),
            'profile_vm_result_sha256': sha(a.profile_vm_result),
            'note': 'no engine Python/native source change; profile proof inherited from native27',
        }
        review['source_delta'] = {
            'basis': 'native27 runtime and frozen request manifests',
            'runtime_modules': [],
            'behaviour': 'host queue, client read bound and launcher operations only; no engine Python/native setting, numeric, workload or tolerance change',
        }
    else:
        review['native20_selection_observation'] = {
            'scope': 'receipt-only exact every-row score bytes and canonical selected-set fingerprints',
            'cpu_image_probe': 'test_e3_native_selection_audit_image.py',
            'cpu_image_log_sha256': sha(a.image_test_log),
            'cpu_image_container': a.image_test_container,
            'coordinator_result_sha256': sha(a.coordinator_result),
            'coordinator_verdict': coordinator['verdict'],
            'coordinator_probe_sha256': coordinator.get('probe_sha256'),
            'coordinator_wrapper_sha256': sha(a.coordinator_input / 'run_cpu.py'),
            'coordinator_import_graph_sha256': sha(a.coordinator_input / 'proof/import-graph.json'),
            'gpu_qualified': False, 'box_qualified': False,
            'timing_caveat': 'exact score and selected-set copy/hash work can perturb performance timing',
        }
        review['source_delta'] = {
            'basis': 'native19 plus diagnostic selection identity receipts',
            'runtime_modules': ['native_selection_audit.py', 'qsa_driver.py', 'wire_v1.py',
                                'worker_bridge.py', 'engine_driver.py'],
            'generated_source': 'ops_qsa_indexer.py',
            'harness_pin': 'identity_declare.S5_LAUNCHER_SHA binds the new exact indexer mount',
            'behaviour': 'no grant, refusal, top-k, numerics, setting, workload or tolerance change',
        }
    review['night_window'] = {'hard_end_utc': a.hard_end,
                              'timeout_s': 14400, 'kill_margin_s': 900,
                              'placeholder': is_placeholder,
                              'operator_window_granted': not is_placeholder}
    dump(review_path, review)
    closure_path = bundle / 'e3_build_closure.json'
    closure = json.loads(closure_path.read_text())
    closure['reference_pins'][indexer_name] = sha(indexer_target)
    closure['builder_sources'] = {
        path.relative_to(bundle / 'e3_builder').as_posix(): sha(path)
        for path in sorted((bundle / 'e3_builder').rglob('*')) if path.is_file()
        and '__pycache__' not in path.parts}
    dump(closure_path, closure)
    rows = {path.relative_to(bundle).as_posix(): sha(path)
            for path in bundle.rglob('*') if path.is_file()
            and path.name != 'SHA256SUMS' and '__pycache__' not in path.parts}
    (bundle / 'SHA256SUMS').write_text(''.join(
        digest + '  ' + name + '\n' for name, digest in sorted(rows.items())))
    manifest = sha(bundle / 'SHA256SUMS')
    target = out / ('q18a-' + variant + '-bundle-' + manifest[:12])
    bundle.rename(target)
    bundle = target
    queue_path = bundle / ('runs/q18a_' + variant + '.draft.sh')
    need(inventory(bundle) == rows, 'prepared bundle changed')
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'VLLM_API_KEY',
                        'ANTHROPIC_AUTH_TOKEN', 'ARM_AUTH_ENV_FILE')
           and not k.startswith(('FLASH_S5_', 'Q18A_', 'S5_'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
        map(str, (bundle / 'e3_builder', bundle / 'e2_builder', bundle))))
    run(['bash', '-n', str(queue_path)],
        out / 'queue-syntax.json', env=env)
    raw, final = out / 'raw', out / 'final'
    run(['python3', '-B', str(bundle / 'e3_builder/make_s5_overlap.py'),
         str(BASE), str(raw), '--reference', str(bundle / 'e3_reference'),
         '--base-manifest-sha256', BASE_SHA, '--native-candidate'],
        out / 'generation.json', env=env)
    generation = json.loads((raw / 'S5-OVERLAP-GENERATION.json').read_text())
    need(generation['armable'] is False, 'raw candidate armed')
    expected = {path.relative_to(raw).as_posix(): sha(path)
                for path in raw.rglob('*') if path.is_file() and
                path.relative_to(raw).as_posix() not in
                ('SHA256SUMS', 'S5-OVERLAP-GENERATION.json')}
    proven_overlay = a.profile_vm_input / 'overlay'
    runtime_compared = [name for name in expected
        if (proven_overlay / name).is_file()
                        and name not in ('e3/decode-probe-evidence.json',
                                         'e3/native-implementation-closure.json',
                                         'run-flash-s5-test.sh')]
    need(runtime_compared and all(expected[name] == sha(proven_overlay / name)
                                  for name in runtime_compared),
         'bundled generated runtime differs from connected self-run overlay')
    runner = RUNNER.read_text().replace('native18', variant).replace('Native18', variant.title())
    runner = runner.replace(
        'Operator order28Oct night requires queue hard end by29Oct08:00CET, owner restores by08:30.',
        'Owner window hard end; owner controls box staging, queue and restoration.')
    need(runner.count('datetime.datetime(2026,10,29,7,0,tzinfo=datetime.timezone.utc)') == 1,
         'runner window anchor')
    runner = runner.replace('datetime.datetime(2026,10,29,7,0,tzinfo=datetime.timezone.utc)',
                            end_expr)
    runner = replace_assignments(runner, {
        'H': "Path('/home/<service-user>/harness/" + manifest[:12] + "')",
        'BUNDLE': repr(manifest), 'REVIEW': repr(generation['implementation_review_sha256']),
        'LAUNCHER': repr(expected['run-flash-s5-test.sh']),
        'EXPECTED_SOURCES': json.dumps(expected, sort_keys=True, indent=4)})
    runner_path = out / ('q18a-' + variant + '-owner-run-' + manifest[:12] + '.py')
    runner_path.write_text(runner)
    default_runtime = subprocess.check_output(
        ['docker', 'info', '--format', '{{.DefaultRuntime}}'], text=True).strip()
    need(default_runtime == 'runc', 'CPU helper default runtime is not runc')
    run(['bash', str(bundle / 'e3_builder/native/build_q17_helpers.sh'), IMAGE,
         str(raw / 'e3/native')], out / 'cpu-helper-build.json', env=env)
    helper_log = (out / 'cpu-helper-build.log').read_text()
    helper_names = re.findall(r'Q17_NATIVE_BUILD PASS retained-container=(\S+)', helper_log)
    need(len(helper_names) == 1, 'CPU helper container receipt absent')
    helper = json.loads(subprocess.check_output(
        ['docker', 'inspect', helper_names[0]], text=True))[0]
    need(helper['State']['ExitCode'] == 0 and not helper['State']['Running']
         and not helper['HostConfig'].get('DeviceRequests')
         and not helper['HostConfig'].get('Devices'), 'CPU helper used device or failed')
    run(['python3', '-B', str(bundle / 'e3_finalize_package.py'), str(raw), str(final)],
        out / 'finalize.json', env=env)
    prior_runtime = [p for p in (OLD_FINAL / 'e3').rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.so')
                     ]
    need(prior_runtime and all(
        sha(p) == sha(final / 'e3' / p.relative_to(OLD_FINAL / 'e3'))
        for p in prior_runtime),
        'native28 finalized engine runtime differs from native27')
    verifier = out / 'verify-owner.py'
    verifier.write_text('''import importlib.util,json,sys
from pathlib import Path
runner,raw,final=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('owner',runner)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
owner.verify_sources(raw,built=True)
owner.OVERLAY=final
owner.verify_final_build()
print('OWNER_VERIFY PASS')
''')
    run(['python3', '-B', str(verifier), str(runner_path), str(raw), str(final)],
        out / 'owner-verifier.json', env=env)
    need(inventory(bundle) == rows, 'input bundle mutated')
    for path in [bundle, *bundle.rglob('*')]:
        path.chmod(0o500 if path.is_dir() else 0o400)
    need(all(not (path.stat().st_mode & 0o222)
             for path in [bundle, *bundle.rglob('*')]),
         'held bundle remains writable')
    runner_path.chmod(0o500)
    receipt = {'schema': variant + '-held-delivery/1', 'verdict': 'PASS',
               'delivery_state': 'HOLD', 'box_go': False,
               'bundle': str(bundle), 'manifest_sha256': manifest,
               'bundle_files': len(rows), 'runner': str(runner_path),
               'runner_sha256': sha(runner_path), 'runner_bytes': runner_path.stat().st_size,
               'review_sha256': generation['implementation_review_sha256'],
               'launcher_sha256': expected['run-flash-s5-test.sh'],
               'source_pins': expected, 'regression_log_sha256': regression_sha,
               'native27_runtime_files_byte_equal': len(prior_runtime),
               'host_selfrun_record_sha256': sha(a.host_selfrun_record),
               'host_selfrun_log_sha256': sha(host_log),
               'host_selfrun_container': host_selfrun['container'],
               'host_observer_sha256': sha(bundle / 'e3_native28_host_observer.py'),
               'ops_selfrun_container': ops_selfrun_container,
               'ops_selfrun_record_sha256': sha(out / 'ops-selfrun.json'),
               'ops_selfrun_log_sha256': sha(out / 'ops-selfrun-command.log'),
               'rule4_continuation_sha256': sha(bundle / 'e3_native28_rule4_continuation.py'),
               'w2_client_sha256': sha(bundle / 'e3_native_performance.py'),
               'identity_declare_sha256': sha(bundle / 'identity_declare.py'),
               'profile_receipt_helper_sha256': sha(bundle / 'e3_native27_profile_receipt.py'),
               'engine_driver_sha256': sha(bundle / 'e3_builder/engine_driver.py'),
               'profile_vm_result_sha256': sha(a.profile_vm_result),
               'profile_vm_probe_sha256': sha(profile_vm_probe),
               'profile_vm_input_manifest_sha256': profile_manifest,
               'profile_hub_result_sha256': sha(a.profile_hub_result),
               'profile_fixture_leaves_sha256': sha(bundle / 'native27_profile_fixture_leaves.json'),
               'row4_source_audit_sha256': sha(ROOT / 'reviews/codex-20261030-native26-row4-source-audit.md'),
               'regression_record_sha256': regression_record_sha,
               'graph_runtime_compared': len(runtime_compared),
               'coordinator_result_sha256': sha(a.coordinator_result),
               'coordinator_input': str(a.coordinator_input.resolve()),
               'coordinator_input_manifest_sha256': coordinator['manifest_sha256'],
               'coordinator_probe_sha256': coordinator.get('probe_sha256'),
               'coordinator_wrapper_sha256': sha(a.coordinator_input / 'run_cpu.py'),
               'coordinator_import_graph_sha256': sha(a.coordinator_input / 'proof/import-graph.json'),
               'coordinator_container': coordinator.get('container'),
               'fixture_leaves_sha256': sha(bundle / 'native21_coordinator_fixture_leaves.json'),
               'row4_fixture_leaves_sha256': sha(bundle / 'native23_row4_pause_fixture_leaves.json'),
               'graph_fixture_leaves_sha256': sha(bundle / 'native24_graph_fixture_leaves.json'),
               'census_fixture_leaves_sha256': sha(bundle / 'native25_census_fixture_leaves.json'),
               'vm_result_sha256': sha(a.vm_result),
               'vm_probe_sha256': vm.get('probe_sha256'),
               'vm_input_manifest_sha256': vm['manifest_sha256'],
               'audited_input_manifest_sha256': audit_sha,
               'audited_selfrun_sha256': sha(a.audited_selfrun),
               'vm_audited_payload_files_equal': len(compared),
               'graph_vm_result_sha256': sha(a.graph_vm_result),
               'graph_vm_probe_sha256': graph_vm.get('probe_sha256'),
               'graph_vm_input_manifest_sha256': graph_input_sha,
               'graph_hub_result_sha256': sha(a.graph_hub_result),
               'graph_hub_probe_sha256': graph_hub.get('probe_sha256'),
               'census_vm_result_sha256': sha(a.census_vm_result),
               'census_vm_probe_sha256': census_vm.get('probe_sha256'),
               'census_vm_input_manifest_sha256': census_input_sha,
               'census_hub_result_sha256': sha(a.census_hub_result),
               'census_hub_probe_sha256': census_hub.get('probe_sha256'),
               'image_test_log_sha256': None if a.disable_audit else sha(a.image_test_log),
               'image_test_container': None if a.disable_audit else a.image_test_container,
               'hard_end_utc': a.hard_end,
               'hard_end_is_placeholder': is_placeholder,
               'cpu_helper_container': helper_names[0],
               'cpu_helper_runtime': default_runtime,
               'cpu_helper_device_requests': helper['HostConfig'].get('DeviceRequests'),
               'timeout_s': 14400, 'kill_margin_s': 900,
               'hardware_qualified': False, 'release_qualified': False}
    receipt['build_command'] = ['python3', '-B', str(Path(__file__).resolve()),
                                *sys.argv[1:]]
    dump(out / 'provenance.json', receipt)
    brief = f'''# {variant.title()} held delivery

Status: **HOLD**. The hard end alone is not a box GO.

- Bundle: `{bundle}` ({len(rows)} listed files plus `SHA256SUMS`)
- Manifest SHA256: `{manifest}`
- Owner runner: `{runner_path}` ({receipt['runner_bytes']} bytes)
- Runner SHA256: `{receipt['runner_sha256']}`
- Review SHA256: `{receipt['review_sha256']}`
- Launcher SHA256: `{receipt['launcher_sha256']}`
- Queue hard end: `{receipt['hard_end_utc']}` (placeholder={receipt['hard_end_is_placeholder']}); timeout {receipt['timeout_s']} s plus {receipt['kill_margin_s']} s kill margin
- Strict regression: {counts[0][0]}/{counts[0][1]}, {len(suites)} suites, 0 skips; basis `{regression_basis}`; log SHA256 `{receipt['regression_log_sha256']}`; committed record SHA256 `{regression_record_sha}`
- Profile-refusal pinned-image CPU self-run: 4/4 tests, 0 skips, runc/no GPU, container `{receipt['host_selfrun_container']}`, record SHA256 `{receipt['host_selfrun_record_sha256']}`; receipt helper SHA256 `{receipt['profile_receipt_helper_sha256']}`
- Native27 one-card VM receipt proof: result SHA256 `{receipt['profile_vm_result_sha256']}`, probe SHA256 `{receipt['profile_vm_probe_sha256']}`, input manifest `{receipt['profile_vm_input_manifest_sha256']}`; exact-input hub CPU-shaped result SHA256 `{receipt['profile_hub_result_sha256']}`; fixture leaves SHA256 `{receipt['profile_fixture_leaves_sha256']}`
- Native27 finalized engine runtime: {receipt['native27_runtime_files_byte_equal']} Python/native files byte-equal; only the generated S5 launcher changes
- Native28 pinned-image hub self-run: five focused tests, zero skips, no GPU, container `{receipt['ops_selfrun_container']}`, record SHA256 `{receipt['ops_selfrun_record_sha256']}`; observer SHA256 `{receipt['host_observer_sha256']}`, rule-4 continuation SHA256 `{receipt['rule4_continuation_sha256']}`
- Row-4 source audit: `reviews/codex-20261030-native26-row4-source-audit.md` SHA256 `{receipt['row4_source_audit_sha256']}`. It found no direct 50 GB file/log emitter in the generated observer path; the Docker json log versus writable-layer cause remains open.
- Coordinator: 12/12 PASS in pinned image, CPU-full; result SHA256 `{receipt['coordinator_result_sha256']}`; probe SHA256 `{receipt['coordinator_probe_sha256']}`; input manifest SHA256 `{receipt['coordinator_input_manifest_sha256']}`
- Fixture leaves: `native21_coordinator_fixture_leaves.json` SHA256 `{receipt['fixture_leaves_sha256']}`; row-4 seam leaves `native23_row4_pause_fixture_leaves.json` SHA256 `{receipt['row4_fixture_leaves_sha256']}`; graph leaves `native24_graph_fixture_leaves.json` SHA256 `{receipt['graph_fixture_leaves_sha256']}`; census leaves `native25_census_fixture_leaves.json` SHA256 `{receipt['census_fixture_leaves_sha256']}`
- CPU helper build: `{helper_names[0]}`, runc, no device requests
- Row-4 VM GPU proof: result SHA256 `{receipt['vm_result_sha256']}`, input manifest `{receipt['vm_input_manifest_sha256']}`; exact audited hub self-run SHA256 `{receipt['audited_selfrun_sha256']}`, audited input `{receipt['audited_input_manifest_sha256']}`, {len(compared)} mounted payload files byte-equal
- Real graph/observer VM GPU proof: result SHA256 `{receipt['graph_vm_result_sha256']}`, probe SHA256 `{receipt['graph_vm_probe_sha256']}`, input manifest `{receipt['graph_vm_input_manifest_sha256']}`; exact hub CPU self-run SHA256 `{receipt['graph_hub_result_sha256']}`
- Real split-bank replay-census VM GPU proof: result SHA256 `{receipt['census_vm_result_sha256']}`, probe SHA256 `{receipt['census_vm_probe_sha256']}`, input manifest `{receipt['census_vm_input_manifest_sha256']}`; exact hub CPU self-run SHA256 `{receipt['census_hub_result_sha256']}`

This is native27 engine runtime with host queue changes only. The S5 launcher caps Docker's ephemeral json-file log at 512 MiB × 4; the queue retains the full followed stream. The host observer records a final root/SizeRw sample and stop intent below 8 GiB before asking the queue to stop. W1/W2 first-advance reads use a declared 1200 s client wait while TTFT remains the measured latency. A rule-4 control hard-cap result remains raw INCONCLUSIVE, but may allow later phases only after the frozen rule-4 grader reproduces that exact reason and all other gates pass. A rule-4 FAIL still withholds later phases. No cap, tolerance or workload value changes. The owner must pre-check and stage this candidate before any box action.

Native23 rule 4 had a real FAIL at stream 0 position 320; native24 passed both rule-4 gates with the same settings, workload and grader. The cause of the earlier outlier remains unproved. Native25 preserves the unchanged numerical gates and does not select between repeated runs.

The first executor-shutdown *marker* and host queue-stop intent are captured; the initiating reason is reported unknown unless the unchanged engine stream explicitly states it. The observer never calls an engine control API.

ETA at delivery (CET): owner pre-check and staging 10–15 min; candidate ARM immediately after owner clearance if the queue's worst-case hard-end guard still admits entry. Native29 is a separate VM-only behavioral investigation and is not a prerequisite for this observability/queue candidate.

Full source pins and self-run metadata are in `provenance.json`; the fixture declaration is inside the bundle.
The exact local builder command is retained as `build_command` in `provenance.json`.
'''
    (out / 'delivery-brief.md').write_text(brief)
    print(variant.upper() + ' HOLD PASS ' + manifest)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('NATIVE28 HOLD FAIL ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        raise
