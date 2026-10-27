#!/usr/bin/env python3
"""CPU-only offline execution of the q18a real-overlap queue.

The queue is exercised end to end with fake PATH commands, a loopback probe
server, a fake W1 driver, fake qualification/lifecycle interfaces and a fake
overlap measurement artifact.  The final report is produced by the real
``e3_overlap_grade.py`` over that artifact, so the queue/grade plumbing is
proved without any GPU, CUDA, docker daemon, network or real box path.

These tests do not prove overlap performance or engine behaviour.  They prove
that the queue refuses the unarmable package, withholds overlap until every
gate is PASS, retains FAIL/INCONCLUSIVE intervals, stops and restores exactly,
and never starts production.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import test_e3_startup_queue as old
from e3_overlap_grade import _receipt_digest

HERE = Path(__file__).resolve().parent
QUEUE = HERE / 'runs' / 'q18a_overlap.draft.sh'
SETTINGS = HERE / 'runs' / 'q18a_overlap_settings.sh'
LABEL = 's5-overlap-80-80'
SHA_W = 'b' * 64
PACKAGE = 'a' * 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FAKE_W1 = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
url, model, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
bin = Path(os.environ['FAKE_BIN'])
if (bin / 'change-next').exists():
    staged = Path(os.environ['FAKE_DEFAULTS'] + '.next')
    if staged.is_file() and not staged.is_symlink():
        staged.write_text(staged.read_text() + 'QUEUE_CHANGED=1\n')
if (bin / 'missing-w1').exists():
    raise SystemExit(2)
rates = [100.0 - i for i in range(8)]
def repeat(route, rate, gap):
    return {'route': route, 'valid': True,
            'metrics': {'stream_rates_tokens_per_s': [rate] * 8},
            'streams': [{'window': {'gaps_ms': [gap, gap + 1.0]}}
                        for _ in range(8)]}
repeats = [repeat('full', 100.0, 5.0), repeat('split', 60.0, 7.0)]
routes = {
    'full': {'route': 'full', 'valid_repeats': 1,
             'aggregate_tokens_per_s_median': 100.0, 'gap_p99_ms': 5.0},
    'split': {'route': 'split', 'valid_repeats': 1,
              'aggregate_tokens_per_s_median': 60.0, 'gap_p99_ms': 7.0},
}
summary = {'verdict': 'COMPLETE',
           'totals': {'valid': 2, 'required': 2, 'contract': 0},
           'routes': routes}
failed = (bin / 'fail-w1').exists()
if failed:
    summary['verdict'] = 'FAIL'
    summary['totals']['contract'] = 1
Path(out).write_text(json.dumps({
    'schema': 2, 'summary': summary, 'repeats': repeats,
    'config': {'WORKLOAD_SHA256': 'SHA_W_PLACEHOLDER'}}))
raise SystemExit(1 if failed else 0)
'''

FAKE_QUAL = r'''#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
for name in ('phase', 'url', 'model', 'run-id', 'epoch', 'out'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
failed = (Path(os.environ['FAKE_BIN']) / ('fail-qual-' + a.phase)).exists()
Path(a.out).write_text(json.dumps({'schema': 1, 'phase': a.phase,
                                   'run_id': a.run_id,
                                   'verdict': 'FAIL' if failed else 'PASS'}))
raise SystemExit(1 if failed else 0)
'''

FAKE_LIFECYCLE = r'''#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
for name in ('url', 'model', 'run-id', 'epoch', 'out'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
failed = (Path(os.environ['FAKE_BIN']) / 'fail-lifecycle').exists()
Path(a.out).write_text(json.dumps({'schema': 1, 'run_id': a.run_id,
                                   'verdict': 'FAIL' if failed else 'PASS'}))
raise SystemExit(1 if failed else 0)
'''

FAKE_CURL = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write('curl ' + ' '.join(sys.argv[1:]) + '\n')
bin = Path(os.environ['FAKE_BIN'])
raise SystemExit(7 if (bin / 'fail-health').exists() or (bin / 'engine-crash').exists() else 0)
'''

FAKE_OVERLAP = r'''#!/usr/bin/env python3
import hashlib, json, os, sys, time
from pathlib import Path
url, model, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
bin = Path(os.environ['FAKE_BIN'])
if (bin / 'missing-overlap').exists():
    raise SystemExit(2)
overlay = Path(os.environ['S5_OVERLAY'])
launcher = overlay / os.environ.get('S5_LAUNCHER_NAME', 'run-flash-s5-test.sh')
generation = json.loads(Path(os.environ['GENERATION']).read_text())
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
worker = 'fixture-worker-1'
owner = {'worker_incarnation': worker, 'engine_epoch': 9,
         'storage_epoch': 1, 'route_generation': 7}

def rank(number, seq):
    return {
        'rank': number, 'ownership': dict(owner),
        'copy': {'buckets': 3, 'bytes': 4096 + number, 'events': 4,
                 'completion': 'queried_event', 'queried': True},
        'attention': {'ops': 5, 'complete': True},
        'state_write': {'proof': 'queried_event', 'queried': True,
                        'writers': ['worker%d' % number], 'joined_streams': 2},
        'pp': {'receive': True, 'consume': True},
        'replay': {'submitted_before': 10 * seq + number,
                   'submitted_after': 10 * seq + number + 4,
                   'completed_before': 8 * seq + number,
                   'completed_after': 8 * seq + number + 2},
    }

def accepted(seq, route, committed, useful, complete):
    return {
        'segment_seq': seq,
        'start_monotonic_ns': seq * 1000000000,
        'end_monotonic_ns': seq * 1000000000 + 500000000,
        'accepted': True, 'path': 'e3_overlap', 'copy_route': route,
        'decode_arrivals': 8,
        'committed_tokens_before': committed[0],
        'committed_tokens_after': committed[1],
        'useful_tokens_before': useful[0],
        'useful_tokens_after': useful[1],
        'prefill_completed': complete,
        'top': {'owner': dict(owner)},
        'ranks': [rank(number, seq) for number in range(4)],
    }

segments = [
    accepted(1, 'sparse', (0, 70000), (0, 68032), False),
    accepted(2, 'dense', (70000, 140000), (68032, 136064), False),
    accepted(3, 'dense', (140000, 200000), (136064, 204096), True),
    {'segment_seq': 4, 'start_monotonic_ns': 4100000000,
     'end_monotonic_ns': 4500000000, 'accepted': False,
     'path': 'stock_forward', 'stock_forward_used': True,
     'fallback_claimed_as_overlap': False},
]
anchor_s, anchor_ns = time.time(), time.monotonic_ns()
window = {'start_ns': anchor_ns - 3500000000,
          'end_ns': anchor_ns + 100000000,
          'basis': 'overlap client measurement window'}
overlap = {
    'valid': True, 'engine_crash': False,
    'invalid_intervals': [{'reasons': ['fixture boundary interval retained']}],
    'summary_basis': {'filtered_invalid': False, 'retry_selected': False,
                      'picked': False,
                      'basis': 'one fixed schedule; no retry and no picking'},
    'claims': {'speed_claim': False, 'retry_selected': False,
               'survivor_averaged': False, 'offered_work': False},
    'interval_counters': {'reset_count': 0},
    'segments': segments,
    'owners': {str(rank): dict(owner) for rank in range(4)},
    'useful_completed_tokens': 204096, 'prompt_tokens': 204096,
    'completed_prefills': 1, 'prefill_wall_s': 250.0,
    'decode_tokens_per_s': 90.0, 'prefill_tokens_per_s': 900.0,
    'fresh_ttft_s': 20.0, 'append_ttft_s': 5.0,
    'valid_completed_turns': 8, 'cancellations': 0, 'errors': 0,
    'workload_sha256': 'SHA_W_PLACEHOLDER', 'window': window,
    'worker_incarnation': worker, 'engine_epoch': 9,
    'storage_epoch': 1, 'route_generation': 7,
    'routes': {'sparse': 1, 'dense': 2, 'serialized': 1,
               'stock_forward_fallback': 1},
    'fallback': {'segments': 1, 'explicit': True,
                 'counted_as_overlap': False},
}
memory = {'reserve_mib': 512,
          'allocator_peaks': {'gpu'+str(rank): {'reserved_bytes': 1000000000,
                              'peak_reserved_bytes': 1200000000,
                              'basis': 'fixture allocator counters'} for rank in range(4)}}
artifact = {
    'schema': 1, 'run_id': os.environ.get('RUN_ID', 'fixture'),
    'target_label': os.environ.get('Q18A_TARGET_LABEL', 'S5_TARGET'),
    'launcher_sha256': sha_file(launcher),
    'env_launcher_sha256': None,
    'package_sha256': generation.get('package_sha256'),
    'lanes': {
        'prefill_full': {'route': 'full', 'valid': True,
                         'serving_baseline': True,
                         'prefill_tokens_per_s': 1500.0,
                         'fresh_ttft_s': 15.0, 'append_ttft_s': 4.0,
                         'workload_sha256': 'SHA_W_PLACEHOLDER'},
        'prefill_split': {'route': 'split', 'valid': True,
                          'serving_baseline': False,
                          'prefill_tokens_per_s': 1000.0,
                          'fresh_ttft_s': 22.0, 'append_ttft_s': 6.0,
                          'workload_sha256': 'SHA_W_PLACEHOLDER'},
    },
    'overlap': overlap, 'memory': memory,
    'wall_anchor_s': anchor_s, 'wall_anchor_ns': anchor_ns,
}
failed = (bin / 'fail-overlap').exists()
crashed = (bin / 'crash-overlap').exists()
if failed:
    overlap['valid'] = False
    overlap['reasons'] = ['fixture overlap FAIL retained']
    artifact['reasons'] = ['fixture overlap FAIL retained']
if crashed:
    overlap['valid'] = False
    overlap['engine_crash'] = True
    artifact['reasons'] = ['fixture engine crash']
    (bin / 'engine-crash').write_text('')
# Keep the sampler running long enough that at least one dated sample surely
# falls inside the declared client measurement window.
time.sleep(2.5)
out.write_text(json.dumps(artifact))
raise SystemExit(1 if (failed or crashed) else 0)
'''.replace('SHA_W_PLACEHOLDER', SHA_W)


class OverlapQueue:
    """The startup fixture plus the q18a candidate/queue interfaces.

    Deliberately not a ``unittest.TestCase`` subclass: ``QueueFlowTests`` binds
    the fixture methods through ``setUp``/``tearDown`` so the historical q17
    tests in ``test_e3_startup_queue`` are not collected twice.
    """

    setUp = old.QueueRun.setUp
    tearDown = old.QueueRun.tearDown
    fake = old.QueueRun.fake

    def assert_no_secret(self, text):
        self.assertNotIn(old.AUTH, text)

    def make_harness(self):
        old.QueueRun.make_harness(self)
        (self.h / 'runs').mkdir(exist_ok=True)
        shutil.copyfile(QUEUE, self.h / 'runs/q18a_overlap.draft.sh')
        shutil.copyfile(SETTINGS, self.h / 'runs/q18a_overlap_settings.sh')
        for name in ('e3_private_auth.py', 'e3_decode_correctness.py',
                     'agent_replay.py', 'e3_overlap_grade.py'):
            shutil.copyfile(HERE / name, self.h / name)
        (self.helper_dir()).mkdir()
        for name in ('q17b-w1.json', 'q18a-w2-overlap.json'):
            shutil.copyfile(HERE / 'workloads' / name,
                            self.helper_dir() / name)
        (self.h / 'e3_w1_decode_driver.py').write_text(FAKE_W1)
        (self.h / 'e3_overlap_performance.py').write_text(FAKE_OVERLAP)
        (self.h / 'e3_q18a_qualification.py').write_text(FAKE_QUAL)
        (self.h / 'e3_q18a_lifecycle.py').write_text(FAKE_LIFECYCLE)
        old.write_manifest(self.h)

    def helper_dir(self):
        return self.h / 'workloads'

    def make_overlay(self):
        old.QueueRun.make_overlay(self)
        overlay = self.overlay
        replay = b'\x7fELF fixture replay observer bytes\n'
        (overlay / 'e3/native/flash_s5_replay_observer.so').write_bytes(replay)
        self.replay = hashlib.sha256(replay).hexdigest()
        self.candidate_sha = sha(overlay / 'run-flash-s5-test.sh')
        readiness = {
            'schema': 1, 'reviewed': True,
            'reviewed_by': 'offline-fixture-reviewer',
            'require_native_ready': True, 'native_ready': True,
            'full_native_overlap': True, 'sparse_dma_lease': True,
            'armable': True, 'package_sha256': PACKAGE,
            'launcher_sha256': self.candidate_sha,
            'path': 'e3/native-overlap-readiness.json',
        }
        readiness['receipt_sha256'] = _receipt_digest(readiness)
        (overlay / 'e3/native-overlap-readiness.json').write_text(
            json.dumps(readiness, sort_keys=True))
        self.readiness = overlay / 'e3/native-overlap-readiness.json'
        generation = json.loads(
            (overlay / 'S5-OVERLAP-GENERATION.json').read_text())
        generation.update({
            'armable': True, 'require_native_ready': True,
            'package_sha256': PACKAGE,
            'default_feature': 'FLASH_S5_OVERLAP=0',
            'incremental_kv_bytes': 0,
            'native_overlap': {'enabled': True, 'sparse_dma_lease': True,
                               'stock_forward_fallback_hidden': False},
        })
        generation['generated_files'] = sorted(set(
            generation['generated_files']
            + ['e3/native-overlap-readiness.json',
               'e3/native/flash_s5_replay_observer.so']))
        (overlay / 'S5-OVERLAP-GENERATION.json').write_text(
            json.dumps(generation, sort_keys=True))
        self.generation = generation
        (self.h / 'identity_declare.py').write_text(
            'S5_LAUNCHER_SHA = %r\n' % self.candidate_sha)
        old.write_manifest(overlay)
        old.write_manifest(self.h)

    def make_requests(self):
        old.QueueRun.make_requests(self)
        auth = self.r / 'private/vllm-api.env'
        auth.parent.mkdir(mode=0o700, exist_ok=True)
        from e3_private_auth import prepare
        prepare(auth, {'VLLM_API_KEY': old.AUTH})
        self.auth_file = auth
        startup = json.loads((self.r / 'startup.json').read_text())
        identity = startup['identity']
        identity['defaults']['FLASH_S5_AUTH_ENV_FILE'] = str(auth)
        identity['env'] = {
            'FLASH_S5_DECODE_ONLY_PROBE': '0', 'FLASH_S5_OVERLAP': '1',
            'FLASH_S5_DECODE_SMS': '44', 'FLASH_S5_E3_NATIVE_OVERLAP': '1',
            'FLASH_S5_SPARSE_DMA_LEASE': '1',
            'FLASH_S5_GREEN_LIB': ('/usr/local/lib/python3.12/dist-packages/'
                                   'vllm/v1/flash_e3/native/flash_s5_green.so'),
            'FLASH_S5_GREEN_LIB_SHA256': self.lib,
            'FLASH_S5_REPLAY_OBSERVER': '1', 'FLASH_S5_ENGINE_OFFERS': '1',
            'FLASH_S5_REPLAY_LIB': ('/usr/local/lib/python3.12/dist-packages/'
                                    'vllm/v1/flash_e3/native/'
                                    'flash_s5_replay_observer.so'),
            'FLASH_S5_REPLAY_LIB_SHA256': self.replay,
            'Q18A_TARGET_LABEL': LABEL,
            'VLLM_API_KEY': {'redacted_length': len(old.AUTH)},
        }
        for mount in identity['mounts']:
            if mount['destination'] == '/work':
                mount['source'] = str(self.overlay / 'run-flash-s5-test.sh')
                mount['content']['entries'][0]['sha256'] = self.candidate_sha
        (self.r / 'startup.json').write_text(
            json.dumps({'schema': 1, 'identity': identity}, sort_keys=True))
        plan = {
            'schema': 1, 'model': 'fixture-model',
            'w1': 'workloads/q17b-w1.json',
            'overlap': 'workloads/q18a-w2-overlap.json',
            'target_label': LABEL,
            'readiness': str(self.readiness), 'reserve_mib': 512,
            'snapshot_max_bytes': 64 * 1024 ** 2,
            'denominators': ('decode=actual full-GPU W1; prefill=matched '
                             'split solo, full-GPU solo also reported'),
        }
        (self.r / 'request-plan.json').write_text(
            json.dumps(plan, sort_keys=True))
        pins = [self.r / 'startup.json', self.r / 'launcher.found',
                self.r / 'request-plan.json',
                self.h / 'runs/q18a_overlap.draft.sh',
                self.h / 'runs/q18a_overlap_settings.sh',
                self.h / 'e3_startup_grade.py',
                self.h / 'e3_startup_mount_policy.json',
                self.h / 'e3_build_closure.json', self.h / 'e3_probe_rpc.py',
                self.h / 'engine_identity.py', self.h / 'qlib.sh',
                self.h / 'e3_private_auth.py',
                self.h / 'e3_overlap_performance.py',
                self.h / 'e3_overlap_grade.py',
                self.h / 'e3_w1_decode_driver.py',
                self.h / 'e3_decode_correctness.py',
                self.h / 'e3_q18a_qualification.py',
                self.h / 'e3_q18a_lifecycle.py',
                self.h / 'agent_replay.py', self.h / 'identity_declare.py',
                self.h / 'workloads/q17b-w1.json',
                self.h / 'workloads/q18a-w2-overlap.json',
                self.overlay / 'SHA256SUMS', self.overlay / 'core/SHA256SUMS',
                self.overlay / 'S5-OVERLAP-GENERATION.json',
                self.overlay / 'run-flash-s5-test.sh',
                self.overlay / 'e3/native-overlap-readiness.json',
                self.overlay / 'e3/native/flash_s5_green.so',
                self.overlay / 'e3/native/flash_s5_replay_observer.so',
                self.setcfg, self.image_file]
        (self.r / 'SHA256SUMS').write_text(
            ''.join(sha(p) + '  ' + str(p.absolute()) + '\n' for p in pins))

    def install_fakes(self):
        env = old.QueueRun.install_fakes(self)
        self.fake('curl', FAKE_CURL)
        env.update({
            'S5_LAUNCHER_NAME': 'run-flash-s5-test.sh',
            'S5_REPLAY_LIB_SHA256': self.replay,
            'Q18A_TARGET_LABEL': LABEL,
            'Q18A_READINESS': str(self.readiness),
            'Q18A_OVERLAP_BOUNDED': '1',
        })
        return env

    def run_queue(self, state=None):
        env = self.install_fakes()
        if state is not None:
            target = self.root / 'systemctl-pre.json'
            target.write_text(json.dumps(state))
            env['FAKE_STATE'] = str(target)
        return subprocess.run(
            ['bash', str(self.h / 'runs/q18a_overlap.draft.sh'), str(self.q)],
            capture_output=True, text=True, timeout=180, env=env)

    def mark(self, name):
        (self.bin / name).write_text('')

    def rehash_requests(self):
        lines = []
        for line in (self.r / 'SHA256SUMS').read_text().splitlines():
            _digest, name = line.split(maxsplit=1)
            lines.append(sha(name) + '  ' + name)
        (self.r / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')

    def step(self, text, name):
        return [line for line in text.splitlines()
                if line.startswith('STEP ' + name + ' ')]

    def read_json(self, name):
        return json.loads((self.q / name).read_text())


class QueueFlowTests(OverlapQueue, unittest.TestCase):
    def test_success_runs_every_phase_and_restores(self):
        done = self.run_queue()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assert_no_secret(done.stdout + done.stderr)
        self.assertIn('Q18A-OVERLAP DONE rc=0 scope=real-e3-overlap', done.stdout)
        self.assertIn('verdict=PASS', done.stdout)
        self.assertEqual(done.stdout.count('Q18A-OVERLAP DONE'), 1)
        self.assertLess(done.stdout.index('STEP row4 PASS'),
                        done.stdout.index('STEP w1_controls PASS'))
        for name in ('pins', 'preflight', 'restart', 'startup',
                     'w1_controls', 'row1', 'rule4_controls',
                     'rule4_candidate', 'row4', 'overlap-gates', 'overlap',
                     'overlap-engine', 'lifecycle', 'engine-stop',
                     'launcher-restore', 'defaults-restore', 'overlap_grade'):
            steps = self.step(done.stdout, name)
            self.assertEqual(len(steps), 1, (name, done.stdout))
            self.assertIn(' PASS', steps[0], (name, done.stdout))
        grade = self.read_json('overlap-grade.json')
        self.assertEqual(grade['verdict'], 'PASS', grade)
        self.assertEqual(grade['routes']['sparse'], 1)
        self.assertEqual(grade['routes']['dense'], 2)
        self.assertEqual(grade['routes']['serialized'], 1)
        self.assertEqual(grade['routes']['stock_forward_fallback'], 1)
        self.assertEqual(grade['routes']['accepted_segments'], 3)
        self.assertEqual(grade['metrics']['useful_completed_tokens'], 204096)
        self.assertEqual(grade['target']['label'], LABEL)
        overlap = self.read_json('overlap.json')
        self.assertTrue(overlap['overlap']['valid'])
        self.assertEqual(overlap['overlap']['invalid_intervals'][0]['reasons'],
                         ['fixture boundary interval retained'])
        self.assertEqual(overlap['overlap']['segments'][-1]['path'],
                         'stock_forward')
        cleanup = self.read_json('cleanup.json')
        self.assertTrue(cleanup['engine_stopped'])
        self.assertFalse(cleanup['production_restored'])
        self.assertEqual(cleanup['launcher_found_sha256'],
                         cleanup['launcher_restored_sha256'])
        self.assertEqual(cleanup['defaults_found_sha256'],
                         cleanup['defaults_restored_sha256'])
        self.assertTrue((self.q / 'client-samples.pressure.jsonl').is_file())
        pressure = [json.loads(line) for line in
                    (self.q / 'client-samples.pressure.jsonl').read_text().splitlines()]
        self.assertTrue(any(row.get('pressure') is True for row in pressure))
        self.assertEqual(self.defaults.read_text().splitlines(),
                         ['GPU_KV_BYTES="1800000000"', 'EXTRA_ENV="OLD=1"',
                          'OVERLAY="/old"', 'S5="0"'])
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')
        calls = self.calls.read_text()
        self.assertIn('restart', calls)
        self.assertIn('stop', calls)
        self.assertFalse([line for line in calls.splitlines()
                          if line.split()[:1] == ['start']])
        for path in self.q.rglob('*'):
            if path.is_file():
                self.assert_no_secret(path.read_text(errors='ignore'))

    def test_armable_false_is_refused_before_any_box_action(self):
        generation = json.loads(
            (self.overlay / 'S5-OVERLAP-GENERATION.json').read_text())
        generation['armable'] = False
        (self.overlay / 'S5-OVERLAP-GENERATION.json').write_text(
            json.dumps(generation, sort_keys=True))
        old.write_manifest(self.overlay)
        self.rehash_requests()
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP pins FAIL', done.stdout)
        self.assertIn('armable:false package refused', done.stdout)
        self.assertEqual(self.step(done.stdout, 'launcher-install'), [])
        self.assertNotIn('restart', self.calls.read_text())
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')
        self.assertFalse((self.q / 'overlap.json').exists())

    def test_qualification_gate_fail_withholds_overlap(self):
        self.mark('fail-qual-rule4_candidate')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP rule4_candidate FAIL', done.stdout)
        self.assertIn('W1 withheld until row1/rule4/placement all PASS', done.stdout)
        self.assertFalse((self.q / 'w1.log').exists())
        gates = self.read_json('gates.json')
        self.assertEqual(gates['outcomes']['rule4_candidate'], 'FAIL')
        self.assertNotEqual(gates['outcomes']['row4'], 'FAIL')
        self.assertIn('STEP overlap-not-run INCONCLUSIVE', done.stdout)
        self.assertEqual([line for line in done.stdout.splitlines()
                          if line.startswith('STEP overlap ') and
                          'overlap-not-run' not in line], [])
        self.assertFalse((self.q / 'overlap.json').read_text().find('"not_run": true') < 0)
        grade = self.read_json('overlap-grade.json')
        self.assertIn(grade['verdict'], ('FAIL', 'INCONCLUSIVE'))
        self.assertTrue((self.q / 'lifecycle.json').is_file())
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],
                         'inactive')

    def test_overlap_fail_is_retained_and_cleanup_runs(self):
        self.mark('fail-overlap')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP overlap FAIL', done.stdout)
        overlap = self.read_json('overlap.json')
        self.assertFalse(overlap['overlap']['valid'])
        self.assertEqual(overlap['overlap']['reasons'],
                         ['fixture overlap FAIL retained'])
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],
                         'inactive')
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')

    def test_engine_crash_skips_lifecycle_and_still_stops_restores(self):
        self.mark('crash-overlap')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP overlap FAIL', done.stdout)
        crash = self.step(done.stdout, 'lifecycle')
        self.assertEqual(len(crash), 1)
        self.assertIn('INCONCLUSIVE', crash[0])
        self.assertIn('not run after an engine crash', crash[0])
        self.assertNotIn('STEP overlap-engine', done.stdout)
        self.assertIn('engine-crashed=1', done.stdout)
        self.assertIn('STEP engine-stop PASS', done.stdout)
        self.assertIn('STEP launcher-restore PASS', done.stdout)
        self.assertFalse(self.read_json('cleanup.json')['production_restored'])
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')

    def test_stop_failure_withholds_both_restores(self):
        self.mark('fail-stop')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP engine-stop FAIL', done.stdout)
        withheld = self.step(done.stdout, 'launcher-restore')
        self.assertTrue(withheld and 'withheld' in withheld[0], done.stdout)
        withheld = self.step(done.stdout, 'defaults-restore')
        self.assertTrue(withheld and 'withheld' in withheld[0], done.stdout)
        self.assertTrue(any('defaults-next-preserve FAIL' in line and
                            'withheld' in line for line in done.stdout.splitlines()))
        cleanup = self.read_json('cleanup.json')
        self.assertEqual(cleanup['engine_state'], 'active')
        self.assertTrue(cleanup['restore_withheld'])
        self.assertFalse(cleanup['engine_stopped'])
        self.assertIn('FLASH_S5_OVERLAP=1', self.task.joinpath(
            'run-flash-prod.sh').read_text() + self.defaults.read_text())
        self.assertFalse(self.read_json('overlap-grade.json')['verdict'] == 'PASS')

    def test_startup_failure_still_restores_exact_files(self):
        self.mark('fail-health')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP restart FAIL', done.stdout)
        self.assertIn('STEP launcher-restore PASS', done.stdout)
        self.assertIn('STEP defaults-restore PASS', done.stdout)
        self.assertEqual(self.defaults.read_text().splitlines(),
                         ['GPU_KV_BYTES="1800000000"', 'EXTRA_ENV="OLD=1"',
                          'OVERLAY="/old"', 'S5="0"'])
        self.assertNotIn('startup-grade.json', os.listdir(self.q))

    def test_missing_w1_artifact_cannot_pass(self):
        self.mark('missing-w1')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('STEP w1_controls INCONCLUSIVE', done.stdout)
        self.assertIn('STEP overlap-not-run INCONCLUSIVE', done.stdout)
        grade = self.read_json('overlap-grade.json')
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE', grade)
        self.assertIn('baseline_invalid:w1_full', grade['missing'])
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],
                         'inactive')

    def test_overlap_artifact_missing_cannot_pass(self):
        self.mark('missing-overlap')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('STEP overlap INCONCLUSIVE', done.stdout)
        lifecycle = self.step(done.stdout, 'lifecycle')
        self.assertEqual(len(lifecycle), 1)
        self.assertIn('PASS', lifecycle[0])
        grade = self.read_json('overlap-grade.json')
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE', grade)
        self.assertIn('overlap_lane_not_valid', grade['missing'])

    def test_changed_owned_next_is_not_moved(self):
        self.mark('change-next')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertTrue(any('defaults-next-preserve FAIL' in line and
                            'withheld' in line for line in done.stdout.splitlines()))
        self.assertTrue((self.defaults.parent / (self.defaults.name + '.next')).is_file())
        self.assertIn('QUEUE_CHANGED=1',
                      (self.defaults.parent / (self.defaults.name + '.next')).read_text())
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],
                         'inactive')

    def test_preexisting_staged_defaults_are_untouched(self):
        staged = self.defaults.parent / (self.defaults.name + '.next')
        staged.write_text('FOREIGN=1\n')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('INCONCLUSIVE defaults-next-exists', done.stdout)
        self.assertEqual(staged.read_text(), 'FOREIGN=1\n')
        self.assertEqual(self.step(done.stdout, 'restart'), [])
        self.assertNotIn('restart', self.calls.read_text())

    def test_existing_output_directory_is_not_modified(self):
        self.q.mkdir()
        sentinel = self.q / 'keep.txt'
        sentinel.write_text('keep\n')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('INCONCLUSIVE output-exists', done.stdout)
        self.assertEqual(sentinel.read_text(), 'keep\n')
        self.assertEqual(sorted(p.name for p in self.q.iterdir()), ['keep.txt'])

    def test_engine_active_is_refused_before_launcher_install(self):
        done = self.run_queue(state={'active': 'active', 'restarts': 0})
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('engine-not-stopped:active', done.stdout)
        self.assertEqual(self.step(done.stdout, 'launcher-install'), [])
        self.assertEqual((self.task / 'run-flash-prod.sh').read_text(),
                         '#!/bin/sh\n# found production launcher\n')
        self.assertIn('"active": "active"', (self.root / 'systemctl-pre.json').read_text())

    def test_arbitrary_env_launcher_pin_is_refused(self):
        env = self.install_fakes()
        env['S5_LAUNCHER_SHA'] = 'e' * 64
        done = subprocess.run(
            ['bash', str(self.h / 'runs/q18a_overlap.draft.sh'), str(self.q)],
            capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('arbitrary-env-launcher-pin', done.stdout)
        self.assertEqual(self.step(done.stdout, 'launcher-install'), [])
        self.assertNotIn('restart', self.calls.read_text())

    def test_missing_overlap_runtime_setting_is_refused(self):
        startup = json.loads((self.r / 'startup.json').read_text())
        startup['identity']['env']['FLASH_S5_OVERLAP'] = '0'
        (self.r / 'startup.json').write_text(json.dumps(startup))
        pins = (self.r / 'SHA256SUMS').read_text().splitlines()
        lines = []
        for line in pins:
            digest, name = line.split(maxsplit=1)
            if name.endswith('startup.json'):
                lines.append(sha(self.r / 'startup.json') + '  ' + name)
            else:
                lines.append(line)
        (self.r / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
        done = self.run_queue()
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('STEP pins FAIL', done.stdout)
        self.assertIn('declared runtime setting FLASH_S5_OVERLAP', done.stdout)
        self.assertEqual(self.step(done.stdout, 'launcher-install'), [])

    def test_tampered_overlap_workload_is_refused(self):
        workload = self.h / 'workloads/q18a-w2-overlap.json'
        workload.write_text(workload.read_text() + '\n')
        done = self.run_queue()
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn('INCONCLUSIVE harness-checksum', done.stdout)
        self.assertEqual(self.step(done.stdout, 'launcher-install'), [])


if __name__ == '__main__':
    unittest.main()
