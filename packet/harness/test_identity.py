#!/usr/bin/env python3
"""Offline identity tests. All Docker invocations go to a temporary mock executable.

The mock runs the actual standard-library container probe against fixture files;
no Docker daemon, engine, GPU, SSH or production paths are accessed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import engine_identity as identity

HERE = Path(__file__).resolve().parent
TMP = HERE.parent.parent / 'tmp'
IMAGE = 'sha256:' + 'a' * 64
REPO_DIGEST = 'example/engine@sha256:' + 'b' * 64
DUMMY = 'fake-sensitive-canary-12345'

# Only this executable can be reached as `docker` from a test. It passes the
# probe unchanged to Python, so hashing, symlinks, exclusions and budget guards
# exercise the same implementation that will run inside the container.
MOCK = '''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
p=Path(os.environ['MOCK_STATE']); d=json.loads(p.read_text()); a=sys.argv[1:]
with open(os.environ['MOCK_LOG'],'a') as f: f.write(json.dumps(a[:5])+'\\n')
if d.get('fail'):
    print(d['fail'], file=sys.stderr); sys.exit(1)
if a[0]=='inspect':
    n=d.get('inspections',0); d['inspections']=n+1; p.write_text(json.dumps(d))
    x=d.get('inspect_after',d['inspect']) if n else d['inspect']
    print(json.dumps([x])); sys.exit(0)
if a[:2]==['image','inspect']:
    print(json.dumps([d['image']])); sys.exit(0)
if a[:2]==['exec','-i']:
    if d.get('break_probe'): print('{}'); sys.exit(0)
    if d.get('change_defaults'):
        Path(d['change_defaults']).write_text('PREFILL_CHUNK=512\\n')
    payload=sys.stdin.read()
    r=subprocess.run([sys.executable, '-c', a[5]], input=payload, text=True, capture_output=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
print('unsupported mock call',file=sys.stderr); sys.exit(1)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_content(data):
    return {'kind': 'file', 'bytes': len(data), 'sha256': sha(data)}


class IdentityTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='identity-test-', dir=TMP))
        self.bin = self.root / 'bin'; self.bin.mkdir()
        mock = self.bin / 'docker'; mock.write_text(MOCK); mock.chmod(0o700)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        MOCK_STATE=str(self.root / 'docker.json'), MOCK_LOG=str(self.root / 'docker.log'),
                        PYTHONDONTWRITEBYTECODE='1')
        self.file = self.root / 'container-qsa.py'; self.file.write_bytes(b'original code\n')
        self.tree = self.root / 'container-work'; self.tree.mkdir()
        (self.tree / 'code.py').write_bytes(b'private code\n')
        (self.tree / 'cache').mkdir(); (self.tree / 'cache' / 'mutable').write_bytes(b'cache')
        self.defaults = self.root / 'defaults'
        self.defaults.write_text('# test fixture\nPREFILL_CHUNK="1024"\nHF_TOKEN="' + DUMMY + '"\nEXTRA_ENV="QSA_HOST_META=1 API_KEY=' + DUMMY + '"\n')
        self.raw_env = ['CUDA_VISIBLE_DEVICES=0,2,1,3', 'QSA_HOST_META=1', 'HF_TOKEN=' + DUMMY]
        self.raw_args = ['/model', '--max-num-batched-tokens', '1024', '--api-key', DUMMY]
        self.state = {'inspect': {'Id': 'c' * 64, 'Image': IMAGE, 'RestartCount': 0, 'Path': 'vllm',
                                 'Args': self.raw_args, 'Config': {'Env': self.raw_env},
                                 'State': {'Running': True, 'Paused': False, 'Restarting': False, 'Status': 'running', 'Pid': 123, 'StartedAt': '2026-10-24T20:00:00Z'},
                                 'Mounts': [{'Type': 'bind', 'Source': '/declared/overlay/qsa.py', 'Destination': str(self.file), 'RW': False, 'Propagation': 'rprivate'},
                                            {'Type': 'bind', 'Source': '/declared/work', 'Destination': str(self.tree), 'RW': True, 'Propagation': 'rprivate'}]},
                      'image': {'Id': IMAGE, 'RepoDigests': [REPO_DIGEST]}}
        self.request = {'schema': identity.REQUEST, 'identity': {
            'image': {'id': IMAGE, 'repo_digests': [REPO_DIGEST]},
            'command': {'path': 'vllm', 'args': ['/model', '--max-num-batched-tokens', '1024', '--api-key', {'redacted_length': len(DUMMY)}]},
            'env': {'CUDA_VISIBLE_DEVICES': '0,2,1,3', 'QSA_HOST_META': '1', 'HF_TOKEN': {'redacted_length': len(DUMMY)}},
            'defaults': {'PREFILL_CHUNK': '1024', 'HF_TOKEN': {'redacted_length': len(DUMMY)},
                         'EXTRA_ENV': {'QSA_HOST_META': '1', 'API_KEY': {'redacted_length': len(DUMMY)}}},
            'mounts': [
                {'type': 'bind', 'source': '/declared/overlay/qsa.py', 'destination': str(self.file), 'rw': False, 'propagation': 'rprivate', 'content': file_content(b'original code\n'), 'watch': []},
                {'type': 'bind', 'source': '/declared/work', 'destination': str(self.tree), 'rw': True, 'propagation': 'rprivate',
                 'content': {'kind': 'directory', 'entries': [{'path': 'code.py', **file_content(b'private code\n')}],
                             'exclusions': [{'path': 'cache', 'reason': 'mutable JIT output, not source code'}]},
                 'watch': [{'path': 'rowcache.off', 'kind': 'absent'}]}]}}
        self.reqpath = self.root / 'requested.json'
        self.save()

    def save(self):
        (self.root / 'docker.json').write_text(json.dumps(self.state))
        self.reqpath.write_text(json.dumps(self.request))

    def capture(self, name='before', extra=(), want=0):
        self.save()
        out = self.root / (name + '.json')
        r = subprocess.run([sys.executable, str(HERE / 'engine_identity.py'), 'capture', str(out), '--request', str(self.reqpath),
                            '--defaults', str(self.defaults), *extra], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, want, r.stdout + r.stderr)
        self.assertNotIn(DUMMY, r.stdout + r.stderr + out.read_text())
        return out

    def check(self, before, after=None, want=0):
        args = [sys.executable, str(HERE / 'engine_identity.py'), 'check', str(before), str(self.reqpath)]
        if after is not None:
            args.append(str(after))
        r = subprocess.run(args, env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, want, r.stdout + r.stderr)
        self.assertNotIn(DUMMY, r.stdout + r.stderr)
        return r

    def test_capture_before_after_through_qlib(self):
        before, after = self.root / 'b.json', self.root / 'a.json'
        shell = 'source "$1"; identity "$2" "$4" --defaults "$5" && identity "$3" "$4" --defaults "$5" && identity_check "$2" "$4" "$3"'
        r = subprocess.run(['bash', '-c', shell, 'test', str(HERE / 'qlib.sh'), str(before), str(after), str(self.reqpath), str(self.defaults)],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn(DUMMY, r.stdout + before.read_text() + after.read_text())
        d = json.loads(before.read_text())
        self.assertEqual(d['identity'], self.request['identity'])
        self.assertEqual(before.stat().st_mode & 0o777, 0o600)
        self.assertIn('"exec", "-i", "' + 'c'*64, (self.root / 'docker.log').read_text())

    def test_stale_host_path_never_used_for_hash(self):
        # Declared host paths do not exist. Hash succeeds from container paths.
        self.capture()

    def test_modified_mount_fails_even_same_size(self):
        self.file.write_bytes(b'modified code\n')
        self.capture(want=1)

    def test_unexpected_file_in_directory_fails(self):
        (self.tree / 'extra.py').write_text('extra')
        self.capture(want=1)

    def test_excluded_cache_can_change(self):
        before = self.capture()
        (self.tree / 'cache' / 'mutable').write_text('changed')
        self.check(before, self.capture('after'))

    def test_watched_runtime_disable_file_fails(self):
        (self.tree / 'rowcache.off').write_text('off')
        self.capture(want=1)

    def test_symlink_hashes_referent(self):
        link = self.tree / 'alias.py'; link.symlink_to(self.file)
        self.request['identity']['mounts'][1]['content']['entries'].insert(0, {'path': 'alias.py', 'kind': 'symlink', 'target': str(self.file), 'referent': file_content(b'original code\n')})
        self.capture()
        self.file.write_bytes(b'different\n')
        self.capture('changed', want=1)

    def test_directory_symlink_does_not_silently_skip_tree(self):
        (self.tree / 'alias').symlink_to(self.tree / 'cache')
        self.capture(want=2)

    def test_missing_unreadable_or_over_budget_never_passes(self):
        self.capture(extra=('--max-bytes', '1'), want=2)
        self.capture('files', extra=('--max-files', '1'), want=2)
        self.state['inspect']['Mounts'][0]['Destination'] = str(self.root / 'missing')
        self.capture('missing', want=2)

    def test_wrong_image_args_env_source_destination_and_rw_fail(self):
        original = copy.deepcopy(self.state)
        mutations = [lambda s: s['image'].update(RepoDigests=[]),
                     lambda s: s['inspect']['Args'].__setitem__(2, '512'),
                     lambda s: s['inspect']['Config']['Env'].append('QSA_ROWCACHE_MIXED=1'),
                     lambda s: s['inspect']['Mounts'][0].update(Source='/different/overlay/qsa.py'),
                     lambda s: s['inspect']['Mounts'][0].update(RW=True)]
        for i, change in enumerate(mutations):
            self.state = copy.deepcopy(original); change(self.state)
            self.capture(str(i), want=1)
        self.state = copy.deepcopy(original)
        self.state['inspect']['Image'] = 'sha256:' + 'd' * 64
        self.state['image']['Id'] = self.state['inspect']['Image']
        self.capture('image', want=1)

    def test_extra_and_missing_mounts_fail(self):
        self.state['inspect']['Mounts'].pop()
        self.capture(want=1)

    def test_duplicate_mount_destination_is_invalid(self):
        self.state['inspect']['Mounts'].append(self.state['inspect']['Mounts'][0])
        self.capture(want=2)

    def test_docker_failure_never_leaks_stderr(self):
        self.state['fail'] = DUMMY
        self.capture(want=2)

    def test_bad_probe_or_stopped_container_is_inconclusive(self):
        self.state['break_probe'] = True
        self.capture(want=2)
        self.state.pop('break_probe')
        self.state['inspect']['State']['Running'] = False
        self.capture('stopped', want=2)

    def test_restart_during_capture_is_inconclusive(self):
        self.state['inspect_after'] = copy.deepcopy(self.state['inspect'])
        self.state['inspect_after']['RestartCount'] = 1
        self.capture(want=2)

    def test_normal_healthcheck_bookkeeping_is_allowed(self):
        self.state['inspect_after'] = copy.deepcopy(self.state['inspect'])
        self.state['inspect_after']['State']['Health'] = {'Status': 'healthy', 'Log': ['different']}
        self.capture()

    def test_same_length_secret_change_within_capture_is_detected(self):
        self.state['inspect_after'] = copy.deepcopy(self.state['inspect'])
        self.state['inspect_after']['Config']['Env'][-1] = 'HF_TOKEN=' + 'z'*len(DUMMY)
        self.capture(want=2)

    def test_default_change_during_capture_is_inconclusive(self):
        self.state['change_defaults'] = str(self.defaults)
        self.capture(want=2)

    def test_same_id_restart_between_measurement_endpoints_fails(self):
        before = self.capture()
        self.state['inspect']['RestartCount'] = 1
        self.state['inspect']['State']['StartedAt'] = '2026-10-24T21:00:00Z'
        self.state['inspect']['State']['Pid'] = 222
        self.check(before, self.capture('after'), want=1)

    def test_changed_container_id_fails(self):
        before = self.capture()
        self.state['inspect']['Id'] = 'e'*64
        self.check(before, self.capture('after'), want=1)

    def test_duplicated_reversed_or_incomplete_artifacts_are_inconclusive(self):
        before, after = self.capture(), self.capture('after')
        self.check(before, before, want=2)
        self.check(after, before, want=2)
        bad = self.root / 'bad.json'; bad.write_text('{}')
        self.check(bad, want=2)
        bad.write_text(json.dumps({'schema': identity.SCHEMA, 'complete': False}))
        self.check(before, bad, want=2)

    def test_request_policy_changes_fail(self):
        before = self.capture()
        self.request['identity']['mounts'][1]['content']['exclusions'][0]['reason'] = 'different scope'
        self.save()
        self.check(before, want=1)

    def test_no_empty_request_can_pass(self):
        self.request['identity']['mounts'] = []
        self.capture(want=2)

    def test_output_never_overwritten(self):
        before = self.capture(); original = before.read_bytes()
        self.capture(want=2)
        self.assertEqual(before.read_bytes(), original)

    def test_defaults_duplicates_and_shell_not_executed(self):
        sentinel = self.root / 'must-not-exist'
        text = f'UNKNOWN=$(touch {sentinel})\nPREFILL_CHUNK=1024\n'
        snap = identity.defaults_snapshot(text.encode())
        self.assertIn('redacted_length', snap['values']['UNKNOWN'])
        self.assertFalse(sentinel.exists())
        with self.assertRaises(identity.EvidenceError):
            identity.defaults_snapshot(b'X=1\nX=2\n')

    def test_redaction_nested_env_json_args_and_comments(self):
        snap = identity.defaults_snapshot(('EXTRA_ARGS="--api-key=' + DUMMY + ' --max-num-seqs 10"\n# ' + DUMMY + '\n').encode())
        args = identity.clean_args(['--api-key=' + DUMMY, '--compilation-config', json.dumps({'api_key': DUMMY, 'mode': 0})])
        env = identity.clean_env({'UNKNOWN': DUMMY, 'NCCL_SECRET': DUMMY, 'QSA_HOST_META': '1'})
        self.assertNotIn(DUMMY, json.dumps([snap, args, env]))
        self.assertEqual(env['QSA_HOST_META'], '1')

    def test_unknown_length_only_operational_values_cannot_pass(self):
        self.state['inspect']['Config']['Env'].append('NEW_OPTION=1')
        self.request['identity']['env']['NEW_OPTION'] = {'redacted_length': 1, 'unclassified': True}
        self.capture(want=2)
        self.state['inspect']['Config']['Env'][-1] = 'NEW_OPTION=2'
        self.capture('same-length', want=2)


if __name__ == '__main__':
    unittest.main()
