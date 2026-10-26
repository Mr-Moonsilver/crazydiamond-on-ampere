"""Full q17b queue lifecycle over fake OS commands; no box/GPU/root."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import hashlib
import test_e3_startup_queue as old
HERE=Path(__file__).resolve().parent
FAKE_CLIENTS='''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--run-id');a,rest=p.parse_known_args()
out=Path(a.out);out.mkdir()
rc=1 if (Path(os.environ['FAKE_BIN'])/'fail-clients').exists() else 0
verdict='FAIL' if rc else 'PASS'
for name in ('correctness.json','placement.json','performance.json'):(out/name).write_text('{}')
if not (Path(os.environ['FAKE_BIN'])/'missing-clients-grade').exists():
 with (out/'clients-grade.json').open('x') as f:json.dump(dict(schema=1,run_id=a.run_id,verdict=verdict,final_route='full',observer_during_performance=False,outcomes=dict(correctness=verdict,placement=verdict,performance=verdict),files=['correctness.json','placement.json','performance.json']),f)
print('Q17B_CLIENTS '+verdict)
raise SystemExit(rc)
'''
class FullQueue(unittest.TestCase):
    setUp=old.QueueRun.setUp
    tearDown=old.QueueRun.tearDown
    fake=old.QueueRun.fake
    assert_no_secret=old.QueueRun.assert_no_secret
    def make_harness(self):
        old.QueueRun.make_harness(self)
        for name in ('q17b_full.draft.sh','q17b_full_settings.sh'):
            shutil.copyfile(HERE/'runs'/name,self.h/'runs'/name)
        shutil.copyfile(HERE/'e3_private_auth.py',self.h/'e3_private_auth.py')
        (self.h/'e3_q17b_clients.py').write_text(FAKE_CLIENTS)
        for name in ('e3_decode_correctness.py','e3_w1_decode_driver.py'):
            shutil.copyfile(HERE/name,self.h/name)
        (self.h/'identity_declare.py').write_text('S5_LAUNCHER_SHA='+repr(hashlib.sha256(old.launcher_text().encode()).hexdigest())+'\n')
        (self.h/'workloads').mkdir()
        for name in ('q17b-correctness.json','q17b-w1.json'):
            shutil.copyfile(HERE/'workloads'/name,self.h/'workloads'/name)
        closure=json.loads((self.h/'e3_build_closure.json').read_text())
        closure['builder_sources']['native/flash_s5_replay_observer.cpp']='f'*64
        (self.h/'e3_build_closure.json').write_text(json.dumps(closure));old.write_manifest(self.h)
    def make_overlay(self):
        old.QueueRun.make_overlay(self)
        (self.overlay/'e3/native/flash_s5_replay_observer.so').write_bytes(b'fake-observer-library')
        self.replay=old.sha(self.overlay/'e3/native/flash_s5_replay_observer.so')
        (self.overlay/'e3/native/q17-helper-build.json').write_text('{}');old.write_manifest(self.overlay)
    def make_requests(self):
        old.QueueRun.make_requests(self)
        d=json.loads((self.r/'startup.json').read_text())
        d['identity']['defaults']['FLASH_S5_AUTH_ENV_FILE']=str(self.r/'private/vllm-api.env')
        d['identity']['env'].update(FLASH_S5_REPLAY_OBSERVER='1',FLASH_S5_REPLAY_LIB_SHA256=self.replay,FLASH_S5_GREEN_LIB='/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native/flash_s5_green.so',FLASH_S5_REPLAY_LIB='/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native/flash_s5_replay_observer.so')
        (self.r/'startup.json').write_text(json.dumps(d))
        (self.r/'request-plan.json').write_text(json.dumps({'model':'fixture-model'}))
        (self.r/'private').mkdir(mode=0o700)
        shutil.copyfile(self.auth_file,self.r/'private/vllm-api.env');(self.r/'private/vllm-api.env').chmod(0o600)
        paths=[Path(line.split(maxsplit=1)[1]) for line in (self.r/'SHA256SUMS').read_text().splitlines()]
        paths += [self.h/'runs/q17b_full.draft.sh',self.h/'runs/q17b_full_settings.sh',self.r/'request-plan.json']
        (self.r/'SHA256SUMS').write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
    def install_fakes(self):
        env=old.QueueRun.install_fakes(self);env.update(S5_REPLAY_LIB_SHA256=self.replay,Q17B_FULL_BOUNDED='1');return env
    def run_queue(self):
        return subprocess.run(['bash',str(self.h/'runs/q17b_full.draft.sh'),str(self.q)],env=self.install_fakes(),capture_output=True,text=True,timeout=120)
    def test_all_phases_then_cleanup(self):
        result=self.run_queue();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('STEP full-clients PASS',result.stdout);self.assertIn('Q17B-FULL DONE rc=0',result.stdout)
        self.assertIn('STEP launcher-restore PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
        self.assert_no_secret(result.stdout+result.stderr)
        self.assertEqual(json.loads((self.q/'queue-grade.json').read_text())['verdict'],'PASS')
    def test_startup_failure_never_runs_clients(self):
        (self.bin/'fail-health').write_text('');result=self.run_queue();self.assertNotEqual(result.returncode,0)
        self.assertNotIn('STEP full-clients',result.stdout);self.assertIn('STEP engine-stop PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
    def test_client_failure_keeps_failure_and_restores(self):
        (self.bin/'fail-clients').write_text('');result=self.run_queue();self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP full-clients FAIL',result.stdout);self.assertIn('STEP engine-log PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
    def test_missing_client_artifact_cannot_pass(self):
        (self.bin/'missing-clients-grade').write_text('');result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP clients-grade-consistency FAIL',result.stdout)
    def test_existing_output_not_modified(self):
        self.q.mkdir();(self.q/'sentinel').write_text('keep');result=self.run_queue();self.assertEqual(result.returncode,2)
        self.assertEqual(sorted(p.name for p in self.q.iterdir()),['sentinel'])
    def test_bad_private_file_refused_before_launcher_install(self):
        (self.r/'private/vllm-api.env').chmod(0o644);result=self.run_queue();self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertNotIn('STEP launcher-install PASS',result.stdout)
    def test_stderr_engine_error_cannot_pass(self):
        original=old.FAKE_DOCKER.replace("print(Path(os.environ['FAKE_LOG']).read_text(), end='')", "print('',end=''); print('Traceback',file=sys.stderr)")
        from unittest import mock
        with mock.patch.object(old,'FAKE_DOCKER',original):result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr);self.assertIn('STEP engine-errors FAIL',result.stdout)
if __name__=='__main__':unittest.main()
