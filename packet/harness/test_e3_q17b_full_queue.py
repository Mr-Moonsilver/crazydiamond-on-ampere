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
if (Path(os.environ['FAKE_BIN'])/'change-next').exists():
 Path(os.environ['FAKE_DEFAULTS']+'.next').write_text('replacement staged defaults')
rc=1 if (Path(os.environ['FAKE_BIN'])/'fail-clients').exists() else 0
verdict='FAIL' if rc else 'PASS'
for name in ('correctness.json','placement.json','performance.json'):(out/name).write_text('{}')
if not (Path(os.environ['FAKE_BIN'])/'missing-clients-grade').exists():
 with (out/'clients-grade.json').open('x') as f:json.dump(dict(schema=1,run_id=a.run_id,verdict=verdict,final_route='full',observer_during_performance=False,outcomes=dict(correctness=verdict,placement=verdict,performance=verdict),files=['correctness.json','placement.json','performance.json']),f)
print('Q17B_CLIENTS '+verdict)
raise SystemExit(rc)
'''
class FullQueue(unittest.TestCase):
    queue_name='q17b_full3.draft.sh'
    settings_name='q17b_full_settings.sh'
    correctness_name='q17b-correctness.json'
    setUp=old.QueueRun.setUp
    tearDown=old.QueueRun.tearDown
    fake=old.QueueRun.fake
    assert_no_secret=old.QueueRun.assert_no_secret
    def make_harness(self):
        old.QueueRun.make_harness(self)
        for name in (self.queue_name,self.settings_name):
            shutil.copyfile(HERE/'runs'/name,self.h/'runs'/name)
        shutil.copyfile(HERE/'e3_private_auth.py',self.h/'e3_private_auth.py')
        (self.h/'e3_q17b_clients.py').write_text(FAKE_CLIENTS)
        for name in ('e3_decode_correctness.py','e3_w1_decode_driver.py'):
            shutil.copyfile(HERE/name,self.h/name)
        (self.h/'identity_declare.py').write_text('S5_LAUNCHER_SHA='+repr(hashlib.sha256(old.launcher_text().encode()).hexdigest())+'\n')
        (self.h/'workloads').mkdir()
        for name in (self.correctness_name,'q17b-w1.json'):
            shutil.copyfile(HERE/'workloads'/name,self.h/'workloads'/name)
        closure=json.loads((self.h/'e3_build_closure.json').read_text())
        closure['builder_sources']['native/flash_s5_replay_observer.cpp']='f'*64
        (self.h/'e3_build_closure.json').write_text(json.dumps(closure));old.write_manifest(self.h)
    def make_overlay(self):
        old.QueueRun.make_overlay(self)
        (self.overlay/'e3/native/flash_s5_replay_observer.so').write_bytes(b'fake-observer-library')
        self.replay=old.sha(self.overlay/'e3/native/flash_s5_replay_observer.so')
        # Match the real generator: helper evidence is CUDA source, not its Python wrapper.
        self.helper=old.sha(self.overlay/'e3/native/flash_s5_green.cu')
        self.assertNotEqual(self.helper,old.sha(self.overlay/'e3/green_context.py'))
        ep=self.overlay/'e3/decode-probe-evidence.json'
        ev=json.loads(ep.read_text());ev['helper_source_sha256']=self.helper
        ep.write_text(json.dumps(ev))
        gp=self.overlay/'S5-OVERLAP-GENERATION.json'
        gen=json.loads(gp.read_text());gen['diagnostic_modes']['decode_only']['evidence']=ev
        gp.write_text(json.dumps(gen))
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
        paths += [self.h/'runs'/self.queue_name,self.h/'runs'/self.settings_name,self.r/'request-plan.json']
        (self.r/'SHA256SUMS').write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
    def install_fakes(self):
        env=old.QueueRun.install_fakes(self);env.update(S5_REPLAY_LIB_SHA256=self.replay,Q17B_FULL_BOUNDED='1');return env
    def run_queue(self):
        return subprocess.run(['bash',str(self.h/'runs'/self.queue_name),str(self.q)],env=self.install_fakes(),capture_output=True,text=True,timeout=120)
    def test_all_phases_then_cleanup(self):
        result=self.run_queue();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('STEP full-clients PASS',result.stdout);self.assertIn('Q17B-FULL DONE rc=0',result.stdout)
        self.assertIn('STEP launcher-restore PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
        self.assert_no_secret(result.stdout+result.stderr)
        self.assertIn('STEP defaults-next-preserve PASS',result.stdout)
        self.assertFalse(Path(str(self.defaults)+'.next').exists())
        self.assertTrue((self.q/'private/defaults.next.generated').is_file())
        self.assertEqual(json.loads((self.q/'queue-grade.json').read_text())['verdict'],'PASS')
    def test_python_wrapper_hash_cannot_certify_native_helper(self):
        ep=self.overlay/'e3/decode-probe-evidence.json'
        ev=json.loads(ep.read_text())
        ev['helper_source_sha256']=old.sha(self.overlay/'e3/green_context.py')
        ep.write_text(json.dumps(ev))
        gp=self.overlay/'S5-OVERLAP-GENERATION.json'
        gen=json.loads(gp.read_text());gen['diagnostic_modes']['decode_only']['evidence']=ev
        gp.write_text(json.dumps(gen));old.write_manifest(self.overlay)
        pins=self.r/'SHA256SUMS'
        paths=[Path(line.split(maxsplit=1)[1]) for line in pins.read_text().splitlines()]
        pins.write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
        result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP pins FAIL',result.stdout)
        self.assertIn('native helper source evidence differs',result.stdout)
        self.assertNotIn('STEP launcher-install',result.stdout)

    def test_startup_failure_never_runs_clients(self):
        (self.bin/'fail-health').write_text('');result=self.run_queue();self.assertNotEqual(result.returncode,0)
        self.assertNotIn('STEP full-clients',result.stdout);self.assertIn('STEP engine-stop PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
        self.assertFalse(Path(str(self.defaults)+'.next').exists())
        self.assertIn('STEP defaults-next-preserve PASS',result.stdout)
    def test_client_failure_keeps_failure_and_restores(self):
        (self.bin/'fail-clients').write_text('');result=self.run_queue();self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP full-clients FAIL',result.stdout);self.assertIn('STEP engine-log PASS',result.stdout);self.assertIn('STEP defaults-restore PASS',result.stdout)
    def test_missing_client_artifact_cannot_pass(self):
        (self.bin/'missing-clients-grade').write_text('');result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP clients-grade-consistency FAIL',result.stdout)
    def test_changed_owned_next_is_not_moved(self):
        (self.bin/'change-next').write_text('')
        result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP defaults-next-preserve FAIL',result.stdout)
        self.assertEqual(Path(str(self.defaults)+'.next').read_text(),'replacement staged defaults')
        self.assertFalse((self.q/'private/defaults.next.generated').exists())
        self.assertIn('STEP defaults-restore PASS',result.stdout)

    def test_preexisting_next_is_preserved(self):
        staged=Path(str(self.defaults)+'.next');staged.write_text('preexisting')
        result=self.run_queue()
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertEqual(staged.read_text(),'preexisting')
        self.assertFalse((self.q/'private/defaults.next.generated').exists())

    def test_failed_setcfg_output_is_preserved_privately(self):
        self.setcfg.write_text(self.setcfg.read_text()+'\nraise SystemExit(1)\n')
        pins=self.r/'SHA256SUMS'
        paths=[Path(line.split(maxsplit=1)[1]) for line in pins.read_text().splitlines()]
        pins.write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
        result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP configure FAIL',result.stdout)
        self.assertIn('STEP defaults-next-preserve PASS',result.stdout)
        self.assertFalse(Path(str(self.defaults)+'.next').exists())
        self.assertTrue((self.q/'private/defaults.next.generated').exists())

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
class PrecontrolQueue(FullQueue):
    queue_name='q17b_full4.draft.sh'
    correctness_name='q17b-correctness-precontrols.json'

    def test_new_identity_changes_only_workload_and_queue_references(self):
        for old_name,new_name in [('q17b_full3.draft.sh',self.queue_name),
                                  ('q17b_full3_declare.sh','q17b_full4_declare.sh')]:
            actual=(HERE/'runs'/new_name).read_text()
            actual='\n'.join(line for line in actual.splitlines()
                             if not line.startswith(('# Supersedes queue3', '# A pre-control mismatch'))) + '\n'
            expected=(HERE/'runs'/old_name).read_text().replace(
                'q17b_full3.draft.sh',self.queue_name).replace(
                'q17b-correctness.json',self.correctness_name)
            self.assertEqual(actual,expected)
        previous=json.loads((HERE/'workloads/q17b-correctness.json').read_text())
        following=json.loads((HERE/'workloads'/self.correctness_name).read_text())
        self.assertEqual(following.pop('route_plan'),['full','full','split','full'])
        previous.pop('route_plan')
        following.pop('workload');previous.pop('workload')
        self.assertEqual(following,previous)

class ProtocolV2Queue(FullQueue):
    queue_name='q17b_full5.draft.sh'
    correctness_name='q17b-correctness-precontrols.json'
    def setUp(self):
        from unittest.mock import patch
        import make_s5_overlap,make_e2_arena
        def current():return old.full11_launcher_text().replace('e3/capture_streams.py ', '')
        self._patches=[patch.object(old,'launcher_text',current),patch.object(old,'LAUNCHER_SHA',hashlib.sha256(current().encode()).hexdigest())]
        for p in self._patches:p.start()
        FullQueue.setUp(self)
    def tearDown(self):
        FullQueue.tearDown(self)
        for p in reversed(self._patches):p.stop()
    def make_harness(self):
        FullQueue.make_harness(self)
        for name in ('e3_teacher_decode_client.py','e3_teacher_decode_grade.py','teacher_forced.py'):
            shutil.copyfile(HERE/name,self.h/name)
        shutil.copyfile(HERE/'workloads/q17b-teacher-decode-v3.json',self.h/'workloads/q17b-teacher-decode-v3.json')
        (self.h/'baselines').mkdir(exist_ok=True)
        shutil.copyfile(HERE/'baselines/q15-teacher-forced-rule3.json',self.h/'baselines/q15-teacher-forced-rule3.json')
        code=FAKE_CLIENTS.replace("('correctness.json','placement.json','performance.json')","('teacher-forced.json','data-invariant.json','correctness.json','placement.json','launch-config.json','performance.json')").replace("outcomes=dict(correctness=verdict,placement=verdict,performance=verdict)","outcomes=dict(teacher_forced=verdict,data_invariant=verdict,placement=verdict,performance=verdict)").replace("files=['correctness.json','placement.json','performance.json']","files=['teacher-forced.json','data-invariant.json','correctness.json','placement.json','launch-config.json','performance.json']")
        (self.h/'e3_q17b_clients.py').write_text(code);old.write_manifest(self.h)
    def make_requests(self):
        FullQueue.make_requests(self)
        p=self.r/'request-plan.json';data=json.loads(p.read_text());data['snapshot_max_bytes']=64*1024**2;p.write_text(json.dumps(data))
        pins=self.r/'SHA256SUMS';paths=[Path(line.split(maxsplit=1)[1]) for line in pins.read_text().splitlines()]
        pins.write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
    def test_byte_and_teacher_grades_required_for_success(self):
        p=self.h/'e3_q17b_clients.py';p.write_text(p.read_text().replace('data_invariant=verdict','data_invariant="INCONCLUSIVE"'));old.write_manifest(self.h)
        result=self.run_queue()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn('STEP clients-grade-consistency FAIL',result.stdout)

class Full11Queue(ProtocolV2Queue):
    queue_name='q17b_full11.draft.sh'
    def setUp(self):
        from unittest.mock import patch
        import sys
        sys.path.insert(0,str(old.ROOT/'patches/s5_engine'))
        import make_s5_overlap,make_e2_arena
        def current():return old.full11_launcher_text()
        self._patches=[patch.object(old,'launcher_text',current),patch.object(old,'LAUNCHER_SHA',hashlib.sha256(current().encode()).hexdigest())]
        for p in self._patches:p.start()
        FullQueue.setUp(self)

    def test_only_identity_changes_from_protocol_v2_queue(self):
        import identity_declare
        expected_pin='f1310837266716ac90875cb637490cf8d08a5a81f3008d46affe57fa4b928b54'
        self.assertNotEqual(identity_declare.S5_LAUNCHER_SHA,expected_pin)  # Current generator has full12 mounts.
        self.assertEqual(old.LAUNCHER_SHA,expected_pin)
        for before,after in [('q17b_full5.draft.sh',self.queue_name),('q17b_full5_declare.sh','q17b_full11_declare.sh')]:
            expected=(HERE/'runs'/before).read_text().replace(
                '68d6c1b9aee62895cb4d92cd8a6d7f1734c0091d25234156382524d0ede4fb62',expected_pin).replace(
                'q17b_full5.draft.sh',self.queue_name).replace('QNAME=${QNAME:-q17b}','QNAME=${QNAME:-q17b-full11}')
            actual=(HERE/'runs'/after).read_text()
            actual='\n'.join(line for line in actual.splitlines() if not line.startswith('# Full11:'))+'\n'
            self.assertEqual(actual,expected)
            subprocess.run(['bash','-n',str(HERE/'runs'/after)],check=True)

class Full12Queue(ProtocolV2Queue):
    queue_name='q17b_full12.draft.sh'
    settings_name='q17b_full12_settings.sh'
    def setUp(self):
        from unittest.mock import patch
        import make_s5_overlap,make_e2_arena
        def current():
            # Full12 is an immutable executed identity. Later native modules
            # belong to a fresh candidate, not to a rewritten full12 queue.
            text=make_s5_overlap._s5_launcher(make_e2_arena.launcher((old.ROOT/'deploy/run-flash-prod.sh').read_text()))
            return text.replace('e3/admission_metadata.py ', '').replace('e3/worker_bridge.py ', '')
        self._patches=[patch.object(old,'launcher_text',current),patch.object(old,'LAUNCHER_SHA',hashlib.sha256(current().encode()).hexdigest())]
        for p in self._patches:p.start()
        FullQueue.setUp(self)
    def make_harness(self):
        ProtocolV2Queue.make_harness(self)
        p=self.h/'e3_q17b_clients.py';s=p.read_text()
        s=s.replace("('teacher-forced.json',", "('engine-accounting.json','teacher-forced.json',",1)
        s=s.replace('schema=1,run_id=a.run_id,verdict=verdict',
                    'schema=1,engine_accounting_required=True,run_id=a.run_id,verdict=verdict',1)
        p.write_text(s);old.write_manifest(self.h)
    def make_requests(self):
        ProtocolV2Queue.make_requests(self)
        p=self.r/'startup.json';d=json.loads(p.read_text())
        d['identity']['env']['FLASH_S5_ENGINE_OFFERS']='1';p.write_text(json.dumps(d))
        pins=self.r/'SHA256SUMS';paths=[Path(line.split(maxsplit=1)[1]) for line in pins.read_text().splitlines()]
        pins.write_text(''.join(old.sha(p)+'  '+str(p)+'\n' for p in paths))
    def test_missing_separate_engine_artifact_refuses_success(self):
        p=self.h/'e3_q17b_clients.py';s=p.read_text().replace("'engine-accounting.json',",'')
        p.write_text(s);old.write_manifest(self.h)
        r=self.run_queue();self.assertEqual(r.returncode,1,r.stdout+r.stderr)
        self.assertIn('STEP clients-grade-consistency FAIL',r.stdout)
    def test_settings_change_only_the_declared_offer_instrument(self):
        old_text=(HERE/'runs/q17b_full_settings.sh').read_text()
        new_text=(HERE/'runs'/self.settings_name).read_text().split('\n',1)[1]
        self.assertEqual(new_text,old_text.replace('FLASH_S5_REPLAY_OBSERVER=1',
                          'FLASH_S5_REPLAY_OBSERVER=1 FLASH_S5_ENGINE_OFFERS=1'))

if __name__=='__main__':unittest.main()
