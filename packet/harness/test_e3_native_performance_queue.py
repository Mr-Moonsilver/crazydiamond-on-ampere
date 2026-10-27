"""Fresh native queue orchestration over fake OS and phase executables only.

No GPU/root/engine/box action. Native packet, numerical, placement, memory,
authentication and package predicates have separate actual-module CPU suites;
these fakes exercise order, path/argument contracts, refusal and owned cleanup.
"""
import json
import datetime
from pathlib import Path
import shutil
import subprocess
import unittest

import test_e3_overlap_performance_queue as old
from test_e3_startup_queue import write_manifest

HERE=Path(__file__).resolve().parent

PHASE_SCRIPT=r'''
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('url','model','run-id','epoch','out','phase','receipt','template','workload','store','request-plan','placement-workload','prefill-plan','window-id','teacher-baseline','snapshot-max-bytes','timeout-s','package','w1','gates','performance','samples'):
 p.add_argument('--'+name)
a=p.parse_args()
name=Path(__file__).stem
root=Path(os.environ['FAKE_BIN'])
def write(path,obj):
 with open(path,'x') as f:json.dump(obj,f)
def fail(key):return (root/('fail-'+key)).exists()
if a.run_id:assert a.run_id==os.environ['RUN_ID']
if a.epoch:assert a.epoch=='9'
if name=='e3_native_startup_grade':
 write(a.out,{'verdict':'PASS','engine_epoch':9})
elif name=='e3_native_request_plans':
 write(a.out,{'schema':'CPU plan fixture'})
 write(a.receipt,{'verdict':'PASS'})
elif name=='e3_q18a_qualification':
 assert all((a.store,a.request_plan,a.workload,a.placement_workload,a.prefill_plan,a.window_id,a.teacher_baseline))
 assert int(a.snapshot_max_bytes)==64*1024**2
 if a.phase=='rule4_candidate':assert os.environ['Q18A_TEACHER_ARTIFACT_SHA']=='b'*64
 if a.phase=='row4':
  assert os.environ['Q18A_TEACHER_ARTIFACT_SHA']=='b'*64
  assert os.environ['Q18A_ROW1_SHA']=='a'*64
 bad=fail('qual-'+a.phase)
 write(a.out,{'verdict':'FAIL' if bad else 'PASS','phase':a.phase,
              'row1_sha256':'a'*64,'teacher_artifact_sha256':'b'*64})
 sys.exit(1 if bad else 0)
elif name=='e3_native_performance':
 gates=json.load(open(a.gates));assert all(v=='PASS' for v in gates['outcomes'].values())
 if (root/'missing-overlap').exists():sys.exit(2)
 bad=fail('overlap')
 write(a.out,{'schema':'s5-native-performance/1','grade':{'verdict':'FAIL' if bad else 'PASS',
              'target':{'status':'SHORTFALL'},'target_is_gate':False}})
 crashed=(root/'crash-overlap').exists()
 if crashed:(root/'engine-crash').touch()
 sys.exit(1 if bad or crashed else 0)
elif name=='e3_native_memory_grade':
 bad=fail('memory');write(a.out,{'verdict':'FAIL' if bad else 'PASS'})
 sys.exit(1 if bad else 0)
elif name=='e3_q18a_lifecycle':
 bad=fail('lifecycle');write(a.out,{'verdict':'FAIL' if bad else 'PASS'})
 sys.exit(1 if bad else 0)
'''

AUTH_SCRIPT=r'''
import json,os,sys,time
from pathlib import Path
base,script,*args=sys.argv[1:];name=Path(script).name
assert base==os.environ['PROBE_URL']
root=Path(os.environ['FAKE_BIN'])
if name=='leaktest.py':
 assert args==['8','3'];print('sessions 8 rounds 3 bad 0 in 1s')
elif name=='leak_partial_tail.py':
 assert args[2:5]==['10','4','7568']
 assert os.environ['MODE']=='inside' and os.environ['GEN']=='200'
 assert os.environ['CANCEL_FRAC']=='0.3' and os.environ['NEED_PARTIAL']=='1'
 Path(args[5]).write_text('{}')
elif name=='pressure2.py':
 assert args[2:5]==['2','200000','1800'] and os.environ['PRESSURE_MAX_TOKENS']=='128'
 Path(args[5]).write_text('{}')
 Path(os.environ['PRESSURE_READY']).write_text(json.dumps({'first_tokens':[1,1]}))
 while not Path(args[6]).exists():time.sleep(.01)
elif name=='pp_canary_bridge.py':
 command,q,*other=args;q=Path(q)
 if command=='prepare':
  assert other[-2:]==['10','2']
  (q/'canary.request').write_text('{}');(q/'canary.result').write_text('{}')
 elif command=='grade':print(json.dumps({'verdict':'PASS'}))
else:raise RuntimeError('unexpected fixture tool')
'''

class NativeQueueTests(old.OverlapQueue,unittest.TestCase):
    step=old.QueueFlowTests.step
    read_json=old.QueueFlowTests.read_json
    mark=old.QueueFlowTests.mark
    rehash_requests=old.QueueFlowTests.rehash_requests

    def make_harness(self):
        super().make_harness()
        for name in ('q18a_native1.draft.sh','q18a_native1_settings.sh'):
            shutil.copyfile(HERE/'runs'/name,self.h/'runs'/name)
        for name in ('e3_native_startup_grade','e3_native_request_plans','e3_q18a_qualification',
                     'e3_native_performance','e3_native_memory_grade','e3_q18a_lifecycle'):
            (self.h/(name+'.py')).write_text(PHASE_SCRIPT)
        (self.h/'e3_native_w1.py').write_text(old.FAKE_W1)
        (self.h/'e3_auth_exec.py').write_text(AUTH_SCRIPT)
        (self.h/'e3_native_runtime_client.py').write_text('''
class NativeRuntimeClient:
 def __init__(self,*a,**k):pass
 def prepare(self,*a):return {'route':'full','CPU_fixture':True}
 def resume(self):return {'resumed':True}
''')
        for name in ('e3_native_qualification_driver.py','e3_native_teacher_sequence.py',
                     'leaktest.py','leak_partial_tail.py','pressure2.py','pp_canary_bridge.py'):
            (self.h/name).write_text('# CPU queue fixture only\n')
        (self.h/'pp_release_grade.py').write_text("import json;print(json.dumps({'verdict':'PASS'}))\n")
        (self.h/'pressure_wait.py').write_text("import pathlib,sys;pathlib.Path(sys.argv[4]).write_text('{}')\n")
        builder=self.h/'e3_builder';builder.mkdir()
        (builder/'decode_probe.py').write_text("GENERATED_SOURCES=()\n")
        (builder/'native_readiness.py').write_text('''
import os
from pathlib import Path
def verify_finalized_package(root,modules):
 if (Path(os.environ['FAKE_BIN'])/'fail-native-closure').exists():raise ValueError('fixture closure refusal')
 return {'native_ready':True,'CPU_fixture':True}
''')
        for name in ('q18a-native-row1-template.json','q18a-native-placement-prefill.json',
                     'q17b-teacher-decode-v3.json','q17b-correctness-precontrols.json'):
            shutil.copyfile(HERE/'workloads'/name,self.h/'workloads'/name)
        (self.h/'baselines').mkdir()
        shutil.copyfile(HERE/'baselines/q15-teacher-forced-rule3.json',self.h/'baselines/q15-teacher-forced-rule3.json')
        write_manifest(self.h)

    def make_requests(self):
        # The owner canary's remote fresh process derives this public path.
        self.r=self.root/'out-requests';self.r.mkdir()
        super().make_requests()
        plan=json.loads((self.r/'request-plan.json').read_text())
        plan['canary_client_auth_file']='/home/<service-user>/.flashnext/canary-client-auth.env'
        (self.r/'request-plan.json').write_text(json.dumps(plan))
        pins=self.r/'SHA256SUMS'
        lines=pins.read_text().replace(str(self.h/'runs/q18a_overlap.draft.sh'),
                                      str(self.h/'runs/q18a_native1.draft.sh'))
        lines=lines.replace(str(self.h/'runs/q18a_overlap_settings.sh'),
                            str(self.h/'runs/q18a_native1_settings.sh'))
        pins.write_text(lines);self.rehash_requests()

    def run_queue(self,env=None):
        settings=self.install_fakes()
        if env:settings.update(env)
        return subprocess.run(['bash',str(self.h/'runs/q18a_native1.draft.sh'),str(self.q)],
                              capture_output=True,text=True,timeout=180,env=settings,cwd=self.root)

    def test_every_native_phase_and_unchanged_hygiene_then_exact_restore(self):
        done=self.run_queue();self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        outcome=self.read_json('queue-grade.json')
        self.assertEqual(outcome['verdict'],'PASS')
        for name in ('startup','row1','rule4_controls','rule4_candidate','row4','w1_controls',
                     'overlap','memory','lifecycle','cross-session-leak','partial-leak',
                     'pressure-lifetime','canary'):
            self.assertEqual(outcome['phase_outcomes'][name],'PASS',done.stdout)
        self.assertLess(done.stdout.index('STEP row4 PASS'),done.stdout.index('STEP w1_controls PASS'))
        self.assertEqual(outcome['measured_performance']['target']['status'],'SHORTFALL')
        self.assertFalse(outcome['release_qualified'])
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],'inactive')
        self.assertEqual((self.task/'run-flash-prod.sh').read_text(),'#!/bin/sh\n# found production launcher\n')
        self.assert_no_secret(done.stdout+done.stderr)

    def withheld(self,phase):
        self.mark('fail-qual-'+phase)
        done=self.run_queue();self.assertEqual(done.returncode,1,done.stdout+done.stderr)
        self.assertNotIn('STEP overlap PASS',done.stdout)
        self.assertFalse((self.q/'w1.json').exists())
        self.assertFalse((self.q/'canary.request').exists())
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],'inactive')

    def test_row1_refusal_withholds_w1_w2_and_hygiene(self):self.withheld('row1')
    def test_precontrol_refusal_withholds_w1_w2_and_hygiene(self):self.withheld('rule4_controls')
    def test_candidate_refusal_withholds_w1_w2_and_hygiene(self):self.withheld('rule4_candidate')
    def test_row4_refusal_withholds_w1_w2_and_hygiene(self):self.withheld('row4')

    def test_native_closure_failure_precedes_launcher_mutation(self):
        self.mark('fail-native-closure');done=self.run_queue()
        self.assertEqual(done.returncode,2,done.stdout+done.stderr)
        self.assertNotIn('STEP launcher-install',done.stdout)
        self.assertNotIn('restart',self.calls.read_text())

    def test_memory_breach_is_hard_fail_and_preserves_measured_performance(self):
        self.mark('fail-memory');done=self.run_queue()
        self.assertEqual(done.returncode,1,done.stdout+done.stderr)
        self.assertEqual(self.read_json('queue-grade.json')['phase_outcomes']['memory'],'FAIL')
        self.assertTrue((self.q/'overlap.json').is_file())
        self.assertEqual(self.read_json('cleanup.json')['engine_state'],'inactive')

    def test_engine_crash_skips_hygiene_and_always_stops(self):
        self.mark('crash-overlap');done=self.run_queue()
        self.assertEqual(done.returncode,1,done.stdout+done.stderr)
        self.assertIn('STEP lifecycle INCONCLUSIVE',done.stdout)
        self.assertFalse((self.q/'canary.request').exists())
        self.assertIn('STEP engine-stop PASS',done.stdout)

    def test_failed_stop_withholds_both_restores(self):
        self.mark('fail-stop');done=self.run_queue()
        self.assertEqual(done.returncode,1,done.stdout+done.stderr)
        cleanup=self.read_json('cleanup.json')
        self.assertEqual(cleanup['engine_state'],'active')
        self.assertTrue(cleanup['restore_withheld'])
        self.assertIn('STEP launcher-restore FAIL',done.stdout)
        self.assertIn('STEP defaults-restore FAIL',done.stdout)

    def test_failed_startup_restores_without_phase_collection(self):
        self.mark('fail-health');done=self.run_queue()
        self.assertEqual(done.returncode,1,done.stdout+done.stderr)
        self.assertNotIn('STEP row1 ',done.stdout)
        self.assertIn('STEP launcher-restore PASS',done.stdout)
        self.assertIn('STEP defaults-restore PASS',done.stdout)

    def test_exact_private_path_is_in_fresh_declare_settings_and_queue(self):
        path='/home/<service-user>/.flashnext/canary-client-auth.env'
        for name in ('q18a_native1.draft.sh','q18a_native1_declare.sh','q18a_native1_settings.sh'):
            text=(HERE/'runs'/name).read_text();self.assertIn(path,text)
            subprocess.run(['bash','-n',str(HERE/'runs'/name)],check=True)

    def test_native2_is_name_only_queue_settings_and_declaration_clone(self):
        # All hardware phases, workloads, gates, bounds and settings remain the
        # already-tested native1 sequence. The declaration/outer-log fixes live
        # in identity and the owner runner; do not slip a runtime delta here.
        for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
            before=(HERE/'runs'/('q18a_native1'+suffix)).read_text()
            after=(HERE/'runs'/('q18a_native2'+suffix)).read_text()
            self.assertEqual(after,before.replace('q18a_native1','q18a_native2')
                             .replace('q18a-native1','q18a-native2'))
            subprocess.run(['bash','-n',str(HERE/'runs'/('q18a_native2'+suffix))],check=True)

    def test_late_real_launch_refuses_before_paths_or_engine_actions(self):
        # Deliberately leave fake-OS mode: the slot guard must return before
        # resolving any actual box path, key or engine interface.
        end=int(datetime.datetime(2026,10,28,6,tzinfo=datetime.timezone.utc).timestamp())
        self.fake('date',"#!/usr/bin/env python3\nimport sys\nprint("+str(end)+" if '-d' in sys.argv else "+str(end-15300+1)+")\n")
        env=self.install_fakes();env['Q17_TEST_OFFLINE']='0'
        # install_fakes does not replace date, and prevents a child re-exec.
        done=subprocess.run(['bash',str(HERE/'runs/q18a_native1.draft.sh')],
            cwd=self.root,env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(done.returncode,2,done.stdout+done.stderr)
        self.assertIn('insufficient-slot-for-full-bound-and-restore',done.stdout)
        self.assertEqual(self.calls.read_text(),'')
        text=(HERE/'runs/q18a_native1.draft.sh').read_text()
        self.assertIn("2026-10-28 06:00:00 UTC",text)


if __name__=='__main__':unittest.main()
