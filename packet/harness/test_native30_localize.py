import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import native30_localize as plugin
import e3_native30_grade as grade

HERE=Path(__file__).parent

class Instrumentation(unittest.TestCase):
    def test_all_ranks_prelaunch_sync_and_original_result(self):
        for rank in range(4):
            output=io.StringIO();order=[]
            class Dispatch:
                native=NS(state=NS(worker_rank=rank));runner=NS(device='cpu-device-fixture')
                def operation(self,command):
                    self.before=json.loads(output.getvalue().splitlines()[-1][len(plugin.PREFIX):]);order.append('launch');return command
            plugin.wrap(Dispatch,'operation','LAYER')
            command=NS(op_seq=3,layer_start=2,layer_end=3,transaction=NS(chunk_start=0,chunk_end=1024,txn_seq=1))
            fake=NS(cuda=NS(synchronize=lambda device:order.append('sync'),memory_reserved=lambda d:17,mem_get_info=lambda d:(23,100)))
            with patch.dict(os.environ,FLASH_NATIVE30_SYNC='native'),patch.dict(sys.modules,torch=fake),contextlib.redirect_stderr(output):
                obj=Dispatch();self.assertIs(obj.operation(command),command)
            self.assertEqual(order,['launch','sync']);self.assertEqual(obj.before['event'],'begin');self.assertEqual(obj.before['rank'],rank)
            rows=[json.loads(s[len(plugin.PREFIX):]) for s in output.getvalue().splitlines()]
            self.assertEqual([r['event'] for r in rows],['begin','sync-begin','barrier-memory','barrier-memory','sync-end','end'])

    def test_sync_failure_identifies_engine_boundary_without_exception_text(self):
        class Worker:
            rank=1
            def execute_model(self,*args,**kwargs):return 3
        plugin.wrap(Worker,'execute_model','DECODE_STEP')
        def bad(device):raise RuntimeError('PRIVATE_ERROR_TEXT')
        fake=NS(cuda=NS(synchronize=bad,memory_reserved=lambda d:17,mem_get_info=lambda d:(23,100)));hooks=NS(decode_scope=lambda obj:contextlib.nullcontext())
        stream=io.StringIO()
        with patch.dict(os.environ,FLASH_NATIVE30_SYNC='decode'),patch.dict(sys.modules,{'torch':fake,'vllm.v1.flash_e3.worker_hooks':hooks}),contextlib.redirect_stderr(stream):
            with self.assertRaises(RuntimeError):Worker().execute_model(None)
        self.assertNotIn('PRIVATE_ERROR_TEXT',stream.getvalue());row=json.loads(stream.getvalue().splitlines()[-1][len(plugin.PREFIX):])
        self.assertEqual((row['classification'],row['rank'],row['step']),('ENGINE',1,1))

    def test_pp_transfer_never_inserts_a_barrier(self):
        class D:
            rank=0
            def send(self,c):return c
        plugin.wrap(D,'send','PP_SEND');c=NS(op_seq=1,layer_start=None,layer_end=None,transaction=NS(chunk_start=0,chunk_end=1024,txn_seq=1))
        with patch.dict(os.environ,FLASH_NATIVE30_SYNC='both'),patch.object(plugin,'sync',side_effect=AssertionError('PP barrier')),contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(D().send(c),c)

    def test_idempotent_wrapper_and_disabled_sync(self):
        class D:
            rank=2
            def execute_model(self,x):return x
        plugin.wrap(D,'execute_model','DECODE_STEP');f=D.execute_model;plugin.wrap(D,'execute_model','DECODE_STEP');self.assertIs(D.execute_model,f)
        with patch.dict(os.environ,FLASH_NATIVE30_SYNC='none'),contextlib.redirect_stderr(io.StringIO()):self.assertEqual(D().execute_model(7),7)

class Queue(unittest.TestCase):
    def test_short_queue_keeps_real_restore_functions_and_hard_bound(self):
        original=(HERE/'runs/q18a_native28.draft.sh').read_text();short=(HERE/'runs/q18a_native30_localize.draft.sh').read_text()
        for name in ('restore_launcher','restore_defaults','stop_engine','write_cleanup'):
            def body(text):return text.split(name+'() {',1)[1].split('\n}\n',1)[0]
            self.assertEqual(body(short),body(original))
        self.assertNotIn('qualification_phase row1',short);self.assertNotIn('qualification_phase rule4',short)
        self.assertEqual(short.count('timeout --signal=TERM --kill-after=20s 480s python3'),1)
        self.assertIn('--kill-after=420s 1200s',short);self.assertIn('time.time()+1200+420',short)
        self.assertEqual(short.count('--out "$Q/localize-grade.json"'),1)
        subprocess.run(['bash','-n',str(HERE/'runs/q18a_native30_localize.draft.sh')],check=True)

    def test_actual_exit_trap_runs_stop_then_restores_on_failure(self):
        text=(HERE/'runs/q18a_native30_localize.draft.sh').read_text()
        body='on_exit() {'+text.split('on_exit() {',1)[1].split('\n}\n',1)[0]+'\n}\n'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);calls=p/'calls';script=p/'test.sh'
            # All external/device operations are explicit CPU shell fixtures.
            stubs='\n'.join(n+'() { echo '+n+' >> "$CALLS"; }' for n in ('stop_pressure','stop_sampler','capture_logs','capture_end','stop_host_observer','host_crash_detected','archive_owned_next','write_cleanup','step','record','finish_overlap'))
            stubs+='\nstop_engine() { echo stopped >> "$CALLS"; ENGINE_STATE=inactive; }\nrestore_launcher() { test "$ENGINE_STATE" = inactive; echo launcher >> "$CALLS"; }\nrestore_defaults() { test "$ENGINE_STATE" = inactive; echo defaults >> "$CALLS"; }\npython3() { return 0; }\n'
            script.write_text('set -u\n'+stubs+body+'\nCALLS="$1"; Q="$2"; H="$2"; Q_OWNED=0; ROOT_FREE_GUARD=0; INTERRUPTED=0; STARTED=1; EVIDENCE_DONE=0; ENGINE_CRASHED=0; DONE=1; FAILS=1; INCS=0\ntrap on_exit EXIT\nexit 1\n')
            out=subprocess.run(['bash',str(script),str(calls),d],capture_output=True,text=True)
            self.assertEqual(out.returncode,1,out.stderr);rows=calls.read_text().splitlines();self.assertLess(rows.index('stopped'),rows.index('launcher'));self.assertLess(rows.index('launcher'),rows.index('defaults'))

    def test_all_rank_grade_and_missing_receipt_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log';rows=[]
            for rank in range(4):
                for label in ('FIRST_ADMISSION','DECODE_STEP'):
                    for event in ('begin','end'):rows.append(plugin.PREFIX+json.dumps(dict(rank=rank,event=event,label=label,call_id=rank*10+len(label))))
            p.write_text('\n'.join(rows));self.assertEqual(grade.grade(p,{'verdict':'PASS'})['verdict'],'PASS')
            p.write_text('\n'.join(rows[:-4]));self.assertEqual(grade.grade(p,{'verdict':'PASS'})['failure_class'],'HARNESS')
            self.assertEqual(grade.grade(p,None,True)['failure_class'],'ENGINE')

class StagingAndPairs(unittest.TestCase):
    def test_last_unmatched_begin_and_memory_are_retained_per_rank(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log'
            rows=[dict(rank=1,event='begin',label='LAYER',call_id=10),
                  dict(rank=1,event='barrier-memory',call_id=10,memory={'free_bytes':23,'reserved_bytes':17}),
                  dict(rank=1,event='exception',call_id=10,error_type='RuntimeError'),
                  dict(rank=1,event='begin',label='DECODE_STEP',call_id=11),
                  dict(rank=1,event='end',label='DECODE_STEP',call_id=11)]
            p.write_text('\n'.join(plugin.PREFIX+json.dumps(r) for r in rows))
            row=grade.grade(p,None)['per_rank']['1']
            self.assertEqual(row['last_unmatched_begin']['call_id'],10)
            self.assertEqual(row['last_memory']['memory'],{'free_bytes':23,'reserved_bytes':17})

    def test_staging_has_no_engine_stop_or_idle_dependency(self):
        import ast
        path=HERE.parent/'tools/native30_owner_functions.py'
        tree=ast.parse(path.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        source=ast.get_source_segment(path.read_text(),functions['preflight'])
        self.assertNotIn('live_guards(',source);self.assertNotIn('systemctl',source);self.assertNotIn('nvidia-smi',source)
        arm=ast.get_source_segment(path.read_text(),functions['live_guards'])
        self.assertIn('stopped_unit(',arm);self.assertIn('nvidia-smi',arm);self.assertNotIn('manifest(',arm)
        declaration=(HERE/'runs/q18a_native30_localize_declare.sh').read_text()
        self.assertNotIn('systemctl is-active',declaration);self.assertNotIn('gpu_idle',declaration)
        self.assertIn('f9fb82486ef3132ac3e6fa9c9ed12fb33225b08e1727727376d3e5ac1a678311',declaration)

class PlacementCLI(unittest.TestCase):
    def test_direct_existing_placement_keeps_no_current_epoch_qualification_claim(self):
        import e3_native30_row4 as short
        import e3_native_qualification_driver as qd
        import e3_q18a_qualification as qual
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'plan.json').write_text('{}');(p/'workload.json').write_text('{}')
            args=NS(workload=str(p/'workload.json'),url='http://127.0.0.1:1',prefill_plan=str(p/'plan.json'),
                snapshot_max_bytes=64*1024**2,store=str(p/'store'),run_id='native30-test',model='m',epoch=7,timeout_s=90,out=str(p/'out.json'))
            calls=[]
            class Placement:
                def __init__(self,**kw):self.kw=kw;calls.append(('construct',kw))
                def run(self,driver,**kw):calls.append(('run',kw));return {'cleanup':{'restored':True}}
            with patch.object(qd,'NativePlacementRunner',Placement),patch.object(qd,'QualificationDriver',return_value=object()),patch.object(qual,'_workload_object',return_value=object()):
                self.assertEqual(short.run(args),0)
            self.assertEqual(len(calls),2);self.assertEqual(calls[0][1]['prefill_plan'],{})
            record=json.loads((p/'out.json').read_text());self.assertFalse(record['release_qualified']);self.assertEqual(record['skipped'],['row1','rule4_controls','rule4_candidate'])
            self.assertNotIn('row1_sha256',record)

if __name__=='__main__':unittest.main()
