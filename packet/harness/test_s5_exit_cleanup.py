import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from s5_resident import PLETransport,close_resident_resources
from s5_exit_regrade import check_log,regrade,warning_times
from s5_value import grade
from test_s5_value import Value


class Cleanup(unittest.TestCase):
    def test_close_releases_cycle_and_buffers_before_contexts(self):
        order=[];alive=[True];refs=[]
        class Resource:
            def __init__(self,name):self.name=name;refs.append(weakref.ref(self))
            def __del__(self):order.append(('release',self.name,alive[0]))
        class Lane:
            def reset(self):pass
            def ple(self):pass
        lane=Lane();lane.host=Resource('kv-host');lane.kv=Resource('kv-view')
        lane.ple_host=Resource('ple-host');lane.ple_uva=Resource('ple-view')
        obj=object.__new__(PLETransport)
        obj.closed=False;obj.pending=True;obj.lane=lane
        obj.original_reset=lane.reset;obj.original_ple=lane.ple
        obj.packed=Resource('packed');obj.raw=Resource('raw');obj.base=Resource('base')
        obj.copy_done=SimpleNamespace(synchronize=lambda:order.append('copy-fence'))
        lane.reset=obj.reset;lane.ple=obj.lookup
        def flush():
            self.assertTrue(all(r() is None for r in refs));self.assertTrue(alive[0]);order.append('flush')
        def destroy():
            self.assertIn('flush',order);alive[0]=False;order.append('destroy-context')
        context=SimpleNamespace(sd=SimpleNamespace(synchronize=lambda:order.append('decode-fence')),
                                sp=SimpleNamespace(synchronize=lambda:order.append('prefill-fence')),close=destroy)
        graph=SimpleNamespace(reset=lambda:order.append('graph-reset'))
        progress=SimpleNamespace(update=lambda name:order.append(name))
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:order.append('device-fence')),
                              accelerator=SimpleNamespace(empty_host_cache=flush))
        with patch.dict(sys.modules,{'torch':torch}):
            result=close_resident_resources([graph],[context],[lane],[obj],progress)
            before=len(order);obj.close();self.assertEqual(len(order),before)
        self.assertTrue(result['host_cache_drained_before_contexts'])
        self.assertLess(order.index('prefill-fence'),order.index('graph-reset'))
        self.assertLess(order.index('graph-reset'),order.index('copy-fence'))
        self.assertTrue(all(x[2] for x in order if isinstance(x,tuple)))
        ref=weakref.ref(obj);obj=None
        self.assertIsNone(ref(),'bound-method transport cycle survived close')
        self.assertIs(lane.reset.__self__,lane)

    def test_fence_failure_preserves_ownership(self):
        obj=object.__new__(PLETransport);obj.closed=False;obj.pending=True
        lane=SimpleNamespace();obj.lane=lane;obj.packed=object()
        obj.copy_done=SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(RuntimeError('fence')))
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}):
            with self.assertRaises(RuntimeError):obj.close()
        self.assertIs(obj.lane,lane);self.assertTrue(obj.pending);self.assertFalse(obj.closed)
        self.assertIsNotNone(obj.packed)

    def test_cache_flush_failure_never_destroys_context(self):
        order=[]
        context=SimpleNamespace(sd=SimpleNamespace(synchronize=lambda:None),
                                sp=SimpleNamespace(synchronize=lambda:None),close=lambda:order.append('destroy'))
        def fail():raise RuntimeError('flush')
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None),accelerator=SimpleNamespace(empty_host_cache=fail))
        with patch.dict(sys.modules,{'torch':torch}):
            with self.assertRaises(RuntimeError):close_resident_resources([], [context], [], [], SimpleNamespace(update=lambda _:None))
        self.assertEqual(order,[])
        torch.accelerator=SimpleNamespace()
        with patch.dict(sys.modules,{'torch':torch}):
            with self.assertRaisesRegex(RuntimeError,'requires'):close_resident_resources([], [context], [], [], None)


class ExitRegrade(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]/'tmp';root.mkdir(exist_ok=True)
        self.path=Path(tempfile.mkdtemp(prefix='s5-exit-',dir=root))
        self.addCleanup(shutil.rmtree,self.path,ignore_errors=True)
        self.raw=Value().fixture();self.raw['rung']='profile-terminal'
        for rank in self.raw['ranks']:
            for name in ('correctness','noop_correctness','resident_correctness'):
                for lane in ('decode','prefill'):
                    rank[name][lane].update(solo_runs=4,distinct_solo_outputs=1,
                        output_matches_solo=True,solo_states_identical=True,finite=True,
                        comparisons=[dict(solo=i,output_bitwise=True) for i in range(4)])
        self.write('result.json',self.raw)
        self.stamped=dict(grade(self.raw),verdict='INCONCLUSIVE',launcher_rc=1,reason='process did not exit cleanly')
        self.write('grade.json',self.stamped);(self.path/'launch-rc').write_text('1\n')
        self.write('handoff.json',dict(own_container_stopped=1,gpu_idle=1,engine_untouched=True))
        self.phases=[]
        for rank,t in ((0,1792903114.599422),(1,1792903114.5618222)):
            c=dict(kind='phase',rank=rank,pid=68+rank,phase='cleanup',wall_s=1792903114.512,seq=1)
            f=dict(c,phase='completed',wall_s=t,seq=2)
            self.write(f'result.json.rank{rank}.progress.json',f)
            (self.path/f'result.json.rank{rank}.progress.jsonl').write_text(json.dumps(c)+'\n'+json.dumps(f)+'\n')
            self.phases.append(dict(rank=rank,pid=68+rank,cleanup_wall_s=c['wall_s'],completed_wall_s=t))
        self.log='\n'.join([
            '[W1025 04:38:34.553282305 CachingHostAllocator.cpp:26] Warning: Exception in pinned allocator free(), rethrowing (function free)',
            '[W1025 04:38:34.553282315 CachingHostAllocator.cpp:26] Warning: Exception in pinned allocator free(), rethrowing (function free)',
            'c10::AcceleratorError', 'CUDA error: context is destroyed', 'CUDA error: context is destroyed',
            'Exception raised from record at /pytorch/c10/cuda/CUDAEvent.h:152',
            'PyGC_Collect', 'Py_FinalizeEx', 'Fatal Python error: Aborted',
            'rank : 0 (local_rank: 0)\nexitcode : -6 (pid: 68)\ntraceback : Signal 6 (SIGABRT)',
            'rank : 1 (local_rank: 1)\nexitcode : -6 (pid: 69)\ntraceback : Signal 6 (SIGABRT)'])
        (self.path/'run.log').write_text(self.log)

    def write(self,name,value): (self.path/name).write_text(json.dumps(value)+'\n')
    def run_grade(self):return regrade(self.path,self.path/'handoff.json')

    def test_qualified_measurement_keeps_process_failure_and_original(self):
        original=(self.path/'grade.json').read_bytes();r=self.run_grade()
        self.assertEqual(r['verdict'],'QUALIFIED_MEASUREMENT');self.assertTrue(r['measurement_accepted'])
        self.assertFalse(r['process_exit_clean']);self.assertEqual(r['original_verdict'],'INCONCLUSIVE')
        self.assertFalse(r['timing']['warnings_after_all_completed'])
        self.assertEqual(r['timing']['warning_utc_ns'][0],1792903114553282305)
        self.assertEqual((self.path/'grade.json').read_bytes(),original)
        self.assertEqual(r['measurement']['ranks'][0]['decode_share_vs_solo'],.5) # soft target miss stays usable

    def test_warning_before_cleanup_even_fractionally_is_rejected(self):
        log=self.log.replace('34.553282305','34.511999999')
        with self.assertRaises(ValueError):check_log(log,self.phases)
        self.assertEqual(warning_times(self.log,1792903114.5)[0]%10**9,553282305)

    def test_invalid_device_context_only_under_same_teardown_evidence(self):
        log=self.log.replace('CUDA error: context is destroyed','CUDA error: invalid device context',1)
        (self.path/'run.log').write_text(log)
        r=self.run_grade();self.assertEqual(r['verdict'],'QUALIFIED_MEASUREMENT')
        self.assertIn('invalid device context',r['timing']['cuda_errors'])
        for bad in (log.replace('Py_FinalizeEx','missing'),log.replace('34.553282305','34.511999999'),
                    log.replace('invalid device context','invalid device ordinal')):
            (self.path/'run.log').write_text(bad)
            self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')

    def test_other_failures_never_qualify(self):
        for text in ('RuntimeError: something unrelated', 'CUDA error: illegal memory access',
                     'NCCL WARN failed', 'watchdog expired', 'Fatal Python error: Segmentation fault'):
            with self.subTest(text=text):
                (self.path/'run.log').write_text(self.log+'\n'+text)
                self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')
        (self.path/'run.log').write_text(self.log.replace('(pid: 69)','(pid: 999)'))
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')

    def test_tail_bitwise_correctness_and_rank_errors_remain_hard(self):
        for failure in ('tail','bitwise','correctness'):
            raw=copy.deepcopy(self.raw)
            if failure=='tail':raw['ranks'][0]['modes']['overlap']['prefill_covers_decode_tail']=False
            elif failure=='bitwise':raw['ranks'][0]['correctness']['decode']['bitwise_first']=False
            else:raw['ranks'][1]['correctness']['passed']=False
            self.write('result.json',raw)
            self.assertEqual(self.run_grade()['verdict'],'FAIL' if failure=='correctness' else 'INCONCLUSIVE')
        self.write('result.json',self.raw)
        self.write('result.json.rank1.error.json',dict(fatal_correctness=True))
        self.assertEqual(self.run_grade()['verdict'],'FAIL')

    def test_matching_later_solo_is_valid_but_novel_output_is_not(self):
        raw=copy.deepcopy(self.raw)
        v=raw['ranks'][0]['resident_correctness']['prefill']
        v.update(bitwise_first=False,distinct_solo_outputs=2)
        v['comparisons']=[dict(solo=i,output_bitwise=i==3) for i in range(4)]
        self.write('result.json',raw)
        self.write('grade.json',dict(grade(raw),verdict='INCONCLUSIVE',launcher_rc=1,reason='process did not exit cleanly'))
        r=self.run_grade();self.assertEqual(r['verdict'],'QUALIFIED_MEASUREMENT')
        self.assertFalse(next(x for x in r['measurement']['ranks'] if x['rank']==0 and x['mode']=='overlap_resident')['bitwise_equal'])
        v['comparisons'][3]['output_bitwise']=False
        self.write('result.json',raw)
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')

    def test_missing_idle_incomplete_rank_and_wrong_downgrade(self):
        self.write('handoff.json',dict(own_container_stopped=1,gpu_idle=0,engine_untouched=True))
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')
        self.write('handoff.json',dict(own_container_stopped=1,gpu_idle=1,engine_untouched=True))
        self.write('grade.json',dict(self.stamped,reason='tail missing'))
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')
        self.write('grade.json',self.stamped)
        (self.path/'launch-rc').write_text('124\n')
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')
        (self.path/'launch-rc').write_text('1\n')
        self.write('result.json.rank1.progress.json',dict(phase='cleanup',rank=1,pid=69))
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')

    def test_malformed_evidence_and_recorded_fail(self):
        self.write('result.json',[])
        self.assertEqual(self.run_grade()['verdict'],'INCONCLUSIVE')
        self.write('grade.json',dict(self.stamped,verdict='FAIL'))
        self.write('result.json.rank1.error.json',dict(fatal_correctness=False))
        self.assertEqual(self.run_grade()['verdict'],'FAIL')

    def test_cli_refuses_overwrite(self):
        script=Path(__file__).with_name('s5_exit_regrade.py')
        cmd=[sys.executable,str(script),str(self.path),'--handoff',str(self.path/'handoff.json'),'--out',str(self.path/'regrade.json')]
        self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)
        before=(self.path/'regrade.json').read_bytes()
        self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
        self.assertEqual((self.path/'regrade.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
