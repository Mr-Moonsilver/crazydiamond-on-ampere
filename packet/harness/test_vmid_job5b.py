"""Real phase + real exclusive writer; only CUDA/native calls are CPU fakes."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import vmid_job5 as j
import vmid_job5b as fresh
import vmid_support as s
from test_vmid_job5 import phase as fixture


class Tensor:
    def data_ptr(self):return 256
    def __eq__(self,other):return self
    def item(self):return True


class Stream:
    cuda_stream=11
    def synchronize(self):pass


class PhaseFakes:
    def __init__(self,n):
        self.n=n;self.created=[];self.closed=[];self.replayed=[];self.observer_closed=0
        self.receipt=fixture(n)
        self.native_lib=NS(vmg_create=self.create,vmg_inventory=self.inventory,vmg_replay=self.replay,vmg_close=self.close_graph)
        self.observer_lib=NS(s5ro_bind_context=lambda *a:0,s5ro_bind_graph=self.bind,
                             s5ro_begin_window=self.begin,s5ro_static=object(),s5ro_end_window=object())
        self.torch=NS(int32='int32',full=lambda *a,**kw:Tensor(),empty=lambda *a,**kw:Tensor(),all=lambda *a:Tensor(),
                      cuda=NS(Stream=Stream,synchronize=lambda:None,max_memory_reserved=lambda *a:2097152,OutOfMemoryError=type('FakeOOM',(RuntimeError,),{})))
        self.binding=NS(_lib=self.observer_lib)
        self.ro=NS(ReplayBinding=lambda *a,**kw:self.binding,
                   ObserverLimits=lambda:NS(as_record=lambda:dict(j.LIMITS)),ObserverSession=self.session,
                   _validate_static=lambda *a:None,_validate_window=lambda *a:None)
        self.split=NS(decode_sms=44,prefill_sms=38,total_sms=82,decode_context_id=7,
                      decode_stream_token=33,decode_stream=Stream(),close=lambda:None)
        self.gc=NS(open_split=lambda *a,**kw:self.split,DriverBinding=lambda *a:None)
    def create(self,*a):
        owner=len(self.created)+1;self.created.append(owner)
        a[-2]._obj.value=owner;a[-1]._obj.value=owner+100
        return 0
    def inventory(self,owner,out,length):
        for i,n in enumerate([1386,1,24,1,7,0,0,0]):out[i]=n
        return 0
    def replay(self,owner):self.replayed.append(owner);return 0
    def close_graph(self,owner):self.closed.append(owner);return 0
    def bind(self,token,graph,slot,out):out._obj.value=1419;return 0
    def begin(self,token,generation,out):out._obj.value=1;return 0
    def report(self,method,args,**kw):
        return copy.deepcopy(self.receipt['static' if method is self.observer_lib.s5ro_static else 'window'])
    def close_observer(self):self.observer_closed+=1
    def session(self,*a):
        return NS(_native_reserved=239419152,_token=1,start=lambda:None,
                  _report_call=self.report,binding=self.binding,close=self.close_observer)
    def run(self,out,name):
        with patch.object(j,'GraphBinding',lambda *a:NS(lib=self.native_lib)),patch.object(j.s,'graph_bytes',lambda *a:b'fake-definition'),patch.object(j.time,'sleep'):
            return j.phase(self.torch,self.ro,self.gc,Path('/cpu-only-native'),
                           {'files':{'flash_s5_replay_observer.so':'fake','flash_s5_replay_observer.cpp':'fake'}},
                           self.n,out,name)


class Tests(unittest.TestCase):
    def test_real_phase_writes_all_snapshots_once_in_both_windows(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for name,n in j.WINDOWS.items():
                fake=PhaseFakes(n);r=fake.run(out,name)
                self.assertNotIn('phase_exception',r['missing'])
                self.assertEqual(r['replays'],n)
                self.assertTrue(r['cleanup_ok'])
                self.assertEqual(fake.created,list(range(1,16)))
                self.assertEqual(fake.closed,list(range(15,0,-1)))
                self.assertEqual(len(fake.replayed),n)
                self.assertEqual(fake.observer_closed,1)
                paths=sorted(out.glob(name+'-progress-*.json'))
                self.assertEqual(len(paths),15)
                counts=sorted(len(json.loads(p.read_text())['graphs']) for p in paths)
                self.assertEqual(counts,list(range(1,16)))
                self.assertEqual(len(json.loads((out/f'{name}-progress-startup_pressure-00.json').read_text())['graphs']),1)
                self.assertEqual(len(json.loads((out/f'{name}-progress-bound-06.json').read_text())['graphs']),15)
                self.assertFalse((out/f'{name}-progress.json').exists())
                self.assertTrue((out/f'{name}-startup.json').exists())
                self.assertTrue((out/f'{name}-window.json').exists())
            self.assertEqual(len(list(out.glob('*-progress-*.json'))),30)

    def test_existing_snapshot_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);existing=out/'short-progress-startup_pressure-00.json'
            s.write(existing,{'previous':'retained'});original=existing.read_bytes()
            fake=PhaseFakes(40);r=fake.run(out,'short')
            self.assertEqual(r['error_type'],'FileExistsError')
            self.assertEqual(r['replays'],0);self.assertTrue(r['cleanup_ok'])
            self.assertEqual(existing.read_bytes(),original)
            self.assertEqual(fake.closed,[1])

    def test_queue_is_pure_identity_rename(self):
        root=Path(__file__).resolve().parent
        original=(root/'runs/vmid_job5.sh').read_text()
        path=root/'runs/vmid_job5b.sh'
        self.assertEqual(path.read_text(),original.replace('job5','job5b').replace('JOB5','JOB5B'))
        subprocess.run(['bash','-n',str(path)],check=True)
        self.assertIs(fresh.job.probe,j.probe);self.assertIs(fresh.job.grade,j.grade)

    def test_sidecar_owns_only_job5b(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'monitor.stop').touch();calls=[]
            def command(args):
                calls.append(args)
                if any('query-compute-apps' in x for x in args):return 0,'GPU-x, 1, python3, 2900\nGPU-x, 2, UnrealEditor, 2700\n'
                if 'top' in args:return 0,'PID\n1\n'
                return 0,''
            with patch.object(s,'command',command):s.monitor(out,'flashnext-job5b')
            self.assertEqual([x for x in calls if 'stop' in x],[['sudo','-n','docker','stop','-t','5','flashnext-job5b']])

if __name__=='__main__':unittest.main()
