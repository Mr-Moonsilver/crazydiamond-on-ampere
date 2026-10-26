import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import vmid_job10 as j

ROOT=Path(__file__).resolve().parents[1]

def case(name):
    traces=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)]
    rows=[dict(trace_sha256=t,position=p,total_len=100+p,input_position=99+p,
               forced_token_exact=True,logprob_sha256='a'*64,serialized_sha256='a'*64)
          for t in traces for p in range(256,512)]
    return dict(name=name,rows=rows,trace_digests=traces,overflow=False,capture_defects=[],
                logits_unchanged=True,restored=True,shift_negative_rejected=True,
                calls=512 if name=='permuted_partial_padding' else 256)

def fixture():
    return dict(image=j.s.IMAGE,teacher_sha256=j.TEACHER_SHA,source_pins=dict(j.PINS),
                memory_fraction=.125,bad_readback_rejected=True,cleanup_ok=True,
                cases=[case(n) for n in j.CASES],failures=[],missing=[])

class VM10Tests(unittest.TestCase):
    def test_complete_fixture_and_missing_coverage(self):
        r=fixture();self.assertEqual(j.grade(r)['verdict'],'PASS')
        r['cases'][0]['rows'].pop();self.assertEqual(j.grade(r)['verdict'],'INCONCLUSIVE')
    def test_mismatch_and_duplicate_are_not_absence(self):
        for key,value in [('forced_token_exact',False),('input_position',7),('logprob_sha256','b'*64)]:
            r=fixture();r['cases'][1]['rows'][0][key]=value
            self.assertEqual(j.grade(r)['verdict'],'FAIL')
        r=fixture();r['cases'][0]['rows'][1]=copy.deepcopy(r['cases'][0]['rows'][0])
        self.assertEqual(j.grade(r)['verdict'],'FAIL')
    def test_unproved_negative_cleanup_or_identity_refuses(self):
        for key,value in [('bad_readback_rejected',False),('cleanup_ok',False),('teacher_sha256','c'*64),('memory_fraction',.75),('source_pins',{})]:
            r=fixture();r[key]=value;self.assertNotEqual(j.grade(r)['verdict'],'PASS')
        r=fixture();r['cases'][0]['shift_negative_rejected']=False
        self.assertEqual(j.grade(r)['verdict'],'FAIL')
    def test_pinned_real_serializer_and_teacher_source(self):
        sys.path.insert(0,str(ROOT/'patches/s5_engine'));import teacher_decode as td
        self.assertEqual(j.s.sha(Path(td.__file__)),j.TEACHER_SHA)
        emit=j.serializers(ROOT/'reference/vllm-src')
        for token in (3,24):
            ids=[token]+list(range(20));values=[-float(token)]+[-float(x) for x in range(20)]
            values[-1]=float('-inf')
            top=emit(ids,values)
            self.assertEqual(j.digest(['a'*64,328,list(top.items())]),td.logprob_digest('a'*64,328,ids,values))
    def test_finalizer_crash_and_monitor_remain_binding(self):
        for rc,monitor,expected in [(139,dict(own_samples=3,errors=0),'FAIL'),
                                   (0,dict(own_samples=0,errors=0),'INCONCLUSIVE'),
                                   (0,dict(own_samples=3,errors=0),'PASS')]:
            out=Path(tempfile.mkdtemp(prefix='vm10-finalize-'))
            j.s.write(out/'probe.json',fixture());j.s.write(out/'monitor.json',monitor)
            j.finalize(out,rc)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['verdict'],expected)
            self.assertEqual(set(r['cases']),set(j.CASES))
    def test_owned_queue_and_backstop(self):
        p=ROOT/'harness/runs/vmid_job10.sh';text=p.read_text()
        subprocess.run(['bash','-n',str(p)],check=True)
        for expected in ('Q=/opt/flashnext/runs/job10','flock -n 9','--name flashnext-job10','--network none','--memory 4g','--gpus all','PYTORCH_CUDA_ALLOC_CONF'):
            self.assertIn(expected,text)
        self.assertNotIn('systemctl',text);self.assertNotIn('flashnext-job9',text)
        self.assertIn('set_per_process_memory_fraction(.125,0)',Path(j.__file__).read_text())
        self.assertIn("'flashnext-job10'",(ROOT/'harness/vmid_support.py').read_text())
if __name__=='__main__':unittest.main()
