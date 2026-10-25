import copy
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from s5_variants import SplitController,score_profile

class Controller(unittest.TestCase):
    def setUp(self):
        self.table={(4,s):dict(validated=True,decode_share=1.,p99_ratio=1.,prefill_share=p) for s,p in [(24,.85),(34,.75),(44,.60)]}
    def test_fences_and_unknown_cost_block_switch(self):
        c=SplitController(self.table)
        for _ in range(5):self.assertFalse(c.choose(4,False,.1,1.)['switch'])
        self.assertFalse(c.choose(4,True,None,1.)['switch'])
        self.assertFalse(c.choose(4,True,2.,1.)['switch'])
        self.assertEqual(c.choose(4,True,.1,1.)['split'],24)
    def test_hysteresis_and_uncalibrated_load(self):
        c=SplitController(self.table)
        for _ in range(3):self.assertFalse(c.choose(4,True,.1,1.)['switch'])
        self.assertTrue(c.choose(4,True,.1,1.)['switch'])
        self.assertEqual(c.choose(5,True,.1,1.)['reason'],'infeasible_or_uncalibrated')
    def test_never_interpolate_or_use_unsafe_candidate(self):
        for v in self.table.values():v['validated']=False
        self.assertEqual(SplitController(self.table).choose(4,True,.1,1.)['reason'],'infeasible_or_uncalibrated')
        self.table[4,24]['decode_share']=1.;self.table[4,24]['validated']=False
        self.assertFalse(SplitController(self.table).choose(4,True,.1,1.)['switch'])
    def test_soft_targets_and_balanced_objective(self):
        self.table[4,24].update(decode_share=.79,prefill_share=.85)
        self.table[4,34].update(decode_share=.95,prefill_share=.50)
        self.table[4,44].update(decode_share=.98,prefill_share=.40)
        c=SplitController(self.table)
        for _ in range(4):r=c.choose(4,True,.1,1.)
        self.assertEqual(r['split'],24) # Below80 still better combined progress.
        self.table[4,34].update(decode_share=.87,prefill_share=.87)
        self.assertEqual(c.choose(4,True,.1,1.)['split'],34) # Keep optimizing.

class Frontier(unittest.TestCase):
    def fixture(self):
        m=dict(decode=dict(decode_steps_s=100.,gap_p99_ms=2.),prefill=dict(prefill_steps_s_upper=10.),
               decode_split=dict(decode_steps_s=99.),overlap=dict(decode_steps_s=96.,gap_p99_ms=2.3,
               prefill_steps_s=8.2,prefill_covers_decode_tail=True,prefill_tail_order_ambiguous=False))
        return dict(ranks=[dict(rank=i,completed=True,audit=dict(verdict='PASS'),modes=copy.deepcopy(m)) for i in (0,1)])
    def test_worst_rank_and_missing_audit(self):
        d=self.fixture();self.assertTrue(score_profile(d)['decode_target_met'])
        d['ranks'][1]['modes']['overlap']['decode_steps_s']=90.
        self.assertTrue(score_profile(d)['decode_target_met']) # Operator target now80.
        d['ranks'][1]['audit']['verdict']='INCONCLUSIVE'
        with self.assertRaises(AssertionError):score_profile(d)
    def test_uncertain_tail_not_pass(self):
        d=self.fixture();d['ranks'][0]['modes']['overlap']['prefill_tail_order_ambiguous']=True
        with self.assertRaises(AssertionError):score_profile(d)
    def test_queue_syntax_and_runtime_scope(self):
        root=Path(__file__).resolve().parent
        for name in ('runs/q_s5_variant.draft.sh','runs/q_s5_ladder.draft.sh','runs/q_s5_model_bisect2.draft.sh','runs/q_s5_terminal_measure.draft.sh','s5_variant_entry.sh'):
            s=(root/name).read_text();self.assertEqual(subprocess.run(['bash','-n'],input=s,text=True).returncode,0)
            for bad in ('systemctl stop','systemctl start','docker rm','--rm'):self.assertNotIn(bad,s)
        self.assertNotIn('globaltimer',(root/'s5_terminal_mode.py').read_text().split('import threading',1)[1])

    def test_terminal_bounds_and_same_lane_gaps(self):
        from s5_terminal_stats import summarize
        def row(a,b):return dict(submitted_s=a,observed_s=b,gpu_ms=(b-a)*1000)
        rows=dict(decode=[row(0,.1),row(.9,1.)],prefill=[row(-.1,.05),row(.2,.4),row(.8,1.1)])
        r=summarize(rows,dict(decode=1.,prefill=1.2),[900.],True,'overlap')
        self.assertEqual(r['decode_steps_s'],2.)
        self.assertEqual((r['prefill_steps_s'],r['prefill_steps_s_upper']),(1.,3.))
        self.assertEqual(r['gap_p99_ms'],900.);self.assertTrue(r['prefill_covers_decode_tail'])
        self.assertFalse(summarize(rows,dict(decode=1.,prefill=1.2),[900.],False,'overlap')['prefill_covers_decode_tail'])
    def test_missing_grade_case_is_kept_and_cannot_pass(self):
        import json
        from s5_bisect_summary import summarize,CASES
        h=Path(__file__).resolve().parent;tmp=h.parent.parent/'tmp';tmp.mkdir(exist_ok=True)
        d=Path(tempfile.mkdtemp(prefix='s5-summary-',dir=tmp))
        r=summarize(d);self.assertEqual(len(r['cases']),7);self.assertEqual(r['verdict'],'INCONCLUSIVE')
        for name in CASES[:-1]:
            p=d/name/'model-terminal';p.mkdir(parents=True)
            (p/'grade.json').write_text(json.dumps(dict(verdict='PASS')))
        (d/'steps.txt').write_text('marlin-green-cap rc=2 connections=8\n')
        r=summarize(d);self.assertEqual(r['cases'][-1]['case_rc'],2)
        self.assertFalse(r['cases'][-1]['grade_present']);self.assertEqual(r['verdict'],'INCONCLUSIVE')
        p=d/CASES[-1]/'model-terminal';p.mkdir(parents=True)
        (p/'result.json.rank0.error.json').write_text(json.dumps(dict(fatal_correctness=True)))
        self.assertEqual(summarize(d)['verdict'],'FAIL')

class Cap(unittest.TestCase):
    def test_fail_closed_and_ablation_shapes(self):
        from s5_marlin_cap import checked_result
        from s5_clock import Inconclusive
        from s5_ablation import configure
        checked_result(24,1,82);checked_result(0,0,0)
        for args in ((24,0,82),(58,1,24)):
            with self.assertRaises(Inconclusive):checked_result(*args)
        s={'layers':4,'mix':['linear_attention']*3+['full_attention']}
        original=copy.deepcopy(s);configure(s,'mixed');self.assertEqual(s,original)
        configure(s,'qsa');self.assertEqual(s['mix'],['full_attention']*4)
        configure(s,'gdn');self.assertEqual(s['mix'],['linear_attention']*4)

    def test_native_interposer_on_cpu_mock(self):
        # Tests the REAL interceptor C++ against fake CUDA headers/runtime. This
        # is no CUDA compile/ABI qualification, and never initializes a GPU.
        h=Path(__file__).resolve().parent
        tmp=h.parent.parent/'tmp';tmp.mkdir(exist_ok=True)
        d=Path(tempfile.mkdtemp(prefix='s5-cap-cpu-',dir=tmp))
        (d/'cuda.h').write_text('''#pragma once
#include <stdint.h>
typedef void* CUstream; typedef void* CUgreenCtx; typedef int CUresult;
struct CUdevResource { struct {unsigned smCount;} sm; };
#define CU_DEV_RESOURCE_TYPE_SM 0
extern "C" int cuStreamGetGreenCtx(CUstream,CUgreenCtx*);
extern "C" int cuGreenCtxGetDevResource(CUgreenCtx,CUdevResource*,int);
''')
        (d/'cuda_runtime_api.h').write_text('''#pragma once
#define CUDARTAPI
enum cudaDeviceAttr {cudaDevAttrMultiProcessorCount=16, other=17};
enum cudaError_t {cudaSuccess=0,cudaErrorUnknown=999};
extern "C" cudaError_t cudaDeviceGetAttribute(int*,cudaDeviceAttr,int);
''')
        (d/'real.cpp').write_text('''#include "cuda.h"
#include "cuda_runtime_api.h"
extern "C" int cuStreamGetGreenCtx(CUstream s,CUgreenCtx* g){*g=s;return 0;}
extern "C" int cuGreenCtxGetDevResource(CUgreenCtx g,CUdevResource* r,int){r->sm.smCount=(uintptr_t)g==1?24:58;return 0;}
extern "C" cudaError_t cudaDeviceGetAttribute(int* v,cudaDeviceAttr,int){*v=82;return cudaSuccess;}
''')
        (d/'run.cpp').write_text('''#include "cuda_runtime_api.h"
#include <dlfcn.h>
#include <stdint.h>
#include <cassert>
#include <thread>
int main(){
 auto begin=(int(*)(uint64_t,int,int*))dlsym(RTLD_DEFAULT,"s5_marlin_begin");
 auto end=(int(*)(int*,int*))dlsym(RTLD_DEFAULT,"s5_marlin_end");
 assert(begin && end);int cap,v,h,o;
 cudaDeviceGetAttribute(&v,cudaDevAttrMultiProcessorCount,0);assert(v==82);
 assert(begin(1,0,&cap)==0 && cap==24);assert(begin(2,0,&cap)!=0);
 std::thread peer([&]{int c,w,n,z;assert(begin(2,0,&c)==0 && c==58);
 cudaDeviceGetAttribute(&w,cudaDevAttrMultiProcessorCount,0);assert(w==58);
 assert(end(&n,&z)==0 && n==1 && z==82);});peer.join();
 cudaDeviceGetAttribute(&v,cudaDevAttrMultiProcessorCount,0);assert(v==24);
 cudaDeviceGetAttribute(&v,cudaDevAttrMultiProcessorCount,1);assert(v==82);
 cudaDeviceGetAttribute(&v,other,0);assert(v==82);
 assert(end(&h,&o)==0 && h==1 && o==82);assert(end(&h,&o)!=0);
 cudaDeviceGetAttribute(&v,cudaDevAttrMultiProcessorCount,0);assert(v==82);
 assert(begin(0,0,&cap)==0 && cap==0);
 cudaDeviceGetAttribute(&v,cudaDevAttrMultiProcessorCount,0);assert(v==82);
 assert(end(&h,&o)==0 && h==0);
}
''')
        def run(cmd,**kwargs):
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=30,**kwargs)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        run(['g++','-shared','-fPIC','-I'+str(d),str(d/'real.cpp'),'-o',str(d/'libmock.so')])
        run(['g++','-std=c++17','-shared','-fPIC','-I'+str(d),str(h/'s5_marlin_cap.cpp'),'-L'+str(d),'-lmock','-ldl','-o',str(d/'cap.so')])
        run(['g++','-I'+str(d),str(d/'run.cpp'),'-L'+str(d),'-lmock','-ldl','-pthread','-o',str(d/'run')])
        run([str(d/'run')],env=dict(os.environ,LD_PRELOAD=str(d/'cap.so'),LD_LIBRARY_PATH=str(d)))

    def test_native_generator_exact_source(self):
        import importlib.util
        h=Path(__file__).resolve().parent
        path=h.parent/'patches/marlin_green/make_marlin_green.py'
        spec=importlib.util.spec_from_file_location('capgen',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        source=h.parent/'reference/vllm-csrc-dc36fcce9/csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu'
        if not source.exists():self.skipTest('optional reference native source absent')
        original=source.read_text();changed=m.transform(original)
        self.assertEqual(changed.split('#include <algorithm>\n',1)[1].replace(m.DELTA,''),original)
        with self.assertRaises(ValueError):m.transform(original+'x')

    def test_fatal_without_result_is_fail(self):
        import json
        h=Path(__file__).resolve().parent;tmp=h.parent.parent/'tmp';tmp.mkdir(exist_ok=True)
        d=Path(tempfile.mkdtemp(prefix='s5-grade-cpu-',dir=tmp));out=d/'result.json'
        Path(str(out)+'.rank0.error.json').write_text(json.dumps(dict(fatal_correctness=True)))
        p=subprocess.run(['python3',str(h/'s5_variants.py'),'--grade',str(out)],capture_output=True,text=True)
        self.assertEqual(p.returncode,1);self.assertEqual(json.loads(p.stdout)['verdict'],'FAIL')
if __name__=='__main__':unittest.main()
