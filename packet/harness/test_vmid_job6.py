"""CPU-only context query ABI, unchanged placement guard, and VM receipt tests."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import vmid_job6 as j
from test_e3_replay_observer import ObserverTestBase,CONTEXT_ID,EXPECTED_SMS

ROOT=Path(__file__).resolve().parents[1]


def diagnostic(route='green',origin='green'):
    cid=7 if route=='green' else 1
    return dict(schema='s5ro-context-diagnostics/1',id_namespace='CUPTI',expected_context_id=7,
        start_ns=1100,end_ns=1400,eligible_kernel_records=3,omitted_records=0,boundary_records=0,row_cap=4096,
        contexts=[dict(context_id=7,parent_context_id=1,is_green=True,sms=44)],
        executions=[dict(slot=0,node_index=0,context_id=cid,stream_id=2,graph_id=11,graph_node_id=101,count=3)],
        nodes=[dict(slot=0,node_index=0,graph_node_id=101,query_rc=0,func_present=True,kern_present=False,
                    ctx_present=True,ctx_semantics='ignored_func_set',context_id_available=False,context_id=0)],
        nodes_omitted=0,census_requested=True,complete=True)


def fixture(origin='green'):
    counts={k:0 for k in ('dropped_records_total','buffers_refused','parse_errors','callback_errors','ambiguous_node_records',
        'unknown_node_records','kind_mismatch_records','incomplete_records','unknown_timestamp_records',
        'invalid_interval_records','boundary_overlap_records')}
    counts['ring_overflows']={k:0 for k in ('aliases','resource','context','kernel','copy')}
    return dict(family='aten',capture_origin=origin,cleanup_ok=True,serialized_stable=True,native_schema_valid=True,
        static_census_stable=True,membership={'ok':True},split=dict(decode=44,prefill=38,total=82),static={'complete':True,'graph_slots':[{'required':{'kernel':1}}]},
        window={'counts':counts,'flags':{'known_wrong_context':True},'expected_context':{'is_green':True,'sm_matching':1}},
        windows={route:dict(output_exact=True,replays=3,diagnostics=diagnostic(route),
                    stream=dict(green_present=route=='green',green_cupti_context_id=7 if route=='green' else 0))
                 for route in ('plain','green')})


class GradeTests(unittest.TestCase):
    def test_negative_is_retained_and_green_misplacement_fails(self):
        r=fixture();self.assertEqual(j.case_grade(r)['verdict'],'PASS')
        r['windows']['green']['diagnostics']['executions'][0]['context_id']=1
        g=j.case_grade(r);self.assertEqual(g['verdict'],'FAIL')
        self.assertIn('green:known_wrong_context',g['failures'])
        r=fixture('plain');r['windows']['plain']['diagnostics']['executions'][0]['context_id']=7
        self.assertIn('plain:ordinary_negative_not_rejected',j.case_grade(r)['failures'])
    def test_loss_missing_mapping_and_multiplicity_never_pass(self):
        for kind in ('loss','missing_node','multiplicity','query','census'):
            r=fixture()
            if kind=='loss':r['window']['counts']['ring_overflows']['kernel']=1
            elif kind=='missing_node':r['windows']['green']['diagnostics']['executions'][0]['node_index']=99
            elif kind=='multiplicity':r['windows']['green']['diagnostics']['executions'][0]['count']=2
            elif kind=='query':r['windows']['green']['diagnostics']['nodes'][0]['query_rc']=1
            else:r['windows']['green']['diagnostics']['complete']=False
            self.assertNotEqual(j.case_grade(r)['verdict'],'PASS',kind)
    def test_ignored_func_ctx_is_not_effective_placement(self):
        d=diagnostic();n=d['nodes'][0];n.update(context_id=1,context_id_available=True)
        x=j.joined(d)[0];self.assertFalse(x['static_context_applicable']);self.assertIsNone(x['static_activity_agree'])
        n['ctx_semantics']='explicit_kernel_context'
        x=j.joined(d)[0];self.assertTrue(x['static_context_applicable']);self.assertFalse(x['static_activity_agree'])
        n['ctx_semantics']='null_current_context_semantics'
        self.assertIsNone(j.joined(d)[0]['static_activity_agree'])
    def test_cross_origin_is_diagnostic_not_an_invented_qualification(self):
        r=fixture('plain');r['windows']['green']['diagnostics']['executions'][0]['context_id']=1
        g=j.case_grade(r);self.assertEqual(g['verdict'],'PASS')
        self.assertEqual(g['windows']['green']['current_context_predicate'],'FAIL')
        self.assertIn('diagnostic',g['windows']['green']['purpose'])
    def test_finalize_keeps_all_case_verdicts_and_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=[]
            for f in j.FAMILIES:
                for o in j.ORIGINS:r=fixture(o);r['family']=f;rows.append(r)
            j.s.write(p/'probe.json',{'cases':rows})
            j.s.write(p/'monitor.json',{'own_samples':2,'errors':0,'own_peak_mib':400,'stop_triggered':False})
            self.assertEqual(j.finalize(p,0),0)
            r=json.loads((p/'result.json').read_text());self.assertEqual(len(r['cases']),8)
            with self.assertRaises(FileExistsError):j.finalize(p,0)
    def test_queue_and_actual_cuda_headers(self):
        path=ROOT/'harness/runs/vmid_job6.sh';s=path.read_text()
        subprocess.run(['bash','-n',str(path)],check=True)
        for word in ('flock -n 9','/opt/flashnext/.job.lock','flashnext-job6','runs/job6','360s',
                     'PYTORCH_CUDA_ALLOC_CONF','--gpus all','--mode monitor'):
            self.assertIn(word,s)
        self.assertNotIn('vllm-flash',s);self.assertNotIn('docker rm',s)
        subprocess.run(['g++','-std=c++17','-fsyntax-only','-I',str(ROOT/'reference/cuda-runtime-13'),
            '-I',str(ROOT/'reference/cupti-13'),str(ROOT/'harness/native/vmid_context_probe.cpp'),
            str(ROOT/'patches/s5_engine/native/flash_s5_replay_observer.cpp')],check=True,capture_output=True)


class Call:
    def __init__(self,fn=lambda *a:0):self.fn=fn
    def __call__(self,*a):return self.fn(*a)
class Stream:
    def __init__(self,token):self.cuda_stream=token
    def synchronize(self):pass
    def __enter__(self):return self
    def __exit__(self,*a):pass
class Graph:
    def __init__(self):self.replays=0;self.closed=False
    def instantiate(self):pass
    def raw_cuda_graph(self):return 123
    def replay(self):self.replays+=1
    def reset(self):self.closed=True
class PhaseTests(unittest.TestCase):
    def test_real_case_same_graph_six_replays_one_window_and_exclusive_artifacts(self):
        for origin in ('plain','green'):
            with self.subTest(origin=origin),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);graphs=[];ticks=iter((1100,1200,1300,1400));closed=[]
                base=fixture(origin)
                def graph(**kw):g=Graph();graphs.append(g);return g
                lib=NS(s5ro_context_diagnostics=Call(),s5ro_static=Call(),s5ro_end_window=Call(),
                       s5ro_bind_context=Call(),s5ro_bind_graph=Call(),
                       s5ro_begin_window=Call(lambda t,g,p:setattr(p._obj,'value',1000) or 0))
                binding=NS(_lib=lib)
                def report(fn,args,**kw):
                    if fn is lib.s5ro_static:return base['static']
                    if fn is lib.s5ro_end_window:return base['window']
                    return diagnostic('plain' if args[1]==1100 else 'green')
                session=NS(start=lambda:None,close=lambda:closed.append('observer'),_token=1,_report_call=report,_native_reserved=123)
                split=NS(decode_stream=Stream(2),decode_context_id=7,decode_sms=44,prefill_sms=38,total_sms=82,
                         membership=lambda **kw:NS(as_record=lambda:{'ok':True}),close=lambda:closed.append('split'))
                host=NS(timestamp=lambda:next(ticks),stream=lambda s:dict(green_present=s.cuda_stream==2,
                        green_cupti_context_id=7 if s.cuda_stream==2 else 0))
                t=NS(cuda=NS(Stream=lambda:Stream(1),synchronize=lambda:None,stream=lambda s:s,
                     CUDAGraph=graph,graph=lambda *a,**kw:Stream(0),max_memory_reserved=lambda x:1024,
                     OutOfMemoryError=MemoryError))
                ro=NS(ReplayBinding=lambda *a,**kw:binding,ObserverLimits=lambda **kw:None,
                      ObserverSession=lambda *a:session,_validate_static=lambda *a:None,
                      _validate_window=lambda *a:None,_validate_context_diagnostics=lambda *a:None)
                gc=NS(open_split=lambda *a,**kw:split,DriverBinding=lambda *a:None)
                with patch.object(j,'HostQueries',return_value=host),patch.object(j,'operation',return_value=(lambda:None,lambda:True)), \
                     patch.object(j.s,'graph_bytes',return_value=b'unchanged-definition'):
                    r=j.case(t,ro,gc,None,Path('/unused'),{'files':{'flash_s5_replay_observer.so':'x',
                        'flash_s5_replay_observer.cpp':'y'}},out,'aten',origin)
                self.assertNotIn('error_type',r)
                self.assertEqual(graphs[-1].replays,6)
                self.assertEqual(closed,['observer','split']);self.assertTrue(all(g.closed for g in graphs))
                self.assertEqual(len(list(out.iterdir())),3)
                self.assertEqual(r['grade']['verdict'],'PASS')


class NativeTests(ObserverTestBase):
    def diagnostic(self,start=0,end=0):
        f=self.binding._lib.s5ro_context_diagnostics
        f.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];f.restype=C.c_int
        return self.session._report_call(f,(self.session._token,start,end))
    def set_params(self,node,rc=0,func=1,kern=0,ctx=0):
        f=self.ctl.stub_node_params;f.argtypes=[C.c_uint64,C.c_int,C.c_int,C.c_int,C.c_uint64]
        f(node,rc,func,kern,ctx)
    def test_node_census_joins_retained_identity_across_two_graphs(self):
        self.make_bank([{'id':11,'nodes':[(101,0)]},{'id':12,'nodes':[(201,0)]}])
        self.new_session(start=True,bind=True)
        self.set_params(self.bank.node_handles[0][0],func=1,kern=1,ctx=999)
        self.set_params(self.bank.node_handles[1][0],func=0,kern=1)
        d=self.diagnostic();self.assertEqual([n['graph_node_id'] for n in d['nodes']],[101,201])
        self.assertEqual([n['ctx_semantics'] for n in d['nodes']],['ignored_func_set','null_current_context_semantics'])
        self.assertTrue(all(not n['context_id_available'] for n in d['nodes']))
        self.set_params(self.bank.node_handles[1][0],rc=1)
        self.assertEqual(self.diagnostic()['nodes'][1]['ctx_semantics'],'unavailable')
    def test_explicit_kernel_context_queries_correct_id(self):
        self.open_clean();ctx=C.c_void_p()
        f=self.ctl.cuCtxFromGreenCtx;f.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];f.restype=C.c_int
        self.assertEqual(f(C.byref(ctx),C.c_void_p(self.green)),0)
        self.set_params(self.bank.node_handles[0][0],func=0,kern=1,ctx=ctx.value)
        n=self.diagnostic()['nodes'][0]
        self.assertEqual(n['ctx_semantics'],'explicit_kernel_context');self.assertEqual(n['context_id'],CONTEXT_ID)
        self.assertTrue(n['context_id_available'])
    def test_wrong_context_still_fails_and_ranges_do_not_reset_proof(self):
        self.open_clean();self.begin()
        self.emit_kernel(11,101,context_id=1,start=1100,end=1200)
        self.emit_kernel(11,102,start=2100,end=2200,completed=2200)
        r=self.end();self.assertEqual(r['verdict'],'FAIL')
        d=r['replay']['context_diagnostics'];self.assertEqual(d['eligible_kernel_records'],2)
        left=self.diagnostic(1050,1250);right=self.diagnostic(2050,2250)
        self.assertEqual(left['executions'][0]['context_id'],1)
        self.assertEqual(right['executions'][0]['context_id'],CONTEXT_ID)
        self.assertEqual(left['executions'][0]['graph_node_id'],101)
        self.assertEqual(right['executions'][0]['graph_node_id'],102)
        self.assertEqual(r['replay']['counts']['known_wrong_context_records'],1)
        self.assertEqual(self.session.state,'ended')


if __name__=='__main__':unittest.main()
