"""Byte-oracle/failure-artifact integration on CPU fakes; no CUDA claim."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from exactness import Exactness,ExactnessError,TorchExactBackend,MEMORY_KEYS
import e2_exact_grade
from test_e2_runner_lifecycle import fixture,batch,candidate


class Table:
    def __init__(self,facts,values):self.facts=facts;self.values=np.array(values,dtype=np.int32)
    def __getitem__(self,key):return self.values[key]


class OracleBackend:
    def __init__(self,hosts):self.hosts=hosts;self.gather_bad=False
    def snapshot(self,a):
        if hasattr(a,'data') and not isinstance(a,np.ndarray):return np.frombuffer(bytes(a.data),dtype=np.uint8).copy()
        return np.array(a,copy=True,order='C')
    def integers(self,a):return list(np.asarray(a).reshape(-1))
    def equal(self,a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint8),b.view(np.uint8))
    def nbytes(self,a):return a.nbytes
    def finite(self,a):return bool(np.isfinite(a).all())
    def differing_bytes(self,a,b):return int(np.count_nonzero(a.view(np.uint8)!=b.view(np.uint8)))
    def gather_reference(self,host,table,lo,hi,staged):
        staged.data[:]=b''.join(host.data[int(p)*4096:(int(p)+1)*4096] for p in table[lo:hi])
        if self.gather_bad:staged.data[0]^=1
    def gather_rows(self,host,source_row,rows,staged,arena_row):
        width=512
        flat=np.frombuffer(host.data,dtype=np.uint8).reshape(-1,width)
        dst=np.frombuffer(staged.data,dtype=np.uint8).reshape(-1,width)
        dst[arena_row:arena_row+rows]=flat[source_row:source_row+rows]
        if self.gather_bad:dst[arena_row]^=1


def oracle_fixture():
    j,m,t,b,d,lines,switch,x,o,args=fixture()
    root=Path(tempfile.mkdtemp(prefix='e2-exactness-',dir=ROOT.parent/"tmp"))
    backend=OracleBackend(d.hosts)
    oracle=Exactness(root,'PP0_TP0',m.layer_groups,backend,lines.append,page_tokens=4,arena_pages=23)
    x.exactness=oracle;d.verify_bucket=oracle.bucket
    a=list(args);a[1]=d.hosts['qsa0'];a[3]=Table(args[3].facts,[1]);a[4]=np.arange(4);a[8]=np.zeros((4,2),dtype=np.float32)
    return j,m,t,b,d,lines,switch,x,o,tuple(a),oracle,backend,root


def baseline(*a):a[8][:]=np.arange(8).reshape(4,2)
def staged(*a,e2):candidate(*a,e2=e2);baseline(*a)


class ExactnessTests(unittest.TestCase):
    def run_layer(self,x,o,t,args,fn=staged,metadata=None):
        with x.rpc(o):
            x.updated({'a':0},t);x.batch(batch())
            x.staged('qsa0',0,0,4,4,args,baseline,fn,
                     gpu_metadata=metadata if metadata is not None else (np.array([0,4]),np.array([4])))

    def test_actual_lifecycle_byte_bucket_output_and_completed_sidecar(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        self.run_layer(x,o,t,a)
        saved=json.loads(v.progress.read_text());s=saved['layers']['qsa0']
        self.assertEqual((s['calls'],s['dense_dma'],s['buckets'],s['bucket_bytes'],s['output_bytes']),(1,1,1,4096,32))
        self.assertFalse(saved['active']);self.assertFalse(saved['failed']);self.assertFalse(v.primary.exists())
        self.assertEqual(m.pool.held_count(),0)
        self.assertFalse(any(word in v.progress.read_text() for word in ('address','request_id','token_ids')))

    def test_inflight_progress_cannot_resemble_prior_completed_proof(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        def observing(*a,e2):
            self.assertTrue(json.loads(v.progress.read_text())['active'])
            staged(*a,e2=e2)
        self.run_layer(x,o,t,a,observing)
        self.assertFalse(json.loads(v.progress.read_text())['active'])

    def test_bad_dma_bytes_leave_immutable_primary_and_quarantine(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture();backend.gather_bad=True
        with self.assertRaises(ExactnessError):self.run_layer(x,o,t,a)
        primary=v.primary.read_bytes();e=json.loads(primary)
        self.assertEqual(e['kind'],'arena_bytes');self.assertEqual(e['evidence']['different_bytes'],1)
        self.assertTrue(m.failed);self.assertGreater(m.pool.held_count(),0)
        with self.assertRaises(ExactnessError):v.fail('later_failure')
        self.assertEqual(v.primary.read_bytes(),primary)
        with self.assertRaises(ExactnessError):v.completed()

    def test_output_byte_mismatch_not_waived_by_model_noise(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        def wrong(*a,e2):staged(*a,e2=e2);a[8][0,0]=np.nextafter(np.float32(0),np.float32(1))
        with self.assertRaises(ExactnessError):self.run_layer(x,o,t,a,wrong)
        e=json.loads(v.primary.read_text())
        self.assertEqual(e['kind'],'qsa_output_bytes');self.assertEqual(e['evidence']['different_bytes'],1)
        self.assertTrue(m.failed)

    def test_gpu_metadata_and_mapping_mismatches_rejected_before_dma(self):
        for kind in ('offset','length','table','rows'):
            j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture();md=(np.array([0,4]),np.array([4]))
            if kind=='offset':md=(np.array([0,3]),np.array([4]))
            if kind=='length':md=(np.array([0,4]),np.array([5]))
            if kind=='table':a[3].values[0]=2
            if kind=='rows':a[4][1]=0
            with self.subTest(kind=kind),self.assertRaises(ExactnessError):self.run_layer(x,o,t,a,metadata=md)
            self.assertFalse(b.events);self.assertTrue(v.primary.exists())

    def test_off_does_not_run_oracle_or_read_gpu_metadata(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture();switch[0]=False
        with x.rpc(o):
            x.updated({'a':0},t);x.batch(batch())
            x.staged('qsa0',0,0,4,4,a,baseline,staged)
        self.assertEqual(v.stats['qsa0']['calls'],0);self.assertFalse(b.events)

    def test_sparse_fallback_output_checked_without_claiming_dma_coverage(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        def sparse(*a,e2):e2.route('sparse_uva',1);baseline(*a)
        self.run_layer(x,o,t,a,sparse)
        self.assertEqual(v.stats['qsa0']['sparse_uva'],1)
        self.assertEqual(v.stats['qsa0']['dense_dma'],0);self.assertEqual(v.stats['qsa0']['buckets'],0)
        self.assertFalse(b.events)

    def test_existing_evidence_cannot_be_reused_as_new_worker(self):
        *_,v,backend,root=oracle_fixture()
        with self.assertRaises(Exception):Exactness(root,'PP0_TP0',v.stats,backend,lambda _:None)

    def test_memory_opt_in_keeps_exact_oracle_and_reports_observation_boundaries(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        v.memory_enabled=True
        calls=[]
        def memory():
            calls.append(len(calls))
            # New reserved peak first appears after reference output, while
            # current allocated is lower. Never mistake current for peak.
            values={key:0 for key in MEMORY_KEYS}
            values['allocated_bytes.all.current']=10
            values['reserved_bytes.all.peak']=200 if len(calls)>=5 else 100
            return values
        backend.memory_stats=memory
        self.run_layer(x,o,t,a)
        saved=json.loads(v.progress.read_text())
        self.assertEqual(saved['layers']['qsa0']['calls'],1)
        self.assertFalse(saved['failed']);self.assertFalse(saved['active'])
        self.assertEqual(saved['memory']['samples'],6)
        self.assertEqual(saved['memory']['last']['boundary'],'layer_completed')
        self.assertEqual(saved['memory']['last']['rpc_sequence'],m.sequence)
        self.assertEqual(saved['memory']['highwater']['reserved_bytes.all.peak'],200)
        logs=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('FLASH_E2_MEMORY ')]
        self.assertEqual([s['boundary'] for s in logs],['before_candidate','output_reference_snapshot'])
        self.assertEqual(logs[-1]['previous_boundary']['boundary'],'output_candidate_snapshot')
        self.assertNotIn('request_id',json.dumps(logs));self.assertNotIn('address',json.dumps(logs))

    def test_memory_disabled_does_not_call_backend_or_add_sidecar_fields(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        backend.memory_stats=lambda:self.fail('disabled memory probe called')
        self.run_layer(x,o,t,a)
        self.assertNotIn('memory',json.loads(v.progress.read_text()))
        self.assertFalse(any(s.startswith('FLASH_E2_MEMORY ') for s in lines))

    def test_memory_missing_counters_are_unknown_not_zero(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        v.memory_enabled=True
        backend.memory_stats=lambda:{'num_alloc_retries':3,'num_ooms':-1}
        self.run_layer(x,o,t,a)
        record=json.loads(v.progress.read_text())['memory']
        self.assertIn('reserved_bytes.all.peak',record['missing'])
        self.assertIsNone(record['last']['counters']['num_ooms'])
        self.assertEqual(record['last']['counters']['num_alloc_retries'],3)
        self.assertNotIn('num_ooms',record['highwater'])

    def test_memory_backend_uses_only_allocator_bookkeeping_and_whitelisted_fields(self):
        from types import SimpleNamespace
        backend=TorchExactBackend.__new__(TorchExactBackend)
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(memory_stats=lambda:{
            'reserved_bytes.all.peak':123,'unrelated_payload':'not forwarded'}))
        record=backend.memory_stats()
        self.assertEqual(set(record),set(MEMORY_KEYS))
        self.assertEqual(record['reserved_bytes.all.peak'],123)
        self.assertIsNone(record['allocated_bytes.all.current'])

    def test_memory_enabled_does_not_waive_exactness_failure(self):
        j,m,t,b,d,lines,switch,x,o,a,v,backend,root=oracle_fixture()
        v.memory_enabled=True
        backend.memory_stats=lambda:{key:0 for key in MEMORY_KEYS}
        backend.gather_bad=True
        with self.assertRaises(ExactnessError):self.run_layer(x,o,t,a)
        self.assertEqual(json.loads(v.primary.read_text())['kind'],'arena_bytes')
        self.assertTrue(m.failed)


def grade_fixture():
    root=Path(tempfile.mkdtemp(prefix='e2-exact-grade-',dir=ROOT.parent/"tmp"))
    for pp in (0,1):
        for tp in (0,1):
            rank=f'PP{pp}_TP{tp}'
            layers={f'model.layers.{i}.attn':dict(calls=2,dense_dma=1,sparse_uva=1,empty=0,buckets=3,
                     bucket_bytes=50*4048*1024,output_bytes=1024,max_length=210000,prefix_hit_calls=1,
                     dense_max_length=200000,dense_multibucket_calls=1) for i in range(3+24*pp,24+24*pp,4)}
            (root/(rank+'.exact.json')).write_text(json.dumps(dict(schema=1,rank=rank,failed=False,
                    active=False,layers=layers,page_tokens=4048,arena_pages=24)))
    return root


class GraderTests(unittest.TestCase):
    def test_dense_only_phase_rejects_sparse_and_still_requires_dense_coverage(self):
        root=grade_fixture()
        self.assertEqual(e2_exact_grade.grade(root,require_dense_only=True)['verdict'],'FAIL')
        for p in root.glob('*.exact.json'):
            d=json.loads(p.read_text())
            for s in d['layers'].values():s.update(calls=1,sparse_uva=0)
            p.write_text(json.dumps(d))
        self.assertEqual(e2_exact_grade.grade(root,require_dense_only=True)['verdict'],'PASS')
        p=root/'PP0_TP0.exact.json';d=json.loads(p.read_text())
        next(iter(d['layers'].values()))['dense_max_length']=10000
        p.write_text(json.dumps(d))
        self.assertEqual(e2_exact_grade.grade(root,require_dense_only=True)['verdict'],'INCONCLUSIVE')

    def test_complete_four_rank_six_layer_proof(self):
        self.assertEqual(e2_exact_grade.grade(grade_fixture())['verdict'],'PASS')

    def test_immutable_failure_wins_over_successful_progress(self):
        root=grade_fixture();(root/'PP1_TP0.failure.json').write_text('{}')
        self.assertEqual(e2_exact_grade.grade(root)['verdict'],'FAIL')

    def test_sparse_only_or_no_long_dense_or_no_prefix_inconclusive(self):
        for kind in ('sparse','short','prefix'):
            root=grade_fixture();p=root/'PP0_TP0.exact.json';d=json.loads(p.read_text());s=next(iter(d['layers'].values()))
            if kind=='sparse':s.update(dense_dma=0,sparse_uva=2,buckets=0,bucket_bytes=0,dense_multibucket_calls=0,dense_max_length=0)
            if kind=='short':s['dense_max_length']=10000
            if kind=='prefix':s['prefix_hit_calls']=0
            p.write_text(json.dumps(d))
            self.assertEqual(e2_exact_grade.grade(root)['verdict'],'INCONCLUSIVE')

    def test_malformed_missing_counter_wrong_placement_and_partial_write_fail(self):
        for kind in ('counter','layer','geometry','active','route-count','nan','bool','bytes'):
            root=grade_fixture();p=root/'PP0_TP0.exact.json';d=json.loads(p.read_text());s=next(iter(d['layers'].values()))
            if kind=='counter':s.pop('bucket_bytes')
            elif kind=='layer':d['layers']['model.layers.47.attn']=d['layers'].pop('model.layers.3.attn')
            elif kind=='geometry':d['arena_pages']=23
            elif kind=='active':d['active']=True
            elif kind=='route-count':s['calls']=3
            elif kind=='nan':s['max_length']=float('nan')
            elif kind=='bool':s['buckets']=True
            else:s['bucket_bytes']=1
            p.write_text(json.dumps(d))
            self.assertEqual(e2_exact_grade.grade(root)['verdict'],'FAIL',kind)

    def test_missing_workers_is_inconclusive_not_pass(self):
        root=Path(tempfile.mkdtemp(prefix='e2-exact-empty-',dir=ROOT.parent/"tmp"))
        self.assertEqual(e2_exact_grade.grade(root)['verdict'],'INCONCLUSIVE')


if __name__=='__main__':unittest.main()
