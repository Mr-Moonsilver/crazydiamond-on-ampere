import copy
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from s5_oracle_queue import validate,bound_summary
from s5_substitution import TensorBytes
from s5_substitution_trace import OpTrace,compare_traces,changed_axes,traced_mm,traced_gdn_outputs
from test_s5_substitution import tensor
from test_s5_resident import Resident

H=Path(__file__).resolve().parent


def fresh(test):
    root=Path(tempfile.mkdtemp(prefix='s5-oracle-',dir=H.parents[1]/'tmp'))
    test.addCleanup(shutil.rmtree,root,ignore_errors=True)
    return root


class OracleQueue(unittest.TestCase):
    def test_failure_trace_after_existing_primary_uses_real_exclusive_writer(self):
        from s5_failure import diagnostic,worker_failure
        from s5_segment import write_json
        from contextlib import redirect_stderr
        import io,hashlib
        class CorrectnessError(RuntimeError):pass
        for existing in (False,True):
            root=fresh(self);path=root/'result.json.rank0.error.json'
            primary=dict(fatal_correctness=True,error='primary mismatch',evidence={'immutable':True})
            if existing:write_json(path,primary)
            calls=[]
            result=diagnostic(path,primary,lambda:calls.append('trace') or dict(completed=True,comparisons=[]))
            self.assertEqual(calls,['trace']);self.assertTrue(result['completed'])
            original=path.read_bytes()
            side=json.loads((root/'result.json.rank0.source-trace.json').read_text())
            self.assertEqual(json.loads(original),primary)
            self.assertEqual(side['primary']['sha256'],hashlib.sha256(original).hexdigest())
            self.assertTrue(side['diagnostic_trace']['completed'])
            with redirect_stderr(io.StringIO()) as log:
                try:raise FileExistsError('secondary diagnostic error')
                except FileExistsError as exc:
                    rc=worker_failure(path,exc,CorrectnessError,{},lambda:calls.append('closed'))
            self.assertEqual(rc,1);self.assertEqual(path.read_bytes(),original)
            self.assertIn('Traceback',log.getvalue());self.assertIn('FileExistsError',log.getvalue())
            self.assertEqual(calls,['trace','closed'])
            self.assertTrue(json.loads((root/'result.json.rank0.worker-exception.json').read_text())['fatal_correctness'])

    def test_trace_exception_is_additive_and_cannot_clear_primary(self):
        from s5_failure import diagnostic,worker_failure
        from s5_segment import write_json
        from contextlib import redirect_stderr
        import io
        root=fresh(self);path=root/'result.json.rank1.error.json'
        write_json(path,dict(fatal_correctness=True,error='primary'))
        original=path.read_bytes()
        def fail():raise RuntimeError('trace failure')
        with redirect_stderr(io.StringIO()) as log:
            result=diagnostic(path,{},fail)
        self.assertFalse(result['completed']);self.assertEqual(result['error_type'],'RuntimeError')
        self.assertIn('Traceback',log.getvalue());self.assertEqual(path.read_bytes(),original)
        self.assertTrue((root/'result.json.rank1.source-trace-start.json').exists())
        self.assertFalse(json.loads((root/'result.json.rank1.source-trace.json').read_text())['diagnostic_trace']['completed'])
        # A worker with no prior correctness failure remains INCONCLUSIVE;
        # even failure of watchdog.close cannot hide its traceback/verdict.
        other=root/'result.json.rank0.error.json'
        with redirect_stderr(io.StringIO()) as log:
            rc=worker_failure(other,ValueError('worker failure'),AssertionError,{},fail)
        self.assertEqual(rc,2);self.assertFalse(json.loads(other.read_text())['fatal_correctness'])
        self.assertIn('ValueError',log.getvalue());self.assertIn('trace failure',log.getvalue())

    def test_reject_bad_inputs_before_any_launch(self):
        for names in ([],['mixed-resident34']*2,['*'],['qsa-resident24','../x'],['$(touch x)']):
            with self.assertRaises(ValueError):validate(names,'0,2','on',6,1)
        for settings in [('0,1','on',6,1),('0,2','ON',6,1),('0,2','on',5,1),('0,2','on',6,2)]:
            with self.assertRaises(ValueError):validate(['mixed-resident34'],*settings)
        root=fresh(self)
        for i,names in enumerate(['*','mixed-resident34 mixed-resident34','qsa-resident24\nmixed-resident34','   ']):
            q=root/str(i)
            p=subprocess.run(['bash',str(H/'runs/q_s5_resident_oracle.draft.sh')],capture_output=True,
                env=dict(os.environ,H=str(H),Q=str(q),S5_CONFIG='unused',S5_OVERLAY='unused',
                         S5_RESIDENT_CASES=names,S5_PAIR='0,2'))
            self.assertEqual(p.returncode,2);self.assertFalse(q.exists())

    def test_quoted_subset_forwarding_missing_case_stays_visible(self):
        root=fresh(self);bundle=root/'bundle';bundle.mkdir();(bundle/'runs').mkdir()
        # No Docker/GPU command exists in this fake child. Run the real parent
        # driver to check order, bounded input forwarding and missing results.
        for name in ('s5_oracle_queue.py','s5_value.py'):(bundle/name).write_text((H/name).read_text())
        (bundle/'runs/q_s5_variant_oracle.draft.sh').write_text('''set -eu
mkdir -p "$Q/profile-terminal"
python3 - <<'PY'
import json,os,pathlib
q=pathlib.Path(os.environ['Q'])
(q/'called.json').write_text(json.dumps({k:os.environ[k] for k in ('S5_MOE','S5_SOURCE_TRIALS','S5_SOURCE_TRACE_ON_FAIL','S5_PAIR','S5_LAYER_KIND','S5_CUBLAS_DETERMINISTIC','S5_DMA_EVERY_LAYER','S5_DMA_BATCHED','S5_RESIDENT_DMA')}))
(q/'handoff.json').write_text(json.dumps(dict(own_container_stopped=1,gpu_idle=1,engine_untouched=True)))
PY
exit 2
''')
        q=root/'out'
        p=subprocess.run(['bash',str(H/'runs/q_s5_resident_oracle.draft.sh')],capture_output=True,text=True,
            env=dict(os.environ,PYTHONPATH=str(H),H=str(bundle),Q=str(q),S5_CONFIG='unused',S5_OVERLAY='unused',
                     S5_RESIDENT_CASES='mixed-resident34 qsa-dma34 dma-every-step34 dma-batched34 arena-direct34',S5_PAIR='1,3',S5_MOE='off',S5_SOURCE_TRIALS='6',S5_SOURCE_TRACE_ON_FAIL='1',S5_CUBLAS_DETERMINISTIC='1'))
        self.assertEqual(p.returncode,2,p.stdout+p.stderr)
        r=json.loads((q/'summary.json').read_text())
        self.assertEqual([c['case'] for c in r['cases']],['mixed-resident34','qsa-dma34','dma-every-step34','dma-batched34','arena-direct34'])
        self.assertTrue(all(c['verdict']=='INCONCLUSIVE' for c in r['cases']))
        for name in r['requested']['cases']:
            d=json.loads((q/name/'called.json').read_text())
            self.assertEqual(d['S5_MOE'],'off');self.assertEqual(d['S5_SOURCE_TRIALS'],'6')
            self.assertEqual(d['S5_SOURCE_TRACE_ON_FAIL'],'1');self.assertEqual(d['S5_PAIR'],'1,3')
            self.assertEqual(d['S5_CUBLAS_DETERMINISTIC'],'1')
            self.assertEqual(d['S5_DMA_EVERY_LAYER'],'1' if name in ('dma-every-step34','dma-batched34','arena-direct34') else '0')
            self.assertEqual(d['S5_DMA_BATCHED'],'1' if name=='dma-batched34' else '0')
            self.assertEqual(d['S5_RESIDENT_DMA'],'0' if name in ('mixed-resident34','arena-direct34') else '1')

    def test_artifact_binding_and_fail_precedence(self):
        root=fresh(self);f=root/'qsa-resident24';(f/'profile-terminal').mkdir(parents=True)
        config=validate(['qsa-resident24'],'0,2','off',6,1)
        d=Resident().fixture();d['pair']='0,2'
        for r in d['ranks']:
            r['ablation']=dict(layer_kind='qsa',moe='off')
            r['cublas_control']=dict(requested=False,deterministic_algorithms=False,workspace_config=None)
            r['value_probe'].update(source_trials=6,source_trace_on_fail=True,resident_dma=False)
            r['resident_substitution']=dict(passed=True,host_runs=6,resident_runs=6,chunks_per_run=3,deterministic_control=True)
        out=f/'profile-terminal/result.json';out.write_text(json.dumps(d))
        (f/'profile-terminal/grade.json').write_text('{"verdict":"PASS"}')
        (f/'handoff.json').write_text('{"own_container_stopped":1,"gpu_idle":1,"engine_untouched":true}')
        self.assertEqual(bound_summary(root,config)['verdict'],'PASS')
        for key,value in [('pair','1,3'),('trials',3),('moe','on'),('trace',0),('blas',1)]:
            bad=dict(config,**{key:value})
            self.assertEqual(bound_summary(root,bad)['verdict'],'INCONCLUSIVE')
        (f/'profile-terminal/result.json.rank1.error.json').write_text('{"fatal_correctness":true}')
        self.assertEqual(bound_summary(root,dict(config,trials=3))['verdict'],'FAIL')

    def test_trace_first_op_and_position_geometry_without_payload(self):
        a=TensorBytes('op20/layer3/qsa.attention','float32',(3,2,4),struct.pack('<24f',*([0.]*24)))
        values=[0.]*24;values[8+1]=2.;values[8+4+1]=2.
        b=TensorBytes(a.name,a.dtype,a.shape,struct.pack('<24f',*values))
        axes=changed_axes(a,b)
        self.assertEqual(axes[0]['first_indices'],[1]);self.assertEqual(axes[0]['max_differences_per_index'],2)
        self.assertEqual(axes[-1]['first_indices'],[1]);self.assertEqual(axes[1]['distinct_indices'],2)
        prefix=tensor('op19/layer3/qsa.projection',3.)
        r=compare_traces([[prefix,a]]*3,[[prefix,b]]*3)
        self.assertEqual(r[0]['first_divergence'],a.name)
        self.assertEqual(r[0]['first_layer'],3);self.assertEqual(r[0]['first_op'],'qsa.attention')
        self.assertEqual(r[0]['differences'][0]['max_ulp'],1073741824)
        text=json.dumps(r,allow_nan=False)
        self.assertNotIn('"data"',text);self.assertNotIn('"values"',text)

    def test_layer3_boundary_is_first_hc_tuple_not_later_hc(self):
        first=[tensor(f'op30/layer3/hc/{i}',float(i)) for i in range(3)]
        later=[tensor(f'op50/layer3/hc/{i}',float(i)) for i in range(3)]
        attention=tensor('op40/layer3/qsa.attention',1.)
        qsa_input=tensor('op31/layer3/qsa.input',1.)
        a=first+[qsa_input,attention]+later;b=copy.deepcopy(a)
        b[4]=tensor(attention.name,2.)
        r=compare_traces([a]*3,[b]*3)[0]
        self.assertTrue(r['layer3_boundary']['all_equal'])
        self.assertEqual(r['first_op'],'qsa.attention')
        b[1]=tensor(first[1].name,4.)
        r=compare_traces([a]*3,[b]*3)[0]
        self.assertFalse(r['layer3_boundary']['all_equal'])
        self.assertEqual(r['layer3_boundary']['status'],dict(hidden_residual='equal',qsa_input='different',injection='equal'))
        missing=compare_traces([[attention]]*3,[[attention]]*3)[0]['layer3_boundary']
        self.assertFalse(missing['complete']);self.assertIsNone(missing['all_equal'])
        # A missing first HC must not be replaced by the second, post-QSA HC.
        missing=compare_traces([[qsa_input,attention]+later]*3,[[qsa_input,attention]+later]*3)[0]['layer3_boundary']
        self.assertFalse(missing['complete']);self.assertIsNone(missing['all_equal'])

    def test_trace_copies_before_mutation_and_layer_numbering(self):
        class T:
            def __init__(self,v):self.v=v
            def clone(self):return T(self.v)
        x=T(1);tr=OpTrace(SimpleNamespace(indices=x))
        with patch.dict(sys.modules,{'torch':SimpleNamespace(Tensor=T)}):
            tr.call('selection',lambda:None)
            for _ in range(4):
                tr.call('hc',lambda:(x,x,x));tr.call('qsa',lambda:x)
                tr.call('hc',lambda:(x,x,x));tr.call('moe',lambda:x)
            tr.call('hc_final',lambda:x)
        x.v=99
        self.assertTrue(all(v.v==1 for _,v in tr.named))
        self.assertEqual([n.split('/')[1] for n,_ in tr.named if n.endswith('/qsa')],['layer0','layer1','layer2','layer3'])

    def test_new_variant_is_only_recorded_env_extension(self):
        old=(H/'runs/q_s5_variant.draft.sh').read_text();new=(H/'runs/q_s5_variant_oracle.draft.sh').read_text()
        for text in ('S5_SOURCE_TRIALS=$trials','S5_SOURCE_TRACE_ON_FAIL=$trace','source_trials=%s source_trace_on_fail=%s'):
            self.assertIn(text,new)
        # The copied lifecycle is deliberately unchanged: inspect rather than
        # running any hardware command in an offline test.
        expected=old[old.index('idle() {'):].replace('  env_args+=(-e "S5_RESIDENT_DMA=$resident_dma")',
            '  env_args+=(-e "S5_RESIDENT_DMA=$resident_dma" -e "S5_SOURCE_TRIALS=$trials" -e "S5_SOURCE_TRACE_ON_FAIL=$trace")\n'
            '  env_args+=(-e "S5_DMA_EVERY_LAYER=$every" -e "S5_DMA_BATCHED=$batch")\n'
            '  env_args+=(-e "S5_CUBLAS_DETERMINISTIC=$blas")\n'
            '  if [[ $blas == 1 ]];then env_args+=(-e CUBLAS_WORKSPACE_CONFIG=:4096:8);fi')
        self.assertEqual(expected,new[new.index('idle() {'):])
        for name in ('q_s5_variant_oracle.draft.sh','q_s5_resident_oracle.draft.sh'):
            self.assertEqual(subprocess.run(['bash','-n',str(H/'runs'/name)]).returncode,0)

    def test_mm_diagnostic_records_inputs_outputs_and_restores_on_failure(self):
        rows=[];weight=object();x=object();y=object()
        def mm(a,b):self.assertIs(a,x);self.assertIs(b,weight);return y
        torch=SimpleNamespace(mm=mm)
        tr=SimpleNamespace(record=lambda name,t:rows.append((name,t)))
        with patch.dict(sys.modules,{'torch':torch}):
            with self.assertRaises(RuntimeError):
                with traced_mm(SimpleNamespace(w=dict(gdn_out=weight)),tr):
                    self.assertIs(torch.mm(x,weight),y);raise RuntimeError()
        self.assertIs(torch.mm,mm)
        self.assertEqual(rows,[('mm.gdn_out.input',x),('mm.gdn_out.output',y)])

    def test_gdn_trace_observes_original_inplace_norm_and_restores(self):
        class T:
            def __init__(self,v):self.v=v
            def reshape(self,*shape):return self
        output=T(3);state=T(7);rows=[]
        def chunk(*a,**kw):return output,state
        def normfn(x,*a,**kw):kw['out'].v=x.v+1;return 'original-return'
        fla=SimpleNamespace(chunk_gated_delta_rule=chunk);norm=SimpleNamespace(layer_norm_fwd=normfn)
        tr=SimpleNamespace(record=lambda name,t:rows.append((name,t.v)))
        with patch('importlib.import_module',side_effect=lambda name:norm if name.endswith('layernorm_guard') else fla):
            with self.assertRaises(RuntimeError):
                with traced_gdn_outputs(SimpleNamespace(t=1024),tr):
                    self.assertEqual(fla.chunk_gated_delta_rule(),(output,state))
                    self.assertEqual(norm.layer_norm_fwd(output,out=output),'original-return')
                    raise RuntimeError()
        self.assertIs(fla.chunk_gated_delta_rule,chunk);self.assertIs(norm.layer_norm_fwd,normfn)
        self.assertEqual(rows,[('gdn.fla_output',3),('gdn.final_state',7),('gdn.pre_norm',3),('gdn.post_norm',4)])


if __name__=='__main__':unittest.main()
