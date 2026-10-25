"""CPU-only sorted-union, staging and reduction-order oracle; not CUDA qualification."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s6gen',ROOT/'patches/sparse_stage/make_sparse_stage.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def old_plan(indices,rows,n,page,arena):
    selected=indices[rows,:-1]
    unique,inverse=np.unique(selected,return_inverse=True)
    pad=len(unique)>0 and unique[0]==-1
    tokens=unique[1:] if pad else unique
    mapped=indices.copy()
    if len(tokens)==0:return tokens,mapped,[]
    if len(tokens)*5<n*4:
        mapped[rows,:-1]=(inverse.reshape(selected.shape)-int(pad))
    else:tokens=np.arange(n)
    pages=(len(tokens)+page-1)//page
    return tokens,mapped,[(lo,min(lo+arena,pages)) for lo in range(0,pages,arena)]

def device_plan(indices,rows,n,page,arena,block=8):
    flags=np.zeros(n,np.int32)
    for v in indices[rows,:-1].flat:
        if v>=0:
            if v>=n:raise ValueError('bad index')
            flags[v]=1
    ranks=np.empty_like(flags);totals=[]
    for lo in range(0,n,block):
        ranks[lo:lo+block]=np.cumsum(flags[lo:lo+block]);totals.append(sum(flags[lo:lo+block]))
    offsets=np.cumsum([0]+totals[:-1]);count=sum(totals);sparse=count*5<n*4
    tokens=np.empty(n,np.int32)
    for lo in range(0,n,block):
        for i in range(lo,min(lo+block,n)):
            if flags[i]:
                pos=offsets[lo//block]+ranks[i]-1;tokens[pos]=i;ranks[i]=pos
    mapped=indices.copy()
    for r in rows:
        for c,v in enumerate(indices[r,:-1]):mapped[r,c]=ranks[v] if sparse and v>=0 else v
    tokens=tokens[:count] if sparse else np.arange(n)
    pages=(len(tokens)+page-1)//page
    spans=[(lo,min(lo+arena,pages)) for lo in range(0,n//page,arena) if lo<pages]
    return tokens,mapped,spans

def staged(plan,indices,rows,kv,q,page):
    tokens,mapped,spans=plan;output=np.zeros((len(rows),q.shape[-1]),np.float32)
    for ri,r in enumerate(rows):
        acc=np.zeros(q.shape[-1],np.float32);normal=np.float32(0);maximum=np.float32(-1e20)
        for lo,hi in spans:
            # Same selection order and bucket reduction as the staged path. CPU
            # float32 is an oracle for scheduling, not Ampere BF16 tensor cores.
            v=mapped[r,:int(indices[r,-1])];v=v[(v>=lo*page)&(v<hi*page)&(v>=0)]
            if len(v):
                staged_rows=kv[tokens[v]]
                scores=staged_rows@q[r];new=np.maximum(maximum,scores.max())
                alpha=np.exp2(maximum-new);prob=np.exp2(scores-new)
                acc=acc*alpha+prob@staged_rows;normal=normal*alpha+prob.sum();maximum=new
        if normal:output[ri]=acc/normal
    return output

class SparseTests(unittest.TestCase):
    def public_wrapper(self, env, calls):
        generated=g.generate((ROOT/'patches/retain/ops_qsa.py.v8.1').read_text())
        wrapper=ast.parse(g.function(generated,'qsa_sparse_paged_attention_staged'))
        fn=wrapper.body[0]
        for a in fn.args.args: a.annotation=None
        fn.returns=None
        def record(route):
            def invoke(*args):
                # Exercise the same dimensional access that crashed q13b,
                # and keep every public argument for identity comparison.
                self.assertEqual(args[2].shape[1]-1,16)
                self.assertEqual(len(args[3].shape),1)
                calls.append((route,args))
                return route
            return invoke
        ns=dict(os=NS(environ=env),_s6_staged_stock=record('stock'),_s6_staged_async=record('async'))
        exec(compile(ast.fix_missing_locations(wrapper),'generated-public-wrapper','exec'),ns)
        args=(object(),object(),NS(shape=(1024,17)),NS(shape=(100,)),
              NS(numel=lambda:1024),49,NS(shape=(24,1,4048,512)),False,object(),None,0)
        return ns['qsa_sparse_paged_attention_staged'],args

    def test_public_order_stock_routes_including_first_prefill(self):
        for case in ('off','early','retained','empty','sparse-off'):
            calls=[];env={'FLASH_QSA_SPARSE_ASYNC':'1'}
            fn,args=self.public_wrapper(env,calls);args=list(args)
            if case=='off':env['FLASH_QSA_SPARSE_ASYNC']='0'
            elif case=='early':args[5]=23
            elif case=='retained':args[9]=object()
            elif case=='empty':args[4]=NS(numel=lambda:0)
            else:env['VLLM_QSA_KVO_SPARSE_PREFILL']='0'
            with self.subTest(case=case):
                self.assertEqual(fn(*args),'stock')
                self.assertEqual(len(calls),1);self.assertEqual(calls[0][0],'stock')
                for actual,expected in zip(calls[0][1],args):self.assertIs(actual,expected)

    def test_public_order_async_route(self):
        calls=[];fn,args=self.public_wrapper({'FLASH_QSA_SPARSE_ASYNC':'1'},calls)
        fn(*args)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][0],'async')
        for actual,expected in zip(calls[0][1],args):self.assertIs(actual,expected)

    def test_dense_sparse_empty_and_boundaries_byte_equal(self):
        rng=np.random.default_rng(71);n=96;page=8;arena=3
        for count in (0,1,23,24,25,48,76,77,96):
            for _ in range(5):
                ids=np.full((5,129),-1,np.int32);rows=np.array([4,1,3],np.int32)
                chosen=rng.choice(n,count,replace=False)
                for r in rows:
                    values=np.concatenate([chosen,chosen[:min(20,count)]])
                    rng.shuffle(values);ids[r,:len(values)]=values;ids[r,-1]=len(values)
                old=old_plan(ids,rows,n,page,arena);new=device_plan(ids,rows,n,page,arena)
                self.assertEqual(old[0].tolist(),new[0].tolist());self.assertEqual(old[2],new[2])
                if count:self.assertEqual(old[1].tobytes(),new[1].tobytes())
                kv=rng.normal(size=(n,8)).astype('f');q=rng.normal(size=(5,8)).astype('f')
                self.assertEqual(staged(old,ids,rows,kv,q,page).tobytes(),staged(new,ids,rows,kv,q,page).tobytes())
    def test_invalid_inputs_not_part_of_contract(self):
        with self.assertRaises(ValueError):device_plan(np.array([[96,1]]),np.array([0]),96,8,3)
    def test_generated_default_is_original_function_and_async_has_no_host_read(self):
        src=(ROOT/'patches/retain/ops_qsa.py.v8.1').read_text();new=g.generate(src)
        stock=g.function(new,'_s6_staged_stock').replace('def _s6_staged_stock(','def qsa_sparse_paged_attention_staged(',1)
        self.assertEqual(stock,g.function(src,'qsa_sparse_paged_attention_staged'))
        for name in ('_s6_staged_async','_s6_union'):
            f=ast.unparse(ast.parse(g.function(new,name)))
            for forbidden in ('torch.unique','.item(','.tolist(','.cpu(','.synchronize(', 'int(unique'):
                self.assertNotIn(forbidden,f)
        ast.parse(new)
        self.assertIn('FLASH_QSA_SPARSE_ASYNC',new);self.assertIn('FLASH_QSA_SPARSE_VERIFY',new)
    def test_missing_rank_and_mismatch_block_verification(self):
        from s6_grade import verify
        rows=[f"(Worker_PP{p}_TP{t}) S6_VERIFY rows=1024 equal=1" for p in range(2) for t in range(2)]
        self.assertTrue(verify('\n'.join(rows)))
        self.assertFalse(verify('\n'.join(rows[:-1])))
        self.assertFalse(verify('\n'.join(rows+[rows[0].replace('equal=1','equal=0')])))

    def test_stale_partial_page_has_an_explicit_count_mask(self):
        out=g.generate((ROOT/'patches/retain/ops_qsa.py.v8.1').read_text())
        self.assertIn('logical_token < tl.where(sparse, count, total_tokens)',g.function(out,'_s6_staged_kernel'))
        self.assertIn('(v<N)',g.function(out,'_s6_remap'))
        # Poison all unselected rows, including the partial final page: the
        # selected staged rows and result must be independent of those bytes.
        ids=np.array([[1,4,7,-1,3]],np.int32);rows=np.array([0]);q=np.ones((1,8),np.float32)
        kv=np.arange(32*8,dtype=np.float32).reshape(32,8)/100
        plan=device_plan(ids,rows,32,8,2)
        original=staged(plan,ids,rows,kv,q,8)
        kv2=kv.copy();mask=np.ones(32,bool);mask[[1,4,7]]=False;kv2[mask]=np.nan
        self.assertEqual(original.tobytes(),staged(plan,ids,rows,kv2,q,8).tobytes())

    def test_aba_requires_gain_beyond_both_controls_and_reports_streams(self):
        import copy
        from s6_grade import performance
        def record(tps):
            w={'agg_tps':tps,'worst_stream_gap_ms':{'p99':100},'per_stream':[{'tps':tps/8} for _ in range(8)]}
            return {'verdict':'PASS','fails':[],'streams':8,'stream_prompt':153537,'long_prompt':200000,
                    'before':copy.deepcopy(w),'during':copy.deepcopy(w),
                    'newcomer':{'cached_tokens':0,'client_ttft_s':27}}
        a=record(100);b=record(104);c=record(103)
        self.assertEqual(performance(a,b,c)['verdict'],'FAIL')
        c=record(101);d=performance(a,b,c)
        self.assertEqual(d['verdict'],'PASS');self.assertEqual(len(d['measurements']['during']['per_stream_tps']),8)
        b['during']['per_stream'][0]['tps']=1
        self.assertEqual(performance(a,b,c)['verdict'],'FAIL')

    def test_unknown_source_rejected(self):
        with self.assertRaises(AssertionError):g.generate('changed')

if __name__=='__main__':unittest.main()
