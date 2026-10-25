"""Offline scheduler decision and generated-hook tests, no vLLM/GPU imports."""
import ast,importlib.util,json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'patches/cost_budget'
def module(name):
    spec=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=module('controller')._FlashCostBudget;G=module('make_cost_budget');M=json.loads((P/'model.json').read_text())
class CostTests(unittest.TestCase):
    def test_tight_deadline_reports_infeasible_keeps_efficient_chunk(self):
        d=C(M,27).choose(150000,1024)
        self.assertEqual(d['status'],'infeasible');self.assertEqual(d['chunk'],1024)
        self.assertGreater(d['min_predicted_ms'],d['budget_ms'])
    def test_feasible_uses_largest_supported_chunk(self):
        c=C(M,1e6);short=c.predict(60000,512)['ms'];long=c.predict(60000,1024)['ms']
        c.deadline=M['decode_gap99_ms']+(short+long)/2
        self.assertEqual(c.choose(60000,1024)['chunk'],512)
        self.assertEqual(c.choose(60000,1024,spent=long)['status'],'infeasible')
    def test_no_decode_and_out_of_domain_are_not_advertised_feasible(self):
        c=C(M,27)
        self.assertEqual(c.choose(60000,1024,has_decode=False)['status'],'no_decode')
        self.assertEqual(c.choose(261000,1024)['status'],'out_of_domain')
    def test_link_and_coverage_affect_copy_cost(self):
        slow=C(M,200);fast=C(M,200,x8=False)
        self.assertGreater(slow.predict(150000,1024,1)['copy_ms'],fast.predict(150000,1024,1)['copy_ms'])
        self.assertGreater(slow.predict(150000,1024,1)['copy_ms'],slow.predict(150000,1024,.5)['copy_ms'])
    def test_finite_progress_and_simulated_rollovers(self):
        for agents in (8,10):
            c=C(M,27);pos=[60000+i*1000 for i in range(agents)]
            for step in range(200):
                i=step%agents;d=c.choose(pos[i],1024);self.assertGreater(d['chunk'],0)
                pos[i]+=d['chunk']
                if pos[i]>200000:pos[i]=0
            self.assertEqual(sum(c.counts.values()),200)
    def test_generated_integration_preserves_short_first_and_align(self):
        src=(ROOT/'reference/vllm-src/v1/core/sched/scheduler.py').read_text();out=G.generate(src,M)
        tree=ast.parse(out);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
        # Execute only the generated hook with fake requests. Preemption removes
        # reservations by deleting the authoritative scheduled-token entry.
        hook=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_flash_cost_chunk')
        namespace={'logger':type('Log',(),{'info':lambda *a:None})()};exec(compile(ast.Module(body=[hook],type_ignores=[]),'hook','exec'),namespace)
        obj=type('Scheduler',(),{})();obj._flash_cost=C(M,160);obj._flash_cost_active=True
        obj._flash_cost_history={'old':150000};obj._flash_cost_calls=5;obj.current_step=1
        req=type('Request',(),{'request_id':'new'})()
        namespace['_flash_cost_chunk'](obj,req,60000,1024,{'old':1024})
        d=namespace['_flash_cost_chunk'](obj,req,60000,1024,{})
        self.assertEqual(d,512)
        obj._flash_cost=None;self.assertEqual(namespace['_flash_cost_chunk'](obj,req,60000,1024,{}),1024)
        for name in ('_mamba_block_aligned_split','_sf_short_pending'):
            original=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==name)
            actual=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(original),ast.dump(actual))
        self.assertEqual(out.count('self._flash_cost_chunk(request,'),2)
if __name__=='__main__':unittest.main()
