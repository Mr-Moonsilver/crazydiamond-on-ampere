"""W26 regressions: counter domains, cache ownership, typed cancellation and source revert mutants."""
import ast,copy,inspect,json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import vision_fit_8x200k as fit
import test_vision_fit_guards as guards
import fit_memory
import fit_8x200k_source as source

class Repairs(unittest.TestCase):
    def geometry(self):
        row=guards.complete();coverage=row['memory']['native_coverage']
        coverage['native_turns']=1
        coverage['geometry_tail']=dict(turns=3,max_r=64,max_rows=64)
        coverage['rows']['live']['stock']['geometry-tail']=192
        coverage['requests']['live']['stock']['geometry-tail']=3
        return row
    def test_disjoint_terminal_classes_and_revert_mutant(self):
        row=self.geometry();self.assertEqual(fit.assess(row)['verdict'],'FIT')
        text=inspect.getsource(fit.base.assess).replace("geometry['requests']>geometry['turns']","geometry['turns']>native['native_turns']")
        namespace=dict(fit.base.__dict__,PREFIXES=fit.PREFIXES,OUTPUT=8192,OUTPUT_CAPS=fit.OUTPUT_CAPS,COLD_HISTORY=190000);exec(text,namespace)
        with patch.object(fit.base,'assess',namespace['assess']):
            self.assertIn('registered-exception-proof-missing',fit.assess(row)['reasons'])
    def test_unowned_or_excess_geometry_still_refuses_and_mutant(self):
        row=self.geometry();row['memory']['native_coverage']['requests']['live']['stock']['geometry-tail']=4
        self.assertIn('registered-exception-proof-missing',fit.assess(row)['reasons'])
        text=inspect.getsource(fit.base.assess).replace("geometry['requests']>geometry['turns']",'False')
        namespace=dict(fit.base.__dict__,PREFIXES=fit.PREFIXES,OUTPUT=8192,OUTPUT_CAPS=fit.OUTPUT_CAPS,COLD_HISTORY=190000);exec(text,namespace)
        with patch.object(fit.base,'assess',namespace['assess']):self.assertEqual(fit.assess(row)['verdict'],'FIT')
        row=self.geometry();row['memory']['native_coverage']['rows']['live']['stock']['geometry-tail']=193
        self.assertIn('registered-exception-proof-missing',fit.assess(row)['reasons'])
    def test_cache_readiness_is_not_a_junction_default_or_foreign_epoch(self):
        for mutation in (lambda h:h.update(start=0,junction=190256),lambda h:h.update(engine_epoch=-1),lambda h:h.update(uid='wrong')):
            row=guards.complete();mutation(row['cold']['cache_readiness'][0])
            self.assertIn('cached-companions-unproven',fit.assess(row)['reasons'])
        original=inspect.getsource(fit.assess);mutant=original.replace("reasons.append('cached-companions-unproven')",'pass')
        namespace=dict(fit.__dict__);exec(mutant,namespace)
        row=guards.complete();row['cold']['cache_readiness'][0]['start']=0
        self.assertEqual(namespace['assess'](row)['verdict'],'FIT')
    def test_first_admission_does_not_become_a_later_computed_frontier(self):
        engine=NS(_flash_s5_driver=NS(engine_epoch=5),scheduler=NS())
        floor=fit_memory.FitFloor(engine,'e6',writer=lambda _:None)
        def admit(start,end):
            tx=NS(request_uid='uid',chunk_start=start,chunk_end=end,txn_seq=1)
            floor.checked=(('uid',start,end),{})
            floor.admitted(NS(transaction=tx,committed_end=start,prompt_end=190320))
        admit(190080,190256);admit(190256,190319)
        self.assertEqual(floor.first_admissions['uid']['start'],190080)
        with self.assertRaisesRegex(ValueError,'exact-floor-permit'):
            floor.admitted(NS(transaction=NS(request_uid='uid',chunk_start=1,chunk_end=2)))
    def test_native_prefill_stats_set_is_once_and_uses_the_actual_hit(self):
        before=(source.PARENT/'e3/scheduler_hooks.py').read_text()
        after=source.transform('e3/scheduler_hooks.py',before)[0]
        tree=ast.parse(after);owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrefillAdmissionController')
        method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_admit_observed')
        assignment=next(n for n in method.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_stats')
        guard=method.body[method.body.index(assignment)+1]
        script=compile(ast.Module(body=[assignment,guard],type_ignores=[]),'real-native-cache-statistics','exec')
        calls=[];request=NS(prefill_stats=NS(set=lambda **kw:calls.append(kw)),num_preemptions=0,num_computed_tokens=0,num_prompt_tokens=190320)
        exec(script,dict(request=request,num_new_computed=190080))
        self.assertEqual(calls,[dict(num_prompt_tokens=190320,num_local_cached_tokens=190080,num_external_cached_tokens=0)])
        request.num_computed_tokens=190256;exec(script,dict(request=request,num_new_computed=0));self.assertEqual(len(calls),1)
    def test_only_explicit_before_first_output_cancel_is_expected(self):
        class Response:
            status=200
            def __iter__(self):return iter(())
        class Connection:
            sock=object()
            def connect(self):pass
            def request(self,*a):pass
            def getresponse(self):return Response()
            def close(self):pass
        args=NS(url='http://fixture',timeout_s=1,model='fixture')
        for requested,expected in ((True,'expected-cancel'),(False,'error')):
            stream=fit.Stream('fixture',[1],[],{},128);stream.cancel_requested=requested
            with patch.object(fit.base,'connection',return_value=(Connection(),'')):
                fit.generate(args,stream)
            self.assertEqual(stream.outcome,expected)
            self.assertEqual(stream.error is None,requested)
        stream=fit.Stream('fixture',[1],[],{},128);stream.cancel_requested=True;stream.first_ns=1
        with patch.object(fit.base,'connection',return_value=(Connection(),'')):fit.generate(args,stream)
        self.assertEqual(stream.outcome,'error')

if __name__=='__main__':unittest.main()
