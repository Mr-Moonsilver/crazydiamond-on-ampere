#!/usr/bin/env python3
"""Source mutants through the same CPU path tests; retained outputs, no device imports here."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=(
 ('package-import','make_s5_overlap',"node.module in (*E3_MODULES, 'startup_vram_budget')",'node.module in E3_MODULES','ContractTests.test_package_qualifies_optional_startup_helper'),
 ('compact-revert','native_resources','for spec in prefill_scratch_specs(chunk_tokens, compact=compact):','for spec in prefill_scratch_specs(chunk_tokens, compact=False):','ContractTests.test_poison_removed_owners_never_allocated'),
 ('wrong-lane','lane_private','if manager._ensure_workspace_size(1) is not self.raw:','if False:','ScorePathTests.test_wrong_workspace_lane_refuses'),
 ('wrong-stream','lane_private',"if type(token) is not int or token != getattr(current, 'cuda_stream', None):",'if False:','ScorePathTests.test_wrong_compute_stream_refuses'),
 ('stale-view','lane_private','if not self.active or generation != self.generation or _SCORE_LEASE.get() != (self, generation):','if not self.active:','ScorePathTests.test_stale_generation_and_outside_lease_refuse'),
 ('reorder','lane_private',"if _SCORE_LEASE.get() is not None:\n        raise LanePrivateError('model/MoE body reordered", "if False:\n        raise LanePrivateError('model/MoE body reordered",'ScorePathTests.test_reordered_model_body_refuses_before_overwrite'),
 ('error-retention','lane_private','self.failed = True  # Keep storage retained; a partial enqueue is never retried.','pass  # mutant: permit reuse after partial enqueue','ScorePathTests.test_partial_indexer_error_retains_and_quarantines'),
 ('query-alias','lane_private','if _aliases(q, self.raw):','if False:','ScorePathTests.test_query_alias_refuses_before_score_write'),
 ('score-scope','qsa_driver','with native_score_scope(), indexer_phase_scope(phases):','with indexer_phase_scope(phases):','ScorePathTests.test_router_built_commands_reach_both_worker_private_paths'),
 ('fixed-view-revert','make_s5_overlap',"'    logits = private_score_tensor(q, num_queries, logits_width)\\n'", "'    logits = None\\n'",'ScoreBytesTests.test_generated_score_function_same_kernel_args_strides_and_bytes'),
 ('startup-sync','startup_vram_budget','    torch.accelerator.synchronize()','    pass  # mutant: no synchronization','PhaseTests.test_real_sampler_synchronizes_before_each_snapshot'),
)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 results=[]
 for name,module,old,new,test in CASES:
  text=(ROOT/'patches/s5_engine'/(module+'.py')).read_text()
  if text.count(old)!=1:raise ValueError('mutant anchor missing or repeated:'+name)
  path=a.out/(module+'-'+name+'.py');path.write_text(text.replace(old,new));compile(path.read_text(),str(path),'exec')
  code="""import importlib.util,sys,unittest
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'harness'),str(root/'patches/s5_engine')]
spec=importlib.util.spec_from_file_location(sys.argv[2],sys.argv[3]);module=importlib.util.module_from_spec(spec);sys.modules[sys.argv[2]]=module;spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromName('test_g4_headroom.'+sys.argv[4]);result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not result.wasSuccessful())
"""
  r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),module,str(path),test],capture_output=True,text=True,timeout=60)
  (a.out/(name+'.stdout')).write_text(r.stdout);(a.out/(name+'.stderr')).write_text(r.stderr)
  caught=r.returncode!=0 and 'FAILED (' in r.stderr and 'ImportError' not in r.stderr and 'SyntaxError' not in r.stderr
  results.append(dict(name=name,test=test,rc=r.returncode,caught=caught))
  print('G4_MUTANT '+name+' '+('CAUGHT' if caught else 'ESCAPED'),flush=True)
 (a.out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
 if not all(r['caught'] for r in results):raise ValueError('headroom-mutant-escaped')
if __name__=='__main__':main()
