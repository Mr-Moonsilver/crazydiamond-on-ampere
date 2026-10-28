#!/usr/bin/env python3
"""CPU source diagnosis only. No runtime changes, device work, engine or box."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import test_e3_sparse_history_dma as f
from test_e3_native_measurement import make_long_stack,ChunkBlocks


def guard_results():
    reference=ROOT/'reference/vllm-src/models/qwen4_exp/nvidia/qsa.py'
    stock=reference.read_text()
    threshold=next(ast.literal_eval(n.value) for n in ast.parse(stock).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_QSA_STAGE_MIN_ROWS' for t in n.targets))
    assert threshold==64
    f._GeneratedHarness.setUpClass()
    generated=f._GeneratedHarness.generated_qsa
    results=[]
    for name in ('_flash_s5_staged_operation','_flash_s5_sparse_staged_operation'):
        tree=ast.Module(body=[f._function_node(generated,name)],type_ignores=[])
        ast.fix_missing_locations(tree);namespace={'__name__':'models.qwen4_exp.nvidia.qsa','__package__':'models.qwen4_exp.nvidia','torch':None}
        exec(compile(tree,'<actual-generated-helper>','exec'),namespace)
        helper=namespace[name]
        for rows in (0,1,7,63,64,65,1024):
            layer=types.SimpleNamespace(_qsa_kv_offload=True,_qsa_stage_min_rows=threshold)
            try:next(helper(None,layer,None,None,None,None,rows,True,None))
            except RuntimeError as exc:message=str(exc)
            else:raise AssertionError('guard did not refuse absent CPU metadata')
            expected=('S5 operation requires the staged dense row threshold' if rows<64
                      else 'S5 operation requires CPU host metadata; no device readback')
            assert message==expected,(name,rows,message)
            results.append(dict(helper=name,rows=rows,minimum_rows=threshold,error=message))
    return dict(stock_threshold=threshold,stock_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                cases=results,uses_real_generated_guard=True,simulated_layer_attributes=True,
                no_threshold_override=True,device_or_kernel_executed=False)


def admission_sequence():
    s=make_long_stack(tokens=98312,layers=48,chunk=1024,manager_cls=ChunkBlocks)
    s.drive(lambda:s.driver._split_ready)
    s.scheduler.waiting.items=[]
    chunks=[]
    for _ in range(200):
        if s.driver.tx is None and not s.controller.active:
            # The stock scheduler owns the final one-token handback. This
            # fixture only drives native chunks, like run_journal itself.
            s.scheduler.waiting.items = [r for r in s.scheduler.waiting.items
                if r.num_computed_tokens < r.num_tokens - 1]
            if not s.scheduler.waiting.items:break
            if s.controller.admit(s.scheduler) is None:raise AssertionError('admission refused')
        admission=s.controller.parked.admission
        chunks.append(dict(start=admission.transaction.chunk_start,end=admission.transaction.chunk_end,
            rows=admission.chunk_tokens,remaining=admission.remaining_tokens,terminal=admission.terminal))
        s.drive(lambda:s.driver.phase=='finalize')
        s.complete_everything();s.drive(lambda:s.driver.tx is None)
    else:raise AssertionError('bounded admission loop did not finish')
    assert len(chunks)==97
    assert all(c['rows']==1024 for c in chunks[:-1])
    assert chunks[-1]==dict(start=98304,end=98311,rows=7,remaining=1,terminal=True)
    return dict(chunks=chunks,source_driver_and_controller=True,
                fake_scheduler_allocator_device_leaves=True,
                actual_pinned_scheduler_alignment_not_executed=True,
                unsupported_short_chunk_admitted=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'tmp/native14-threshold-diagnosis.json')
    p=parser.parse_args().out
    j=dict(status='SOURCE_DEFECT_REPRODUCED',guards=guard_results(),admission=admission_sequence(),
        runtime_changed=False,scope='source eligibility mismatch; exact box failing row count was not retained')
    p.open('x').write(json.dumps(j,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=j['status'],threshold=64,last_chunk=j['admission']['chunks'][-1],chunks=97,guard_cases=len(j['guards']['cases']))))


if __name__=='__main__':main()
