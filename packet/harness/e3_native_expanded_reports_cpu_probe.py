#!/usr/bin/env python3
"""Mounted runtime CPU source proof. Device/scheduler inputs are explicit fakes."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from e3_native_import_probe import check,error


def collect(graph):
    import torch
    from vllm.v1.flash_e3 import engine_driver as ed,native_measurement as nm
    # The reviewed fixture imports bare names. These aliases name exactly the
    # already mounted modules, with no replacement classes or implementations.
    aliases=[]
    for row in graph['modules']:
        name=row['module']
        if name.startswith('vllm.v1.flash_e3.') and row['active_mount']:
            mod=importlib.import_module(name)
            leaf=name.rsplit('.',1)[-1]
            if leaf not in sys.modules:sys.modules[leaf]=mod
            assert sys.modules[leaf] is mod
            aliases.append(leaf)
    from test_e3_native_measurement import make_long_stack
    from vllm.v1.flash_e3.lane_dispatch import LaneDispatcher
    from vllm.v1.flash_e3.worker_bridge import NativeWorkerBridge
    methods=(ed.NativeSegmentDriver._record,ed.NativeSegmentDriver._ingest_report,
             LaneDispatcher.submit,NativeWorkerBridge._build_report)
    s=make_long_stack(tokens=2,layers=48)
    assert type(s.driver) is ed.NativeSegmentDriver
    assert all(type(w.dispatcher) is LaneDispatcher and type(w._flash_s5_bridge) is NativeWorkerBridge
               for w in s.workers.values())
    start=int(s.driver._monotonic_ns())
    s.drive(lambda:s.driver._split_ready)
    for rank in (0,1):s.worker(rank).lane.pause_counts[0]=200
    s.drive(lambda:s.driver.phase=='finalize',limit=2500)
    s.complete_everything();s.drive(lambda:s.driver.tx is None)
    end=int(s.driver._monotonic_ns())+1
    evidence=s.driver.measurement_census.evidence(start,end,now_ns=end+1)
    assert evidence['completeness']['complete'] is True,evidence['reasons']
    assert evidence['segments'][0]['segment_count']==258
    assert evidence['segments'][0]['stage_segment_counts']=={'0':229,'1':29}
    assert [len(r['proofs']) for r in evidence['completions']]==[230,230,31,31]
    assert s.driver.error is None and s.driver._journal_dropped>0
    assert evidence['completeness']['raw_journal_flag_mutated'] is False
    assert s.driver._last_native_closure['complete'] is True
    assert methods==(ed.NativeSegmentDriver._record,ed.NativeSegmentDriver._ingest_report,
                     LaneDispatcher.submit,NativeWorkerBridge._build_report)
    reports=[row for row in s.driver._journal if row['kind']=='report'
             and isinstance(row['completed_op_seqs'],dict)]
    assert reports
    for report in reports:
        for field in ('completed_op_seqs','pending_op_seqs','proofs'):
            decoded=nm.decode_report_vector(report[field])
            assert nm.encode_report_vector(decoded)==report[field]
    # This is the exact unchanged native13 refusal; the source metadata guard
    # still applies to ordinary non-report input, rather than widening64.
    try:ed._bounded_value(list(range(65)))
    except ed.DriverError as exc:assert 'sequence is too long' in str(exc)
    else:raise AssertionError('legacy generic metadata limit widened')
    caps=s.driver.measurement_census.stats()['capacity']
    assert caps['segments_total_limit']==128*8192
    assert caps['proofs_per_rank_total_limit']==128*8192
    assert caps['completed_per_rank_total_limit']==128*8192
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',accepted_segments=258,stage_segments={'0':229,'1':29},
        queried_proofs=[230,230,31,31],measurement=evidence,
        native_closure=s.driver._last_native_closure,raw_journal_dropped=s.driver._journal_dropped,
        paged_reports=len(reports),sample_paged_report=reports[-1],
        generic64_refusal_preserved=True,aggregate_budget_unchanged=True,
        runtime_classes_actual_mounted=True,runtime_methods_patched=False,
        startup_flags_patched=False,fixture_import_aliases=aliases,
        simulated_device_and_scheduler_inputs=True,hardware_qualified=False,
        source_shas={name:hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                     for name,mod in (('engine_driver',ed),('native_measurement',nm))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    graph=json.loads(a.graph.read_text())
    try:
        j=check(graph)
        if j['verdict']=='PASS':j['expanded_reports']=collect(graph)
    except Exception as exc:
        j=locals().get('j',dict(failures=[],missing=[],counts={}))
        j.update(verdict='FAIL',rc=1,exception=error(exc));j['failures'].append('expanded-source-report-proof')
    j.update(graph_sha256=hashlib.sha256(a.graph.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_qualified=False,hardware_qualified=False,
        scope='actual mounted driver/dispatcher/bridge and telemetry, explicit CPU fake device/scheduler leaves; no GPU/model/real MQ/TP/PP/performance')
    with a.out.open('x') as f:json.dump(j,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:j.get(k) for k in ('verdict','rc','failures','missing')}))
    return j['rc']


if __name__=='__main__':raise SystemExit(main())
