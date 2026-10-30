#!/usr/bin/env python3
"""One row-4 placement diagnostic; never invent current-epoch qualification."""
import argparse
import json
from pathlib import Path
import time


def run(args):
    from e3_native_qualification_driver import QualificationDriver,NativePlacementRunner,QualificationRefused
    from e3_q18a_qualification import _workload_object
    placement=NativePlacementRunner(workload=_workload_object(args.workload),url=args.url,
        prefill_plan=json.loads(Path(args.prefill_plan).read_text()),snapshot_max_bytes=args.snapshot_max_bytes)
    driver=QualificationDriver(args.store,run_id=args.run_id,model=args.model,
        engine_epoch=args.epoch,url=args.url,placement=placement,
        snapshot_max_bytes=args.snapshot_max_bytes,timeout_s=args.timeout_s)
    record=dict(schema='native30-row4-localization/1',scope='one existing row4 placement sequence; not current-epoch qualification',
        engine_epoch=args.epoch,run_id=args.run_id,started_ns=time.monotonic_ns(),release_qualified=False,
        skipped=['row1','rule4_controls','rule4_candidate'],skip_basis='owner-authorized prior identical-engine receipts; no prior artifact relabeled to this epoch')
    try:
        result=placement.run(driver,window_id=args.run_id+'-placement',run_id=args.run_id,engine_epoch=args.epoch,model=args.model)
        record.update(verdict='PASS',failure_class='NONE',evidence=result)
        rc=0
    except QualificationRefused as exc:
        record.update(verdict='INCONCLUSIVE',failure_class='HARNESS',reason=exc.code,evidence=exc.evidence,cleanup=exc.cleanup,
            classification_basis='placement refused or engine response unavailable; retained rank/host fault receipts override to ENGINE')
        rc=2
    except Exception as exc:
        record.update(verdict='FAIL',failure_class='HARNESS',error_type=type(exc).__name__)
        rc=1
    record['finished_ns']=time.monotonic_ns()
    with Path(args.out).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    return rc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('url','model','run-id','store','workload','prefill-plan','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--snapshot-max-bytes',type=int,required=True)
    p.add_argument('--timeout-s',type=float,default=90)
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
