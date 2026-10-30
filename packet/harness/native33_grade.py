"""Bounded offline attribution summary; never promotes diagnostic success to safety."""
import argparse
import json
from pathlib import Path


def grade(paths):
    first=None;graphs=[];changes=[];limits=malformed=begins=ends=unavailable=0
    for path in paths:
        with Path(path).open(errors='replace') as f:
            for line in f:
                p=line.find('NATIVE30 ')
                if p<0:continue
                try:r=json.loads(line[p+9:])
                except ValueError:malformed+=1;continue
                event=r.get('event','')
                begins+=event=='native33-launch-begin';ends+=event=='native33-launch-end'
                limits+=event=='native33-receipt-limit'
                unavailable+=event=='native33-metadata-unavailable'
                if event in ('native33-launch-exception','native33-boundary-exception'):
                    if first is None or r.get('monotonic_ns',0)<first.get('monotonic_ns',0):first=r
                if event=='native33-graph-identity' and len(graphs)<16:graphs.append(r)
                if event=='native33-graph-pointer' and r.get('pointer_changed') is True and len(changes)<64:changes.append(r)
    attribution=('NAMED_TRITON_KERNEL' if first and first.get('kernel_name') else
                 'OPAQUE_GRAPH_REPLAY' if first and first.get('launch_kind')=='cuda-graph-replay' else
                 'OPERATOR_OR_BOUNDARY_ONLY' if first else 'NO_FAULT_RECORDED')
    return dict(schema='native33-attribution/1',verdict='FAIL' if first else 'INCONCLUSIVE',
                classification='ENGINE' if first else 'HARNESS' if limits or malformed or unavailable else None,
                attribution=attribution,first_exception=first,graph_identities=graphs,
                pointer_changes=changes,launch_begins=begins,launch_ends=ends,
                receipt_limits=limits,malformed=malformed,metadata_unavailable=unavailable,
                scope='Diagnostic window only. Named Triton site is not an access-PC proof; '
                      'opaque graph internal kernel stays UNKNOWN. Pointer inventory differences '
                      'do not prove the graph read set or causality. No fault under fences does '
                      'not prove the unchanged asynchronous engine safe.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--log',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=grade(a.log)
    with Path(a.out).open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(1 if r['verdict']=='FAIL' else 2)
