"""Bounded causal-arm reading; device errors take precedence over host success."""
import argparse
import hashlib
import json
from pathlib import Path


def grade(paths,arm_record=None):
    first=None;prior=[];inputs=[];completed=[];malformed=lost=limits=unavailable=0;pending={}
    def consume(row):
        nonlocal first,limits,unavailable
        event=row.get('event','')
        if event in ('native34-indexer-prior-work-exception','native34-indexer-kernel-exception',
                     'native34-phase-exception','native34-phase-completion-exception','exception'):
            if first is None or row.get('monotonic_ns',0)<first.get('monotonic_ns',0):first=row
        if event=='native34-indexer-prior-work-drain-complete' and len(prior)<4:prior.append(row)
        if event=='native34-indexer-launch-inputs' and len(inputs)<4:inputs.append(row)
        if event=='native34-indexer-kernel-end' and len(completed)<4:completed.append(row)
        limits+=event in ('native34-receipt-limit','host-receipt-limit')
        unavailable+=event=='native34-unavailable'
    for name in paths:
        with Path(name).open(errors='replace') as stream:
            for line in stream:
                if 'NATIVE30 ' not in line:continue
                try:row=json.loads(line.split('NATIVE30 ',1)[1])
                except (TypeError,ValueError):malformed+=1;continue
                if row.get('event')!='host-fragment':consume(row);continue
                if not row.get('logical_event','').startswith('native34-'):continue
                key=(row.get('pid'),row.get('rank'),row.get('record_id'))
                if (not isinstance(row.get('parts'),int) or not 0<row['parts']<=128
                        or row.get('part') not in range(row['parts']) or len(pending)>=64):
                    lost+=1;continue
                parts=pending.setdefault(key,{})
                parts[row['part']]=row
                if len(parts)!=row['parts']:continue
                try:
                    text=''.join(parts[i]['data'] for i in range(row['parts']))
                    if len(text.encode())>128*1024 or hashlib.sha256(text.encode()).hexdigest()!=row['logical_sha256']:
                        raise ValueError('fragment-digest')
                    value=json.loads(text);value.update(pid=row['pid'],monotonic_ns=parts[0]['monotonic_ns'])
                    consume(value)
                except (KeyError,ValueError,TypeError):lost+=1
                del pending[key]
    lost+=len(pending)
    arm=json.loads(Path(arm_record).read_text()) if arm_record and Path(arm_record).is_file() else {}
    failure_class='ENGINE' if first else 'HARNESS' if malformed or lost or limits or unavailable or arm.get('verdict')!='PASS' else None
    verdict='FAIL' if first else 'PASS' if failure_class is None else 'INCONCLUSIVE'
    attribution='NO_DEVICE_FAULT_RECORDED'
    if first:
        attribution=('PRIOR_RANK1_WORK_BEFORE_TARGET' if first['event']=='native34-indexer-prior-work-exception'
                     else 'BRACKETED_INDEXER_LAUNCH_OR_FENCE' if first['event']=='native34-indexer-kernel-exception'
                     and any(r.get('rank')==1 and r.get('monotonic_ns',0)<first.get('monotonic_ns',0) for r in prior)
                     else 'OVERLAPPED_PHASE_OR_BOUNDARY')
    if not first and arm.get('arm')=='I' and not any(r.get('rank')==1 and r.get('rank1_post_fence') is True for r in completed):
        verdict='INCONCLUSIVE';failure_class='HARNESS';attribution='I_DISCRIMINATOR_NOT_PROVED'
    return dict(schema='native34-causal-reading/1',verdict=verdict,failure_class=failure_class,
        arm=arm.get('arm'),release_qualified=False,attribution=attribution,first_exception=first,
        prior_work_drain_receipts=prior,indexer_inputs=inputs,indexer_completed=completed,
        malformed=malformed,missing_fragments=lost,receipt_limits=limits,metadata_unavailable=unavailable,
        arm_record_verdict=arm.get('verdict'),arm_record_reason=arm.get('reason'),
        scope='PASS means bounded causal arm only. No access-PC proof, peer-access exclusion, numerical or placement pass. '
              'Declared owners and source providers are not observed last device writers. No fix established.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',action='append',required=True)
    p.add_argument('--arm-record');p.add_argument('--out',required=True);a=p.parse_args()
    r=grade(a.log,a.arm_record)
    with Path(a.out).open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r['verdict']=='PASS' else 1 if r['verdict']=='FAIL' else 2)
