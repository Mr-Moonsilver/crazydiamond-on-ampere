#!/usr/bin/env python3
"""Bounded native34 causal arm; preserves preconditioning, never qualifies row4."""
import argparse
import json
from pathlib import Path
import time

from e3_native_qualification_driver import (
    QualificationDriver,QualificationRefused,QualificationError,NativePlacementRunner,
    freeze_prefill_plan,correctness,PLACEMENT_STREAMS,PLACEMENT_PHASE,RANKS,
)


class Native34PlacementRunner(NativePlacementRunner):
    def marker(self,driver,action,epoch,arm=None):
        value=dict(action=action)
        if arm is not None:value['arm']=arm
        reply=driver._worker('row4','teacher_evidence',dict(action='teacher_evidence',native34=value))
        rows=self._qualifications(reply,action='teacher_evidence',epoch=epoch)
        proofs={rank:rows[rank].get('native34') for rank in RANKS}
        if any(type(proof) is not dict for proof in proofs.values()):
            raise QualificationRefused('native34-worker-proof-missing')
        return proofs

    def control_a(self,driver,states,epoch):
        starts={state.index:state.count() for state in states}
        self.marker(driver,'A_START',epoch)
        driver._session('row4','resume',require_epoch=False)
        deadline=time.monotonic()+min(30.,driver.timeout_s)
        while time.monotonic()<deadline:
            if any(state.error is not None or state.malformed is not None or state.finish_reason is not None for state in states):
                raise QualificationRefused('native34-control-a-live-cohort-lost')
            if all(state.count()>=starts[state.index]+2 for state in states):break
            time.sleep(.005)
        else:raise QualificationRefused('native34-control-a-no-progress')
        # The real pause is global and the real barrier drains owned ordinary,
        # output-copy and PP consumers before the per-graph event queries.
        # Re-entering the already-owned split route supplies an awaited global
        # keep-pause without changing banks. A_END performs its explicit stream
        # drain BEFORE the original qualification replay-census query.
        paused=driver._session('row4','pause_enter_split')
        proofs=self.marker(driver,'A_END',epoch)
        if any(proof.get('verdict')!='PASS' for proof in proofs.values()):
            raise QualificationRefused('native34-control-a-unproved',evidence=dict(workers=proofs))
        return dict(verdict='PASS',workers=proofs,pause=paused)

    def run(self,driver,*,window_id,run_id,engine_epoch,model,arm):
        if arm not in ('S','O','I'):raise ValueError('native34-arm-invalid')
        full_bank,split_bank=driver.route_banks()
        prefill=freeze_prefill_plan(self.prefill_plan,run_id=run_id,model=model)
        states=[];prefill_states=[];transition_attempted=False;result=dict(schema='native34-arm/1',arm=arm,
            run_id=run_id,engine_epoch=engine_epoch,verdict='INCONCLUSIVE',
            release_qualified=False,scope='bounded first native prefix and both captured decode cohorts',
            preserves=['INT4','TP2xPP2','row1/rule4 preconditioning','eight live decoders','native slot8/cap10','txn100 chunk[0,1024)'])
        try:
            prepared=driver._session('row4','prepare',target_route='full')
            self._require_acks(prepared,bank=full_bank,label='prepare-full',epoch=engine_epoch)
            frontend=driver._frontend('row4',False,None)
            if frontend.get('enabled') is not False:raise QualificationRefused('shipping-path-instrument-enabled')
            configured=driver._worker('row4','teacher_configure',dict(action='teacher_configure',
                enabled=True,snapshot_max_bytes=self.snapshot_max_bytes))
            self._qualifications(configured,action='teacher_configure',epoch=engine_epoch)
            cohort=correctness.build_cohort(self.workload)
            if len(cohort)!=PLACEMENT_STREAMS:raise QualificationRefused('native34-cohort-shape')
            salt=correctness.build_cache_salt(run_id,9000)
            bodies=[correctness.request_body(model,row['prompt'],self.workload,salt) for row in cohort]
            # A deferred guided-decoding sample would need a later peer command;
            # this diagnostic accepts the exact plain original workload only.
            if any(any(key in body for key in ('structured_outputs','guided_decoding','response_format')) for body in bodies):
                raise QualificationRefused('native34-deferred-sampling-workload-refused')
            states=correctness._start_streams(self.url,model,bodies,cohort,self.workload)
            driver._session('row4','resume',require_epoch=False)
            opened=correctness._wait_open(states,self.workload);result['full_open']=opened
            if opened.get('complete') is not True:raise QualificationRefused('placement-open-incomplete')
            # No native phase is supplied until A completes. No prefill request
            # exists during this split-only segment.
            transition_attempted=True
            split=driver._session('row4','pause_enter_split')
            self._require_acks(split,bank=split_bank,label='enter-split',epoch=engine_epoch)
            result['control_a']=self.control_a(driver,states,engine_epoch)
            result['arm_acks']=self.marker(driver,'ARM',engine_epoch,arm)
            admission=driver._session('row4','admission',phase=PLACEMENT_PHASE)
            if admission.get('armed') is not True:raise QualificationRefused('native34-admission-not-armed')
            if any(state.finish_reason is not None or state.error is not None for state in states):
                raise QualificationRefused('native34-cohort-finished-before-native-admission')
            result['decode_counts_at_admission']={str(state.index):state.count() for state in states}
            prefill_states=correctness._start_streams(self.url,model,prefill['bodies'],prefill['cohort'],prefill['workload'])
            driver._session('row4','resume',require_epoch=False)
            deadline=time.monotonic()+driver.timeout_s
            while time.monotonic()<deadline:
                status=driver._session('row4','status')
                causal=status.get('native34') or {}
                if causal.get('terminal') is True:
                    result['causal']=causal;break
                if any(state.error is not None or state.finish_reason is not None for state in states):
                    raise QualificationRefused('native34-live-cohort-lost-before-bounded-cut',evidence=dict(causal=causal))
                time.sleep(.01)
            else:raise QualificationRefused('native34-bounded-prefix-timeout')
            result.update(verdict='PASS',failure_class='NONE',
                interpretation='this arm completed its bounded protocol; no fault mechanism or release pass inferred')
        except QualificationRefused as exc:
            result.update(reason=exc.code,failure_class='HARNESS',refusal=exc.evidence)
        except Exception as exc:
            result.update(verdict='FAIL',failure_class='HARNESS',error_type=type(exc).__name__,
                classification_basis='retained device fault receipts override to ENGINE')
        finally:
            prefill_alive=correctness._stop_streams(result,prefill_states,prefill['workload'])
            decode_alive=correctness._stop_streams(result,states,self.workload)
            result['reader_joins']=dict(prefill=prefill_alive,decode=decode_alive)
            result['cleanup']=(self._attempt_cleanup(driver,engine_epoch) if transition_attempted else
                               dict(attempted=False,restored=False,reason='no split transition attempted'))
            if prefill_alive or decode_alive or (transition_attempted and result['cleanup'].get('restored') is not True):
                result['verdict']='INCONCLUSIVE';result['failure_class']='HARNESS'
                result.setdefault('reason','native34-cleanup-incomplete')
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('url','model','run-id','store','workload','prefill-plan','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=('S','O','I'),required=True)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--snapshot-max-bytes',type=int,required=True)
    p.add_argument('--timeout-s',type=float,default=90.)
    a=p.parse_args(argv)
    from e3_q18a_qualification import _workload_object
    try:
        workload=_workload_object(json.loads(Path(a.workload).read_text()))
        prefill=json.loads(Path(a.prefill_plan).read_text())
        placement=Native34PlacementRunner(workload=workload,url=a.url,prefill_plan=prefill,
                                          snapshot_max_bytes=a.snapshot_max_bytes)
        driver=QualificationDriver(a.store,run_id=a.run_id,model=a.model,engine_epoch=a.epoch,
            url=a.url,placement=placement,snapshot_max_bytes=a.snapshot_max_bytes,timeout_s=a.timeout_s)
        result=placement.run(driver,window_id=a.run_id+'-causal',run_id=a.run_id,
            engine_epoch=a.epoch,model=a.model,arm=a.arm)
    except (ValueError,TypeError,OSError,QualificationError) as exc:
        result=dict(schema='native34-arm/1',arm=a.arm,run_id=a.run_id,engine_epoch=a.epoch,
            verdict='INCONCLUSIVE',failure_class='HARNESS',release_qualified=False,
            reason=exc.code if isinstance(exc,QualificationRefused) else 'native34-input-contract',error_type=type(exc).__name__)
    with Path(a.out).open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    return 0 if result['verdict']=='PASS' else 2 if result['verdict']=='INCONCLUSIVE' else 1


if __name__=='__main__':raise SystemExit(main())
