"""Native35 R/O reading: the native34 causal grade plus the native-private writer catalogue.

Arm O (q18a-native35-o) is graded exactly as native34 O3. Arm R additionally
requires, for both PP0 ranks, that the seven indirect operands matched host
intent and that the builder-written operands' last catalogued writer is the
native chunk's own one-request build, with no ordinary (multi-request) build
ever catalogued on that storage. Readback is evidence; it repairs nothing.
"""
import argparse
import json
from pathlib import Path

from native34_grade import grade as causal_grade
from native34_readback import KEYS

BUILDER_WRITTEN=('k_work_metadata_ptr','logical_positions_ptr','state_slots_ptr','compressed_slots_ptr')
CHUNK_WRITTEN=('query_start_loc_ptr','pos_ptr','state_table_ptr')
NATIVE_QUERY_STARTS=[0,1024]


def native_writer(values):
    failures,missing,rows=[],[],{}
    for rank in (0,1):
        for name in KEYS:
            key='rank%d:%s'%(rank,name)
            found=[r for r in values if r.get('rank')==rank and r.get('argument')==name]
            if len(found)!=1:
                missing.append(key+':indirect-value-record');continue
            row=found[0];chain=row.get('writer_chain') or {}
            history=chain.get('history') or [];last=chain.get('last_catalogued_writer')
            rows[key]=dict(value_verdict=row.get('verdict'),history_length=len(history),
                last_catalogued_writer={k:last.get(k) for k in ('provider','kernel','num_reqs','num_tokens','query_start_loc_cpu','mode','step','stream','completion_observed','transaction','chunk','request_index')} if type(last) is dict else None,
                history_overflow=chain.get('history_overflow'))
            if row.get('verdict')!='PASS':failures.append(key+':value-differs-from-host-intent')
            if chain.get('observed') is not True or type(last) is not dict:
                missing.append(key+':no-catalogued-writer');continue
            if chain.get('history_overflow') is True:missing.append(key+':writer-history-overflow')
            if name in BUILDER_WRITTEN:
                ordinary=[h for h in history if h.get('provider')=='QSAMetadataBuilder.build' and h.get('num_reqs')!=1]
                if ordinary:failures.append(key+':ordinary-build-catalogued-on-native-storage')
                native=(last.get('provider')=='QSAMetadataBuilder.build' and last.get('kernel')=='_build_qsa_metadata_kernel'
                        and last.get('num_reqs')==1 and last.get('query_start_loc_cpu')==NATIVE_QUERY_STARTS)
                if not native:
                    (failures if last.get('provider')=='QSAMetadataBuilder.build' else missing).append(key+':last-writer-not-native-one-request-build')
                elif last.get('completion_observed') is not True:
                    missing.append(key+':native-build-completion-not-observed')
            else:
                native=(last.get('provider')=='LaneChunk.__init__' and last.get('transaction')==100 and last.get('chunk')==[0,1024])
                if not native:missing.append(key+':last-writer-not-lane-chunk-constructor')
    coverage=not missing and set(rows)=={'rank%d:%s'%(r,n) for r in (0,1) for n in KEYS}
    return dict(verdict='FAIL' if failures else 'INCONCLUSIVE' if missing or not coverage else 'PASS',
        failures=failures,missing=missing,operands=rows,coverage=coverage,
        scope='host-issued writer catalogue and current-stream completion events; peer and unregistered device writers remain unproved; no fix inferred from a clean catalogue')


def grade(paths,arm_record=None,require_readback=False,require_native_writer=False):
    result=causal_grade(paths,arm_record,require_readback or require_native_writer)
    result.update(schema='native35-causal-reading/1',native_writer_required=require_native_writer,native_writer=None)
    if require_native_writer:
        writer=native_writer(result.get('indirect_values') or [])
        result['native_writer']=writer
        if result['first_exception'] is None:
            if writer['verdict']=='FAIL' and result['verdict']!='FAIL':
                result.update(verdict='FAIL',failure_class='ENGINE',attribution='ORDINARY_WRITER_ON_NATIVE_STORAGE_OR_VALUE_MISMATCH')
            elif writer['verdict']=='INCONCLUSIVE' and result['verdict']=='PASS':
                result.update(verdict='INCONCLUSIVE',failure_class='HARNESS',attribution='NATIVE_PRIVATE_WRITER_NOT_PROVED')
    result['scope']=('PASS means the bounded O-style arm completed without a device fault and, when required, '
        'every indirect operand matched host intent with the native-private storage last written by the native '
        'one-request build on both PP0 ranks. No access-PC proof, no peer-write exclusion, no numerical, '
        'placement or performance pass. A clean arm is evidence for the deciding O arm, not a release pass.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--log',action='append',required=True)
    p.add_argument('--require-readback',action='store_true');p.add_argument('--require-native-writer',action='store_true')
    p.add_argument('--arm-record');p.add_argument('--out',required=True);a=p.parse_args()
    r=grade(a.log,a.arm_record,a.require_readback,a.require_native_writer)
    with Path(a.out).open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r['verdict']=='PASS' else 1 if r['verdict']=='FAIL' else 2)
