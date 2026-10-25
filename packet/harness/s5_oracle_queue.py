"""Strict subset inputs and artifact binding for the unarmed source-oracle queue."""
import argparse
import json
from pathlib import Path
from s5_resident import CASE_CONFIG
from s5_value import summarize,table


def validate(names,pair,moe,trials,trace,blas=0):
    if not names or len(set(names))!=len(names) or any(n not in CASE_CONFIG for n in names):
        raise ValueError('requires nonempty unique canonical case names')
    if pair not in ('0,2','1,3') or moe not in ('on','off'):
        raise ValueError('unknown pair/MoE mode')
    if type(trials) is not int or trials not in (3,6) or type(trace) is not int or trace not in (0,1):
        raise ValueError('unreviewed trial/trace setting')
    if type(blas) is not int or blas not in (0,1):raise ValueError('unknown cuBLAS control')
    return dict(cases=list(names),pair=pair,moe=moe,trials=trials,trace=trace,blas=blas)


def bound_summary(root,config):
    root=Path(root);r=summarize(root,tuple(config['cases']))
    for case in r['cases']:
        try:
            folder=root/case['case'];d=json.loads((folder/'profile-terminal/result.json').read_text())
            if d['pair']!=config['pair']:raise ValueError('pair')
            for rank in d['ranks']:
                blas=rank['cublas_control']
                if (blas['requested'] is not bool(config['blas'])
                    or blas['deterministic_algorithms'] is not bool(config['blas'])
                    or (config['blas'] and blas['workspace_config']!=':4096:8')):
                    raise ValueError('cuBLAS control not applied')
                p=rank['value_probe'];check=rank['resident_substitution']
                if (rank['ablation']['moe']!=config['moe'] or p['source_trials']!=config['trials']
                    or p['source_trace_on_fail'] is not bool(config['trace'])
                    or check['host_runs']!=config['trials'] or check['resident_runs']!=config['trials']
                    or check['chunks_per_run']!=3 or check['passed'] is not True):
                    raise ValueError('oracle configuration/evidence')
                if config['moe']=='off' and check.get('deterministic_control') is not True:
                    raise ValueError('MoE-off deterministic control')
            handoff=json.loads((folder/'handoff.json').read_text())
            if not (handoff['own_container_stopped']==1 and handoff['gpu_idle']==1 and handoff['engine_untouched'] is True):
                case['verdict']='FAIL';raise ValueError('cleanup')
            case['oracle_binding']='matched'
        except (OSError,KeyError,TypeError,ValueError):
            case['oracle_binding']='missing_or_mismatched'
            if case['verdict']!='FAIL':case['verdict']='INCONCLUSIVE'
    r['verdict']='FAIL' if any(c['verdict']=='FAIL' for c in r['cases']) else 'INCONCLUSIVE' if any(c['verdict']!='PASS' for c in r['cases']) else 'PASS'
    r['requested']=config
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--pair',required=True);p.add_argument('--moe',required=True)
    p.add_argument('--trials',required=True,type=int);p.add_argument('--trace',required=True,type=int)
    p.add_argument('--blas',type=int,default=0)
    p.add_argument('names',nargs='*');a=p.parse_args()
    try:c=validate(a.names,a.pair,a.moe,a.trials,a.trace,a.blas)
    except ValueError as e:p.error(str(e))
    if a.root is None:print(' '.join(c['cases']));return 0
    r=bound_summary(a.root,c)
    (a.root/'summary.json').write_text(json.dumps(r)+'\n');(a.root/'summary.md').write_text(table(r))
    print(table(r),end='');print('S5 ORACLE '+r['verdict'])
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[r['verdict']]


if __name__=='__main__':raise SystemExit(main())
