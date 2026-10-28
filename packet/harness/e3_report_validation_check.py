"""CPU equivalence/cost check against the exact frozen report validator."""
import ast
import hashlib
from pathlib import Path
import time
import types

BASELINE_SHA='4f3c29949fddd0944275387cd1d6efd977f90d4d35f6a5e2d5191aa23cc9fb85'


def check(wire,baseline_path):
    baseline_path=Path(baseline_path)
    source=baseline_path.read_text()
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==BASELINE_SHA
    # A separate reference subclass; no live class or runtime method is patched.
    namespace=dict(vars(wire));exec(compile(source,str(baseline_path),'exec'),namespace)
    old=type('FrozenWorkerCompletionReport',(wire.WorkerCompletionReport,),
             {'__post_init__':namespace['__post_init__']})
    tx=wire.TransactionId(1,'report-validation-cpu',1,0,1,1)
    base=dict(transaction=tx,worker_rank=0,completed_op_seqs=(0,1),pending_op_seqs=(2,3),
              completed_sequence=0,proofs=(),cursor=wire.CompletionCursor('layer',0,0),
              state_epoch=1,host_submit_ns=((0,1),(1,2),(2,3),(3,4)))
    mutations=[{},dict(host_submit_ns=()),dict(completed_op_seqs=(),pending_op_seqs=()),
        dict(host_submit_ns=((4,4),)),dict(host_submit_ns=((0,-1),)),
        dict(host_submit_ns=((0,1),(0,2))),dict(host_submit_ns=((1,1),(0,2))),
        dict(host_submit_ns=((True,1),)),dict(host_submit_ns=((0,True),)),
        dict(host_submit_ns=((0,),)),dict(host_submit_ns=('bad',)),dict(host_submit_ns=[]),
        dict(pending_op_seqs=(3,2)),dict(pending_op_seqs=(1,2)),
        dict(pending_op_seqs=(2,'bad')),dict(transaction=object()),
        dict(worker_rank=True),dict(completed_sequence=-1),dict(state_epoch=0)]
    cases=[]
    def outcome(cls,kwargs):
        try:cls(**kwargs)
        except Exception as exc:return (type(exc).__name__,str(exc))
        return ('accepted',None)
    for delta in mutations:
        kw=base|delta
        prior,current=outcome(old,kw),outcome(wire.WorkerCompletionReport,kw)
        assert prior==current,(delta,prior,current)
        cases.append(dict(changed_fields=sorted(delta),result=prior[0],message=prior[1]))
    timings=[]
    for size in (128,1024,4096):
        kw=base|dict(completed_op_seqs=(),pending_op_seqs=tuple(range(size)),
                     host_submit_ns=tuple((i,i+1) for i in range(size)))
        row=dict(operations=size)
        for name,cls in [('baseline',old),('current',wire.WorkerCompletionReport)]:
            start=time.perf_counter_ns();cls(**kw);row[name+'_ns']=time.perf_counter_ns()-start
        timings.append(row)
    return dict(status='PASS',cases=cases,timings=timings,baseline_sha256=BASELINE_SHA,
                runtime_methods_patched=False,numerical_or_protocol_field_changes=False,
                scope='report acceptance and rejection messages, CPU construction time; not engine throughput')
