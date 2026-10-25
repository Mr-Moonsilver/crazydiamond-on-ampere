"""S5 model liveness ablation: no globaltimer, anchor or per-op graph events.

Same release cadence and prefill-tail control as run_mode; not a performance
result. Lane-local completion events and host issue logs are retained.
"""
import threading
import time

class NullTrace:
    def call(self,kind,fn):return fn()
    def times(self):return {}
    events=()


class HostTrace(NullTrace):
    def __init__(self,progress,step):self.progress=progress;self.step=step;self.op=0
    def call(self,kind,fn):
        self.op+=1
        self.progress.event('prefill','op-enter',step=self.step,op=self.op,name=kind)
        out=fn()
        self.progress.event('prefill','op-issued',step=self.step,op=self.op,name=kind)
        return out


def run(dec,pre,graph,sd,sp,ctl,progress,steps=200,mode='overlap',op_logs=True):
    import torch
    import torch.distributed as dist
    from s5_terminal_stats import summarize
    device=torch.cuda.current_device();dec.reset();pre.reset();torch.cuda.synchronize();dist.barrier(group=ctl)
    ready=threading.Event();done=threading.Event();errors=[];rows={'decode':[],'prefill':[]}
    events={'decode':[],'prefill':[]};current_pre=[];lock=threading.Lock();tail=[]
    def execute(lane,stream,fn,i,notify=None):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        submitted=time.perf_counter();progress.event(lane,'step-enter',step=i)
        with torch.cuda.stream(stream):a.record();fn();b.record()
        if lane=='prefill':
            with lock:current_pre[:]=[a,b]
        progress.event(lane,'step-issued',step=i)
        if notify:notify()
        while not b.query():time.sleep(.0001)
        observed=time.perf_counter()
        if lane=='decode' and i==steps-1 and mode=='overlap':
            with lock:pair=list(current_pre)
            # CPU queries only: no wait inserted into either CUDA stream.
            tail.append(bool(pair and pair[0].query() and not pair[1].query()))
        events[lane].append((a,b));rows[lane].append(dict(step=i,submitted_s=submitted,observed_s=observed,gpu_ms=a.elapsed_time(b)))
        progress.event(lane,'step-complete',step=i)
    def decoder():
        try:
            torch.cuda.set_device(device)
            if mode=='overlap':ready.wait()
            origin=time.perf_counter()
            for i in range(steps):
                release=origin+i*.0095*4/24
                while time.perf_counter()<release:time.sleep(.0001)
                if mode=='serial':execute('prefill',sd,lambda:pre.forward(NullTrace()),i)
                execute('decode',sd,graph.replay,i)
        except BaseException as e:errors.append(e)
        finally:done.set()
    def prefiller():
        try:
            torch.cuda.set_device(device)
            for i in range(10000):
                flag=torch.tensor([int(done.is_set())],dtype=torch.int32)
                dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=ctl)
                if int(flag[0]):break
                pre.reduce.step=i
                execute('prefill',sp,lambda:pre.forward(HostTrace(progress,i) if op_logs else NullTrace()),i,ready.set)
            else:raise RuntimeError('prefill ticket limit')
        except BaseException as e:errors.append(e);ready.set()
    if mode=='prefill':
        for i in range(max(30,steps//4)):execute('prefill',sp,lambda:pre.forward(NullTrace()),i)
    elif mode in ('decode','serial'):decoder()
    elif mode=='overlap':
        p=threading.Thread(target=prefiller,daemon=True);d=threading.Thread(target=decoder,daemon=True)
        p.start();d.start();d.join();p.join()
    else:raise ValueError('unknown terminal mode')
    if errors:raise errors[0]
    torch.cuda.synchronize()
    if mode!='prefill' and len(rows['decode'])!=steps:raise RuntimeError('missing decode progress')
    if mode!='decode' and not rows['prefill']:raise RuntimeError('missing prefill progress')
    durations={k:v[0][0].elapsed_time(v[-1][1])/1000 for k,v in events.items() if v}
    gaps=[a[1].elapsed_time(b[1]) for a,b in zip(events['decode'],events['decode'][1:])]
    return summarize(rows,durations,gaps,bool(tail and tail[0]),mode)
