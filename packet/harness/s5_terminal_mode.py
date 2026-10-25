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


def run(dec,pre,graph,sd,sp,ctl,progress,steps=30):
    import torch
    import torch.distributed as dist
    from s5_onepair_gpu import timed
    device=torch.cuda.current_device();dec.reset();pre.reset();torch.cuda.synchronize();dist.barrier(group=ctl)
    ready=threading.Event();done=threading.Event();errors=[];rows={'decode':[],'prefill':[]}
    def decoder():
        try:
            torch.cuda.set_device(device);ready.wait();origin=time.perf_counter()
            for i in range(steps):
                release=origin+i*.0095*4/24
                while time.perf_counter()<release:time.sleep(.0001)
                rows['decode'].append(timed(graph.replay,sd,progress,'decode',i))
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
                # Signal readiness after enqueue, before the terminal wait.
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                progress.event('prefill','step-enter',step=i)
                with torch.cuda.stream(sp):a.record();pre.forward(NullTrace());b.record()
                progress.event('prefill','step-issued',step=i);ready.set()
                while not b.query():time.sleep(.0001)
                rows['prefill'].append(a.elapsed_time(b));progress.event('prefill','step-complete',step=i)
            else:raise RuntimeError('prefill ticket limit')
        except BaseException as e:errors.append(e);ready.set()
    p=threading.Thread(target=prefiller,daemon=True);d=threading.Thread(target=decoder,daemon=True)
    p.start();d.start();d.join();p.join()
    if errors:raise errors[0]
    torch.cuda.synchronize()
    if len(rows['decode'])!=steps or not rows['prefill']:raise RuntimeError('missing lane progress')
    return rows
