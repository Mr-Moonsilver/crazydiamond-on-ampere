#!/usr/bin/env python3
"""S5 feasibility v2 (24 Oct review): decode + prefill overlap on a real TP pair, with collectives and host traffic.

Two processes = one TP pair (launch with torch.distributed.run --nproc_per_node 2; CUDA_VISIBLE_DEVICES = the pair,
e.g. "1,3" in PCI order = the x8 GPU 02:00.0 + 42:00.0). Per "layer" of a step:
  decode  : 8-row bf16 GEMMs over its share of weights + UVA gather of DEC_HOST_KB from pinned host RAM (host-KV row
            reads) + all-reduce of the 8 x 4096 output (own NCCL communicator)
  prefill : 1024 x 4096 x 14336 bf16 GEMM + UVA gather of PRE_HOST_MB pinned history (staging) + all-reduce of the
            1024 x 4096 output (its own NCCL communicator; NCCL kernels spin while waiting, holding SMs)
Decode steps are issued at the engine's cadence (PERIOD ms). Measured: host-visible decode token gaps (time between
completions of consecutive steps, as a client sees them) p50/p99/max, and prefill layers completed inside the
window where decode runs (common overlap window only). Modes: alone (decode only), prefill alone, plain streams,
decode-high-priority streams. Green partitions are measured by greenctx_feasibility.py on single GPUs.
Correctness is enforced: the all-reduced results are checked against a reference and any mismatch aborts (exit 1).
UVA is required: no fallback copy mode. Results: one JSON per rank 0 in OUT.
usage: python3 -m torch.distributed.run --nproc_per_node 2 overlap_tp.py OUT.json
       (env: PERIOD=9.5 LAYERS=6 DEC_W_MB=220 DEC_HOST_KB=512 PRE_HOST_MB=3 STEPS=400)
"""
import json, os, sys, time
import torch
import torch.distributed as dist

E = lambda k, d: type(d)(os.environ.get(k, d))
PERIOD, LAYERS, STEPS, PRE_QUEUE = E("PERIOD", 9.5), E("LAYERS", 6), E("STEPS", 400), E("PRE_QUEUE", 8)
DEC_W_MB, DEC_HOST_KB, PRE_HOST_MB = E("DEC_W_MB", 220), E("DEC_HOST_KB", 512), E("PRE_HOST_MB", 3)
out_path = sys.argv[1]

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
dev = torch.cuda.current_device()
g_dec = dist.new_group([0, 1], backend="nccl")  # one communicator per stream: no cross-stream ordering hazards
g_pre = dist.new_group([0, 1], backend="nccl")

try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
except Exception as e:
    print("UVA import failed -- required, aborting:", e)
    sys.exit(1)

K, N = 4096, 14336
torch.manual_seed(1234 + rank)
n_dec_w = max(1, DEC_W_MB * 2**20 // (K * K * 2))  # decode: K x K slices (attention/MoE projection sized)
Wd = [torch.randn(K, K, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(n_dec_w)]
xd = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
A = torch.randn(1024, K, device=dev, dtype=torch.bfloat16) * 0.01
B = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.01
Bo = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.01
host = torch.randn(512 * 2**20 // 2, dtype=torch.bfloat16).pin_memory().view(-1, 1024)
uva = get_accelerator_view_from_cpu_tensor(host)
dec_rows = DEC_HOST_KB * 1024 // 2048
pre_rows = PRE_HOST_MB * 2**20 // 2048
dec_idx = torch.randint(0, host.shape[0], (dec_rows,), device=dev)
pre_idx = torch.randint(0, host.shape[0], (pre_rows,), device=dev)
dec_g = torch.empty(dec_rows, 1024, device=dev, dtype=torch.bfloat16)
pre_g = torch.empty(pre_rows, 1024, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()  # everything above is ready before other streams read it


def decode_layer(i):
    y = torch.mm(xd, Wd[i % n_dec_w])
    torch.index_select(uva, 0, dec_idx, out=dec_g)
    dist.all_reduce(y, group=g_dec)
    return y


def prefill_layer():
    h = torch.mm(torch.mm(A, B), Bo)
    torch.index_select(uva, 0, pre_idx, out=pre_g)
    dist.all_reduce(h, group=g_pre)
    return h


# correctness: all-reduced results on side streams equal a reference computed on the default stream
ref_d = torch.mm(xd, Wd[0]); dist.all_reduce(ref_d, group=g_dec)
ref_p = torch.mm(torch.mm(A, B), Bo); dist.all_reduce(ref_p, group=g_pre)
torch.cuda.synchronize()


def check_streams(sd, sp):
    with torch.cuda.stream(sd):
        d = decode_layer(0)
    with torch.cuda.stream(sp):
        p = prefill_layer()
    torch.cuda.synchronize()
    ok = torch.equal(d, ref_d) and torch.equal(p, ref_p)
    flag = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(flag)
    if int(flag) != world:
        print(f"rank {rank}: CORRECTNESS FAIL on side streams -- aborting")
        sys.exit(1)


def run(mode):
    """mode: decode | prefill | plain | prio. One host loop issues decode steps at the cadence and keeps PRE_QUEUE
    prefill layers queued, polling completions (host timestamps) between issues. Returns rank-local metrics."""
    import collections
    if mode == "prio":
        sd, sp = torch.cuda.Stream(priority=-1), torch.cuda.Stream(priority=0)
    else:
        sd, sp = torch.cuda.Stream(), torch.cuda.Stream()
    check_streams(sd, sp)
    dist.barrier(); torch.cuda.synchronize()
    with_dec, with_pre = mode != "prefill", mode != "decode"
    pend_d, pend_p = collections.deque(), collections.deque()
    done_d, done_p = [], []

    def poll():
        now = time.perf_counter()
        while pend_d and pend_d[0].query():
            pend_d.popleft(); done_d.append(now)
        while pend_p and pend_p[0].query():
            pend_p.popleft(); done_p.append(now)
        while with_pre and len(pend_p) < PRE_QUEUE:
            with torch.cuda.stream(sp):
                prefill_layer()
                e = torch.cuda.Event(); e.record(sp)
            pend_p.append(e)

    t0 = time.perf_counter()
    for i in range(STEPS):
        while (time.perf_counter() - t0) * 1000 < i * PERIOD:
            poll()
        if with_dec:
            with torch.cuda.stream(sd):
                for l in range(LAYERS):
                    decode_layer(l)
                e = torch.cuda.Event(); e.record(sd)
            pend_d.append(e)
        poll()
    t1 = time.perf_counter()
    while pend_d:
        poll()
    t_end = done_d[-1] if done_d else t1
    torch.cuda.synchronize()
    span = t_end - t0
    res = {"mode": mode, "window_s": round(span, 3)}
    if done_d:
        gaps = sorted((b - a) * 1000 for a, b in zip(done_d, done_d[1:]))
        res.update({"gap_p50": round(gaps[len(gaps) // 2], 2), "gap_p99": round(gaps[int(len(gaps) * 0.99) - 1], 2),
                    "gap_max": round(gaps[-1], 2), "steps": len(done_d)})
    if with_pre:
        n = sum(1 for t in done_p if t0 <= t <= t_end)
        res["prefill_layers_in_window"] = n
        res["prefill_layers_per_s"] = round(n / span, 1)
    return res


results = [run(m) for m in ("decode", "prefill", "plain", "prio")]
base = results[0]
alone_rate = results[1].get("prefill_layers_per_s") or 0
for r in results[2:]:
    r["gap_p50_x"] = round(r["gap_p50"] / base["gap_p50"], 2)
    r["gap_p99_x"] = round(r["gap_p99"] / base["gap_p99"], 2)
    r["prefill_share"] = round(r["prefill_layers_per_s"] / alone_rate, 3) if alone_rate else None
if rank == 0:
    meta = {"devices": [torch.cuda.get_device_name(dev)], "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "period_ms": PERIOD, "layers": LAYERS, "dec_w_mb": DEC_W_MB, "dec_host_kb": DEC_HOST_KB,
            "pre_host_mb": PRE_HOST_MB, "steps": STEPS, "uva": True}
    json.dump({"meta": meta, "results": results}, open(out_path, "w"), indent=1)
    for r in results:
        print(json.dumps(r))
dist.barrier()
dist.destroy_process_group()
