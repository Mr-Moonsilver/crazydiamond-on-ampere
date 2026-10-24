#!/usr/bin/env python3
"""S5 feasibility v2 (24 Oct review): decode + prefill overlap on a real TP pair, with collectives and host traffic.

Two processes = one TP pair (launch with torch.distributed.run --nproc_per_node 2; CUDA_VISIBLE_DEVICES = the pair,
e.g. "1,3" in PCI order = the x8 GPU 02:00.0 + 42:00.0). Per "layer" of a step:
  decode  : 8-row bf16 GEMMs over its share of weights + UVA gather of DEC_HOST_KB from pinned host RAM (host-KV row
            reads) + all-reduce of the 8 x 4096 output (own NCCL communicator)
  prefill : 1024 x 4096 x 14336 bf16 GEMM + UVA gather of PRE_HOST_MB pinned history (staging) + all-reduce of the
            1024 x 4096 output (its own NCCL communicator; NCCL kernels spin while waiting, holding SMs)
Decode steps are released at the engine's cadence (PERIOD ms). Measured: host-visible decode token gaps (time between
completions of consecutive steps, as a client sees them) p50/p99/max; lateness = completion - nominal release
(includes CPU submission delay and queueing, Codex review P1) p50/p99/max and steps finishing more than one PERIOD
late; prefill layers completed inside the window where decode runs (common overlap window only). Each mode runs
REPEATS times in shuffled order after a warm-up pass; medians of the repeats are reported with the spread. Modes: alone (decode only), prefill alone, plain streams,
decode-high-priority streams. Green partitions are measured by greenctx_feasibility.py on single GPUs.
Correctness is enforced: the all-reduced results are checked against a reference and any mismatch aborts (exit 1).
UVA is required: no fallback copy mode. Results: one JSON per rank 0 in OUT.
v2b (24 Oct, first run): with 6 layers x 32 MiB the decode step took 0.6 ms alone, far below the engine's ~9 ms per
stage step (W1 424 tok/s = 8 streams x 53 tok/s, two PP cohorts), so the overlap result was not representative.
Defaults now model one pipeline stage: LAYERS=24, each reading its own DEC_W_MB of weights (24 x 140 MiB ~ 3.4 GB per
step, ~3.7 ms at HBM rate), every QSA_EVERY-th layer (6 of 24) also gathering DEC_HOST_KB from host (~36 MB per step,
~2.8 ms at the x8 link's 13 GB/s), plus 24 all-reduces. The decode-alone step time is reported (late_p50 of mode
decode) so the calibration is visible; prefill alone also reports TFLOP/s from a host-timed synchronized run as a
cross-check of the event counting (the first run implied ~97 TFLOP/s, above the 3090's 71 BF16 spec).
usage: python3 -m torch.distributed.run --nproc_per_node 2 overlap_tp.py OUT.json
       (env: PERIOD=9.5 LAYERS=24 QSA_EVERY=4 DEC_W_MB=140 DEC_HOST_KB=6144 PRE_HOST_MB=3 STEPS=400 REPEATS=3)
"""
import json, os, sys, time
import torch
import torch.distributed as dist

E = lambda k, d: type(d)(os.environ.get(k, d))
PERIOD, LAYERS, STEPS, PRE_MAXQ = E("PERIOD", 9.5), E("LAYERS", 24), E("STEPS", 400), E("PRE_MAXQ", 8)
DEC_W_MB, DEC_HOST_KB, PRE_HOST_MB = E("DEC_W_MB", 140), E("DEC_HOST_KB", 6144), E("PRE_HOST_MB", 3)
QSA_EVERY = E("QSA_EVERY", 4)
out_path = sys.argv[1]

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
dev = torch.cuda.current_device()
g_dec = dist.new_group([0, 1], backend="nccl")  # one communicator per stream: no cross-stream ordering hazards
g_pre = dist.new_group([0, 1], backend="nccl")
g_ctl = dist.new_group([0, 1], backend="gloo")  # host control path: agreed prefill watermark, never behind GPU work

try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
except Exception as e:
    print("UVA import failed -- required, aborting:", e)
    sys.exit(1)

# Green contexts (SM partitions, driver API via ctypes; same helper as greenctx_feasibility.py): GREEN_SMS="24,34,44"
# adds, per count c, mode gdec<c> (decode alone on c SMs) and green<c> (decode on c SMs, prefill on the rest). The
# NCCL collectives still run on torch's NCCL stream in the primary context, i.e. outside the partition.
import ctypes
GREEN = [int(x) for x in os.environ.get("GREEN_SMS", "").split(",") if x]
_cu, _RES = (ctypes.CDLL("libcuda.so.1") if GREEN else None), 144


def _ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


_green = {}


def green_streams(dec_sms):
    """(decode stream on dec_sms SMs, prefill stream on the rest, actual counts); one pair of contexts per count."""
    if dec_sms in _green:
        return _green[dec_sms]
    cudev = ctypes.c_int()
    _ck(_cu.cuDeviceGet(ctypes.byref(cudev), dev), "cuDeviceGet")
    whole, group, rest = ((ctypes.c_ubyte * _RES)() for _ in range(3))
    _ck(_cu.cuDeviceGetDevResource(cudev, whole, 1), "cuDeviceGetDevResource")
    n = ctypes.c_uint(1)
    _ck(_cu.cuDevSmResourceSplitByCount(group, ctypes.byref(n), whole, rest, 0, ctypes.c_uint(dec_sms)), "split")
    out = []
    for res in (group, rest):
        desc, g, st = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        _ck(_cu.cuDevResourceGenerateDesc(ctypes.byref(desc), res, 1), "desc")
        _ck(_cu.cuGreenCtxCreate(ctypes.byref(g), desc, cudev, 1), "cuGreenCtxCreate")
        _ck(_cu.cuGreenCtxStreamCreate(ctypes.byref(st), g, 1, 0), "cuGreenCtxStreamCreate")
        out.append(torch.cuda.ExternalStream(st.value, device=dev))
    smc = lambda b: int.from_bytes(bytes(b[96:100]), "little")
    gc, rc, total = smc(group), smc(rest), torch.cuda.get_device_properties(dev).multi_processor_count
    # Codex round 4: the count is a minimum, so validate what came back before any trial uses it
    if n.value != 1 or gc < dec_sms or gc <= 0 or rc <= 0 or gc + rc != total:
        raise RuntimeError(f"green split {dec_sms}: groups {n.value}, {gc} + {rc} SMs of {total}")
    print(f"rank {rank} green split {dec_sms}: {gc} + {rc} of {total} SMs", flush=True)
    _green[dec_sms] = (out[0], out[1], gc, rc)
    return _green[dec_sms]


K, N = 4096, 14336
torch.manual_seed(1234 + rank)
n_dec_w = LAYERS  # decode: one weight matrix per layer, DEC_W_MB each, so every step streams all of them from HBM
Nd = max(K, DEC_W_MB * 2**20 // (K * 2))
Wd = [torch.randn(K, Nd, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(n_dec_w)]
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
    y = torch.mm(xd, Wd[i % n_dec_w])[:, :K].contiguous()
    if i % QSA_EVERY == 0:
        torch.index_select(uva, 0, dec_idx, out=dec_g)
    dist.all_reduce(y, group=g_dec)
    return y


def prefill_layer():
    h = torch.mm(torch.mm(A, B), Bo)
    torch.index_select(uva, 0, pre_idx, out=pre_g)
    dist.all_reduce(h, group=g_pre)
    return h


# correctness: all-reduced results on side streams equal a reference computed on the default stream
ref_d = torch.mm(xd, Wd[0])[:, :K].contiguous(); dist.all_reduce(ref_d, group=g_dec)
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
    """mode: decode | prefill | plain | prio. Collective schedule is identical on both ranks (Codex round 2, P1): at
    every release i the host issues the decode step (if any) and then refills prefill to PRE_MAXQ outstanding layers
    against a watermark both ranks agree on (gloo MIN of completed counts), in that order, on both ranks. Completions are only observed
    (host timestamps, between issues) and every issued operation is drained before returning. Returns metrics."""
    import collections
    if mode == "prio":
        sd, sp = torch.cuda.Stream(priority=-1), torch.cuda.Stream(priority=0)
    elif mode.startswith(("green", "gdec")):
        sd, sp, _, _ = green_streams(int(mode.lstrip("gredcn")))
    else:
        sd, sp = torch.cuda.Stream(), torch.cuda.Stream()
    check_streams(sd, sp)
    dist.barrier(); torch.cuda.synchronize()
    with_dec, with_pre = mode != "prefill", mode != "decode" and not mode.startswith("gdec")
    pend_d, pend_p = collections.deque(), collections.deque()
    done_d, done_p, release, issued_p, maxq = [], [], [], 0, 0

    def poll():
        now = time.perf_counter()
        while pend_d and pend_d[0].query():
            pend_d.popleft(); done_d.append(now)
        while pend_p and pend_p[0].query():
            pend_p.popleft(); done_p.append(now)

    def refill(wm):
        nonlocal issued_p, maxq
        with torch.cuda.stream(sp):
            for _ in range(max(0, PRE_MAXQ - (issued_p - wm))):
                prefill_layer()
                e = torch.cuda.Event(); e.record(sp)
                pend_p.append(e)
                issued_p += 1
        maxq = max(maxq, issued_p - wm)

    t0 = time.perf_counter()
    for i in range(STEPS):
        while (time.perf_counter() - t0) * 1000 < i * PERIOD:
            poll()
        release.append(t0 + i * PERIOD / 1000)
        if with_dec:
            with torch.cuda.stream(sd):
                for l in range(LAYERS):
                    decode_layer(l)
                e = torch.cuda.Event(); e.record(sd)
            pend_d.append(e)
        if with_pre:
            # Codex round 3: refill to PRE_MAXQ outstanding layers against the slower rank's completed count (MIN
            # over the gloo group), so both ranks issue the same tickets and the backlog never exceeds PRE_MAXQ
            poll()
            wm = torch.tensor([len(done_p), 0], dtype=torch.int64)
            dist.all_reduce(wm, op=dist.ReduceOp.MIN, group=g_ctl)
            refill(int(wm[0]))
        poll()
    t1 = time.perf_counter()
    # Codex round 4: the prefill keeps its bounded backlog until BOTH ranks have completed the last decode, so the
    # decode tail runs under the same load as the rest of the window. Each round agrees (MIN) the watermark and an
    # all-done flag, so both ranks run the same rounds and issue the same tickets.
    rounds = 0
    while True:
        tw = time.perf_counter()
        while (time.perf_counter() - tw) * 1000 < PERIOD / 4:
            poll()
        if not with_pre:
            if not pend_d:
                break
            continue
        st = torch.tensor([len(done_p), 0 if pend_d else 1], dtype=torch.int64)
        dist.all_reduce(st, op=dist.ReduceOp.MIN, group=g_ctl)
        rounds += 1
        if int(st[1]):
            break
        refill(int(st[0]))
    t_end = done_d[-1] if done_d else t1
    td = time.perf_counter()
    torch.cuda.synchronize()  # drains the prefill backlog (<= PRE_MAXQ): same number of collectives on both ranks
    drain_s = time.perf_counter() - td
    poll()
    span = t_end - t0
    res = {"mode": mode, "rank": rank, "window_s": round(span, 3), "prefill_issued": issued_p,
           "prefill_max_outstanding": maxq, "drain_s": round(drain_s, 3), "tail_s": round(t_end - t1, 3),
           "tail_rounds": rounds, "last_prefill_done_s": round(done_p[-1] - t0, 3) if done_p else None}
    if done_d:
        gaps = sorted((b - a) * 1000 for a, b in zip(done_d, done_d[1:]))
        res.update({"gap_p50": round(gaps[len(gaps) // 2], 2), "gap_p99": round(gaps[int(len(gaps) * 0.99) - 1], 2),
                    "gap_max": round(gaps[-1], 2), "steps": len(done_d),
                    "decode_steps_per_s": round(len(done_d) / span, 1)})
        late = sorted((d - r) * 1000 for d, r in zip(done_d, release))
        res.update({"late_p50": round(late[len(late) // 2], 2), "late_p99": round(late[int(len(late) * 0.99) - 1], 2),
                    "late_max": round(late[-1], 2), "late_gt_period": sum(1 for x in late if x > PERIOD),
                    "late_gt_2period": sum(1 for x in late if x > 2 * PERIOD)})
    if with_pre:
        n = sum(1 for t in done_p if t0 <= t <= t_end)
        res["prefill_layers_in_window"] = n
        res["prefill_layers_per_s"] = round(n / span, 1)
    return res


import random


def prefill_wall(n=40):
    """Host-timed prefill layers (synchronized): TFLOP/s cross-check of the event-based counts."""
    dist.barrier(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        prefill_layer()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return round(n / dt, 1), round(n * 4 * 1024 * K * N / dt / 1e12, 1)


MODES = ("decode", "prefill", "plain", "prio") + tuple(f"{k}{c}" for c in GREEN for k in ("gdec", "green"))
REPEATS = E("REPEATS", 3)
for m in MODES:  # warm-up pass (allocator, NCCL, clocks), discarded
    run(m)
order = [m for m in MODES for _ in range(REPEATS)]
random.Random(42).shuffle(order)  # same order on both ranks: the collectives must match
raw = {m: [] for m in MODES}
for m in order:
    raw[m].append(run(m))


WORST = ("gap_p50", "gap_p99", "prefill_max_outstanding", "drain_s", "gap_max", "late_p50", "late_p99", "late_max", "late_gt_period", "late_gt_2period")


def worst(m, k):
    """per repeat, the worst rank's value (max for latency-like keys, min for rates)"""
    per = []
    for j in range(len(raw[m])):
        vals = [allraw[r][m][j].get(k) for r in range(world) if allraw[r][m][j].get(k) is not None]
        if vals:
            per.append(max(vals) if k in WORST else min(vals))
    return per


def med(m, k):
    per = sorted(worst(m, k))
    return per[len(per) // 2] if per else None


allraw = [None] * world
dist.all_gather_object(allraw, raw)  # both ranks' trials (Codex round 2, P2: rank 1 was discarded)
results = []
for m in MODES:
    keys = sorted({k for r in raw[m] for k in r if isinstance(r[k], (int, float))})
    r = {"mode": m, **{k: med(m, k) for k in keys}, "spread_worst_rank": {k: worst(m, k) for k in
                                                              ("gap_p99", "late_p99", "prefill_layers_per_s")}}
    results.append(r)
base, alone_rate = results[0], results[1].get("prefill_layers_per_s") or 0
for r in results[2:]:
    r["gap_p50_x"] = round(r["gap_p50"] / base["gap_p50"], 2)
    r["gap_p99_x"] = round(r["gap_p99"] / base["gap_p99"], 2)
    r["late_p99_x"] = round(r["late_p99"] / base["late_p99"], 2)
    r["decode_rate_share"] = round(r["decode_steps_per_s"] / base["decode_steps_per_s"], 3)
    r["prefill_share"] = (round(r["prefill_layers_per_s"] / alone_rate, 3)
                          if alone_rate and r.get("prefill_layers_per_s") is not None else None)
    if r["mode"].startswith(("green", "gdec")):
        r["sm_split"] = green_streams(int(r["mode"].lstrip("gredcn")))[2:]
wall_rate, wall_tflops = prefill_wall()
if rank == 0:
    meta = {"prefill_wall_rank0_layers_per_s": wall_rate, "prefill_wall_rank0_tflops": wall_tflops, "prefill_wall_note": "dense BF16 torch.mm, rank 0 only; these 3090s run gpu-unlock (README): stock 71 TFLOP/s is not the ceiling", "qsa_every": QSA_EVERY,"devices": [torch.cuda.get_device_name(dev)], "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "period_ms": PERIOD, "layers": LAYERS, "dec_w_mb": DEC_W_MB, "dec_host_kb": DEC_HOST_KB,
            "pre_host_mb": PRE_HOST_MB, "pre_maxq": PRE_MAXQ, "results_are": "median over repeats of the worst rank", "steps": STEPS, "repeats": REPEATS, "order": order, "uva": True,
            "priority_range": torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range")
            else None}
    json.dump({"meta": meta, "results": results, "raw_per_rank": allraw}, open(out_path, "w"), indent=1)
    for r in results:
        print(json.dumps(r))
dist.barrier()
dist.destroy_process_group()
