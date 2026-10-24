#!/usr/bin/env python3
"""S5 feasibility (standalone, no engine): can decode keep its speed while prefill runs beside it on one GPU?

One GPU is split with CUDA green contexts (driver API via ctypes): `dec` SMs for a decode-shaped load, the rest for a
prefill-shaped load. Both run at the same time on their own streams.
  decode step  : 8-row bf16 GEMMs streaming W_DEC_MB of weights from HBM (memory-bound, like batch-8 MoE/attention
                 projections) + a gather of HOST_MB from pinned host RAM over PCIe (like the host-KV row reads)
  prefill op   : 1024 x 4096 x 14336 bf16 GEMM (compute-bound, like a 1,024-token chunk)
Reported per configuration: decode steps/s and prefill TFLOP/s, alone (full GPU), alone on its partition, and
concurrent; plus the default case of two ordinary streams without partitioning. The S5 question is answered by
"concurrent decode on its partition" vs "decode alone on the full GPU" and the prefill rate left over.
usage: greenctx_feasibility.py [DEC_SM_COUNTS e.g. 16,24,34,42]  (env: W_DEC_MB=900 HOST_MB=24 SECONDS=4)
"""
import ctypes, os, sys, time
import torch

W_DEC_MB = int(os.environ.get("W_DEC_MB", "900"))
HOST_MB = int(os.environ.get("HOST_MB", "24"))
SECONDS = float(os.environ.get("SECONDS", "4"))
dev = torch.cuda.current_device()
torch.cuda.init()
cu = ctypes.CDLL("libcuda.so.1")

RES = 144  # sizeof(CUdevResource) in the CUDA 12.x driver ABI: type(4) + padding(92) + union(48)
CU_DEV_RESOURCE_TYPE_SM = 1


def ck(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what}: {r} {s.value}")


def sm_count(buf, i=0):
    return int.from_bytes(bytes(buf[i * RES + 96:i * RES + 100]), "little")


ck(cu.cuInit(0), "cuInit")
cudev = ctypes.c_int()
ck(cu.cuDeviceGet(ctypes.byref(cudev), dev), "cuDeviceGet")
drv = ctypes.c_int()
cu.cuDriverGetVersion(ctypes.byref(drv))
whole = (ctypes.c_ubyte * RES)()
ck(cu.cuDeviceGetDevResource(cudev, whole, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource")
TOTAL_SM = sm_count(whole)
print(f"device {dev} {torch.cuda.get_device_name(dev)}, driver API {drv.value}, SMs {TOTAL_SM}", flush=True)


def green_streams(dec_sms):
    """Two streams: one on a green context with dec_sms SMs, one on a green context with the remaining SMs."""
    group = (ctypes.c_ubyte * RES)()
    rest = (ctypes.c_ubyte * RES)()
    n = ctypes.c_uint(1)
    ck(cu.cuDevSmResourceSplitByCount(group, ctypes.byref(n), whole, rest, 0, ctypes.c_uint(dec_sms)), "split")
    out = []
    for res in (group, rest):
        desc = ctypes.c_void_p()
        ck(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), res, 1), "desc")
        g = ctypes.c_void_p()
        ck(cu.cuGreenCtxCreate(ctypes.byref(g), desc, cudev, 1), "cuGreenCtxCreate")  # CU_GREEN_CTX_DEFAULT_STREAM
        s = ctypes.c_void_p()
        ck(cu.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), "cuGreenCtxStreamCreate")  # non-blocking
        out.append(torch.cuda.ExternalStream(s.value, device=dev))
    return out, sm_count(group), sm_count(rest)


# ---- workloads
K, N = 4096, 14336
n_w = max(1, W_DEC_MB * 2**20 // (K * N * 2))
W = [torch.randn(K, N, device=dev, dtype=torch.bfloat16) for _ in range(n_w)]
xd = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
host = torch.randn(256 * 2**20 // 2, dtype=torch.bfloat16).pin_memory().view(-1, 1024)  # 512 MiB pinned
host_rows = HOST_MB * 2**20 // (1024 * 2)
host_idx = torch.randint(0, host.shape[0], (host_rows,), device=dev)
host_uva = None
try:  # UVA view of pinned memory, gathered by a GPU kernel over PCIe (as the engine does)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host_uva = get_accelerator_view_from_cpu_tensor(host)
except Exception:
    pass
gather_out = torch.empty(host_rows, 1024, device=dev, dtype=torch.bfloat16)
A = torch.randn(1024, K, device=dev, dtype=torch.bfloat16)
B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
prefill_flop = 2 * 1024 * K * N


def decode_step():
    for w in W:
        torch.mm(xd, w)
    if host_uva is not None:
        torch.index_select(host_uva, 0, host_idx, out=gather_out)
    else:
        gather_out.copy_(host[:host_rows], non_blocking=True)


def prefill_op():
    torch.mm(A, B)


def run(dec_stream, pre_stream, do_dec, do_pre):
    """Enqueue for SECONDS of work on each stream (calibrated), return (decode steps/s, prefill TFLOP/s)."""
    res = {}
    plan = []
    if do_dec:
        plan.append(("dec", dec_stream, decode_step))
    if do_pre:
        plan.append(("pre", pre_stream, prefill_op))
    # calibrate: time 3 iterations alone on its stream
    iters = {}
    for name, s, fn in plan:
        with torch.cuda.stream(s):
            fn()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(3):
                fn()
            e1.record()
        e1.synchronize()
        iters[name] = max(3, int(SECONDS * 1000 / (e0.elapsed_time(e1) / 3)))
    torch.cuda.synchronize()
    ev = {}
    for name, s, fn in plan:
        with torch.cuda.stream(s):
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
        ev[name] = [e0, None]
    # interleave enqueue so both queues stay deep
    left = dict(iters)
    while any(left.values()):
        for name, s, fn in plan:
            if left[name]:
                with torch.cuda.stream(s):
                    for _ in range(min(4, left[name])):
                        fn()
                left[name] -= min(4, left[name])
    for name, s, fn in plan:
        with torch.cuda.stream(s):
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
        ev[name][1] = e1
    torch.cuda.synchronize()
    for name in ev:
        t = ev[name][0].elapsed_time(ev[name][1]) / 1000
        res[name] = iters[name] / t if name == "dec" else iters[name] * prefill_flop / t / 1e12
    return res.get("dec"), res.get("pre")


# correctness of launches on green-context streams (memory is shared with torch's primary context)
s_def_a, s_def_b = torch.cuda.Stream(), torch.cuda.Stream()
ref = torch.mm(A[:64], B)
(gs, gd, gr) = green_streams(TOTAL_SM // 2)
with torch.cuda.stream(gs[0]):
    got = torch.mm(A[:64], B)
torch.cuda.synchronize()
print("green-context launch correct:", torch.equal(ref, got), f"(split {gd}/{gr})", flush=True)

d_full, _ = run(s_def_a, s_def_b, True, False)
_, p_full = run(s_def_a, s_def_b, False, True)
d_both, p_both = run(s_def_a, s_def_b, True, True)
print(f"full GPU alone      : decode {d_full:8.1f} steps/s | prefill {p_full:6.1f} TFLOP/s", flush=True)
print(f"two plain streams   : decode {d_both:8.1f} steps/s ({d_both / d_full:5.1%}) | prefill {p_both:6.1f} "
      f"TFLOP/s ({p_both / p_full:5.1%})   <- no partitioning", flush=True)
counts = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "12,20,28,36,42").split(",")]
for c in counts:
    (sd, sp), nd, npre = green_streams(c)
    d_alone, _ = run(sd, sp, True, False)
    _, p_alone = run(sd, sp, False, True)
    d_c, p_c = run(sd, sp, True, True)
    print(f"split {nd:2d}/{npre:2d} SMs : decode alone {d_alone:8.1f} ({d_alone / d_full:5.1%}) concurrent {d_c:8.1f} "
          f"({d_c / d_full:5.1%}) | prefill alone {p_alone:6.1f} ({p_alone / p_full:5.1%}) concurrent {p_c:6.1f} "
          f"({p_c / p_full:5.1%})", flush=True)
print("host gather:", "UVA index_select" if host_uva is not None else "copy_ fallback", f"{HOST_MB} MiB/step; "
      f"weights {n_w} x {K}x{N} bf16 = {n_w * K * N * 2 / 2**20:.0f} MiB/step")
