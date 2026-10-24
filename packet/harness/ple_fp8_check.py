#!/usr/bin/env python3
"""FP8 PLE preflight on one GPU (seconds; run inside the engine image, engine stopped): can the production PLE
lookup kernel copy rows of a float8_e4m3fn pinned table, directly and as uint8 byte views (make_plefp8.py)?

Builds a small pinned host table (ROWS x 160, float8_e4m3fn from random BF16 / scale), a UVA view of it, and looks up
random ids (some outside the rank's vocabulary range -> zero rows) with _lookup_ple_embedding_from_pinned_kernel from
the image's ngram_embedding module (MODULE). Variant "fp8": fp8 pointers as the unpatched model would pass them;
variant "bytes": uint8 views. Each variant: compile/launch error (recorded, not fatal) or a byte-exact comparison with
torch indexing on the host table; dequantized rows are also compared with the BF16 source (error stats, informative).
Verdict: PASS if "bytes" is byte-exact (the patched overlay works); "fp8" is reported either way (whether the patch is
needed). FAIL if "bytes" errors or differs. Exit 0 PASS, 1 FAIL.
usage: ple_fp8_check.py OUT.json  (env ROWS=500000 N=16384 MODULE=vllm.models.qwen4_exp.nvidia.ngram_embedding)
"""
import importlib, json, os, sys, traceback
import torch

ROWS, N = int(os.environ.get("ROWS", 500000)), int(os.environ.get("N", 16384))
MODULE = os.environ.get("MODULE", "vllm.models.qwen4_exp.nvidia.ngram_embedding")
D, LO = 160, 1000  # row width (PLE head dim), first id this "rank" owns
out_path = sys.argv[1]
mod = importlib.import_module(MODULE)
kernel = mod._lookup_ple_embedding_from_pinned_kernel
import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

torch.manual_seed(0)
dev = torch.device("cuda", 0)
src = (torch.randn(ROWS, D) * 0.02).to(torch.bfloat16)
scale = src.float().abs().max() / 448.0
host = (src.float() / scale).to(torch.float8_e4m3fn).pin_memory()
uva = get_accelerator_view_from_cpu_tensor(host)
hi = LO + ROWS  # this "rank" owns ids [LO, hi)
ids = torch.randint(LO, hi, (N,), device=dev)
ids[: N // 16] = torch.randint(0, LO, (N // 16,), device=dev)  # below and above the range: must come back as zeros
ids[N // 16: N // 8] = torch.randint(hi, hi + LO, (N // 16,), device=dev)
local = (ids - LO).clamp(0, ROWS - 1).cpu()
inr = ((ids >= LO) & (ids < hi)).cpu()
want = host.view(torch.uint8)[local].clone()
want[~inr] = 0
res = {"config": {"ROWS": ROWS, "N": N, "D": D, "module": MODULE, "device": torch.cuda.get_device_name(0),
                  "cc": list(torch.cuda.get_device_capability(0)), "triton": triton.__version__}}
for name in ("fp8", "bytes"):
    out = torch.empty(N, D, dtype=torch.float8_e4m3fn, device=dev)
    w, o = (uva, out) if name == "fp8" else (uva.view(torch.uint8), out.view(torch.uint8))
    try:
        kernel[(N,)](w, ids, o, D, LO, hi, BLOCK_D=triton.next_power_of_2(D))
        torch.cuda.synchronize()
        got = out.view(torch.uint8).cpu()
        r = {"ok": True, "byte_exact": bool(torch.equal(got, want)), "differing_bytes": int((got != want).sum())}
        deq = (out.float() * scale.to(dev)).cpu()[inr]
        ref = src.float()[local[inr]]
        err = (deq - ref).abs()
        r.update(max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                 rel_rms=float(((deq - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()))
    except Exception as e:  # noqa: BLE001 - recorded: whether the unpatched kernel takes fp8 is a result
        r = {"ok": False, "error": repr(e)[:600], "trace": traceback.format_exc()[-1500:]}
    res[name] = r
b = res["bytes"]
res["final"] = "PASS" if b.get("ok") and b.get("byte_exact") else "FAIL"
res["patch_needed"] = not (res["fp8"].get("ok") and res["fp8"].get("byte_exact"))
json.dump(res, open(out_path, "w"), indent=1)
print("ple_fp8_check", json.dumps({k: res[k] for k in ("final", "patch_needed")}),
      "| fp8", {k: v for k, v in res["fp8"].items() if k != "trace"}, "| bytes", {k: v for k, v in b.items() if k != "trace"})
sys.exit(0 if res["final"] == "PASS" else 1)
