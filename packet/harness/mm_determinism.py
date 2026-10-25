#!/usr/bin/env python3
"""torch.mm bitwise determinism on the S5 slice's exact GEMM shapes (q11u mixed-slice noise localisation).

One GPU, one process, no collectives, engine stopped. For every torch.mm shape the slice issues
(s5_segment_gpu.py weights(), T=1024 tokens) identical bf16 inputs are multiplied CALLS times and every result is
compared bitwise with call 0: differing elements, max ordered-bit (ULP) distance, max |diff|. Modes:
  default       as the slice calls it
  --perturb     a second stream runs large GEMMs throughout, so cuBLAS kernels see SM/timing contention
  --deterministic  torch.use_deterministic_algorithms(True); the caller must set CUBLAS_WORKSPACE_CONFIG=:4096:8
                   before the process starts (cuBLAS reads it at init), so this mode is a separate invocation.
Reports counts/ULPs only, no payloads. Exit 0 = ran, 2 = could not run.
usage: mm_determinism.py --out FILE [--calls 64] [--perturb] [--deterministic]
"""
import argparse
import json
import sys
import time

SHAPES = {  # name: (M, K, N) for torch.mm(x[M,K], w[K,N])
    "gdn_in": (1024, 2560, 8192), "gdn_ba": (1024, 2560, 48), "gdn_out": (1024, 3072, 2560),
    "qsa_in": (1024, 2560, 6656), "qsa_out": (1024, 3072, 2560),
    "router": (1024, 2560, 512), "shared_up": (1024, 2560, 640), "shared_down": (1024, 320, 2560),
    "shared_gate": (1024, 2560, 1), "hc_down": (1024, 10240, 336), "hc_up": (1024, 320, 10240),
}


def ordered_bits(t):
    """bf16 -> int32 ordered representation (sign-magnitude to a monotone integer)."""
    import torch
    v = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(v < 0, -(v & 0x7FFF), v)


def compare(ref, out):
    import torch
    same = torch.equal(ref, out)
    if same:
        return dict(equal=True, differing_elements=0, max_ulp=0, max_abs=0.0)
    diff = ref != out
    ulp = (ordered_bits(ref) - ordered_bits(out)).abs()
    return dict(equal=False, differing_elements=int(diff.sum()), max_ulp=int(ulp.max()),
                max_abs=float((ref.float() - out.float()).abs().max()),
                differing_rows=int(diff.any(dim=1).sum()), differing_cols=int(diff.any(dim=0).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--calls", type=int, default=64)
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    import os
    import torch
    report = dict(mode="deterministic" if a.deterministic else ("perturb" if a.perturb else "default"),
                  calls=a.calls, torch=torch.__version__, cuda=torch.version.cuda,
                  cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  visible=os.environ.get("CUDA_VISIBLE_DEVICES"), shapes={}, started=time.time())
    try:
        if a.deterministic:
            torch.use_deterministic_algorithms(True)
        dev = torch.device("cuda", 0)
        report["device"] = torch.cuda.get_device_name(dev)
        report["allow_bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        report["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        side = torch.cuda.Stream() if a.perturb else None
        pa = None
        if side is not None:
            pa = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        torch.manual_seed(871)
        for name, (m, k, n) in SHAPES.items():
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * .01
            w = torch.randn(k, n, device=dev, dtype=torch.bfloat16) * .01
            ref = torch.mm(x, w).clone()
            torch.cuda.synchronize()
            rows = []
            for i in range(a.calls):
                if side is not None:
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            torch.mm(pa, pa)
                out = torch.mm(x, w)
                torch.cuda.synchronize()
                c = compare(ref, out)
                if not c["equal"]:
                    rows.append(dict(call=i, **c))
            report["shapes"][name] = dict(shape=[m, k, n], calls=a.calls, differing_calls=len(rows),
                                          worst=max(rows, key=lambda r: r["differing_elements"]) if rows else None,
                                          distinct_results=None)
            # how many distinct bit patterns appeared (binary noise shows as 2)
            seen = {ref.view(torch.int16).cpu().numpy().tobytes()}
            for i in range(min(a.calls, 16)):
                if side is not None:
                    with torch.cuda.stream(side):
                        torch.mm(pa, pa)
                seen.add(torch.mm(x, w).view(torch.int16).cpu().numpy().tobytes())
            report["shapes"][name]["distinct_results"] = len(seen)
        torch.cuda.synchronize()
        report["finished"] = time.time()
        report["any_nondeterministic"] = any(v["differing_calls"] for v in report["shapes"].values())
    except Exception as e:  # report the class only; no payloads or paths
        report["error"] = type(e).__name__
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"MM_DET mode={report['mode']} ERROR {type(e).__name__}", flush=True)
        return 2
    json.dump(report, open(a.out, "w"), indent=1)
    for name, v in report["shapes"].items():
        w = v["worst"] or {}
        print(f"MM_DET mode={report['mode']} shape={name} {v['shape']} differing_calls={v['differing_calls']}/{v['calls']} "
              f"distinct={v['distinct_results']} worst_elems={w.get('differing_elements', 0)} max_ulp={w.get('max_ulp', 0)} "
              f"rows={w.get('differing_rows', 0)} cols={w.get('differing_cols', 0)}", flush=True)
    print(f"MM_DET mode={report['mode']} any_nondeterministic={report['any_nondeterministic']} "
          f"bf16_reduced_reduction={report['allow_bf16_reduced_precision_reduction']} "
          f"workspace={report['cublas_workspace_config']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
