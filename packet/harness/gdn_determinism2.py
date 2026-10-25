#!/usr/bin/env python3
"""GDN prefill determinism micro-test v2 (q11x 25 Oct: conv cache row 1, since row 0 is the null block the
conv kernel skips -- the q8e run read an uninitialised conv output; original text follows). (q8d 24 Oct: the S5 slice's layer-0 GDN prefill output differs between
identical runs alone, op 4 of the bisect; layer-0 final state equal in q8c).

One process, one GPU at a time, no collectives. The slice's GDN prefill path (mm -> causal_conv1d_fn ->
fused_post_conv_prep -> chunk_gated_delta_rule -> gated RMSNorm -> mm) is run RUNS times on identical inputs with a
synchronize and a clone after every stage, including each FLA internal (cumsum, kkt, solve_tril, w/u, h, o).
Variants:
  plain   as the slice calls it
  poison  free cached memory is filled with 0xFF (NaN in bf16/fp32) before every run: an uninitialised read shows up
          as NaN or as a run-to-run difference
  noidx   poison, and chunk_indices/chunk_offsets left to FLA (None) instead of precomputed
Reports, per variant and stage, which runs differ from run 1, max |diff|, NaN count, and the first differing stage.
usage: gdn_determinism.py --devices 0,2 --out /q/gdn-det.json   (exit 0 when it ran, 2 when it could not)
"""
import argparse
import json
import sys
import traceback
from collections import OrderedDict

import torch

T, HK, HV, DK, DV = 1024, 8, 24, 128, 128


def build(device):
    torch.manual_seed(871)
    r = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16) * .01
    w = {"gdn_in": r(2560, 8192), "gdn_ba": r(2560, 48), "gdn_out": r(3072, 2560), "conv": r(5120, 4),
         "A": torch.zeros(HV, device=device), "dt": torch.zeros(HV, device=device),
         "norm": torch.ones(DK, device=device, dtype=torch.bfloat16)}
    x = torch.randn(T, 2560, device=device, dtype=torch.bfloat16)
    return w, x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"runs": args.runs, "devices": {}}
    rc = 0
    try:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
        import vllm.third_party.flash_linear_attention.ops.chunk as chunk
        from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
        from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
        rec = None

        def clone(o):
            if isinstance(o, torch.Tensor):
                return [o.detach().clone()]
            return [t.detach().clone() for t in (o or ()) if isinstance(t, torch.Tensor)]

        def put(name, o):
            torch.cuda.synchronize()
            rec[name] = clone(o)

        for name in ("chunk_local_cumsum", "chunk_scaled_dot_kkt_fwd", "solve_tril", "recompute_w_u_fwd",
                     "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o"):
            def wrap(fn, name=name):
                def f(*a, **kw):
                    out = fn(*a, **kw)
                    put("fla." + name, out)
                    return out
                return f
            setattr(chunk, name, wrap(getattr(chunk, name)))

        for dev in [int(d) for d in args.devices.split(",")]:
            torch.cuda.set_device(dev)
            w, x = build(dev)
            starts = torch.tensor([0, T], dtype=torch.int32)
            cu = starts.to(dev)
            idx, off = prepare_chunk_indices(starts, 64).to(dev), prepare_chunk_offsets(starts, 64).to(dev)
            ids = torch.ones(1, device=dev, dtype=torch.int32)  # row 1: row 0 is NULL_BLOCK_ID (cdc8375)
            initial = torch.ones(1, device=dev, dtype=torch.bool)
            dres = {"name": torch.cuda.get_device_name(dev)}
            for variant in ("plain", "poison", "noidx"):
                runs = []
                for _ in range(args.runs):
                    if variant != "plain":
                        junk = torch.empty(2 << 30, dtype=torch.uint8, device=dev).fill_(255)
                        del junk
                    state = torch.zeros(1, HV, DV, DK, device=dev, dtype=torch.float32)
                    conv = torch.zeros(2, 5120, 3, device=dev, dtype=torch.bfloat16)
                    rec = OrderedDict()
                    mixed = torch.mm(x, w["gdn_in"]); put("mm_in", mixed)
                    packed, z = mixed.split([5120, 3072], dim=-1)
                    packed = packed.contiguous()
                    ba = torch.mm(x, w["gdn_ba"])
                    b, a = (v.contiguous() for v in ba.split(HV, dim=-1))
                    z = z.reshape(T, HV, DV).contiguous()
                    cv = causal_conv1d_fn(packed.T, w["conv"], None, conv, cu, cache_indices=ids,
                                          has_initial_state=initial, activation="silu", validate_data=False).T.contiguous()
                    put("conv1d", cv); put("conv_state", conv[1:])
                    q, k, v, g, beta = fused_post_conv_prep(cv, a, b, w["A"], w["dt"], HK, DK, DV)
                    put("post_conv_prep", (q, k, v, g, beta))
                    i_, o_ = (idx, off) if variant != "noidx" else (None, None)
                    out, fin = chunk.chunk_gated_delta_rule(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                                            g.unsqueeze(0), beta.unsqueeze(0), initial_state=state,
                                                            output_final_state=True, cu_seqlens=cu,
                                                            use_qk_l2norm_in_kernel=False, chunk_indices=i_,
                                                            chunk_offsets=o_)
                    put("chunk_out", out); put("final_state", fin); put("h0_after", state)
                    out = out.reshape(T, HV, DV)
                    layer_norm_fwd(out.reshape(-1, DV), w["norm"], None, 1e-6, z=z.reshape(-1, DV),
                                   out=out.reshape(-1, DV), group_size=DV, norm_before_gate=True, is_rms_norm=True,
                                   activation="silu")
                    put("gated_norm", out)
                    put("mm_out", torch.mm(out.reshape(T, 3072), w["gdn_out"]))
                    runs.append(rec)
                stages = []
                for name, ref in runs[0].items():
                    bad, md, nans = [], 0.0, 0
                    for j, r in enumerate(runs[1:], 2):
                        got = r[name]
                        if not all(torch.equal(p, q_) for p, q_ in zip(ref, got)):
                            bad.append(j)
                            md = max(md, max(((p.float() - q_.float()).abs().nan_to_num(float("inf")).max().item()
                                              for p, q_ in zip(ref, got)), default=0.0))
                    for r in runs:
                        nans += sum(int(torch.isnan(t.float()).sum().item()) for t in r[name])
                    stages.append({"stage": name, "differs_in_runs": bad, "max_abs": md, "nan_total": nans,
                                   "ref_absmax": max(p.float().abs().max().item() for p in ref) if ref else None})
                first = next((s for s in stages if s["differs_in_runs"] or s["nan_total"]), None)
                dres[variant] = {"first": first, "stages": stages}
                print(json.dumps({"device": dev, "variant": variant, "first": first}), flush=True)
            report["devices"][str(dev)] = dres
    except BaseException as e:  # noqa: BLE001
        report.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        print(report["error"], flush=True)
        rc = 2
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    return rc


if __name__ == "__main__":
    sys.exit(main())
