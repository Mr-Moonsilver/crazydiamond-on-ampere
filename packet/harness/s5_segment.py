#!/usr/bin/env python3
"""S5 production-kernel slice. --plan / --combine use only the Python standard library.

GPU execution is for the operator's stopped-engine queue, never the hub. One
measured trial/mode after warmup. Outputs are exclusive-create, never overwritten.
GO is a representative-slice feasibility result, not an engine release verdict.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

MODES = ("decode", "prefill", "serial", "prio", "green24", "green34", "green44")
PAIRS = ("0,2", "1,3")
EXIT = {"GO": 0, "NO-GO": 3, "INCONCLUSIVE": 2, "FAIL": 1}


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)


def plan(config, layers=4, decode_rows=4, partition=(24, 24)):
    c, q = config["text_config"], config["quantization_config"]
    expected = {"hidden_size": 2560, "hc_count": 4, "head_dim": 256,
                "num_attention_heads": 24, "num_key_value_heads": 2,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "num_experts": 512,
                "num_experts_per_tok": 10, "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640, "hc_lowrank": 320,
                "indexer_budget": 2048, "indexer_compress_ratio": 4}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("config differs from the reviewed Flash-Next shapes")
    if (q.get("bits"), q.get("group_size"), q.get("sym")) != (4, 128, True):
        raise ValueError("requires symmetric W4A16 group-128")
    if c.get("mamba_ssm_dtype") != "float32" or c.get("ple_layer_ids") != [2]:
        raise ValueError("requires FP32 recurrent state and PLE layer 2")
    mix = ["linear_attention"] * 3 + ["full_attention"]
    if c["layer_types"] != mix * 12:
        raise ValueError("requires the 3 GDN : 1 QSA layer order")
    if type(layers) is not int or layers not in (4, 8, 12):
        raise ValueError("slice layers must be 4, 8 or 12")
    if type(decode_rows) is not int or decode_rows not in (4, 8):
        raise ValueError("decode rows must be 4 (8 sessions / 2 PP cohorts) or 8")
    if tuple(partition) not in ((24, 24), (26, 22)):
        raise ValueError("reviewed PP partitions are 24,24 and 26,22")
    stages = [c["layer_types"][:partition[0]], c["layer_types"][partition[0]:]]
    stage_counts = {pair:{"gdn": stage.count("linear_attention"), "qsa":stage.count("full_attention")}
                    for pair,stage in zip(PAIRS, stages)}
    return {"schema": 2, "pp_partition":list(partition), "layers": layers, "mix": mix * (layers // 4),
            "decode_rows": decode_rows, "streams": 8, "pp_cohorts": 2,
            "chunk": 1024, "history_decode": 153537, "history_prefill": 200000,
            "page_size": 4048, "selection_width": 2051, "rowcache_slots": 10240,
            "hidden": 2560, "hc": 4, "local_q_heads": 12, "local_kv_heads": 1,
            "local_gdn_key_heads": 8, "local_gdn_value_heads": 24,
            "gdn_state": [decode_rows, 24, 128, 128], "gdn_state_dtype": "float32",
            "local_experts": 256, "global_experts": 512, "topk": 10,
            "marlin_w13": [256, 160, 2560], "marlin_w2": [256, 40, 5120],
            "marlin_scales13": [256, 20, 1280], "marlin_scales2": [256, 5, 2560],
            "quant_type": "uint4b8", "zero_points": None,
            "expert_weight_bytes": 256 * 2560 * 640 * 3 // 2,
            "parked_hc_bytes": 1024 * (2560 * 5 + 4) * 2,
            "ple_rows_per_token": 16, "ple_row_bytes": 320, "ple_host_mib": 256,
            "modes": list(MODES), "splits": [24, 34, 44], "pairs": list(PAIRS),
            "warmup": 1, "measured_repeats": 1, "vram_limit_gib": 18,
            "calibration": {"decode_ms": 9.0, "prefill_ms": 131.0, "tolerance": .25,
                            "reference": "x8 stage; x16 uses same conservative 25% band"},
            "stage_counts": stage_counts,
            "scope": "kernel slice, fixed synthetic input distribution, not a model forward",
            "approximations": ["synthetic QSA selections replace indexer computation",
                               "no QSA RoPE/indexer/KV insertion or PLE convolution/projections",
                               "same weight bank reused across slice layers (well above 6 MiB L2)",
                               "one PLE lookup per slice on both pairs per brief; stage 1 has no PLE in engine",
                               "BF16 dense projections use torch.mm; skinny-GEMM dispatch is not reproduced"],
            "required_runtime_evidence": ["cuda_graph_equal", "finite", "reduction_equal",
                                           "gdn_dispatch_evidence", "calibrated", "sm_graph_restriction"]}


def grade_pair(data):
    """Fail correctness before considering missing calibration. Never bless partial mode/rank sets."""
    reasons = []
    if data.get("correctness") is False or data.get("fatal_correctness"):
        return {"verdict": "FAIL", "reasons": ["numerical/reduction/graph correctness failed"]}
    if data.get("pair") not in PAIRS:
        reasons.append("unknown pair")
    ranks = data.get("ranks", [])
    if len(ranks) != 2 or {r.get("rank") for r in ranks} != {0, 1}:
        reasons.append("both distinct TP ranks required")
    for r in ranks:
        if r.get("cuda_visible_devices")!=data.get("pair") or not r.get("pci_bus_id"):
            reasons.append("rank device identity does not establish the declared pair")
        for key in ("cuda_graph_equal", "finite", "reduction_equal"):
            if r.get(key) is False:
                return {"verdict": "FAIL", "reasons": [f"rank {r.get('rank')} {key} failed"]}
            if r.get(key) is not True:
                reasons.append(f"rank {r.get('rank')} missing {key}")
        if r.get("gdn_dispatch_evidence") is not True:
            reasons.append("selected GDN implementation not bound to production dispatch evidence")
        if r.get("clock_units_validated") is not True:
            reasons.append("cross-context GPU clock scale not validated")
        clock = r.get("clock_check", {})
        if clock.get("schema") != 2 or clock.get("valid") is not True:
            reasons.append("missing multi-window clock/granularity evidence")
        peak = r.get("peak_vram_gib")
        if not finite(peak, True) or peak > 18:
            reasons.append("VRAM evidence absent or exceeds 18 GiB")
        cal = r.get("calibration", {})
        for lane, target in (("decode", 9.), ("prefill", 131.)):
            val = cal.get(lane + "_stage_ms")
            if not finite(val, True) or abs(val / target - 1) > .25:
                reasons.append(f"{lane} calibration {val} vs {target} ms (25% band)")
        modes = r.get("modes", {})
        if set(modes) != set(MODES):
            reasons.append("incomplete mode set")
        for name in MODES:
            m = modes.get(name, {})
            pcie = m.get("device_pcie",{})
            if (pcie.get("available") is not True or type(pcie.get("samples")) is not int
                    or pcie["samples"]<2 or not finite(pcie.get("mean_rx_GB_s")) or pcie["mean_rx_GB_s"]<0):
                reasons.append(f"{name}: missing device PCIe counter evidence")
            if m.get("duration_source") != "same-stream CUDA events":
                reasons.append(f"{name}: durations must use in-stream CUDA events")
            if m.get("prefill_tail_order_ambiguous") and name not in ("decode", "prefill", "serial"):
                reasons.append(f"{name}: coarse clock cannot establish prefill tail coverage")
            if name != "decode":
                lo, hi = m.get("prefill_steps_s"), m.get("prefill_steps_s_upper")
                if not finite(hi) or not finite(lo) or hi < lo:
                    reasons.append(f"{name}: missing/invalid cross-context boundary bounds")
            if m.get("correct") is False:
                return {"verdict": "FAIL", "reasons": [f"{name}: output mismatch"]}
            if m.get("correct") is not True:
                reasons.append(f"{name}: missing correctness")
            required = (["prefill_steps_s"] if name == "prefill" else
                        ["decode_steps_s", "gap_p99_ms", "gap_p50_ms", "gap_max_ms"])
            if name not in ("decode", "prefill"):
                required += ["prefill_steps_s"]
                if name != "serial" and m.get("prefill_covers_decode_tail") is not True:
                    reasons.append(f"{name}: prefill did not cover the complete decode window")
            if any(not finite(m.get(k), True) for k in required):
                reasons.append(f"{name}: missing/invalid timing or rate")
            if name.startswith("green"):
                if m.get("sm_graph_restriction") is not True:
                    reasons.append(f"{name}: graph SM restriction not established")
                split = m.get("sm_split", [])
                if (len(split) != 2 or any(type(n) is not int or n <= 0 for n in split)
                        or sum(split) != r.get("device_sms") or split[0] < int(name[5:])):
                    reasons.append(f"{name}: invalid SM split")
    if reasons:
        return {"verdict": "INCONCLUSIVE", "reasons": sorted(set(reasons))}
    candidates = []
    for mode in ("green24", "green34", "green44"):
        shares = [r["modes"][mode]["decode_steps_s"] / r["modes"]["decode"]["decode_steps_s"] for r in ranks]
        gaps = [r["modes"][mode]["gap_p99_ms"] / r["modes"]["decode"]["gap_p99_ms"] for r in ranks]
        pre = [r["modes"][mode]["prefill_steps_s"] / r["modes"]["prefill"]["prefill_steps_s_upper"] for r in ranks]
        pre_upper = [r["modes"][mode]["prefill_steps_s_upper"] / r["modes"]["prefill"]["prefill_steps_s"] for r in ranks]
        total = [(shares[i]+pre[i]) / (
            r["modes"]["serial"]["decode_steps_s"]/r["modes"]["decode"]["decode_steps_s"] +
            r["modes"]["serial"]["prefill_steps_s_upper"]/r["modes"]["prefill"]["prefill_steps_s"])
                 for i,r in enumerate(ranks)]
        total_upper = [(shares[i]+pre_upper[i]) / (
            r["modes"]["serial"]["decode_steps_s"]/r["modes"]["decode"]["decode_steps_s"] +
            r["modes"]["serial"]["prefill_steps_s"]/r["modes"]["prefill"]["prefill_steps_s_upper"])
                 for i,r in enumerate(ranks)]
        candidates.append({"mode": mode, "decode_share": min(shares), "p99_ratio": max(gaps),
                           "prefill_share": min(pre), "prefill_share_upper":min(pre_upper),
                           "normalized_work_vs_serial":min(total), "normalized_work_vs_serial_upper":min(total_upper),
                           "decode_gate": min(shares) >= .95 and max(gaps) <= 1.2})
    eligible = [m for m in candidates if m["decode_gate"] and m["normalized_work_vs_serial"]>=1]
    best = max(eligible, key=lambda m: m["prefill_share"]) if eligible else None
    verdict = "GO" if best and best["prefill_share"] >= .4 else "NO-GO"
    if verdict == "NO-GO" and any(m["decode_gate"] and m["prefill_share_upper"] >= .4
                                  and m["normalized_work_vs_serial_upper"] >= 1 for m in candidates):
        return {"verdict":"INCONCLUSIVE", "candidates":candidates,
                "reasons":["coarse timestamp boundary uncertainty crosses a service/work threshold"]}
    return {"verdict": verdict, "best": best, "candidates": candidates,
            "reasons": [] if verdict == "GO" else
            ["measured service/p99/prefill-share gate missed; consult per-class timings and requested host bytes",
             "resource causality cannot be inferred from a throughput ratio alone"]}


def combine(results):
    if any(r.get("fatal_correctness") or r.get("correctness") is False
           or any(x.get(k) is False for x in r.get("ranks",[]) for k in
                  ("cuda_graph_equal","finite","reduction_equal")) for r in results):
        return {"verdict":"FAIL","reasons":["recorded correctness failure takes precedence"]}
    if len(results) != 2 or {r.get("pair") for r in results} != set(PAIRS):
        return {"verdict": "INCONCLUSIVE", "reasons": ["exactly both pairs required"]}
    # Identical workload/settings across pairs; device identities and derived stage counts can differ.
    if results[0].get("run_spec") != results[1].get("run_spec") or not results[0].get("run_spec"):
        return {"verdict": "INCONCLUSIVE", "reasons": ["pair run specifications differ or are absent"]}
    devices = [r.get("pci_bus_id") for d in results for r in d.get("ranks",[])]
    if len(devices)!=4 or len(set(devices))!=4 or None in devices:
        return {"verdict":"INCONCLUSIVE","reasons":["four distinct physical GPUs must be established"]}
    grades = {r["pair"]: grade_pair(r) for r in results}
    vv = [r["verdict"] for r in grades.values()]
    v = next((x for x in ("FAIL", "INCONCLUSIVE", "NO-GO") if x in vv), "GO")
    return {"verdict": v, "pairs": grades}


def write_json(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def parse_modes(value):
    modes = value.split(",")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise argparse.ArgumentTypeError("unique comma-separated S5 modes required")
    return modes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--combine", nargs=2, metavar=("X16_JSON", "X8_JSON"))
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "reference/model-config.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pair", choices=PAIRS)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--decode-rows", type=int, default=4)
    ap.add_argument("--modes", type=parse_modes, default=list(MODES),
                    help="exact mode subset; subset results are diagnostic INCONCLUSIVE, never GO")
    ap.add_argument("--mode-timeout", type=float, default=120., help="seconds per mode phase, plus 5s hard grace")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pp-partition", choices=("24,24", "26,22"), default="24,24")
    ap.add_argument("--gdn-decode", choices=("cuda", "packed"), default="cuda")
    ap.add_argument("--dispatch-evidence", type=Path, help="JSON {kernel, source_sha256, evidence}; inspected production trace/dispatch record")
    args = ap.parse_args()
    if args.combine:
        found_errors = []
        try:
            for p in args.combine:
                path = Path(p)
                found_errors.extend(json.loads(e.read_text()) for e in path.parent.glob(path.name+".rank*.error.json"))
            if any(e.get("fatal_correctness") for e in found_errors):
                result = {"verdict":"FAIL","reasons":["per-rank artifact records a correctness failure"]}
                if args.out:write_json(args.out,result)
                print(json.dumps(result,indent=2))
                return 1
            records = []
            for p in args.combine:
                path = Path(p)
                records.append(json.loads(path.read_text()))
            result = ({"verdict":"INCONCLUSIVE","reasons":["per-rank failure artifacts present"]}
                      if found_errors else combine(records))
        except (OSError, ValueError, KeyError, TypeError) as e:
            result = {"verdict":"INCONCLUSIVE","reasons":["missing/invalid pair evidence: "+str(e)]}
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return EXIT[result["verdict"]]
    cfg = json.loads(args.config.read_text())
    spec = plan(cfg, args.layers, args.decode_rows, tuple(map(int,args.pp_partition.split(","))))
    if not math.isfinite(args.mode_timeout) or args.mode_timeout <= 0:
        ap.error("--mode-timeout must be finite and positive")
    spec["modes"] = args.modes
    spec["collectives"] = {"decode":"CustomAllreduce", "prefill":"PyNccl"}
    spec["mode_timeout_s"] = args.mode_timeout
    spec["model_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.plan:
        print(json.dumps(spec, indent=2))
        return 0
    if args.out is None or args.pair is None or args.steps < 30:
        ap.error("GPU run requires --out, --pair and --steps >= 30")
    from s5_segment_gpu import run
    return run(args, spec)


if __name__ == "__main__":
    sys.exit(main())
