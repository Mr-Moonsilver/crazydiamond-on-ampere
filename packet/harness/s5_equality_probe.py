#!/usr/bin/env python3
"""S5 equality diagnostic (q8b 21:13: "concurrent prefill state differs from eager", decode bit-equal).

Question: is the prefill slice simply not bit-deterministic run to run (gate design issue), or does running it
next to decode change it (a real cross-lane hazard)? Same Slice/weights/Reducer as s5_segment_gpu, ordinary
streams only, no green contexts, no timing.

  A  prefill alone, twice on the same stream            -> run-to-run determinism
  B  prefill alone on a second stream                     -> stream dependence
  C  prefill next to eager decode (threads, two streams)  -> concurrency, eager
  D  prefill next to the captured decode graph            -> the failing q8b case
  plus: decode alone twice, and whether reset() restores the lane inputs (x, ple ids, selection indices).

Per tensor (last_output, GDN state per layer, conv state per layer): torch.equal, max |diff|, max relative diff.
Launch: torchrun --nproc_per_node 2 s5_equality_probe.py --config CFG --pair 0,2 --out /q/eq-0-2.json
GPU slot required; exit 0 when every comparison ran (whatever it found), 2 when the probe could not run.
"""
import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_segment import plan  # noqa: E402
import s5_segment_gpu as g  # noqa: E402



def names(lane):
    n = len(lane.state)
    return ["last_output"] + [f"gdn_state[{i}]" for i in range(n)] + [f"conv_state[{i}]" for i in range(n)]


def diff(ref, got, labels):
    rows = []
    for name, a, b in zip(labels, ref, got):
        if a.shape != b.shape:
            rows.append({"tensor": name, "equal": False, "error": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"})
            continue
        af, bf = a.float(), b.float()
        d = (af - bf).abs()
        scale = af.abs().max().item()
        rows.append({"tensor": name, "equal": bool(torch.equal(a, b)), "max_abs": d.max().item(),
                     "max_rel": d.max().item() / scale if scale else None, "ref_absmax": scale,
                     "differing_elements": int((d > 0).sum().item()), "elements": a.numel()})
    first = next((r["tensor"] for r in rows if not r["equal"]), None)
    return {"all_equal": first is None, "first_differing": first, "tensors": rows}


def eager(lane, stream):
    lane.reset()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        lane.forward(g.Trace())
    stream.synchronize()
    return g.snapshot(lane)


def inputs(lane):
    keep = {"x": lane.x, "ple_ids": lane.ple_ids, "indices": lane.indices, "counter": lane.counter}
    return {k: v.clone() for k, v in keep.items()}


def concurrent(dec, pre, sd, sp, decode_call):
    dec.reset(); pre.reset(); torch.cuda.synchronize()
    errors = []

    def p():
        try:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            with torch.cuda.stream(sp):
                pre.forward(g.Trace())
            sp.synchronize()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    worker = threading.Thread(target=p)
    worker.start()
    with torch.cuda.stream(sd):
        decode_call()
    sd.synchronize(); worker.join()
    if errors:
        raise errors[0]
    return g.snapshot(dec), g.snapshot(pre)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pair", choices=("0,2", "1,3"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--gdn-decode", choices=("cuda", "packed"), default="packed")
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    out = args.out.with_name(f"{args.out.stem}.rank{rank}.json")
    report = {"rank": rank, "pair": args.pair, "phase": "setup"}
    try:
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        dist.init_process_group("gloo")
        dg = dist.new_group([0, 1], backend="gloo"); pg = dist.new_group([0, 1], backend="gloo")
        refgroup = dist.new_group([0, 1], backend="nccl")
        spec = plan(json.loads(args.config.read_text()), args.layers, 4, (24, 24))
        torch.manual_seed(871)
        rd, rp = g.Reducer(dg, device), g.Reducer(pg, device)
        rd.check(refgroup); rp.check(refgroup)
        w = g.weights(spec, rank)
        dec = g.Slice(spec, rank, rd, False, w, args.gdn_decode)
        pre = g.Slice(spec, rank, rp, True, w, args.gdn_decode)
        L = names(pre)
        s1, s2, sd = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        report["device_name"] = torch.cuda.get_device_name(device)
        report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()

        report["phase"] = "A: prefill alone x2"
        before = inputs(pre)
        ref = eager(pre, s1)
        after = inputs(pre)
        report["reset_restores_prefill_inputs"] = {k: bool(torch.equal(before[k], after[k])) for k in before}
        report["A_prefill_alone_repeat"] = diff(ref, eager(pre, s1), L)
        report["A3_prefill_alone_third"] = diff(ref, eager(pre, s1), L)
        report["phase"] = "B: prefill alone, other stream"
        report["B_prefill_other_stream"] = diff(ref, eager(pre, s2), L)

        report["phase"] = "decode alone x2"
        dbefore = inputs(dec)
        dref = eager(dec, sd)
        report["reset_restores_decode_inputs"] = {k: bool(torch.equal(dbefore[k], v)) for k, v in inputs(dec).items()}
        report["decode_alone_repeat"] = diff(dref, eager(dec, sd), names(dec))

        report["phase"] = "C: prefill next to eager decode"
        dgot, pgot = concurrent(dec, pre, sd, s2, lambda: dec.forward(g.Trace()))
        report["C_decode_vs_alone"] = diff(dref, dgot, names(dec))
        report["C_prefill_vs_alone"] = diff(ref, pgot, L)

        report["phase"] = "D: prefill next to decode graph"
        graph, _, _ = g.make_graph(dec, sd)
        dgot, pgot = concurrent(dec, pre, sd, s2, graph.replay)
        report["D_decode_graph_vs_alone"] = diff(dref, dgot, names(dec))
        report["D_prefill_vs_alone"] = diff(ref, pgot, L)
        report["D_prefill_after_vs_alone"] = diff(ref, eager(pre, s2), L)
        report["phase"] = "done"

        a, c, d_ = (report[k]["all_equal"] for k in ("A_prefill_alone_repeat", "C_prefill_vs_alone",
                                                       "D_prefill_vs_alone"))
        report["reading"] = ("prefill not bit-deterministic even alone: q8b gate needs a tolerance or deterministic"
                             " kernels, not a hazard" if not a else
                             "prefill deterministic alone but differs next to decode: cross-lane hazard" if not (c and d_)
                             else "all equal here: q8b difference not reproduced; check graph/mode order")
        rc = 0
    except BaseException as e:  # noqa: BLE001
        report.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        rc = 2
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"rank": rank, "phase": report["phase"], "reading": report.get("reading"),
                      "error": report.get("error")}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
