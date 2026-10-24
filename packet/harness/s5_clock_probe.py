#!/usr/bin/env python3
"""Standalone S5 clock probe. GPU SLOT REQUIRED; this script never controls engines.

Example in the pinned image, only after flash-next grants the GPU slot:
  CUDA_VISIBLE_DEVICES=0 python3 /h/s5_clock_probe.py --out-dir /q/clock-new
One visible GPU by default; --devices 0,1,... probes sequentially, about 8 seconds
of validation/device plus compilation. No vLLM imports or inference kernels.
--plan is CPU-only. Outputs are exclusive-create; previous runs stay intact.
"""
import argparse
import json
from pathlib import Path
import time

from s5_clock import DeviceClock, Inconclusive, build_command, build_helper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--devices", default="0", help="visible CUDA ordinals, sequential")
    ap.add_argument("--seconds", type=float, default=8, help="minimum validation time/device, excludes build")
    ap.add_argument("--cudart", choices=("shared", "static"), default="shared", help="static is an explicit diagnostic A/B only")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    if args.plan:
        print(json.dumps({"command": build_command("s5_green.cu", "probe-green.so", (8, 6), args.cudart),
                          "windows_ms":[10,50,200], "devices":args.devices,
                          "duration_source":"same-stream CUDA events", "gpu_required":True}, indent=2))
        return 0
    if args.out_dir is None or not 0 <= args.seconds <= 60:
        ap.error("--out-dir and 0 <= --seconds <= 60 required")
    devices = [int(v) for v in args.devices.split(",")]
    if len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        ap.error("unique nonnegative visible ordinals required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    import torch  # --plan and unit tests never initialize CUDA
    worst = 0
    for device in devices:
        row = {"device":device, "validations":[]}
        try:
            torch.cuda.set_device(device)
            row.update(device_name=torch.cuda.get_device_name(device), torch_cuda=torch.version.cuda)
            library = args.out_dir / f"gpu-{device}-green.so"
            row["build"] = build_helper(library, torch.cuda.get_device_capability(device), args.cudart)
            clock = DeviceClock(library)
            row["pci_bus_id"] = clock.pci_bus_id(device)
            start = time.monotonic()
            while not row["validations"] or time.monotonic()-start < args.seconds:
                check = clock.validate()
                row["validations"].append(check)
                print(json.dumps({"device":device,"pci_bus_id":row["pci_bus_id"],
                      "median_ratio":check["median_ratio"], "granularity_ns":check["observed_granularity_ns"],
                      "window_medians":check["window_medians"], "coarse":check["coarse"]}), flush=True)
            row["elapsed_s"] = time.monotonic()-start
            row["verdict"] = "PASS"
        except Exception as e:
            row.update(verdict="INCONCLUSIVE", error=type(e).__name__+": "+str(e),
                       error_evidence=getattr(e,"evidence",{}))
            worst = 2
        with (args.out_dir / f"gpu-{device}.json").open("x") as f:
            json.dump(row, f, indent=2, allow_nan=False); f.write("\n")
        print(json.dumps({"device":device,"verdict":row["verdict"],"error":row.get("error")}), flush=True)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
