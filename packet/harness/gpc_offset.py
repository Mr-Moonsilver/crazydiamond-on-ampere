#!/usr/bin/env python3
"""Core-clock V/F curve offset (MHz) per GPU via the host NVML copy, runtime only (not persisted).
usage: gpc_offset.py                 -> show offset, SM clock, power, power-cap state
       gpc_offset.py --set O0 O1 O2 O3   (PCI order = vLLM index); every change is appended to offset-changes-20261022.log"""
import ctypes, importlib.util, sys, time
ROOT = "/root/gpu-tuning-20261021"
spec = importlib.util.spec_from_file_location("pynvml", f"{ROOT}/host-nvml/pynvml.py")
N = importlib.util.module_from_spec(spec); sys.modules["pynvml"] = N; spec.loader.exec_module(N)
N.nvmlLib = ctypes.CDLL(f"{ROOT}/host-nvml/libnvidia-ml.so.580.173.02"); N.nvmlInit()
hs = sorted((N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())), key=lambda h: N.nvmlDeviceGetPciInfo(h).busId)
def show():
    for i, h in enumerate(hs):
        capped = bool(N.nvmlDeviceGetCurrentClocksEventReasons(h) & N.nvmlClocksEventReasonSwPowerCap)
        print(f"gpu {i}: offset {N.nvmlDeviceGetGpcClkVfOffset(h):+d} MHz, SM {N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)} MHz, "
              f"{N.nvmlDeviceGetPowerUsage(h) / 1000:.0f} W, power-capped {capped}")
if len(sys.argv) == 6 and sys.argv[1] == "--set":
    want = [int(x) for x in sys.argv[2:]]
    assert all(-200 <= w <= 300 for w in want), "refusing offsets outside -200..+300 MHz"
    before = [N.nvmlDeviceGetGpcClkVfOffset(h) for h in hs]
    for h, w in zip(hs, want):
        N.nvmlDeviceSetGpcClkVfOffset(h, w)
    after = [N.nvmlDeviceGetGpcClkVfOffset(h) for h in hs]
    with open(f"{ROOT}/offset-changes-20261022.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} GPC clock VF offset {before} -> {after} MHz (runtime only, live undervolt test)\n")
    assert after == want, f"offsets not applied: {after}"
show()
N.nvmlShutdown()
