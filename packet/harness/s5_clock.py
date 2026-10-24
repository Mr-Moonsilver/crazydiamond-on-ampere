"""S5 timer/build utilities; importing this module does not load torch or CUDA.

Durations always use same-stream CUDA events. Validated globaltimer ticks are
only an approximate same-device ordering clock, never a cross-GPU clock.
"""
import ctypes
import hashlib
import math
from pathlib import Path
import shutil
import statistics
import subprocess

WINDOWS_MS = (10, 50, 200)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class Inconclusive(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = json_safe(evidence or {})


def build_command(source, library, capability, cudart="shared"):
    if capability != (8, 6):
        raise Inconclusive(f"S5 helper requires reviewed GA102 sm_86, got {capability}")
    if cudart not in ("shared", "static"):
        raise ValueError("cudart must be shared or static")
    # A native cubin ONLY: do not silently JIT a foreign-target globaltimer.
    return ["nvcc", "-O2", "-std=c++17", "-shared", "-Xcompiler=-fPIC",
            "--generate-code=arch=compute_86,code=sm_86", "--cudart=" + cudart,
            "-lcuda", str(source), "-o", str(library)]


def build_helper(library, capability, cudart="shared"):
    library = Path(library)
    if library.exists():
        raise Inconclusive("refuse to replace existing green library: " + str(library))
    source = Path(__file__).with_name("s5_green.cu")
    command = build_command(source, library, capability, cudart)
    evidence = {"command": command, "cudart": cudart, "arch": "sm_86", "ptx_fallback": False,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    try:
        evidence["nvcc_version"] = subprocess.run(["nvcc", "--version"], capture_output=True,
                                                   text=True, timeout=10).stdout.strip()
        p = subprocess.run(command, capture_output=True, text=True, timeout=180)
        evidence.update(rc=p.returncode, log=(p.stdout + p.stderr)[-6000:])
        if p.returncode:
            raise Inconclusive("header-built green shim failed", evidence)
        evidence["sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        if shutil.which("readelf"):
            elf = subprocess.run(["readelf", "-d", str(library)], capture_output=True, text=True, timeout=10)
            evidence["elf_needed"] = [x.strip() for x in elf.stdout.splitlines() if "NEEDED" in x]
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Inconclusive("helper build failed: " + str(e), evidence) from e
    return evidence


def analyze_clock(samples, granularity):
    """Frozen offline rules, also applied by the standalone probe.

    Minimum observed positive increment is an empirical resolution bound, not a
    claim that every hardware tick was observed. Reject stopped/reversed clocks.
    Coarse endpoints get a 2-increment allowance, not a free pass on a wrong rate.
    """
    g = granularity.get("min_nonzero_ticks")
    evidence = {"schema": 2, "samples": samples, "granularity": granularity,
                "duration_source": "same-stream CUDA events",
                "ordering_source": "same-device globaltimer; ambiguous boundaries excluded"}
    problems = []
    if type(g) is not int or g <= 0 or granularity.get("backwards", 0) or not granularity.get("nonzero_deltas", 0):
        problems.append("globaltimer did not demonstrate monotonic nonzero resolution")
    valid = []
    for s in samples:
        ev, ticks = s.get("cuda_event_ms"), s.get("raw_ticks", [])
        if (type(ev) not in (float, int) or not math.isfinite(ev) or ev <= 0
                or not isinstance(ticks, (list, tuple)) or len(ticks) != 2 or any(type(t) is not int or t < 0 for t in ticks)
                or ticks[1] <= ticks[0]):
            problems.append("invalid event time or non-increasing raw ticks")
            continue
        ms = (ticks[1] - ticks[0]) / 1e6
        s.update(globaltimer_ms=ms, ratio=ms / ev)
        valid.append(s)
    ratios = [s["ratio"] for s in valid]
    median = statistics.median(ratios) if ratios else None
    evidence["median_ratio"] = median
    evidence["window_medians"] = {}
    gms = g / 1e6 if type(g) is int and g > 0 else 0
    for target in WINDOWS_MS:
        rows = [s for s in valid if s.get("target_ms") == target]
        if not rows:
            problems.append(f"missing {target} ms window")
            continue
        ratio = statistics.median(s["ratio"] for s in rows)
        ev = statistics.median(s["cuda_event_ms"] for s in rows)
        evidence["window_medians"][str(target)] = {"ratio": ratio, "cuda_event_ms": ev}
        if not .5 * target <= ev <= 2 * target:
            problems.append(f"{target} ms window was not exercised (observed {ev:g} ms)")
        # Require every window median and the overall median; one noisy sample
        # cannot reject a stable clock, nor can long windows hide a biased short one.
        if abs(ratio - 1) > .10 + 2 * gms / ev:
            problems.append(f"{target} ms globaltimer scale disagrees with CUDA events")
    if median is None or abs(median - 1) > .10:
        problems.append("median globaltimer scale disagrees with CUDA events")
    if gms > 1:
        problems.append("observed resolution exceeds 1 ms: ordering is too coarse for this slice")
    evidence.update(observed_granularity_ns=g, coarse=bool(gms >= .001),
                    ordering_uncertainty_ns=2 * g if type(g) is int and g > 0 else None,
                    valid=not problems, problems=problems)
    if problems:
        # Values are in both the human error and structured evidence; never hide
        # the measurements behind a bare 'scale disagrees' exception again.
        measurements = [{k: s.get(k) for k in ("target_ms", "globaltimer_ms", "cuda_event_ms", "raw_ticks")}
                        for s in samples]
        raise Inconclusive("; ".join(problems) + f"; median_ratio={median}; measurements={measurements}", evidence)
    return evidence


def sm_restriction(membership, split, observed_decode, observed_prefill, total):
    """Coverage belongs to the long occupancy probe; a short probe needs containment."""
    a, b = map(set, membership)
    d, p = set(observed_decode), set(observed_prefill)
    valid_ids = all(type(x) is int and x >= 0 for x in a | b | d | p)
    # SM IDs need not be a dense range [0, SM count).
    return (valid_ids and len(a) == split[0] and len(b) == split[1]
            and sum(split) == total and not a & b and bool(d) and bool(p)
            and d <= a and p <= b)


def ordering_window(drows, prows, uncertainty):
    """Definite/possible prefill inclusion in a decode window, using raw ticks."""
    primary = drows if drows else prows
    if not primary or type(uncertainty) is not int or uncertainty < 0:
        raise Inconclusive("missing window or invalid clock uncertainty")
    for row in drows + prows:
        ts = row.get("raw_ticks")
        if (not isinstance(ts, list) or len(ts) != 2 or any(type(t) is not int or t < 0 for t in ts)
                or ts[1] < ts[0]):
            raise Inconclusive("invalid/non-monotonic run timestamps", {"row":row})
    for rows in (drows, prows):
        if any(b["raw_ticks"][0] < a["raw_ticks"][1] for a,b in zip(rows,rows[1:])):
            raise Inconclusive("timestamp reversal in serial lane")
    first, last = primary[0]["raw_ticks"][0], primary[-1]["raw_ticks"][1]
    if not drows:
        inside = possible = prows
    else:
        inside = [r for r in prows if r["raw_ticks"][0] >= first+uncertainty
                  and r["raw_ticks"][1] <= last-uncertainty]
        possible = [r for r in prows if r["raw_ticks"][0] >= first-uncertainty
                    and r["raw_ticks"][1] <= last+uncertainty]
    delta = prows[-1]["raw_ticks"][1]-last if prows else None
    return first, last, inside, possible, delta


class DeviceClock:
    def __init__(self, library):
        import torch
        self.torch = torch
        self.lib = ctypes.CDLL(str(library))
        U, I = ctypes.c_uint64, ctypes.c_int
        self.lib.s5_stamp.argtypes = [U, U]
        self.lib.s5_stamp.restype = I
        self.lib.s5_samples.argtypes = [U, I, U, U]
        self.lib.s5_samples.restype = I
        self.lib.s5_pci_bus_id.argtypes = [I, ctypes.c_char_p, I]
        self.lib.s5_pci_bus_id.restype = I
        self.lib.s5_runtime.argtypes = [ctypes.POINTER(I)] * 3
        self.lib.s5_runtime.restype = I
        runtime, driver, device = I(), I(), I()
        self.check(self.lib.s5_runtime(ctypes.byref(runtime), ctypes.byref(driver), ctypes.byref(device)), "runtime")
        self.runtime = {"runtime_version": runtime.value, "driver_version": driver.value, "device": device.value,
                        "torch_cuda": torch.version.cuda}
        if device.value != torch.cuda.current_device():
            raise Inconclusive("helper runtime and torch disagree on current device", self.runtime)
        self.evidence = None
        self.cycles_per_ms = None

    def check(self, rc, operation):
        if rc:
            raise Inconclusive(f"{operation} CUDA status={rc}", {"operation": operation, "status": rc})

    def pci_bus_id(self, device):
        buf = ctypes.create_string_buffer(32)
        self.check(self.lib.s5_pci_bus_id(device, buf, 32), "PCI bus ID")
        return buf.value.decode()

    def stamp(self, tensor, stream):
        self.check(self.lib.s5_stamp(tensor.data_ptr(), stream.cuda_stream), "globaltimer stamp")

    def validate(self, stream=None, repeats=3):
        torch = self.torch
        stream = stream if stream is not None else torch.cuda.Stream()
        ticks = torch.empty(2, dtype=torch.int64, device="cuda")
        gran = torch.empty(9 + 128, dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        # Warm each kernel, event allocation and runtime module BEFORE timing or
        # capture. No JIT/lazy-loading host gap inside the measured window.
        with torch.cuda.stream(stream):
            self.stamp(ticks, stream)
            torch.cuda._sleep(100000)
            self.check(self.lib.s5_samples(gran.data_ptr(), 128, 100000, stream.cuda_stream), "granularity warmup")
        stream.synchronize()
        a, b = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        # Initialize events outside capture, then calibrate _sleep with one
        # pre-enqueued graph; even the pilot excludes Python launch gaps.
        with torch.cuda.stream(stream):
            a.record(); b.record()
        stream.synchronize()
        pilot_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pilot_graph, stream=stream):
            a.record(); torch.cuda._sleep(10000000); b.record()
        for _ in range(2):
            with torch.cuda.stream(stream):
                pilot_graph.replay()
            stream.synchronize()
        pilot = a.elapsed_time(b)
        pilot_graph.reset()
        if not math.isfinite(pilot) or pilot <= 0:
            raise Inconclusive("invalid sleep calibration", {"pilot_cuda_event_ms": pilot})
        self.cycles_per_ms = 10000000 / pilot
        if not 10_000 <= self.cycles_per_ms <= 10_000_000:
            raise Inconclusive("implausible sleep calibration; refusing unbounded spin",
                               {"pilot_cuda_event_ms":pilot,"cycles_per_ms":self.cycles_per_ms})
        samples = []
        for target in WINDOWS_MS:
            cycles = max(1, int(self.cycles_per_ms * target))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                a.record(); self.stamp(ticks, stream)
                torch.cuda._sleep(cycles)
                self.stamp(ticks[1:], stream); b.record()
            # Graph-enqueued work avoids Python dispatch gaps. One replay warms
            # graph upload; the following replays are the measured windows.
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            for _ in range(repeats):
                with torch.cuda.stream(stream):
                    ticks.fill_(-1)  # stale warm-up stamps must never validate a broken capture
                    graph.replay()
                stream.synchronize()
                raw = [int(t) for t in ticks.cpu().tolist()]
                samples.append({"target_ms": target, "sleep_cycles": cycles,
                                "cuda_event_ms": a.elapsed_time(b), "raw_ticks": raw})
            graph.reset()
        with torch.cuda.stream(stream):
            self.check(self.lib.s5_samples(gran.data_ptr(), 128, int(self.cycles_per_ms * 50), stream.cuda_stream),
                       "granularity sample")
        stream.synchronize()
        raw = [int(t) for t in gran.cpu().tolist()]
        resolution = dict(zip(("first_tick", "last_tick", "min_nonzero_ticks", "max_nonzero_ticks",
                               "zero_deltas", "nonzero_deltas", "backwards", "reads", "compiled_arch"), raw[:9]))
        resolution["raw_back_to_back_ticks"] = raw[9:]
        try:
            evidence = analyze_clock(samples, resolution)
        except Inconclusive as e:
            e.evidence.update(runtime=self.runtime, pilot_cuda_event_ms=pilot)
            raise
        if resolution["compiled_arch"] != 860:
            raise Inconclusive("helper executed a foreign-target cubin", evidence)
        evidence.update(runtime=self.runtime, pilot_cuda_event_ms=pilot, graph_replay_validated=True)
        self.evidence = evidence
        return evidence

    def check_order(self, streams):
        """Causally chained stamps across contexts, never event subtraction across them."""
        torch = self.torch
        ticks = torch.empty(len(streams), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        previous = None
        events = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                if previous is not None:
                    stream.wait_event(previous)
                torch.cuda._sleep(max(1, int(self.cycles_per_ms * 3)))
                self.stamp(ticks[i:], stream)
                previous = torch.cuda.Event()
                previous.record()
                events.append(previous)
        events[-1].synchronize()
        raw = [int(t) for t in ticks.cpu().tolist()]
        if any(b <= a for a, b in zip(raw, raw[1:])):
            raise Inconclusive("cross-context causal timestamp ordering failed", {"raw_ticks": raw})
        return {"valid": True, "raw_ticks": raw, "method": "event dependencies + 3 ms gaps; no cross-context elapsed_time"}
