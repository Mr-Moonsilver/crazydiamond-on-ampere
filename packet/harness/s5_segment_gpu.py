"""GPU-only implementation loaded by s5_segment.py; never imported by --plan/tests.

Real kernel calls, synthetic weights/selections. See the report for omitted model
work. An import/ABI/capture/calibration problem is INCONCLUSIVE, not a substitute
BF16 proxy. No service control or network client is implemented here.
"""
from collections import defaultdict
from contextlib import nullcontext
import ctypes
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback

import torch
import torch.distributed as dist

from s5_segment import EXIT, MODES, grade_pair, write_json


from s5_clock import DeviceClock, Inconclusive, build_helper, sm_restriction, ordering_window


class CorrectnessError(RuntimeError):
    pass


class PcieSampler:
    """NVML device counters, ~20 Hz; physical aggregate, no per-stream attribution."""
    def __init__(self, bus_id):
        self.rows, self.error = [], None
        self.stop = threading.Event()
        self.thread = None
        self.bus_id = bus_id

    def start(self):
        try:
            from vllm.third_party import pynvml as nv
            nv.nvmlInit()
            handle = nv.nvmlDeviceGetHandleByPciBusId(self.bus_id)
        except Exception as e:
            self.error = type(e).__name__+": "+str(e)
            return
        def sample():
            try:
                while not self.stop.is_set():
                    self.rows.append({"host_s":time.perf_counter(),
                                      "rx_KiB_s":nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_RX_BYTES),
                                      "tx_KiB_s":nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_TX_BYTES)})
                    self.stop.wait(.05)
            except Exception as e:
                self.error = type(e).__name__+": "+str(e)
            finally:
                nv.nvmlShutdown()
        self.thread = threading.Thread(target=sample,daemon=True);self.thread.start()

    def finish(self, window=None):
        self.stop.set()
        if self.thread:self.thread.join()
        rows = [r for r in self.rows if window is None or window[0]<=r["host_s"]<=window[1]]
        return {"available":self.error is None and len(rows)>=2,"error":self.error,
                "pci_bus_id":self.bus_id,"samples":len(rows),"raw":self.rows,"host_window":window,
                "mean_rx_GB_s":sum(r["rx_KiB_s"] for r in rows)/len(rows)*1024/1e9 if rows else None,
                "meaning":"NVML rolling device PCIe counters; includes other traffic; no per-lane attribution"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(xs, p):
    if not xs:
        raise Inconclusive("empty timing series")
    return sorted(xs)[max(0, math.ceil(p * len(xs)) - 1)]


def checked(condition, message):
    if not condition:
        raise CorrectnessError(message)


class Green:
    """Header-built ABI and SM-ID sampling; one instance for each validated split."""
    def __init__(self, library, sms, device):
        self.lib = ctypes.CDLL(str(library))
        U, I, P = ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
        self.lib.s5_create.argtypes = [I, ctypes.c_uint, ctypes.POINTER(P), ctypes.POINTER(U),
                                      ctypes.POINTER(U), ctypes.POINTER(I), ctypes.POINTER(I)]
        self.lib.s5_create.restype = I
        self.lib.s5_probe.argtypes = [U, I, U, U]
        self.lib.s5_probe.restype = I
        self.lib.s5_close.argtypes = [P]
        self.lib.s5_close.restype = I
        self.handle, ds, ps, dc, pc = P(), U(), U(), I(), I()
        rc = self.lib.s5_create(device, sms, ctypes.byref(self.handle), ctypes.byref(ds),
                                ctypes.byref(ps), ctypes.byref(dc), ctypes.byref(pc))
        if rc:
            raise Inconclusive(f"green context create CUresult={rc}")
        self.split = [dc.value, pc.value]
        self.sd = torch.cuda.ExternalStream(ds.value, device=device)
        self.sp = torch.cuda.ExternalStream(ps.value, device=device)
        self.nsm = torch.cuda.get_device_properties(device).multi_processor_count
        if sum(self.split) != self.nsm or dc.value < sms or min(self.split) <= 0:
            raise Inconclusive(f"invalid split {self.split}, total {self.nsm}")

    def probe(self, out, stream, cycles=0):
        rc = self.lib.s5_probe(out.data_ptr(), out.numel(), stream.cuda_stream, cycles)
        if rc:
            raise Inconclusive(f"SM probe CUDA status={rc}")

    def membership(self):
        sets = []
        self.membership_evidence = []
        for stream, wanted in zip((self.sd, self.sp), self.split):
            seen, rounds = set(), []
            for _ in range(3):
                out = torch.full((self.nsm * 32,), -1, dtype=torch.int32, device="cuda")
                torch.cuda.synchronize()  # initialization precedes the external stream
                with torch.cuda.stream(stream):
                    self.probe(out, stream, 100000)
                stream.synchronize()
                sample = set(out.cpu().tolist())
                if -1 in sample:
                    raise Inconclusive("SM membership probe left unwritten output")
                seen |= sample
                rounds.append(sorted(sample))
                if len(seen) >= wanted:
                    break
            sets.append(seen)
            self.membership_evidence.append(rounds)
        ok = (len(sets[0]) == self.split[0] and len(sets[1]) == self.split[1]
              and not sets[0] & sets[1] and len(sets[0] | sets[1]) == self.nsm)
        return sets, ok

    def close(self):
        rc = self.lib.s5_close(self.handle)
        if rc:
            raise Inconclusive(f"green cleanup CUresult={rc}")


class Reducer:
    """Separate per-lane custom-AR/communicator instances and scratch.

    PyNccl is an explicit reported fallback. Both launch on the caller's current
    stream; torch.distributed NCCL's private stream is used only by the oracle.
    """
    def __init__(self, group, device):
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        self.ca = CustomAllreduce(group, device=device)
        self.nc = PyNcclCommunicator(group, device=device)
        self.paths = set()

    def __call__(self, x):
        if not self.ca.disabled and self.ca.should_custom_ar(x):
            self.paths.add("CustomAllreduce")
            out = self.ca.custom_all_reduce(x)
        else:
            self.paths.add("PyNccl")
            out = self.nc.all_reduce(x,stream=torch.cuda.current_stream())
        if out is None:
            raise Inconclusive("no custom/PyNccl all-reduce available; no torch-NCCL fallback")
        return out

    def capture(self):
        return self.ca.capture() if not self.ca.disabled else nullcontext()

    def check(self, refgroup):
        for n in (4, 1024):
            inp = torch.full((n, 2560), float(dist.get_rank() + 1), device="cuda", dtype=torch.bfloat16)
            expected = inp.clone()
            dist.all_reduce(expected, group=refgroup)
            actual = self(inp)
            torch.cuda.synchronize()
            checked(torch.equal(actual, expected), "reduction differs from torch.distributed reference")


class Trace:
    def __init__(self):
        self.events = []

    def call(self, kind, fn):
        a, b = torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)
        a.record()
        out = fn()
        b.record()
        self.events.append((kind, a, b))
        return out

    def times(self):
        totals = defaultdict(float)
        for kind, a, b in self.events:
            totals[kind] += a.elapsed_time(b)
        return dict(totals)


class Slice:
    def __init__(self, spec, rank, reducer, prefill, weights, gdn_decode):
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        self.spec, self.rank, self.reduce, self.pre, self.w, self.gdn_decode = spec, rank, reducer, prefill, weights, gdn_decode
        self.t = 1024 if prefill else spec["decode_rows"]
        self.reqs = 1 if prefill else self.t
        self.dtype = torch.bfloat16
        self.x = torch.randn(self.t, 10240, device="cuda", dtype=self.dtype) * .02
        self.initial_x = self.x.clone()
        # Cache line 0 is vLLM's NULL_BLOCK_ID: the conv kernels skip it and leave the output unwritten (q8e 24 Oct:
        # prefill output was uninitialised memory). The engine never schedules line 0, so lines 1..reqs are used here.
        self.state = [torch.zeros(self.reqs + 1, 24, 128, 128, device="cuda", dtype=torch.float32)
                      for _ in range(spec["layers"])]
        self.conv = [torch.zeros(self.reqs + 1, 5120, 3, device="cuda", dtype=self.dtype)
                     for _ in range(spec["layers"])]
        self.state_ids = torch.arange(self.reqs, device="cuda", dtype=torch.int32)
        self.cache_ids = self.state_ids + 1
        self.cu = torch.tensor([0, self.t], device="cuda", dtype=torch.int32) if prefill else torch.arange(self.t+1, device="cuda", dtype=torch.int32)
        self.chunk_indices = self.chunk_offsets = None
        if prefill:
            from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
            starts = torch.tensor([0,self.t],dtype=torch.int32)
            self.chunk_indices = prepare_chunk_indices(starts,64).to("cuda")
            self.chunk_offsets = prepare_chunk_offsets(starts,64).to("cuda")
        self.accepted = torch.ones(self.reqs, dtype=torch.int32, device="cuda")
        self.initial = torch.ones(self.reqs, dtype=torch.bool, device="cuda")
        self.workspace = marlin_make_workspace_new(torch.device("cuda", torch.cuda.current_device()), 4)
        self.m13 = torch.empty(self.t * 10 * 2560, dtype=self.dtype, device="cuda")
        self.m2 = torch.empty(self.t * 10, 640, dtype=self.dtype, device="cuda")
        self.history = spec["history_prefill"] if prefill else spec["history_decode"]
        self.pages = math.ceil(self.history / 4048)
        self.host = torch.empty(self.pages * self.reqs, 1, 4048, 512, dtype=self.dtype, pin_memory=True)
        self.host.normal_(0, .03)
        self.kv = get_accelerator_view_from_cpu_tensor(self.host)
        self.table = torch.arange(self.pages * self.reqs, device="cuda", dtype=torch.int32).view(self.reqs, self.pages)
        self.token_to_req = torch.zeros(self.t, device="cuda", dtype=torch.int32) if prefill else self.state_ids
        # Synthetic selection workload: 60% history coverage prefill; rotating decode rows.
        # This replaces the indexer, not the production attention kernel.
        self.base_indices = torch.randint(0, int(self.history * (.6 if prefill else 1)),
                                         (self.t, 2051), device="cuda", dtype=torch.int32)
        self.indices = torch.empty(self.t, 2052, device="cuda", dtype=torch.int32)
        self.counter = torch.zeros((), dtype=torch.int32, device="cuda")
        self.rows = torch.arange(self.t, dtype=torch.int64, device="cuda")
        self.arena = torch.empty(23, 1, 4048, 512, dtype=self.dtype, device="cuda") if prefill else None
        self.rowcache = None
        self.rowcaches = {}
        if not prefill:
            self.rowcache = (torch.empty(self.reqs, 10240, 1, 512, dtype=self.dtype, device="cuda"),
                             torch.full((self.reqs, 10240), -1, dtype=torch.int32, device="cuda"),
                             torch.zeros(2, dtype=torch.int64, device="cuda"),
                             torch.empty(self.reqs, 10240, dtype=torch.int32, device="cuda"),
                             self.state_ids.clone(), torch.zeros(self.reqs, dtype=torch.int64, device="cuda"),
                             self.state_ids.clone(), torch.zeros(1, dtype=torch.int64, device="cuda"))
            for i in range(3, spec["layers"], 4):
                self.rowcaches[i] = tuple(t.clone() for t in self.rowcache)
        self.ple_host = torch.empty(256*2**20//320, 160, dtype=self.dtype, pin_memory=True).normal_(0, .02)
        self.ple_uva = get_accelerator_view_from_cpu_tensor(self.ple_host)
        self.ple_ids = torch.randint(self.ple_host.shape[0], (self.t * 16,), device="cuda", dtype=torch.int64)
        self.ple_base = self.ple_ids.clone()
        self.ple_output = torch.empty(self.t * 16, 160, dtype=self.dtype, device="cuda")
        self.last_output = None
        self.prefill_unique = int(torch.unique(self.base_indices).numel()) if prefill else 0
        # Sparse union path moves only this unique set, not every dense 200k row.
        self.host_bytes_prefill = self.prefill_unique * 1024 * (spec["layers"] // 4)

    def reset(self):
        for tensor in self.state + self.conv:
            tensor.zero_()
        self.counter.zero_()
        for rc in self.rowcaches.values():
            rc[1].fill_(-1)
            rc[2].zero_()

    def selection(self):
        # Update captured inputs on GPU; do not benchmark an eternally warm static row set.
        if self.pre:
            self.indices[:, :-1].copy_(self.base_indices)
        else:
            self.indices[:, :-1].copy_(self.base_indices)
            self.indices[:, -206:-1].copy_((self.base_indices[:, -205:] + self.counter * 17) % self.history)
        self.indices[:, -1].fill_(2051)
        self.ple_ids.copy_((self.ple_base + self.counter*104729) % self.ple_host.shape[0])
        self.counter.add_(1)

    def hc(self, hidden, prev, injection):
        from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm, grouped_gemma_rmsnorm, hc_silu, hc_gate_mix
        if prev is None:
            xn = grouped_gemma_rmsnorm(hidden, self.w["hc_norm"], 1e-6, 4)
        else:
            hidden, xn = hc_combine_norm(hidden, prev, injection, self.w["hc_norm"], 1e-6, 4)
        low = torch.mm(xn, self.w["hc_down"])
        injection = low[:, 320:324]
        gate = torch.mm(hc_silu(low[:, :320], 4), self.w["hc_up"])
        return hidden, hc_gate_mix(xn, gate, 4), injection

    def gdn(self, x, i):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep, chunk_gated_delta_rule
        from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
        from vllm import _custom_ops as ops
        mixed = torch.mm(x, self.w["gdn_in"])
        packed, z = mixed.split([5120, 3072], dim=-1)
        packed = packed.contiguous()
        ba = torch.mm(x, self.w["gdn_ba"])
        b, a = (v.contiguous() for v in ba.split(24, dim=-1))
        z = z.reshape(self.t, 24, 128).contiguous()
        if self.pre:
            cv = causal_conv1d_fn(packed.T, self.w["conv"], None, self.conv[i], self.cu,
                                  cache_indices=self.cache_ids, has_initial_state=self.initial,
                                  activation="silu", validate_data=False).T.contiguous()
            q,k,v,g,beta = fused_post_conv_prep(cv,a,b,self.w["A"],self.w["dt"],8,128,128)
            out, state = chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
                            g.unsqueeze(0),beta.unsqueeze(0),initial_state=self.state[i][1:],
                            output_final_state=True,cu_seqlens=self.cu,use_qk_l2norm_in_kernel=False,
                            chunk_indices=self.chunk_indices,chunk_offsets=self.chunk_offsets)
            self.state[i][1:].copy_(state)
            out = out.reshape(self.t,24,128)
        else:
            cv = causal_conv1d_update(packed,self.conv[i],self.w["conv"],None,"silu",
                                      conv_state_indices=self.cache_ids,validate_data=False)
            out = torch.empty(self.t,24,128,device="cuda",dtype=self.dtype)
            if self.gdn_decode == "cuda":
                ops.fused_gdn_decode_post_conv_mtp(cv,a,b,self.w["A"],self.w["dt"],
                       self.cache_ids[:,None],self.cu,self.accepted,self.state[i],z,self.w["gdn_norm"],
                       out=out,scale=128**-.5,norm_eps=1e-6,output_gate_activation="silu")
            else:
                fused_recurrent_gated_delta_rule_packed_decode(cv,a,b,self.w["A"],self.w["dt"],
                       128**-.5,self.state[i],out.unsqueeze(1),self.cache_ids,True)
        if self.pre or self.gdn_decode == "packed":
            layer_norm_fwd(out.reshape(-1,128),self.w["gdn_norm"],None,1e-6,
                           z=z.reshape(-1,128),out=out.reshape(-1,128),group_size=128,
                           norm_before_gate=True,is_rms_norm=True,activation="silu")
        return torch.mm(out.reshape(self.t,3072),self.w["gdn_out"])

    def qsa(self, x, i):
        from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention, qsa_sparse_paged_attention_staged
        projected = torch.mm(x, self.w["qsa_in"])
        q, gate = projected[:, :3072], projected[:, 3584:]
        q = q.reshape(self.t,12,256).contiguous()
        out = torch.empty_like(q)
        if self.pre:
            qsa_sparse_paged_attention_staged(q,self.kv,self.indices,self.table[0],self.rows,
                                             self.pages-1,self.arena,True,out)
        else:
            k,v = self.kv.transpose(1,2).split(256,dim=-1)
            qsa_sparse_paged_attention(q,k,v,self.indices,self.table,self.token_to_req,False,
                                      out=out,rowcache=self.rowcaches[i])
        return torch.mm(out.flatten(1)*torch.sigmoid(gate),self.w["qsa_out"])

    def moe(self, x):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        from vllm.scalar_type import scalar_types
        logits = torch.mm(x,self.w["router"])
        topw, topi, _ = fused_topk(x,logits,10,True)
        y = fused_marlin_moe(x,self.w["w13"],self.w["w2"],None,None,
                             self.w["s13"],self.w["s2"],topw,topi,scalar_types.uint4b8.id,
                             global_num_experts=512,expert_map=self.w["emap"],workspace=self.workspace,
                             intermediate_cache13=self.m13,intermediate_cache2=self.m2)
        gu = torch.mm(x,self.w["shared_up"])
        a,b = gu.chunk(2,dim=-1)
        shared = torch.mm(torch.nn.functional.silu(a)*b,self.w["shared_down"])
        return y + shared * torch.sigmoid(torch.mm(x,self.w["shared_gate"]))

    def ple(self):
        from vllm.models.qwen4_exp.nvidia.ngram_embedding import _lookup_ple_embedding_from_pinned_kernel
        _lookup_ple_embedding_from_pinned_kernel[(self.ple_ids.numel(),)](
            self.ple_uva,self.ple_ids,self.ple_output,160,0,self.ple_host.shape[0],BLOCK_D=256)
        return self.ple_output

    def forward(self, trace):
        from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine
        trace.call("selection", self.selection)
        trace.call("ple", self.ple)
        trace.call("ple_allreduce",lambda:self.reduce(self.ple_output.reshape(self.t,2560)))
        hidden, block, inj = self.x, None, None
        for i, kind in enumerate(self.spec["mix"]):
            hidden, x, inj = trace.call("hc", lambda: self.hc(hidden,block,inj))
            x = trace.call("gdn" if kind == "linear_attention" else "qsa",
                           lambda: self.gdn(x,i) if kind == "linear_attention" else self.qsa(x,i))
            x = trace.call("allreduce", lambda: self.reduce(x))
            hidden, x, inj = trace.call("hc", lambda: self.hc(hidden,x,inj))
            block = trace.call("moe", lambda: self.moe(x))
            block = trace.call("allreduce", lambda: self.reduce(block))
        self.last_output = trace.call("hc_final", lambda: hc_combine(hidden,block,inj,4))
        return self.last_output


def weights(spec, rank):
    def rand(*shape):
        return torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*.01
    # Already-Marlin-layout random nibbles are valid uint4b8 values; constant
    # group scales make permutation immaterial. No g_idx / zero-point array.
    w = {"w13": torch.randint(-(2**31),2**31-1,spec["marlin_w13"],device="cuda",dtype=torch.int32),
         "w2": torch.randint(-(2**31),2**31-1,spec["marlin_w2"],device="cuda",dtype=torch.int32),
         "s13": torch.full(spec["marlin_scales13"],.01,device="cuda",dtype=torch.bfloat16),
         "s2": torch.full(spec["marlin_scales2"],.01,device="cuda",dtype=torch.bfloat16),
         "emap": torch.full((512,),-1,device="cuda",dtype=torch.int32)}
    w["emap"][rank*256:(rank+1)*256] = torch.arange(256,device="cuda",dtype=torch.int32)
    w.update(hc_norm=torch.zeros(10240,device="cuda",dtype=torch.bfloat16),
             hc_down=rand(10240,336),hc_up=rand(320,10240),
             router=rand(2560,512),shared_up=rand(2560,640),shared_down=rand(320,2560),shared_gate=rand(2560,1),
             gdn_in=rand(2560,8192),gdn_ba=rand(2560,48),gdn_out=rand(3072,2560),
             conv=rand(5120,4),A=torch.zeros(24,device="cuda",dtype=torch.float32),
             dt=torch.zeros(24,device="cuda",dtype=torch.float32),
             gdn_norm=torch.ones(128,device="cuda",dtype=torch.bfloat16),
             qsa_in=rand(2560,6656),qsa_out=rand(3072,2560))
    return w


def stage_time(parts, spec, pair):
    """Counts-weighted extrapolation. PLE is once, not multiplied by stage/slice."""
    ng, nq = spec["stage_counts"][pair]["gdn"], spec["stage_counts"][pair]["qsa"]
    layers = spec["layers"]
    return (parts.get("gdn",0)*ng/(layers*3/4) + parts.get("qsa",0)*nq/(layers/4)
            + sum(parts.get(k,0) for k in ("hc","moe","allreduce"))*(ng+nq)/layers
            + parts.get("ple",0) + parts.get("ple_allreduce",0) + parts.get("selection",0)
            + parts.get("hc_final",0))


def state_equal(a, b):
    return len(a) == len(b) and all(torch.equal(x,y) for x,y in zip(a,b))


def snapshot(lane):
    return [lane.last_output.clone()] + [t.clone() for t in lane.state+lane.conv]


class DecodeGraph:
    def __init__(self, graph, lane, output):
        self.graph, self.lane, self.output = graph, lane, output

    def replay(self):
        self.graph.replay()
        self.lane.last_output = self.output

    def reset(self):
        self.graph.reset()


def make_graph(dec, sd, green=None):
    # Warm the exact eager ops and SM probe before capture. Observe allocation
    # readiness explicitly when the stream belongs to a different green context.
    torch.cuda.synchronize()
    smout = torch.full((green.nsm*32,), -1, dtype=torch.int32, device="cuda") if green else None
    torch.cuda.synchronize()
    with torch.cuda.stream(sd):
        dec.reset()
        dec.forward(Trace())
        if green:
            green.probe(smout, sd)
    sd.synchronize()
    dec.reset(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    trace = Trace()
    with dec.reduce.capture():
        with torch.cuda.graph(graph, stream=sd):
            dec.forward(trace)
            if green:
                green.probe(smout,sd)
    torch.cuda.synchronize()
    if smout is not None:
        smout.fill_(-1)  # the first replay must prove the probe was captured
        torch.cuda.synchronize()
    return DecodeGraph(graph,dec,dec.last_output), trace, smout


def verify_mode(dec, pre, graph, sd, sp):
    """One eager reference and concurrent replay from identical states/inputs.

    Compare actual output AND persistent GDN/conv state, not just a status flag.
    No bitwise requirement across different batch shapes: these shapes are fixed.
    """
    dec.reset(); pre.reset(); torch.cuda.synchronize()
    with torch.cuda.stream(sd):
        dec.forward(Trace())
    sd.synchronize()
    refd = snapshot(dec)
    with torch.cuda.stream(sp):
        pre.forward(Trace())
    sp.synchronize()
    refp = snapshot(pre)
    dec.reset(); pre.reset(); torch.cuda.synchronize()
    errors = []
    def p():
        try:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            with torch.cuda.stream(sp):
                pre.forward(Trace())
            sp.synchronize()
        except BaseException as e:
            errors.append(e)
    worker = threading.Thread(target=p)
    worker.start()
    with torch.cuda.stream(sd):
        graph.replay()
    sd.synchronize(); worker.join()
    if errors:
        raise errors[0]
    checked(state_equal(snapshot(dec),refd), "decode graph/concurrent state differs from eager")
    checked(state_equal(snapshot(pre),refp), "concurrent prefill state differs from eager")
    checked(all(bool(torch.isfinite(x).all()) for x in snapshot(dec)+snapshot(pre)), "nonfinite warm outputs/state")


def run_mode(name, dec, pre, graph, dtrace, sd, sp, ctl, steps, period, clock, green=None):
    dec.reset(); pre.reset(); torch.cuda.synchronize(); dist.barrier(group=ctl)
    ready = threading.Event()
    ddone = threading.Event()
    drows, prows, errors = [], [], []
    device = torch.cuda.current_device()
    anchor = torch.cuda.Event(enable_timing=True)
    clock_origin = torch.empty(1,dtype=torch.int64,device="cuda")
    clock.stamp(clock_origin,torch.cuda.current_stream())
    anchor.record(); anchor.synchronize()
    clock_zero = int(clock_origin.item())
    origin = time.perf_counter()
    sp.wait_event(anchor); sd.wait_event(anchor)
    sm_pre = torch.full((green.nsm*32,), -1, dtype=torch.int32, device="cuda") if green else None
    torch.cuda.synchronize()
    event_rows = {"decode": [], "prefill": []}

    def execute(lane, stream, is_decode, release, issued=None):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        trace = dtrace if is_decode else Trace()
        stamps = torch.empty(2,dtype=torch.int64,device="cuda")
        submitted = time.perf_counter()-origin
        with torch.cuda.stream(stream):
            clock.stamp(stamps,stream)
            start.record()
            if is_decode:
                graph.replay()
            else:
                lane.forward(trace)
                if green:
                    green.probe(sm_pre,stream)
            clock.stamp(stamps[1:],stream)
            end.record()
        if issued:
            issued()
        while not end.query():
            time.sleep(.0001)
        observed = time.perf_counter()-origin
        ticks = stamps.cpu().tolist()
        event_rows["decode" if is_decode else "prefill"].append((start, end))
        return {"raw_ticks": ticks, "cuda_event_ms": start.elapsed_time(end), "release_s": release, "submitted_s": submitted, "observed_s": observed,
                "gpu_start_s": (ticks[0]-clock_zero)/1e9,
                "gpu_end_s": (ticks[1]-clock_zero)/1e9, "parts_ms": trace.times()}

    def decoder():
        try:
            torch.cuda.set_device(device)
            if name not in ("decode","serial"):
                ready.wait()
            release_origin = time.perf_counter()-origin
            for i in range(steps):
                release = release_origin+i*period
                while time.perf_counter()-origin < release:
                    time.sleep(.0001)
                if name == "serial":
                    prows.append(execute(pre,sd,False,release))
                drows.append(execute(dec,sd,True,release))
        except BaseException as e:
            errors.append(e)
        finally:
            ddone.set()

    def prefiller():
        try:
            torch.cuda.set_device(device)
            for _ in range(10000):
                # Both ranks issue the same prefill tickets and drain the decode
                # tail before stopping. Separate control and data communicators.
                done = torch.tensor([int(ddone.is_set())],dtype=torch.int32)
                dist.all_reduce(done,op=dist.ReduceOp.MIN,group=ctl)
                if int(done[0]):
                    break
                prows.append(execute(pre,sp,False,0.,issued=ready.set))
            else:
                raise Inconclusive("prefill ticket limit exceeded")
        except BaseException as e:
            errors.append(e); ready.set()

    if name == "prefill":
        for _ in range(max(30,steps//4)):
            prows.append(execute(pre,sp,False,0.))
    elif name in ("decode","serial"):
        decoder()
    else:
        p = threading.Thread(target=prefiller); d = threading.Thread(target=decoder)
        p.start(); d.start(); d.join(); p.join()
    torch.cuda.synchronize()
    if errors:
        raise errors[0]
    checked(all(bool(torch.isfinite(t).all()) for lane in (dec,pre)
                for t in lane.state+lane.conv + ([lane.last_output] if lane.last_output is not None else [])),
            f"{name}: nonfinite output/state")
    for r in dec.rowcaches.values():
        checked(bool((r[2]>=0).all()), "negative row-cache counters")
    primary_events = event_rows["decode" if drows else "prefill"]
    uncertainty = clock.evidence["ordering_uncertainty_ns"]
    window_first, window_last, inside, possible, tail_delta = ordering_window(drows, prows, uncertainty)
    window_start = (window_first-clock_zero)/1e9
    window_end = (window_last-clock_zero)/1e9
    # Same-stream elapsed time remains precise even if globaltimer ticks coarsely.
    duration = primary_events[0][0].elapsed_time(primary_events[-1][1])/1000
    if not math.isfinite(duration) or duration <= 0:
        raise Inconclusive("invalid same-stream measurement window")
    result = {"correct": True,"window_s": duration,"window_start_s":window_start,
              "duration_source":"same-stream CUDA events", "ordering_uncertainty_ns":uncertainty,
              "host_window":[origin+window_start,origin+window_end],
              "host_window_note":"approximate globaltimer/host anchor; NVML contextual only",
              "raw_decode": drows,"raw_prefill": prows,
              "prefill_steps_s": len(inside)/duration if prows else 0,
              "prefill_steps_s_upper": len(possible)/duration if prows else 0,
              "prefill_boundary_ambiguous":len(possible)-len(inside),
              "prefill_completed_in_window": len(inside),
              "prefill_covers_decode_tail": bool(tail_delta is not None and tail_delta>uncertainty),
              "prefill_tail_order_ambiguous":bool(tail_delta is not None and abs(tail_delta)<=uncertainty),
              "prefill_tail_s": max(0,tail_delta/1e9) if tail_delta is not None else 0}
    if name == "serial":
        # Serial baseline intentionally completes P before D; the final decode
        # has no simultaneously executing P. Report this instead of a false load gate.
        result["prefill_covers_decode_tail"] = False
    if drows:
        gaps = [1000*(b["observed_s"]-a["observed_s"]) for a,b in zip(drows,drows[1:])]
        de = event_rows["decode"]
        gpu_gaps = [a[1].elapsed_time(b[1]) for a,b in zip(de,de[1:])]
        late = [1000*(r["observed_s"]-r["release_s"]) for r in drows]
        result.update(decode_steps_s=len(drows)/duration,gap_p50_ms=percentile(gaps,.5),
                      gap_p99_ms=percentile(gaps,.99),gap_max_ms=max(gaps),gpu_gap_p99_ms=percentile(gpu_gaps,.99),
                      late_p50_ms=percentile(late,.5),late_p99_ms=percentile(late,.99),late_max_ms=max(late),
                      late_gt_period=sum(x>period*1000 for x in late))
    for lane, rows in (("decode",drows),("prefill",prows)):
        names = {k for r in rows for k in r["parts_ms"]}
        result[lane+"_parts_ms"] = {k: sum(r["parts_ms"].get(k,0) for r in rows)/len(rows) for k in names}
    selected, misses = (sum(int(rc[2][j]) for rc in dec.rowcaches.values()) for j in (0,1))
    result["host_reads"] = {
        "meaning": "logical source-byte demand, NOT PCIe hardware-counter traffic",
        "decode_selected_rows": selected,"decode_miss_rows": misses,
        # Colliding losers fall back to host in attention; winners copy once.
        # Duplicate IDs may request a row repeatedly. Treat this as source demand,
        # not unique physical transactions or achieved link utilization.
        "decode_kv_requested_bytes": misses*1024,
        "decode_requested_GB_s": misses*1024/duration/1e9,
        "prefill_kv_requested_bytes": pre.host_bytes_prefill*len(inside),
        "prefill_requested_GB_s": pre.host_bytes_prefill*len(inside)/duration/1e9,
        "ple_decode_bytes": len(drows)*dec.t*16*320,
        "ple_prefill_bytes": len(inside)*pre.t*16*320,
        "physical_pcie_bytes": None,
    }
    if green:
        result["sm_split"] = green.split
        result["prefill_observed_sms"] = sorted(set(sm_pre.cpu().tolist()))
    return result


def run(args, spec):
    rank = int(os.environ.get("RANK","0"))
    report = {"pair":args.pair,"ranks":[],"run_spec":dict(spec,gdn_decode=args.gdn_decode,steps=args.steps)}
    output = args.out
    result = {"rank":rank}
    contexts = []
    try:
        if os.environ.get("WORLD_SIZE") != "2":
            raise Inconclusive("launch with torch.distributed.run --nproc_per_node 2")
        if (os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
                or os.environ.get("CUDA_VISIBLE_DEVICES")!=args.pair):
            raise Inconclusive("declared pair must match CUDA_VISIBLE_DEVICES in PCI_BUS_ID order")
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        dist.init_process_group("gloo")
        ctl = dist.new_group([0,1],backend="gloo")
        dg = dist.new_group([0,1],backend="gloo"); pg = dist.new_group([0,1],backend="gloo")
        refgroup = dist.new_group([0,1],backend="nccl")
        library = output.parent / (output.stem+"-green.so")
        compiled = [None]
        if rank == 0:
            try:
                compiled[0] = build_helper(library, torch.cuda.get_device_capability())
            except Inconclusive as e:
                compiled[0] = {"rc":2,"log":str(e),"evidence":e.evidence}
        dist.broadcast_object_list(compiled,src=0)
        result["helper_build"] = compiled[0]
        if compiled[0]["rc"]:
            raise Inconclusive("header-built green shim failed: "+compiled[0]["log"], compiled[0])
        result["green_shim_sha256"] = digest(library)
        result["phase"] = "clock-validation"
        clock = DeviceClock(library)
        result["clock_check"] = clock.validate()
        result["clock_units_validated"] = True
        result["phase"] = "source-validation"
        expected = json.loads(Path(__file__).with_name("s5_source_manifest.json").read_text())
        if expected["model_config_sha256"] != spec["model_config_sha256"]:
            raise Inconclusive("model config hash differs from the reviewed config")
        result["source_sha256"] = {}
        for module,want in expected["modules"].items():
            actual = digest(importlib.import_module(module).__file__)
            result["source_sha256"][module] = actual
            if actual != want:
                raise Inconclusive(f"reviewed source hash mismatch: {module}")
        torch.manual_seed(871)
        torch.cuda.reset_peak_memory_stats()
        result["phase"] = "reducers-and-kernel-setup"
        rd, rp = Reducer(dg,device), Reducer(pg,device)
        rd.check(refgroup); rp.check(refgroup)
        result["reduction_equal"] = True
        w = weights(spec,rank)
        dec, pre = Slice(spec,rank,rd,False,w,args.gdn_decode), Slice(spec,rank,rp,True,w,args.gdn_decode)
        source = importlib.import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        source_hash = digest(source.__file__)
        evidence = json.loads(args.dispatch_evidence.read_text()) if args.dispatch_evidence else {}
        result["gdn_dispatch_evidence"] = (evidence.get("kernel")==args.gdn_decode
                                            and evidence.get("source_sha256")==source_hash
                                            and bool(evidence.get("evidence")))
        result["dispatch_evidence_record"] = evidence
        result["gdn_source_sha256"] = source_hash
        result["device_sms"] = torch.cuda.get_device_properties(device).multi_processor_count
        result["device_name"] = torch.cuda.get_device_name(device)
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        result["pci_bus_id"] = clock.pci_bus_id(device)
        result["modes"] = {}
        result["setup_peak_vram_gib"] = torch.cuda.max_memory_reserved()/2**30
        # Slice cadence scales by the layer fraction, not by measured slowed-down
        # performance. Keep the raw slice timings; no claimed full-model token rate.
        period = .0095*spec["layers"]/24
        for mode in MODES:
            result["phase"] = mode + ":setup"
            torch.cuda.reset_peak_memory_stats()
            green = None
            if mode.startswith("green"):
                green = Green(library,int(mode[5:]),device)
                contexts.append(green)
                membership, complete = green.membership()
                if not complete:
                    raise Inconclusive("SM membership coverage incomplete or overlapping",
                                       {"split":green.split,"samples":green.membership_evidence})
                sd,sp = green.sd,green.sp
            else:
                sd = torch.cuda.Stream(priority=-1 if mode=="prio" else 0)
                sp = torch.cuda.Stream()
            result["phase"] = mode + ":stream-clock-and-graph-smoke"
            checks = [clock.validate(sd, repeats=1), clock.validate(sp, repeats=1)]
            result.setdefault("stream_clock_checks", {})[mode] = checks
            # Use the least precise observed stream for boundary decisions.
            clock.evidence = max(checks + [result["clock_check"]], key=lambda c:c["ordering_uncertainty_ns"])
            result.setdefault("stream_order_checks", {})[mode] = clock.check_order(
                [torch.cuda.current_stream(), sd, sp, torch.cuda.current_stream()])
            result["phase"] = mode + ":kernel-graph-capture"
            graph,dtrace,smout = make_graph(dec,sd,green)
            result["phase"] = mode + ":kernel-graph-equality"
            verify_mode(dec,pre,graph,sd,sp)
            result["cuda_graph_equal"] = result["finite"] = True
            result["phase"] = mode + ":measurement"
            # One warmup trial, then exactly one measured trial.
            run_mode(mode,dec,pre,graph,dtrace,sd,sp,ctl,4,period,clock,green)
            counters = PcieSampler(clock.pci_bus_id(device));counters.start()
            value = None
            try:
                value = run_mode(mode,dec,pre,graph,dtrace,sd,sp,ctl,args.steps,period,clock,green)
            finally:
                pcie = counters.finish(value["host_window"] if value else None)
            value["device_pcie"] = pcie
            if green:
                observed = set(smout.cpu().tolist())
                pp = set(value["prefill_observed_sms"])
                value["decode_graph_observed_sms"] = sorted(observed)
                value["sm_membership"] = [sorted(x) for x in membership]
                value["sm_membership_probe_rounds"] = green.membership_evidence
                value["sm_graph_restriction"] = complete and sm_restriction(
                    membership, green.split, observed, pp, green.nsm)
            result["modes"][mode] = value
            value["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
            value["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            if torch.cuda.max_memory_reserved()/2**30 > 18:
                raise Inconclusive("peak reserved VRAM exceeded 18 GiB")
            graph.reset()
        result["calibration"] = {
            "decode_stage_ms":stage_time(result["modes"]["decode"]["decode_parts_ms"],spec,args.pair),
            "prefill_stage_ms":stage_time(result["modes"]["prefill"]["prefill_parts_ms"],spec,args.pair),
            "note":"class-count extrapolation at declared PP partition, not a full-stage measurement; eager CPU gaps excluded"}
        result["peak_vram_gib"] = max(result["setup_peak_vram_gib"],max(v["peak_reserved_gib"] for v in result["modes"].values()))
        result["peak_allocated_gib"] = max(v["peak_allocated_gib"] for v in result["modes"].values())
        result["reduction_paths"] = {"decode":sorted(rd.paths),"prefill":sorted(rp.paths)}
        for value in result["modes"].values():
            value["class_slowdown_vs_alone"] = {}
            for lane in ("decode","prefill"):
                base = result["modes"][lane][lane+"_parts_ms"]
                value["class_slowdown_vs_alone"][lane] = {k:v/base[k] for k,v in value[lane+"_parts_ms"].items()
                                                          if base.get(k,0)>0}
        # No success artifact is written before resource cleanup has succeeded.
        torch.cuda.synchronize()
        for ctx in contexts:
            ctx.close()
        gathered = [None,None]
        dist.all_gather_object(gathered,result)
        report["ranks"] = gathered
        report["correctness"] = True
        report["grade"] = grade_pair(report)
        verdict = report["grade"]["verdict"]
        if rank == 0:
            write_json(output,report)
            print(json.dumps(report["grade"],indent=2),flush=True)
        dist.destroy_process_group()
        return EXIT[verdict]
    except BaseException as e:
        verdict = "FAIL" if isinstance(e,CorrectnessError) else "INCONCLUSIVE"
        report.update(fatal_correctness=isinstance(e,CorrectnessError),error=type(e).__name__+": "+str(e),
                      grade={"verdict":verdict},partial_rank=result,
                      error_evidence=getattr(e,"evidence",{}), traceback=traceback.format_exc())
        # Per-rank failures survive distributed peer/timeout problems, without
        # replacing a previous artifact. The combiner will reject partial evidence.
        failure = output.with_name(output.name+f".rank{rank}.error.json")
        if not failure.exists():
            write_json(failure,report)
        print(json.dumps({"rank":rank,"verdict":verdict,"error":report["error"]}),flush=True)
        return EXIT[verdict]
