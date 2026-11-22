#!/usr/bin/env python3
"""Unarmed fifth-GPU probe. Default/--plan uses stdlib only; --run is explicit.

Loads ONLY real BF16 vision weights into the pinned Qwen3 tower (TP1/PP1).
No server, LM, queue, network transport or service control. One warmup and one
measurement per image. Runtime ABI and GPU execution still need qualification.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

TOWER_SHA = "ba8eb6603978073b54b4f09453e0741cb5534321fce60cd8cd3f9bd5ddec724a"
MODEL_COMMIT = "dc36fcce902a63eab06c1b93a5c4a5ee178a0c56"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vision_config(config):
    v = config["vision_config"]
    required = {"hidden_size": 1152, "out_hidden_size": 2560, "patch_size": 16,
                "temporal_patch_size": 2, "spatial_merge_size": 2, "depth": 27,
                "num_heads": 16, "in_channels": 3}
    if any(v.get(k) != val for k, val in required.items()) or v.get("deepstack_visual_indexes", []):
        raise ValueError("probe is pinned to Flash-Next's BF16 tower, no deepstack")
    return v


def embedding_budget(grid, width=2560):
    if len(grid) != 3 or any(type(x) is not int or x <= 0 for x in grid):
        raise ValueError("positive integer T,H,W grid required")
    if grid[1] % 2 or grid[2] % 2:
        raise ValueError("spatial grid must be divisible by merge size 2")
    tokens = math.prod(grid) // 4
    size = tokens * width * 2
    return {"grid_thw": grid, "visual_tokens": tokens, "bf16_bytes": size,
            "MiB": size / 2**20, "raw_base64_bytes_lower_bound": 4*((size+2)//3),
            "ideal_binary_one_copy_ms": {str(g): size*8/(g*1e9)*1000 for g in (1, 10)},
            "ideal_binary_tp2_copies_ms": {str(g): size*16/(g*1e9)*1000 for g in (1, 10)}}


def select_vision_weights(index, model):
    """Reject ambiguous prefixes and paths outside the checkpoint. No LM reads."""
    weights = index["weight_map"]
    prefixes = [p for p in ("visual.", "model.visual.") if any(k.startswith(p) for k in weights)]
    if len(prefixes) != 1:
        raise ValueError("need exactly one checkpoint vision prefix")
    prefix, root, selected = prefixes[0], Path(model).resolve(), {}
    for name, shard in weights.items():
        if not name.startswith(prefix):
            continue
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or path.suffix != ".safetensors":
            raise ValueError("vision shard outside checkpoint or not safetensors")
        selected[name] = (name[len(prefix):], path)
    return selected


def plan(config, sizes):
    vision_config(config)
    if any(s <= 0 or s % 32 for s in sizes):
        raise ValueError("planned square dimensions must be positive multiples of 32")
    return {"mode": "plan_only", "gpu_used": False, "source_commit": MODEL_COMMIT,
            "assumptions": "dimensions AFTER processor resize, BF16, one still image; transport only, no encode/queue/LM time",
            "images": [dict(resized_square=s, **embedding_budget([1, s//16, s//16])) for s in sizes]}


def run_probe(args, config, outdir):
    # This is the only GPU-import path. Never called by --plan or selftest.
    import inspect
    import torch
    import vllm
    from PIL import Image
    from safetensors import safe_open
    from transformers import Qwen3VLProcessor
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel)
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.multimodal.media.image import ImageEmbeddingMediaIO

    if torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one agreed-slot GPU via CUDA_VISIBLE_DEVICES")
    source = inspect.getfile(Qwen3_VisionTransformer)
    if digest(source) != TOWER_SHA:
        raise ValueError("tower source does not match pinned image")
    if "dc36fcce9" not in vllm.__version__:
        raise ValueError("wrong vLLM build; run in pinned ad067db5 image")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.set_default_dtype(torch.bfloat16)
    cfg = vision_config(config)
    selected = select_vision_weights(json.loads((args.model / "model.safetensors.index.json").read_text()), args.model)
    tensor_hash = hashlib.sha256()
    counts = {"tensors": 0, "bytes": 0}

    def weights():
        # CPU slice reads only these tensors even when a shard also holds LM weights.
        by_shard = {}
        for name, (local, path) in selected.items():
            by_shard.setdefault(path, []).append((name, local))
        for path in sorted(by_shard):
            with safe_open(path, framework="pt", device="cpu") as f:
                for name, local in sorted(by_shard[path]):
                    tensor = f.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError("vision checkpoint is not entirely BF16")
                    counts["tensors"] += 1
                    counts["bytes"] += tensor.nbytes
                    tensor_hash.update(name.encode() + b"\0")
                    tensor_hash.update(str(tuple(tensor.shape)).encode() + b"\0")
                    tensor_hash.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                    yield local, tensor

    result = {"mode": "tower_only", "source_commit": MODEL_COMMIT,
              "vllm_version": vllm.__version__, "tower_sha256": TOWER_SHA,
              "config_sha256": digest(args.model / "config.json"),
              "device": torch.cuda.get_device_name(0), "tp": 1, "pp": 1,
              "lm_loaded": False, "repetitions": 1, "warmups": 1,
              "images": []}
    def save(phase):
        result["phase"] = phase
        (outdir / "probe.json").write_text(json.dumps(result, indent=2) + "\n")

    save("initializing")
    # Minimal config: no ModelConfig/EngineArgs/LLM construction or LM allocations.
    vc = VllmConfig()
    with set_current_vllm_config(vc), torch.inference_mode():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                     distributed_init_method="file://" + str(outdir / "rendezvous"))
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        with torch.device("cuda"):
            tower = Qwen3_VisionTransformer(SimpleNamespace(**cfg),
                                           norm_eps=config["text_config"]["rms_norm_eps"],
                                           quant_config=None, prefix="model.visual").eval()
        loaded = tower.load_weights(weights())
        required = set(dict(tower.named_parameters()))
        if required - loaded:
            raise ValueError("incomplete vision load: " + ",".join(sorted(required-loaded)))
        torch.cuda.synchronize()
        result["weights"] = dict(counts, sha256=tensor_hash.hexdigest(),
                                 parameters=len(required), load_wall_s=time.perf_counter()-start,
                                 allocated_delta_bytes=torch.cuda.memory_allocated()-before)
        result["attention_backend"] = str(tower.attn_backend)
        save("weights_loaded")
        # Same explicit HF processor family selected by Qwen3VLProcessingInfo.
        processor = Qwen3VLProcessor.from_pretrained(str(args.model), local_files_only=True,
                                                     use_fast=True)
        result["processor_class"] = type(processor.image_processor).__name__
        for size in args.sizes:
            # Deterministic synthetic RGB content; real checkpoint and processor.
            # This measures cost, not visual understanding/accuracy.
            image = Image.new("RGB", (size, size), color=(37, 121, 209))
            t = time.perf_counter()
            inputs = processor.image_processor(images=[image], return_tensors="pt")
            process_ms = (time.perf_counter()-t)*1000
            grid = inputs["image_grid_thw"].tolist()
            if len(grid) != 1:
                raise ValueError("expected exactly one image")
            budget = embedding_budget(grid[0])
            if budget["visual_tokens"] > args.max_visual_tokens:
                raise ValueError("processor output exceeds probe token bound")
            pixels_cpu = inputs["pixel_values"].to(dtype=torch.bfloat16).pin_memory()
            torch.cuda.synchronize()
            t = time.perf_counter()
            pixels = pixels_cpu.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter()-t)*1000
            warm = tower(pixels, grid_thw=grid)
            torch.cuda.synchronize()
            # Delete no files; release the warmup tensor before measuring activation peaks.
            warm = None
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            t = time.perf_counter()
            begin.record()
            output = tower(pixels, grid_thw=grid)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter()-t)*1000
            peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            if list(output.shape) != [budget["visual_tokens"], 2560] or output.dtype != torch.bfloat16:
                raise ValueError("unexpected output shape or dtype")
            t = time.perf_counter()
            host = torch.empty(output.shape, dtype=output.dtype, device="cpu", pin_memory=True)
            host.copy_(output, non_blocking=True)
            torch.cuda.synchronize()
            d2h_ms = (time.perf_counter()-t)*1000
            if not torch.isfinite(host).all():
                raise ValueError("nonfinite encoder output")
            # Actual pinned media codec, including roundtrip; never persist payload.
            io = ImageEmbeddingMediaIO()
            t = time.perf_counter()
            encoded = io.encode_base64(host)
            serialized_ms = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            decoded = io.load_base64("", encoded)
            decode_ms = (time.perf_counter()-t)*1000
            if decoded.dtype != host.dtype or not torch.equal(decoded, host):
                raise ValueError("embedding media codec roundtrip differs")
            result["images"].append(dict(budget, input_square=size,
                preprocess_wall_ms=process_ms, h2d_wall_ms=h2d_ms,
                encode_cuda_event_ms=begin.elapsed_time(end), encode_wall_ms=wall_ms,
                d2h_wall_ms=d2h_ms, codec_encode_ms=serialized_ms, codec_decode_ms=decode_ms,
                base64_payload_bytes=len(encoded.encode("ascii")),
                allocated_before_encode_bytes=resident, peak_allocated_bytes=peak,
                encode_incremental_peak_bytes=peak-resident, peak_reserved_bytes=reserved,
                output_sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()))
            # Keep no earlier output/activation while the next image is measured.
            output = host = decoded = encoded = pixels = pixels_cpu = inputs = None
            save("measuring")
    result["qualification"] = "MEASURED_TOWER_ONLY; no remote route, language decode or image quality test"
    save("complete")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true", help="GPU slot must be granted by queue owner")
    p.add_argument("--config", type=Path, help="plan JSON config; --run uses MODEL/config.json")
    p.add_argument("--model", type=Path, help="local checkpoint, read-only; no downloads")
    p.add_argument("--out", type=Path, help="fresh result directory, required for --run")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 1536])
    p.add_argument("--max-visual-tokens", type=int, default=4096)
    a = p.parse_args(argv)
    if a.run and (not a.model or not a.out):
        p.error("--run requires --model and --out")
    path = a.model / "config.json" if a.model else a.config
    if path is None:
        p.error("--config or --model required")
    config = json.loads(path.read_text())
    estimate = plan(config, a.sizes)
    if not a.run:
        print(json.dumps(estimate, indent=2))
        return 0
    # Fresh artifact path; no overwrite/deletion of previous runs.
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("VLLM_CACHE_ROOT", str(a.out / "vllm-cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(a.out / "triton-cache"))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(a.out / "flashinfer-cache"))
    try:
        result = run_probe(a, config, a.out)
    except Exception as e:
        # No exception string: external libraries may put paths/payloads in it.
        import traceback
        (a.out / "error.json").write_text(json.dumps({"verdict": "INCONCLUSIVE",
            "error_type": type(e).__name__, "phase_source": "probe.json",
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(e.__traceback__)]}) + "\n")
        print("INCONCLUSIVE: " + type(e).__name__ + "; inspect probe.json phase")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
