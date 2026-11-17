"""One pinned BF16 tower; constructed/called only by the encoder worker thread."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from vision_speech_guard import ALLOCATOR_LIMIT_MIB
from vision_tower_probe import TOWER_SHA, digest, embedding_budget, select_vision_weights, vision_config

MODEL_MANIFEST_SHA = '8b333e2380aac6ed98c6ed5700474972cefb919cf126949fc31f70b47a39ca93'
VISION_WEIGHT_BYTES = 897862112


class VisionBackend:
    def __init__(self, model, run_dir):
        import inspect
        import torch
        import vllm
        from safetensors import safe_open
        from transformers import Qwen3VLProcessor
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
        from vllm.multimodal.media.image import ImageEmbeddingMediaIO

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
        from vision_encoder_bundle import verify
        self.manifest = verify(model)
        if self.manifest != MODEL_MANIFEST_SHA:
            raise ValueError('vision weight manifest differs from staged owner receipt')
        if torch.cuda.device_count() != 1 or 'dc36fcce9' not in vllm.__version__:
            raise ValueError('one agreed GPU and pinned vLLM build required')
        if digest(inspect.getfile(Qwen3_VisionTransformer)) != TOWER_SHA:
            raise ValueError('vision tower source differs')
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_LIMIT_MIB * 2**20 / total, 0)
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.bfloat16)
        config = json.loads((model / 'config.json').read_text())
        cfg = vision_config(config)
        cfg_runtime = VllmConfig()
        cfg_runtime.compilation_config.compile_mm_encoder = False
        cfg_runtime.compilation_config.cudagraph_mm_encoder = False
        self.context = set_current_vllm_config(cfg_runtime)
        self.context.__enter__()
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend='nccl',
            distributed_init_method='file://' + str(run_dir / 'rendezvous'))
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with torch.device('cuda'):
            self.tower = Qwen3_VisionTransformer(SimpleNamespace(**cfg),
                norm_eps=config['text_config']['rms_norm_eps'], quant_config=None,
                prefix='model.visual').eval()
        selected = select_vision_weights(json.loads((model / 'model.safetensors.index.json').read_text()), model)
        counts = {'tensors': 0, 'bytes': 0}

        def weights():
            for path in sorted({entry[1] for entry in selected.values()}):
                with safe_open(path, framework='pt', device='cpu') as source:
                    for name, (local, shard) in sorted(selected.items()):
                        if shard != path:
                            continue
                        tensor = source.get_tensor(name)
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError('non-BF16 vision tensor')
                        counts['tensors'] += 1; counts['bytes'] += tensor.nbytes
                        yield local, tensor

        loaded = self.tower.load_weights(weights())
        if (counts['tensors'] != 333 or counts['bytes'] != VISION_WEIGHT_BYTES
                or set(dict(self.tower.named_parameters())) - loaded):
            raise ValueError('incomplete vision tensor load')
        self.processor = Qwen3VLProcessor.from_pretrained(str(model), local_files_only=True, use_fast=True)
        self.io = ImageEmbeddingMediaIO()
        self.torch = torch
        self.weights = counts
        self.backend = str(self.tower.attn_backend)
        if not any(name in self.backend.upper() for name in ('FLASH_ATTN', 'FLASHINFER')):
            raise ValueError('unqualified attention fallback; no quadratic attention permitted')
        torch.cuda.synchronize()
        self.encodes = 0

    def encode(self, image_bytes):
        from PIL import Image
        torch = self.torch
        started = time.monotonic()
        image = Image.open(io.BytesIO(image_bytes))
        if image.width * image.height > 4096**2 or image.width <= 0 or image.height <= 0:
            raise ValueError('image dimensions exceed packet cap')
        if getattr(image,'n_frames',1)!=1:raise ValueError('animated images and video are refused')
        image = image.convert('RGB')
        inputs = self.processor.image_processor(images=[image], return_tensors='pt',
            min_pixels=32**2, max_pixels=1024**2)
        grids = inputs['image_grid_thw'].tolist()
        if len(grids) != 1:
            raise ValueError('one still image required')
        budget = embedding_budget(grids[0])
        if budget['visual_tokens'] > 1024 or budget['visual_tokens'] <= 0:
            raise ValueError('post-processor grid exceeds 1024 visual tokens')
        with torch.inference_mode():
            pixels = inputs['pixel_values'].to(device='cuda', dtype=torch.bfloat16)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            begin.record()
            output = self.tower(pixels, grid_thw=grids)
            end.record()
            host = output.to('cpu')
            end.synchronize()
        if list(host.shape) != [budget['visual_tokens'], 2560] or host.dtype != torch.bfloat16:
            raise ValueError('unexpected encoder shape/dtype')
        if not bool(torch.isfinite(host).all()):
            raise ValueError('nonfinite encoder output')
        encoded = self.io.encode_base64(host)
        grid_tensor = torch.tensor(grids, dtype=torch.int64)
        encoded_grid = self.io.encode_base64(grid_tensor)
        if not torch.equal(self.io.load_base64('', encoded), host):
            raise ValueError('embedding codec not byte-exact')
        self.encodes += 1
        return dict(schema='flash-next-image-embeds/1', manifest_sha256=self.manifest,
            image_grid_thw=grids, shape=list(host.shape), dtype='bfloat16',
            embedding_sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest(),
            content=dict(type='image_embeds', image_embeds=dict(image_embeds=encoded,
                image_grid_thw=encoded_grid)),
            metrics=dict(encode_event_ms=begin.elapsed_time(end), wall_ms=(time.monotonic()-started)*1000,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), allocator_limit_mib=ALLOCATOR_LIMIT_MIB,
                visual_tokens=budget['visual_tokens'], output_bytes=host.nbytes))
