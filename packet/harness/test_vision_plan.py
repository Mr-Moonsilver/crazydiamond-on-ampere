"""Offline invariants for the fifth-GPU probe; no torch/CUDA import."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vision_tower_probe import embedding_budget, plan, select_vision_weights, vision_config

HERE = Path(__file__).resolve().parent
CFG = {"vision_config": {"hidden_size": 1152, "out_hidden_size": 2560,
    "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
    "depth": 27, "num_heads": 16, "in_channels": 3, "deepstack_visual_indexes": []}}


class VisionPlanTests(unittest.TestCase):
    def test_bytes_and_network_fanout(self):
        b = embedding_budget([1, 64, 64])
        self.assertEqual(b["visual_tokens"], 1024)
        self.assertEqual(b["bf16_bytes"], 5242880)
        self.assertAlmostEqual(b["ideal_binary_one_copy_ms"]["1"], 41.94304)
        self.assertAlmostEqual(b["ideal_binary_tp2_copies_ms"]["10"], 8.388608)
        self.assertEqual(embedding_budget([1, 96, 96])["MiB"], 11.25)

    def test_grid_validation_rectangles_and_time(self):
        for grid in ([1, 63, 64], [1, 64, 0], [1, 64], [True, 64, 64]):
            with self.assertRaises(ValueError):
                embedding_budget(grid)
        self.assertEqual(embedding_budget([2, 32, 64])["visual_tokens"], 1024)

    def test_model_drift_rejected(self):
        self.assertEqual(vision_config(CFG)["out_hidden_size"], 2560)
        for change in ({"deepstack_visual_indexes": [8, 16, 24]}, {"out_hidden_size": 4096}):
            with self.assertRaises(ValueError):
                vision_config({"vision_config": CFG["vision_config"] | change})

    def test_never_selects_lm_tensors_or_outside_shards(self):
        index = {"weight_map": {"model.visual.patch_embed.proj.weight": "a.safetensors",
                                "model.language_model.layers.0.weight": "b.safetensors"}}
        selected = select_vision_weights(index, "/model")
        self.assertEqual(len(selected), 1)
        self.assertEqual(next(iter(selected.values()))[0], "patch_embed.proj.weight")
        for mapping in ({"visual.weight": "../a.safetensors"},
                        {"model.visual.weight": "a.bin"},
                        {"visual.weight": "a.safetensors", "model.visual.weight": "a.safetensors"},
                        {"model.language_model.weight": "a.safetensors"}):
            with self.assertRaises(ValueError):
                select_vision_weights({"weight_map": mapping}, "/model")

    def test_plan_imports_no_gpu_libraries(self):
        # Separate Python, prevent import of ALL non-stdlib optional dependencies.
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config.json"
            config.write_text(json.dumps(CFG))
            code = """
import sys, runpy
class NoGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','vllm','transformers','PIL','safetensors'):
            raise AssertionError('GPU/dependency import during plan: '+fullname)
sys.meta_path.insert(0, NoGPU())
sys.argv = [sys.argv[1], '--config', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
            r = subprocess.run([sys.executable, "-c", code, str(HERE / "vision_tower_probe.py"), str(config)],
                               text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(json.loads(r.stdout)["gpu_used"])

    def test_no_lm_loader_service_or_remote_calls(self):
        tree = ast.parse((HERE / "vision_tower_probe.py").read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertFalse(names & {"LLM", "EngineArgs", "AsyncLLM", "requests", "subprocess"})
        self.assertEqual(plan(CFG, [512])["images"][0]["bf16_bytes"], 1310720)


if __name__ == "__main__":
    unittest.main()
