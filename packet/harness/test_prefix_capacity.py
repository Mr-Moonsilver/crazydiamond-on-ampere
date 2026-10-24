#!/usr/bin/env python3
"""Small hand-computable capacity cases; no torch, box, engine or network."""
import ast
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent_replay
from prefix_capacity import (CheckpointLRU, Layout, consume_text_rng, packed_layout, payloads,
                             replay_config, simulate, turns)


def source_packing(block_tokens, partition, source_path=None, enabled=None, details=False):
    """Execute the IMAGE planner functions; fake spec objects supply only metadata.

    No vLLM/torch imports. Adapters reproduce this model's uniform-type rules,
    not arbitrary model backends; page sizes come from the independently tested
    payload calculation. The packing, projection and byte maximum are real code.
    """
    @dataclass(frozen=True)
    class Spec:
        block_size: int
        page_size_bytes: int
        tp_replicated: bool = False

    class MambaSpec(Spec):
        pass

    class CircularBufferSpec(Spec):
        pass

    class SlidingWindowSpec(Spec):
        pass

    @dataclass
    class Uniform:
        block_size: int
        kv_cache_specs: dict

        @property
        def first_spec(self):
            return next(iter(self.kv_cache_specs.values()))

        @classmethod
        def is_uniform_type(cls, specs):
            first = next(iter(specs.values()))
            return all(type(s) is type(first) and s.block_size == first.block_size and
                       (not isinstance(s, MambaSpec) or
                        (s.page_size_bytes, s.tp_replicated) == (first.page_size_bytes, first.tp_replicated))
                       for s in specs.values())

        @classmethod
        def from_specs(cls, specs):
            return cls(next(iter(specs.values())).block_size, specs) if cls.is_uniform_type(specs) else None

        def get_max_layers_per_page_size(self):
            counts = defaultdict(int)
            for s in self.kv_cache_specs.values():
                counts[s.page_size_bytes] += 1
            return max(counts.values())

    @dataclass
    class Group:
        layer_names: list
        kv_cache_spec: object
        is_eagle_group: bool = False

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((source_path or root / "reference/vllm-src/v1/core/kv_cache_utils.py").read_text())
    names = {"_approximate_gcd", "_get_packed_kv_cache_groups", "_project_kv_cache_groups_to_worker",
             "_get_per_layer_spec", "_get_kv_cache_bytes_per_block"}
    if source_path:
        names.update({"_flash_pp_group_count", "_flash_log_group_layout"})
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *functions], type_ignores=[])
    ns = {"defaultdict": defaultdict, "cdiv": lambda a, b: (a + b - 1) // b,
          "UniformTypeKVCacheSpecs": Uniform, "KVCacheGroupSpec": Group, "MambaSpec": MambaSpec,
          "CircularBufferSpec": CircularBufferSpec, "SlidingWindowSpec": SlidingWindowSpec,
          "HiSparseHotSpec": type("Unused", (), {}), "_glm5_next_tensor_layout": lambda g: None,
          "_annotate_eagle_groups": lambda *a, **kw: None, "_warn_if_unannotated_eagle_mamba": lambda *a: None,
          "_is_deepseek_v4_eagle": lambda c: False,
          "_flash_os": SimpleNamespace(environ={"FLASH_PP_GROUPING": enabled or "0"}),
          "_flash_json": json, "logger": SimpleNamespace(info=lambda *a: None)}
    exec(compile(ast.fix_missing_locations(module), "image_planner_functions", "exec"), ns)
    specs = {}
    for i in range(48):
        if i == 2:
            specs[f"{i}.ple"] = MambaSpec(block_tokens, 1603584, True)  # padded by QSA offload selector
        if i % 4 == 3:
            specs[f"{i}.main"] = Spec(block_tokens, 2 * block_tokens)
            specs[f"{i}.index"] = Spec(block_tokens, 64 * block_tokens)
            specs[f"{i}.ring"] = CircularBufferSpec(4, 1120)
        else:
            specs[f"{i}.gdn"] = MambaSpec(block_tokens, 1603584)
    config = SimpleNamespace(cache_config=SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_block_outermost=True)))
    workers, start = [], 0
    for size in partition:
        owned = {n: s for n, s in specs.items() if start <= int(n.split('.')[0]) < start + size}
        workers.extend([owned, owned.copy()])  # Two TP ranks on each PP stage.
        start += size
    kwargs = {"worker_layer_names": [set(w) for w in workers]} if source_path else {}
    groups = ns["_get_packed_kv_cache_groups"](config, specs, **kwargs)
    pages, start = [], 0
    for size in partition:
        owned = {n: s for n, s in specs.items() if start <= int(n.split('.')[0]) < start + size}
        projected = ns["_project_kv_cache_groups_to_worker"](groups, owned)
        pages.append(ns["_get_kv_cache_bytes_per_block"](projected))
        start += size
    if details:
        return ns, specs, groups, workers, pages
    return sum(isinstance(g.kv_cache_spec.first_spec, MambaSpec) for g in groups), pages


class PrefixCapacityTests(unittest.TestCase):
    def test_image_default_is_semantic_retention(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "reference/vllm-src/config/cache.py").read_text())
        field = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name)
                     and n.target.id == "prefix_cache_retention_interval")
        self.assertIsInstance(field.value, ast.Call)
        default = next(k.value for k in field.value.keywords if k.arg == "default")
        self.assertEqual(ast.literal_eval(default), 0)

    def test_automatic_block_size_is_image_function_result(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "reference/vllm-src/platforms/interface.py").read_text())
        names = {"_qsa_offload_bytes_per_token_per_layer", "_qsa_offload_block_size"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                     "image_block_size_selector", "exec"), ns)
        text = json.loads((root / "reference/model-config.json").read_text())["text_config"]
        cfg = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**text),
                              dtype=SimpleNamespace(itemsize=2)),
                              parallel_config=SimpleNamespace(pipeline_parallel_size=2), speculative_config=None)
        for split in (24, 26):
            util = SimpleNamespace(get_pp_indices=lambda n, rank, count: (0, split) if rank == 0 else (split, n))
            with patch.dict(sys.modules, {"vllm.distributed.utils": util}), patch.dict(os.environ, {"VLLM_QSA_KV_OFFLOAD": "1"}):
                self.assertEqual(ns["_qsa_offload_block_size"](cfg, 1603584), 4048)

    def test_idle_agents_consume_cache_without_raising_usage(self):
        eight = Layout(37).working_set(8, 8, 117000)
        ten = Layout(37).working_set(10, 8, 117000)
        self.assertEqual(eight["pinned_blocks"], 536)
        self.assertEqual(ten["pinned_blocks"], 536)
        self.assertEqual((eight["headroom_blocks"], ten["headroom_blocks"]), (211, 5))
        proposed = Layout(21).working_set(10, 8, 117000)
        self.assertEqual(proposed["headroom_blocks"], 453)
        self.assertEqual(Layout(37).working_set(8, 8, 150000)["cached_vectors_fit"], 19)
        self.assertEqual(Layout(37).working_set(10, 10, 150000)["cached_vectors_fit"], 15)
        self.assertEqual(Layout(21).working_set(8, 8, 150000)["cached_vectors_fit"], 40)
        self.assertEqual(Layout(21).working_set(10, 10, 150000)["cached_vectors_fit"], 35)

    def test_reachable_retention_mask_matches_image_method(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "reference/vllm-src/v1/core/single_type_kv_cache_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaManager")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reachable_block_mask")
        method.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  method], type_ignores=[])
        spec_type = type("MambaSpec", (), {})
        ns = {"MambaSpec": spec_type}
        exec(compile(ast.fix_missing_locations(module), "image_retention_method", "exec"), ns)
        spec = spec_type()
        spec.block_size = 4048
        for prompt in (7392, 7568, 8096, 8097, 9504, 60000, 150000):
            mask = ns["reachable_block_mask"](None, 0, 40, 176, spec, False, 0, (prompt - 1,))
            ends = {(i + 1) * 4048 for i, keep in enumerate(mask) if keep}
            expected = (prompt - 1) // 176 * 176 // 4048 * 4048
            self.assertEqual(ends, {expected} if expected else set())
        self.assertIsNone(ns["reachable_block_mask"](None, 0, 40, 176, spec, False, None, (7567,)))

    def test_reachable_policy_skips_interior_pollution(self):
        layout = Layout(2, pool_bytes=15, page_bytes=1, block_tokens=10, hash_tokens=2,
                        history_groups=0, ring_groups=0)
        rs = [dict(session=(a, 1), prompt=p, reply=1, previous_history=prior, round=r, kind=kind)
              for a, p, prior, r, kind in [(0, 102, 0, 0, "fresh"), (1, 102, 0, 0, "fresh"),
                                          (0, 114, 103, 1, "append")]]
        d = simulate(layout, rs, 4, policy="reachable", warmup_rounds=1)
        self.assertAlmostEqual(d["predicted_append_hit_rate"], 102 / 114)

    def test_packing_matches_image_functions(self):
        for block in (4048, 4064, 4224, 8096):
            for part in ((24, 24), (26, 22)):
                with self.subTest(block=block, partition=part):
                    groups, pages = source_packing(block, part)
                    modeled = packed_layout(block, part)
                    self.assertEqual(groups, modeled["state_groups"])
                    self.assertEqual(pages, [s["page_bytes"] for s in modeled["per_stage"]])

    def test_pp_packing_proposal_preserves_single_rank_page_budget(self):
        for part, groups in (((24, 24), 19), ((26, 22), 21)):
            p = packed_layout(partition=part, pp_aware_proposal=True)
            self.assertEqual(p["state_groups"], groups)
            self.assertEqual(p["page_bytes"], 1603584)
            self.assertTrue(all(max(s["gdn_layers_per_group"]) == 1 for s in p["per_stage"]))

    def test_block_size_threshold_and_uneven_pp_penalty(self):
        self.assertEqual(packed_layout(4048)["state_groups"], 37)
        self.assertEqual(packed_layout(4064)["state_groups"], 19)
        self.assertEqual(packed_layout(4064, (26, 22))["page_bytes"], 3207168)
        self.assertEqual(packed_layout(4064, (24, 24))["page_bytes"], 1609344)

    def test_eviction_priority_of_consumed_partial_tail(self):
        c = CheckpointLRU(4, 2)
        c.publish("a", 100)
        c.publish("b", 100)
        c.latest("a", 100)  # young after reuse
        c.demote("a", 100)  # overlay demotes after replacement publication
        c.publish("c", 100)
        self.assertEqual(c.latest("a", 100), 0)
        self.assertEqual(c.latest("b", 100), 100)

    def test_exact_image_shapes_and_uneven_pp(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "reference/model-config.json").read_text())
        p = payloads(config)
        self.assertEqual((p["gdn_conv_bytes"], p["gdn_recurrent_bytes"], p["gdn_state_bytes"]),
                         (30720, 1572864, 1603584))
        self.assertEqual(p["ple_state_bytes_replicated"], 184320)
        self.assertEqual([s["gdn_snapshot_bytes"] for s in p["stages"]], [28864512, 28864512])
        self.assertEqual([s["qsa_host_bytes_per_token"] for s in p["stages"]], [6144, 6144])
        p = payloads(config, partition=(26, 22))
        self.assertEqual([s["gdn_layers"] for s in p["stages"]], [20, 16])
        self.assertEqual([s["qsa_layers"] for s in p["stages"]], [6, 6])

    def test_pool_rounding_null_and_host_cap(self):
        layout = Layout(37)
        self.assertEqual((layout.pool_blocks, layout.usable_blocks), (1340, 1339))
        self.assertTrue(layout.ledger()["host_cap_passes"])
        self.assertFalse(Layout(37, pool_bytes=2400000000).ledger()["host_cap_passes"])
        limit = layout.ledger()["max_pool_bytes_at_fixed_page_size_under_host_cap"]
        self.assertTrue(Layout(37, pool_bytes=limit).ledger()["host_cap_passes"])
        self.assertFalse(Layout(37, pool_bytes=limit + 1).ledger()["host_cap_passes"])

    def test_last_vectors_fit_10_at_150k_but_not_3_copies(self):
        layout = Layout(37)
        two_vectors = 10 * (38 + 1 + 2 * 37)
        three_vectors = 10 * (38 + 1 + 3 * 37)
        self.assertLess(two_vectors, layout.usable_blocks)
        self.assertGreater(three_vectors, layout.usable_blocks)
        self.assertLess(8 * (38 + 1 + 3 * 37), layout.usable_blocks)
        # Holding all boundary snapshots fails for EIGHT, not just ten.
        self.assertGreater(8 * 37 * (150000 // 4048), layout.usable_blocks)

    def test_one_missing_group_invalidates_latest_only(self):
        c = CheckpointLRU(3, 2)
        c.publish("a", 100)
        c.publish("a", 200)
        self.assertEqual(c.latest("a", 150), 0)  # only one group of 100 survived
        self.assertEqual(c.latest("a", 250), 200)
        self.assertEqual(c.latest("foreign", 250), 0)

    def test_lookup_is_all_groups_and_touch_changes_eviction_order(self):
        c = CheckpointLRU(4, 2)
        c.publish("a", 100)
        c.publish("b", 100)
        self.assertEqual(c.latest("a", 100), 100)
        c.publish("c", 100)
        self.assertEqual(c.latest("b", 100), 0)
        self.assertEqual(c.latest("a", 100), 100)

    def test_zero_capacity_is_miss_not_crash(self):
        c = CheckpointLRU(0, 2)
        c.publish("a", 100)
        self.assertEqual(c.latest("a", 100), 0)

    def test_weighted_appends_not_mean_of_percentages(self):
        layout = Layout(1, pool_bytes=101, page_bytes=1, block_tokens=10, hash_tokens=2,
                        history_groups=0, ring_groups=0)
        rs = [dict(session=(a, 1), prompt=p, reply=2, previous_history=prior, round=r, kind=kind)
              for a, p, prior, r, kind in [(0, 100, 0, 0, "fresh"), (1, 20, 0, 0, "fresh"),
                                          (0, 110, 102, 1, "append"), (1, 40, 22, 1, "append")]]
        d = simulate(layout, rs, 0, warmup_rounds=1)
        self.assertEqual(d["predicted_cached_tokens"], 120)
        self.assertAlmostEqual(d["predicted_append_hit_rate"], 120 / 150)
        self.assertEqual(d["append_turns"], 2)

    def test_40_percent_active_does_not_mean_no_eviction(self):
        layout = Layout(2, pool_bytes=11, page_bytes=1, block_tokens=10, hash_tokens=2,
                        history_groups=0, ring_groups=0)
        rs = [dict(session=(a, 1), prompt=p, reply=1, previous_history=prior, round=r, kind=kind)
              for a, p, prior, r, kind in [(0, 100, 0, 0, "fresh"), (1, 100, 0, 0, "fresh"),
                                          (0, 110, 101, 1, "append")]]
        d = simulate(layout, rs, 4, policy="dense", warmup_rounds=1)
        self.assertEqual(d["pinned_usage_fraction"], .4)
        self.assertEqual(d["predicted_append_hit_rate"], 0)
        self.assertGreater(d["group_block_evictions"], 0)

    def test_latest_policy_retains_other_agents(self):
        layout = Layout(2, pool_bytes=11, page_bytes=1, block_tokens=10, hash_tokens=2,
                        history_groups=0, ring_groups=0)
        rs = [dict(session=(a, 1), prompt=p, reply=1, previous_history=prior, round=r, kind=kind)
              for a, p, prior, r, kind in [(0, 100, 0, 0, "fresh"), (1, 100, 0, 0, "fresh"),
                                          (0, 110, 101, 1, "append")]]
        d = simulate(layout, rs, 4, policy="latest", warmup_rounds=1)
        self.assertAlmostEqual(d["predicted_append_hit_rate"], 100 / 110)

    def test_infeasible_reserve_and_missing_samples_not_predictions(self):
        self.assertIsNone(simulate(Layout(37), [], 1339)["predicted_append_hit_rate"])
        d = simulate(Layout(37), [], 0)
        self.assertEqual(d["status"], "NO_APPEND_SAMPLES")
        self.assertIsNone(d["predicted_append_hit_rate"])

    def test_rng_matches_actual_agent_replay_text(self):
        a, b = random.Random(2000), random.Random(2000)
        for n in (0, 1, 22, 500, 4097):
            consume_text_rng(a, n)
            agent_replay.text(b, n)
            self.assertEqual(a.getstate(), b.getstate())

    def test_rollover_new_session_cannot_hit_old(self):
        cfg = replay_config(dict(AGENTS=1, START_MIN=20, START_MAX=20, APPEND_MIN=2, APPEND_MAX=2,
                                 REPLY_MIN=2, REPLY_MAX=2, ROLL_AT=21))
        rs = list(turns(cfg, 3))
        self.assertEqual([r["session"] for r in rs], [(0, 1), (0, 2), (0, 3)])
        self.assertTrue(all(r["kind"] == "fresh" for r in rs))

    def test_input_validation(self):
        for kwargs in (dict(state_groups=0), dict(state_groups=37, hash_tokens=177),
                       dict(state_groups=37, pool_bytes=1), dict(state_groups=37, history_groups=-1)):
            with self.assertRaises(ValueError):
                Layout(**kwargs)
        for cfg in (dict(AGENTS=0), dict(START_MIN=10, START_MAX=5), dict(TOOL_MIN=float("nan"))):
            with self.assertRaises(ValueError):
                replay_config(cfg)


if __name__ == "__main__":
    unittest.main()
