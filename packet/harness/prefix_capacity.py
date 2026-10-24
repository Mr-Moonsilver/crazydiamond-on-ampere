#!/usr/bin/env python3
"""Offline hybrid-prefix capacity / reuse-distance model, NOT a vLLM emulator.

The CLI takes the allocator's STATE GROUP COUNT, not the local layer count.
Page size / packing / pinned reserve are explicit inputs. The supplied image's
packing algorithm is reconstructed separately by packed_layout (and checked
against its actual Python function with spec adapters in the tests).
Run --help; see reviews/codex-20261024h.md for assumptions and source derivations.

Example (source-derived packing at B=4048, not a recovered runtime manifest):
  python3 harness/prefix_capacity.py --state-groups 37 --agents 8
  python3 harness/prefix_capacity.py --state-groups 37 --agents 10 --policy latest

Agent lengths use agent_replay's seeded RNG consumption, but token targets are
not tokenizer counts. Turns are round-robin, instantaneous cache transactions.
QSA history and running/COW state are represented by a fixed pinned reserve;
only the remaining checkpoint LRU is simulated. No scheduler, timing, hashing,
copy races, GPU work, or performance predictions. Retained checkpoints require
ALL state groups. Hit rates are token-weighted, over append turns only.
"""
import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random


GIB = 1 << 30
WORDS = ("def class return import self value result error file path test assert config request response token cache "
         "block layer index buffer stream kernel queue batch thread lock state update commit branch merge diff patch "
         "line column module package build deploy").split()


def positive(name, value, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if zero else 1}")


def payloads(config, tp=2, partition=(24, 24), ssm_bytes=4):
    """Raw payload, not padded pool charges. MTP=0, BF16 conv and QSA."""
    c = config.get("text_config", config)
    positive("tp", tp)
    if ssm_bytes not in (2, 4):
        raise ValueError("ssm_bytes must be 2 or 4; verify effective cache dtype")
    if sum(partition) != c["num_hidden_layers"] or any(n <= 0 for n in partition):
        raise ValueError("partition must cover the model")
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    if kh % tp or vh % tp or c["num_key_value_heads"] % tp:
        raise ValueError("TP must divide GDN and QSA heads")
    conv = (2 * kh * kd + vh * vd) // tp * (c["linear_conv_kernel_dim"] - 1) * 2
    recurrent = vh // tp * kd * vd * ssm_bytes
    ple = c["hidden_size"] * c["hc_count"] * (c["ple_conv_kernel_size"] - 1) * c["ngram_size"] * 2
    host_per_layer_token = c["num_key_value_heads"] // tp * c["head_dim"] * 2 * 2
    if c["indexer_kv_heads"] != 1 or c["indexer_head_dim"] != 128 or c["indexer_compress_ratio"] != 4:
        raise ValueError("this payload model covers the supplied 128-dim, 4:1 QSA indexer only")
    stages, start = [], 0
    for layers in partition:
        kinds = c["layer_types"][start:start + layers]
        gdn, qsa = kinds.count("linear_attention"), kinds.count("full_attention")
        ple_layers = sum(start <= n < start + layers for n in c["ple_layer_ids"])
        stages.append({"layers": layers, "gdn_layers": gdn, "qsa_layers": qsa,
                       "gdn_snapshot_bytes": gdn * (conv + recurrent),
                       "ple_snapshot_bytes": ple_layers * ple,
                       "qsa_host_bytes_per_token": qsa * host_per_layer_token,
                       "qsa_gpu_bytes_per_token": qsa * (2 + 128 * 2 // 4)})
        start += layers
    return {"tp": tp, "ssm_bytes": ssm_bytes, "mtp": 0,
            "gdn_conv_bytes": conv, "gdn_recurrent_bytes": recurrent,
            "gdn_state_bytes": conv + recurrent, "ple_state_bytes_replicated": ple,
            "qsa_indexer_bytes_per_token_per_layer": 64,
            "qsa_gpu_placeholder_bytes_per_token_per_layer": 2,
            "qsa_raw_ring_bytes_per_layer": {"without_positions": 1024, "with_mrope": 1120},
            "stages": stages}


def packed_layout(block_tokens=4048, partition=(26, 22), pp_aware_proposal=False):
    """Flash BF16-indexer/MTP0 specialization of kv_cache_utils.py:1983-2116.

    Global attention has 12 repeats of (2*B + 64*B) bytes. The repeating-pattern
    floor is 12; _approximate_gcd([12, 36, 1, 12], lower_bound=12) selects 12.
    GDN groups then split to fit anchor_bytes; PLE and rings each fit one group.
    Project the strided GDN grouping onto PP ranks before sizing their slab.
    pp_aware_proposal is a NEW grouping, not an existing engine switch.
    """
    positive("block_tokens", block_tokens)
    if block_tokens % 16 or sum(partition) != 48 or any(n <= 0 for n in partition):
        raise ValueError("requires a multiple-of-16 QSA block and a 48-layer partition")
    # The QSA offload selector sets mamba_page_size_padded to max(GDN, PLE),
    # so PLE's raw 184,320-byte state is charged a GDN-sized page too.
    gdn_bytes = ple_bytes = 1603584
    anchor = max(12 * 66 * block_tokens, gdn_bytes, ple_bytes, 1120)
    groups = max(3, math.ceil(36 / max(1, anchor // gdn_bytes)))
    gdn_layers = [i for i in range(48) if i % 4 != 3]
    offsets, offset = [], 0
    for layers in partition:
        offsets.append((offset, offset + layers))
        offset += layers
    if pp_aware_proposal:
        # Target one GDN state/rank/group. Rank projection makes opposite-stage
        # states compatible even if their combined GLOBAL bytes exceed anchor.
        groups = max(sum(lo <= n < hi for n in gdn_layers) for lo, hi in offsets)
    per_stage = []
    for lo, hi in offsets:
        counts = [sum(lo <= n < hi for n in gdn_layers[g::groups]) for g in range(groups)]
        qsa = sum(n % 4 == 3 for n in range(lo, hi))
        page = max(qsa * 66 * block_tokens, max(counts) * gdn_bytes,
                   ple_bytes if lo <= 2 < hi else 0, qsa * 1120)
        per_stage.append({"gdn_layers_per_group": counts, "qsa_layers": qsa, "page_bytes": page})
    return {"basis": "proposed_PP_aware_grouping" if pp_aware_proposal else "image_packing_algorithm",
            "block_tokens": block_tokens, "partition": list(partition), "anchor_bytes_global": anchor,
            "gdn_groups": groups, "state_groups": groups + 1, "history_groups": 1, "ring_groups": 1,
            "page_bytes": max(s["page_bytes"] for s in per_stage), "per_stage": per_stage,
            "runtime_manifest_verified": False}


@dataclass(frozen=True)
class Layout:
    state_groups: int
    pool_bytes: int = 2150000000
    page_bytes: int = 1603584
    block_tokens: int = 4048
    hash_tokens: int = 176
    history_groups: int = 1
    ring_groups: int = 1
    host_bytes_per_token_rank: int = 6144
    host_cap_gib: int = 33

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(name, value, zero=name in ("history_groups", "ring_groups"))
        if self.block_tokens % self.hash_tokens:
            raise ValueError("hash_tokens must divide block_tokens")
        if self.pool_blocks < 2:
            raise ValueError("pool must hold a null block and a usable block")

    @property
    def pool_blocks(self):
        return self.pool_bytes // self.page_bytes

    @property
    def usable_blocks(self):
        return self.pool_blocks - 1  # BlockPool reserves a shared null block.

    def history_blocks(self, tokens):
        return math.ceil(tokens / self.block_tokens) * self.history_groups

    def reserve(self, agents, history_tokens, working_copies):
        # Deliberately excludes cached resume vectors: those occupy the LRU.
        return agents * (self.history_blocks(history_tokens) + self.ring_groups +
                         working_copies * self.state_groups)

    def working_set(self, live_agents, active_requests, history_tokens, cached_vectors=2):
        """Separate active refs from cached idle histories; one private vector/active request."""
        positive("live_agents", live_agents)
        positive("active_requests", active_requests, zero=True)
        positive("history_tokens", history_tokens)
        positive("cached_vectors", cached_vectors, zero=True)
        if active_requests > live_agents:
            raise ValueError("active_requests cannot exceed live_agents")
        history = self.history_blocks(history_tokens)
        pinned = active_requests * (history + self.state_groups + self.ring_groups)
        idle_history = (live_agents - active_requests) * history
        available = self.usable_blocks - pinned - idle_history
        return {"live_agents": live_agents, "active_requests": active_requests,
                "history_tokens": history_tokens, "cached_vectors_per_agent": cached_vectors,
                "pinned_blocks": pinned, "reported_usage_fraction": pinned / self.usable_blocks,
                "idle_cached_history_blocks": idle_history,
                "cached_vectors_fit": max(0, available // self.state_groups),
                "headroom_blocks": available - cached_vectors * live_agents * self.state_groups,
                "all_checkpoint_vectors_fit_without_margin": self.usable_blocks // self.state_groups}

    def ledger(self):
        host_page = self.block_tokens * self.host_bytes_per_token_rank
        max_host_blocks = self.host_cap_gib * GIB // host_page
        return {**asdict(self), "layout_verified": False,
                "pool_blocks": self.pool_blocks, "usable_blocks": self.usable_blocks,
                "host_bytes_per_rank": self.pool_blocks * host_page,
                "host_cap_passes": self.pool_blocks <= max_host_blocks,
                "max_blocks_under_host_cap": max_host_blocks,
                "max_pool_bytes_at_fixed_page_size_under_host_cap": (max_host_blocks + 1) * self.page_bytes - 1,
                "one_checkpoint_pool_charge_bytes": self.state_groups * self.page_bytes,
                "working_set_examples": [self.working_set(n, active, history)
                                         for history in (117000, 150000) for n in (8, 10)
                                         for active in sorted({8, n})],
                "capacity_rows": [
                    {"history_tokens": n, "state_vectors_per_agent": copies,
                     "blocks_per_agent": per_agent,
                     "agents_fit_no_margin": self.usable_blocks // per_agent,
                     "required_bytes_8": (8 * per_agent + 1) * self.page_bytes,
                     "required_bytes_10": (10 * per_agent + 1) * self.page_bytes}
                    for n in (60000, 105000, 150000, 200000, 262144)
                    for copies in (1, 2, 3)
                    for per_agent in [self.history_blocks(n) + self.ring_groups + copies * self.state_groups]]}


class CheckpointLRU:
    """Individual group blocks: losing even one group invalidates that resume point.

    Fixed pinned blocks live OUTSIDE this cache. A lookup touches only a complete
    checkpoint. Publishing models blocks becoming evictable at a boundary; it is
    not a model of their CUDA ownership/copy lifetime.
    """
    def __init__(self, blocks, groups):
        positive("blocks", blocks, zero=True)
        positive("groups", groups)
        self.blocks, self.groups = blocks, groups
        self.lru = OrderedDict()
        self.evictions = 0

    def latest(self, session, limit):
        candidates = sorted({k[1] for k in self.lru if k[0] == session and k[1] <= limit}, reverse=True)
        for end in candidates:
            keys = [(session, end, g) for g in range(self.groups)]
            if all(k in self.lru for k in keys):
                for k in keys:
                    self.lru.move_to_end(k)
                return end
        return 0

    def publish(self, session, end):
        for group in range(self.groups):
            key = (session, end, group)
            self.lru[key] = None
            self.lru.move_to_end(key)
            while len(self.lru) > self.blocks:
                self.lru.popitem(last=False)
                self.evictions += 1

    def supersede(self, session, keep):
        # Hypothetical session-aware policy, NOT an existing vLLM option.
        for key in list(self.lru):
            if key[0] == session and key[1] != keep:
                self.lru.pop(key)

    def demote(self, session, end):
        for group in range(self.groups):
            key = (session, end, group)
            if key in self.lru:
                self.lru.move_to_end(key, last=False)


def replay_config(overrides=None):
    defaults = dict(AGENTS=8, SEED=2, START_MIN=60000, START_MAX=150000,
                    APPEND_MIN=500, APPEND_MAX=5000, REPLY_MIN=200, REPLY_MAX=1000,
                    TOOL_MIN=1.0, TOOL_MAX=5.0, ROLL_AT=200000)
    cfg = {**defaults, **(overrides or {})}
    for key in ("AGENTS", "START_MIN", "START_MAX", "APPEND_MIN", "APPEND_MAX", "REPLY_MIN", "REPLY_MAX", "ROLL_AT"):
        positive(key, cfg[key])
    positive("SEED", cfg["SEED"], zero=True)
    for lo, hi in (("START_MIN", "START_MAX"), ("APPEND_MIN", "APPEND_MAX"), ("REPLY_MIN", "REPLY_MAX"), ("TOOL_MIN", "TOOL_MAX")):
        if not (math.isfinite(cfg[lo]) and math.isfinite(cfg[hi]) and 0 <= cfg[lo] <= cfg[hi]):
            raise ValueError(f"invalid range {lo}..{hi}")
    return cfg


def consume_text_rng(rng, tokens):
    # Exactly agent_replay.text's RNG calls, without storing synthetic prose.
    for _ in range(max(1, int(tokens / 2.2))):
        rng.choice(WORDS)
        if rng.random() < 0.3:
            rng.randint(0, 9999)


def turns(cfg, rounds):
    positive("rounds", rounds)
    rngs = [random.Random(cfg["SEED"] * 1000 + i) for i in range(cfg["AGENTS"])]
    histories, sessions = [0] * cfg["AGENTS"], [0] * cfg["AGENTS"]
    for round_no in range(rounds):
        for agent, rng in enumerate(rngs):
            fresh = not histories[agent] or histories[agent] > cfg["ROLL_AT"]
            if fresh:
                sessions[agent] += 1
                n = rng.randint(cfg["START_MIN"], cfg["START_MAX"])
                prompt = n
            else:
                n = rng.randint(cfg["APPEND_MIN"], cfg["APPEND_MAX"])
                prompt = histories[agent] + n
            consume_text_rng(rng, n)
            reply = rng.randint(cfg["REPLY_MIN"], cfg["REPLY_MAX"])
            yield {"agent": agent, "session": (agent, sessions[agent]), "round": round_no,
                   "kind": "fresh" if fresh else "append", "prompt": prompt, "reply": reply,
                   "previous_history": 0 if fresh else histories[agent]}
            histories[agent] = prompt + reply
            rng.uniform(cfg["TOOL_MIN"], cfg["TOOL_MAX"])


def simulate(layout, records, pinned_blocks, policy="reachable", warmup_rounds=5, bootstrap_latest=False):
    positive("pinned_blocks", pinned_blocks, zero=True)
    positive("warmup_rounds", warmup_rounds, zero=True)
    if policy not in ("dense", "evict-first", "reachable", "latest"):
        raise ValueError("policy must be dense, evict-first, reachable or latest")
    if pinned_blocks >= layout.usable_blocks:
        return {"status": "INFEASIBLE_RESERVE", "predicted_append_hit_rate": None}
    cache = CheckpointLRU(layout.usable_blocks - pinned_blocks, layout.state_groups)
    prompt_sum = hit_sum = ideal_sum = count = misses = 0
    for r in records:
        # No common-prefix sharing across generated sessions. Hash equality,
        # retokenization and partially missing QSA histories are not simulated.
        session, prompt = tuple(r["session"]), r["prompt"]
        hit = cache.latest(session, prompt - 1)
        if r["kind"] == "append" and r["round"] >= warmup_rounds:
            prompt_sum += prompt
            hit_sum += hit
            ideal_sum += max(0, min(prompt - 1, r["previous_history"] - 1))
            count += 1
            misses += hit == 0
        processed = prompt + r["reply"] - 1
        ends = set(range((hit // layout.block_tokens + 1) * layout.block_tokens,
                         processed + 1, layout.block_tokens))
        if policy == "reachable":
            # Coordinator.get_replay_boundaries: prompt-1 at MTP0, then
            # MambaManager.reachable_block_mask(... retention_interval=0).
            reachable = ((prompt - 1) // layout.hash_tokens * layout.hash_tokens
                         // layout.block_tokens * layout.block_tokens)
            ends.intersection_update({reachable})
        tail = prompt // layout.hash_tokens * layout.hash_tokens
        if tail > hit:
            ends.add(tail)
        if (policy == "latest" or (bootstrap_latest and r["round"] == 0)) and ends:
            # Skip interior cache publications and retire superseded session
            # checkpoints AFTER the replacement is published. Working scratch
            # is already reserved and must still obey the real copy protocol.
            ends = {max(ends)}
        for end in sorted(ends):
            cache.publish(session, end)
            if (policy in ("evict-first", "reachable") and end == tail and tail % layout.block_tokens
                    and hit % layout.block_tokens and tail > hit):
                cache.demote(session, hit)
        if policy == "latest" and ends:
            cache.supersede(session, max(ends))
    return {"status": "CONDITIONAL_MODEL" if count else "NO_APPEND_SAMPLES",
            "policy": policy, "pinned_blocks": pinned_blocks,
            "bootstrap_latest": bootstrap_latest,
            "pinned_usage_fraction": pinned_blocks / layout.usable_blocks,
            "checkpoint_cache_blocks": cache.blocks, "checkpoint_vectors_fit": cache.blocks // layout.state_groups,
            "append_turns": count, "append_prompt_tokens": prompt_sum,
            "predicted_cached_tokens": hit_sum,
            "predicted_append_hit_rate": hit_sum / prompt_sum if prompt_sum else None,
            "ideal_prefix_hit_upper_bound": ideal_sum / prompt_sum if prompt_sum else None,
            "zero_hit_append_fraction": misses / count if count else None,
            "group_block_evictions": cache.evictions}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-groups", type=int, required=True, help="scheduler checkpoint group count; verify from planner")
    for name, default in (("pool-bytes", 2150000000), ("page-bytes", 1603584), ("block-tokens", 4048),
                          ("hash-tokens", 176), ("history-groups", 1), ("ring-groups", 1),
                          ("host-bytes-per-token-rank", 6144), ("host-cap-gib", 33)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--config", type=Path, help="agent_replay JSON (config key) or config object; turns are not replayed")
    p.add_argument("--agents", type=int)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--policy", choices=("dense", "evict-first", "reachable", "latest"), default="reachable")
    p.add_argument("--bootstrap-latest", action="store_true",
                   help="seed only newest checkpoints in round 0; a warm-cache sensitivity, not the real cold start")
    p.add_argument("--reserve-history", type=int, default=150000, help="fixed history size used for all pinned agent slots")
    p.add_argument("--working-copies", type=int, default=1, help="private running/COW vectors per agent, excluding cached resume")
    p.add_argument("--pinned-blocks", type=int, help="override analytical reserve, e.g. 536 for ~40%% of 1339")
    p.add_argument("--model-config", type=Path, default=Path(__file__).resolve().parents[1] / "reference/model-config.json")
    args = p.parse_args()
    try:
        layout = Layout(**{k: getattr(args, k) for k in Layout.__dataclass_fields__})
        cfg = replay_config({k: type(v)(os.environ[k]) for k, v in replay_config().items() if k in os.environ})
        if args.config:
            loaded = json.loads(args.config.read_text())
            cfg = replay_config({**cfg, **loaded.get("config", loaded)})
        if args.agents is not None:
            cfg = replay_config({**cfg, "AGENTS": args.agents})
        positive("reserve_history", args.reserve_history)
        positive("working_copies", args.working_copies, zero=True)
        positive("rounds", args.rounds)
        positive("warmup_rounds", args.warmup_rounds, zero=True)
        if args.warmup_rounds >= args.rounds:
            raise ValueError("warmup_rounds must be smaller than rounds")
        reserve = args.pinned_blocks if args.pinned_blocks is not None else layout.reserve(
            cfg["AGENTS"], args.reserve_history, args.working_copies)
        result = {"kind": "offline_sensitivity_model_not_engine_prediction", "layout": layout.ledger(),
                  "config": cfg, "rounds": args.rounds, "warmup_rounds": args.warmup_rounds,
                  "reserve_history": args.reserve_history, "working_copies": args.working_copies,
                  "prediction": simulate(layout, turns(cfg, args.rounds), reserve, args.policy,
                                         args.warmup_rounds, args.bootstrap_latest),
                  "source_packing_examples": [packed_layout(b, part, proposal)
                                              for b in (4048, 4064, 4224, 8096)
                                              for part in ((24, 24), (26, 22)) for proposal in (False, True)],
                  "limitations": ["Source packing is reconstructed; confirm the engine's actual layout/settings.",
                                  "Fixed pinned reserve assumes complete QSA history residency for live sessions.",
                                  "Round-robin transactions; no wall-clock scheduler or cold-prefill feedback.",
                                  "Token targets, not tokenizer counts; no template changes or prefix hash collisions.",
                                  "Does not model page-crossing checkpoint loss, CoW lifetime or copy scratch peaks.",
                                  "Evict-first models tail demotion only, not copy lifetimes; latest is a proposed policy.",
                                  "No exact reproduction of q5 and no prediction of TTFT or turns/h."]}
        if args.model_config.exists():
            model = json.loads(args.model_config.read_text())
            result["payload_24_24"] = payloads(model)
            result["payload_26_22"] = payloads(model, partition=(26, 22))
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
