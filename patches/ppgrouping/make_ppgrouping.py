#!/usr/bin/env python3
"""Build the opt-in PP grouping + read-only pool telemetry overlay. CPU only.

Usage: make_ppgrouping.py reference/vllm-src OUTPUT
Pinned image cache utils/manager; overlay-v16 block_pool. Never patches inputs.
OUTPUT/core contains the three drop-in files and SHA256SUMS.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

BASES = {
    'kv_cache_utils.py': 'cc43b6a737d269483d1111bef1e61c1bcd0a91d8bd4d11ccf8acbc2523fa9b18',
    'block_pool.py': '7f781378710e616624f87d95ccc7b245e1e741c85f3751268e672c2a262038e8',
    'kv_cache_manager.py': '58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa',
}


def sub(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f'expected one anchor, found {s.count(old)}: {old[:80]}')
    return s.replace(old, new)


GROUP_HELPER = '''
# FLASH PP GROUPING: ownership comes from the engine's worker specs, not names.
def _flash_pp_group_count(spec, worker_layer_names):
    if not isinstance(spec.first_spec, MambaSpec) or spec.first_spec.tp_replicated:
        return None  # PLE and all attention/ring buckets retain stock grouping.
    if not worker_layer_names:
        raise ValueError("FLASH_PP_GROUPING requires actual worker ownership maps")
    names = list(spec.kv_cache_specs)
    owners = [set(names).intersection(worker) for worker in worker_layer_names]
    if set().union(*owners) != set(names):
        raise ValueError("FLASH_PP_GROUPING: incomplete worker ownership")
    count = max(map(len, owners))
    # Contiguous PP partitions with strided global layers have <=1 state/rank.
    # Fail closed for a different layer order/topology instead of widening pages.
    for i in range(count):
        stripe = set(names[i::count])
        if any(len(stripe.intersection(owner)) > 1 for owner in owners):
            raise ValueError("FLASH_PP_GROUPING: stripe has multiple states on a worker")
    return count


def _flash_log_group_layout(configs):
    # One record per worker, emitted by the planner after final pool equalization.
    for worker, config in enumerate(configs):
        groups = []
        for gid, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            first = spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            groups.append({"id": gid, "kind": type(first).__name__,
                           "layers": group.layer_names, "empty": not group.layer_names,
                           "local_bytes": sum(_get_per_layer_spec(group, n).page_size_bytes
                                              for n in group.layer_names),
                           "block_tokens": spec.block_size})
        logger.info("FLASH_PP_LAYOUT %s", _flash_json.dumps({
            "worker_index": worker, "enabled": _flash_os.environ.get("FLASH_PP_GROUPING") == "1",
            "page_bytes": _get_kv_cache_bytes_per_block(config.kv_cache_groups),
            "num_blocks": config.num_blocks, "groups": groups}, sort_keys=True))


'''


POOL_HELPERS = '''
    def _flash_count(self, group, counter, n=1):
        row = self._flash_pool_counts.setdefault(group, {
            "hits": 0, "misses": 0, "not_consulted": 0,
            "evictions": 0, "evicted_hashes": 0})
        row[counter] += n

    def _flash_pool_snapshot(self):
        free_cached = referenced_cached = free_observed = 0
        occupancy = {}
        for block in self.blocks:
            if block.is_null:
                continue
            free = block.ref_cnt == 0
            free_observed += free
            hashes = set(self.cached_block_hashes_by_block.get(block.block_id, ()))
            if block.block_hash is not None:
                hashes.add(block.block_hash)
            if hashes:
                free_cached += free
                referenced_cached += not free
            for gid in {get_group_id(h) for h in hashes}:
                row = occupancy.setdefault(gid, {"free_cached": 0, "referenced_cached": 0})
                row["free_cached" if free else "referenced_cached"] += 1
        free = self.get_num_free_blocks()
        return {"step": self._flash_pool_step, "usable": self.num_gpu_blocks - 1,
                "free": free, "free_cached": free_cached,
                "free_unhashed": free - free_cached,
                "referenced": self.num_gpu_blocks - 1 - free,
                "referenced_cached": referenced_cached,
                "free_count_consistent": free == free_observed,
                "groups": {gid: {**self._flash_pool_counts.get(gid, {
                    "hits": 0, "misses": 0, "not_consulted": 0,
                    "evictions": 0, "evicted_hashes": 0}), **occupancy.get(gid, {
                    "free_cached": 0, "referenced_cached": 0})}
                    for gid in sorted(set(self._flash_pool_counts) | set(occupancy))}}

    def flash_pool_step(self):
        # Scheduler calls, not allocation attempts or GPU forwards. Read-only.
        if not self._flash_pool_stats_every:
            return
        self._flash_pool_step += 1
        if self._flash_pool_step % self._flash_pool_stats_every == 0:
            logger.info("FLASH_POOL_STATS %s", _flash_json.dumps(
                self._flash_pool_snapshot(), sort_keys=True))

'''


def transform(name, s):
    if name == 'kv_cache_utils.py':
        s = sub(s, 'def _get_packed_kv_cache_groups(\n',
                'import os as _flash_os\nimport json as _flash_json\n\n' + GROUP_HELPER +
                'def _get_packed_kv_cache_groups(\n')
        for func, returns in [('_get_packed_kv_cache_groups', ' | None'), ('get_kv_cache_groups', '')]:
            old = f'def {func}(\n    vllm_config: VllmConfig,\n    kv_cache_spec: dict[str, KVCacheSpec],\n) -> list[KVCacheGroupSpec]{returns}:'
            s = sub(s, old, old.replace('\n) ->', '\n    *, worker_layer_names=None,\n) ->'))
        s = sub(s, '        if num_groups == 1:\n            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))',
                '        if _flash_os.environ.get("FLASH_PP_GROUPING") == "1":\n'
                '            pp_count = _flash_pp_group_count(spec, worker_layer_names)\n'
                '            if pp_count is not None:\n'
                '                num_groups = pp_count\n'
                '        if num_groups == 1:\n            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))')
        s = sub(s, '_get_packed_kv_cache_groups(vllm_config, filtered_spec):',
                '_get_packed_kv_cache_groups(\n        vllm_config, filtered_spec, worker_layer_names=worker_layer_names\n    ):')
        s = sub(s, 'global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)',
                'global_kv_cache_groups = get_kv_cache_groups(\n'
                '        vllm_config, merged_kv_cache_specs,\n'
                '        worker_layer_names=([set(worker) for worker in kv_cache_specs]\n'
                '                            if _flash_os.environ.get("FLASH_PP_GROUPING") == "1" else None),\n'
                '    )')
        s = sub(s, '    return kv_cache_configs\n',
                '    if (_flash_os.environ.get("FLASH_PP_GROUPING") == "1"\n'
                '            or int(_flash_os.environ.get("FLASH_POOL_STATS", "0")) > 0):\n'
                '        _flash_log_group_layout(kv_cache_configs)\n'
                '    return kv_cache_configs\n')
    elif name == 'block_pool.py':
        s = sub(s, 'logger = init_logger(__name__)\n',
                'import os as _flash_os\nimport json as _flash_json\n\nlogger = init_logger(__name__)\n')
        s = sub(s, '        self.num_gpu_blocks = num_gpu_blocks\n',
                '        self._flash_pool_stats_every = int(_flash_os.environ.get("FLASH_POOL_STATS", "0"))\n'
                '        if self._flash_pool_stats_every < 0:\n'
                '            raise ValueError("FLASH_POOL_STATS must be a nonnegative integer")\n'
                '        self._flash_pool_step = 0\n'
                '        self._flash_pool_counts = {}\n'
                '        self.num_gpu_blocks = num_gpu_blocks\n')
        s = sub(s, '    def get_cached_block(\n', POOL_HELPERS + '    def get_cached_block(\n')
        s = sub(s, '        for group_id in kv_cache_group_ids:\n',
                '        for index, group_id in enumerate(kv_cache_group_ids):\n')
        s = sub(s, '            if not block:\n                return None\n            cached_blocks.append(block)',
                '            if self._flash_pool_stats_every:\n'
                '                self._flash_count(group_id, "hits" if block else "misses")\n'
                '                if not block:\n'
                '                    for skipped in kv_cache_group_ids[index + 1:]:\n'
                '                        self._flash_count(skipped, "not_consulted")\n'
                '            if not block:\n                return None\n            cached_blocks.append(block)')
        s = sub(s, '        self._emit_block_removed_events(evicted_hashes)\n        return True',
                '        if self._flash_pool_stats_every:\n'
                '            for gid in {get_group_id(h) for h in evicted_hashes}:\n'
                '                self._flash_count(gid, "evictions")\n'
                '            for h in evicted_hashes:\n'
                '                self._flash_count(get_group_id(h), "evicted_hashes")\n'
                '        self._emit_block_removed_events(evicted_hashes)\n        return True')
    else:
        s = sub(s, '        self.coordinator.new_step_starts()\n',
                '        self.block_pool.flash_pool_step()\n        self.coordinator.new_step_starts()\n')
    ast.parse(s)
    return s


def build(source, output):
    originals = {}
    for name, expected in BASES.items():
        raw = (source / 'v1/core' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{name}: wrong source checksum; no output written')
        originals[name] = raw.decode()
    result = {name: transform(name, text) for name, text in originals.items()}
    core = output / 'core'
    core.mkdir(parents=True, exist_ok=True)
    for name, text in result.items():
        (core / name).write_text(text)
    hashes = {name: hashlib.sha256(s.encode()).hexdigest() for name, s in result.items()}
    (core / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sorted(hashes.items())))
    (output / 'provenance.json').write_text(json.dumps({'base_sha256': BASES, 'output_sha256': hashes}, indent=2) + '\n')
    (output / 'changes.diff').write_text(''.join(''.join(difflib.unified_diff(
        originals[n].splitlines(True), s.splitlines(True), fromfile='base/' + n, tofile='core/' + n, n=0))
        for n, s in result.items()))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    build(a.source, a.output)
