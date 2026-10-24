#!/usr/bin/env python3
"""Run the actual patched planner/pool code on CPU metadata; no GPU or torch."""
import ast
import dataclasses
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_prefix_capacity import source_packing
from prefix_warm_cold import CONFIG, grade
from pp_stats_grade import grade as grade_stats

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'patches/ppgrouping/overlay/core'
REF = ROOT / 'reference/vllm-src'
sys.path.insert(0, str(ROOT / 'patches/ppgrouping'))
import make_ppgrouping as generator


def definitions(path, names, ns):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(body) == len(names), names - {n.name for n in body}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)


def pool_class(patched):
    logs = []
    ns = dict(__name__=__name__, dataclass=dataclasses.dataclass, field=dataclasses.field,
              BlockHash=bytes, BlockHashWithGroupId=bytes, MEDIUM_GPU='GPU',
              _flash_os=os, _flash_json=json,
              logger=SimpleNamespace(info=lambda *args: logs.append(args)))
    definitions(REF / 'v1/core/kv_cache_utils.py',
                {'KVCacheBlock', 'FreeKVCacheBlockQueue', 'make_block_hash_with_group_id',
                 'get_group_id', 'get_block_hash'}, ns)
    definitions((CORE if patched else REF / 'v1/core') / 'block_pool.py',
                {'BlockHashToBlockMap', 'BlockPool'}, ns)
    return ns, logs


class PPGroupingTests(unittest.TestCase):
    def test_generated_files_match_pinned_sources(self):
        for name, digest in generator.BASES.items():
            original = (REF / 'v1/core' / name).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), digest)
            self.assertEqual(generator.transform(name, original.decode()), (CORE / name).read_text())

    def test_off_is_stock_and_on_projects_all_four_workers(self):
        for part, count in (((24, 24), 18), ((26, 22), 20)):
            stock = source_packing(4048, part, details=True)
            off = source_packing(4048, part, CORE / 'kv_cache_utils.py', '0', True)
            signature = lambda groups: [(g.layer_names, [(n, s.page_size_bytes, s.tp_replicated)
                         for n, s in g.kv_cache_spec.kv_cache_specs.items()]) for g in groups]
            self.assertEqual(signature(stock[2]), signature(off[2]))
            ns, specs, groups, workers, pages = source_packing(4048, part, CORE / 'kv_cache_utils.py', '1', True)
            self.assertEqual(len(groups), count + 3)  # GDN + PLE + history + ring
            self.assertEqual(pages, [1603584, 1603584])
            self.assertEqual(2150000000 // max(pages), 1340)
            self.assertEqual(sorted(n for g in groups for n in g.layer_names), sorted(specs))
            for wi, worker in enumerate(workers):
                local = ns['_project_kv_cache_groups_to_worker'](groups, worker)
                self.assertEqual(len(local), len(groups))
                self.assertEqual(sorted(n for g in local for n in g.layer_names), sorted(worker))
                self.assertEqual(ns['_get_kv_cache_bytes_per_block'](local), 1603584)
                for gid, group in enumerate(local):
                    self.assertEqual(group.layer_names, [n for n in groups[gid].layer_names if n in worker])
                    if any(n.endswith('.gdn') for n in group.layer_names):
                        self.assertEqual(len(group.layer_names), 1)
                expected_empty = 0 if wi < 2 else (1 if part == (24, 24) else 5)
                self.assertEqual(sum(not g.layer_names for g in local), expected_empty)

    def test_only_exact_one_enables(self):
        for value in ('', '0', 'true', '2'):
            self.assertEqual(source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', value)[0], 37)

    def test_top_level_planner_threads_actual_worker_ownership(self):
        ns, _, _, workers, _ = source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', '1', True)
        definitions(CORE / 'kv_cache_utils.py', {'get_kv_cache_groups', 'get_kv_cache_configs'}, ns)
        ns.update(is_kv_cache_type_attention_free=lambda s: False,
                  get_hisparse_kv_cache_groups=lambda *a: None,
                  is_kv_cache_spec_uniform=lambda s: False,
                  _get_kv_cache_groups_glm5_next=lambda *a: None,
                  HiddenStateCacheSpec=type('UnusedHidden', (), {}),
                  SlidingWindowSpec=type('UnusedWindow', (), {}),
                  KVCacheSpecRegistry=SimpleNamespace(check_kv_cache_spec_registry=lambda s: None),
                  partial=partial, _pool_bytes_per_block=ns['_get_kv_cache_bytes_per_block'],
                  _check_enough_kv_cache_memory=lambda *a: None,
                  _max_memory_usage_bytes_from_groups=lambda *a: None,
                  _estimate_max_model_len_from_groups=lambda *a: None,
                  get_kv_cache_config_from_groups=lambda cfg, groups, memory: SimpleNamespace(
                      num_blocks=memory // ns['_get_kv_cache_bytes_per_block'](groups), kv_cache_groups=groups))
        cfg = SimpleNamespace(scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            speculative_config=None, attention_config=SimpleNamespace(hisparse_config=None),
            model_config=SimpleNamespace(original_max_model_len=262144, max_model_len=262144),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None,
                get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_block_outermost=True)))
        result = ns['get_kv_cache_configs'](cfg, workers, [2150000000] * 4)
        self.assertEqual([len(c.kv_cache_groups) for c in result], [23] * 4)
        self.assertEqual([c.num_blocks for c in result], [1340] * 4)

    def test_bad_ownership_fails_without_silent_page_growth(self):
        ns, _, groups, workers, _ = source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', '1', True)
        spec = next(g.kv_cache_spec for g in groups if any(n.endswith('.gdn') for n in g.layer_names))
        with self.assertRaises(ValueError):
            ns['_flash_pp_group_count'](spec, None)
        with self.assertRaises(ValueError):
            ns['_flash_pp_group_count'](spec, [set()])
        # Four names, two per worker, but stripes overlap a worker twice.
        spec = ns['UniformTypeKVCacheSpecs'](4048, {str(i): spec.first_spec for i in range(4)})
        with self.assertRaises(ValueError):
            ns['_flash_pp_group_count'](spec, [{'0', '2'}, {'1', '3'}])

    def test_mamba_consumer_preserves_global_ids_and_skips_empty_groups(self):
        ns, _, groups, workers, _ = source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', '1', True)
        definitions(REF / 'v1/worker/mamba_utils.py',
                    {'_get_mamba_spec_for_layer', 'get_mamba_groups', 'get_mamba_group_ids'}, ns)
        for worker in workers:
            local = ns['_project_kv_cache_groups_to_worker'](groups, worker)
            ids = ns['get_mamba_group_ids'](ns['get_mamba_groups'](SimpleNamespace(kv_cache_groups=local)))
            expected = [i for i, g in enumerate(local) if g.layer_names and
                        isinstance(g.kv_cache_spec.first_spec, ns['MambaSpec'])]
            self.assertEqual(ids, expected)

    def test_worker_layout_log_includes_empty_slots_and_pool(self):
        ns, _, groups, workers, _ = source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', '1', True)
        logs = []
        ns['logger'] = SimpleNamespace(info=lambda fmt, row: logs.append(json.loads(row)))
        ns['_flash_log_group_layout']([SimpleNamespace(num_blocks=1340,
            kv_cache_groups=ns['_project_kv_cache_groups_to_worker'](groups, w)) for w in workers])
        self.assertEqual([r['worker_index'] for r in logs], list(range(4)))
        self.assertEqual([r['page_bytes'] for r in logs], [1603584] * 4)
        self.assertEqual(sum(g['empty'] for g in logs[-1]['groups']), 5)

    def test_pool_evidence_gate_rejects_wrong_or_missing_layouts(self):
        for enabled in (False, True):
            ns, _, groups, workers, _ = source_packing(4048, (26, 22), CORE / 'kv_cache_utils.py', str(int(enabled)), True)
            lines = []
            ns['logger'] = SimpleNamespace(info=lambda fmt, row: lines.append('2026-10-24T22:00:00Z ' + fmt % row))
            ns['_flash_log_group_layout']([SimpleNamespace(num_blocks=1340,
                kv_cache_groups=ns['_project_kv_cache_groups_to_worker'](groups, w)) for w in workers])
            sample = {'step': 200, 'usable': 1339, 'free': 803, 'free_cached': 600, 'free_unhashed': 203,
                      'referenced': 536, 'referenced_cached': 0, 'free_count_consistent': True,
                      'groups': {'0': dict(hits=4, misses=3, not_consulted=0, evictions=2, evicted_hashes=2)}}
            lines.append('FLASH_POOL_STATS ' + json.dumps(sample))
            text = '\n'.join(lines)
            self.assertEqual(grade_stats(text, enabled)['verdict'], 'PASS')
            for bad in ('\n'.join(lines[:-1]), '\n'.join(lines[1:]), text + '\n' + lines[-1],
                        text.replace('1603584', '3207168'), text.replace('"free": 803', '"free": 804')):
                self.assertEqual(grade_stats(bad, enabled)['verdict'], 'FAIL')


class PoolStatsTests(unittest.TestCase):
    def make(self, stats='2', patched=True):
        ns, logs = pool_class(patched)
        with patch.dict(os.environ, {'FLASH_POOL_STATS': stats}):
            pool = ns['BlockPool'](12, True, 176)
        return pool, logs, ns

    def seed(self, pool, ns, block, group, suffix=b'private-content-hash'):
        pool._insert_block_hash(ns['make_block_hash_with_group_id'](suffix, group), block, num_tokens=176)

    def state(self, pool):
        return ([(b.block_id, b.ref_cnt, b.block_hash, b.block_hash_num_tokens) for b in pool.blocks],
                [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
                dict(pool.cached_block_hashes_by_block), set(pool._evict_first), pool.num_demoted)

    def test_counts_and_first_missing_group_not_suffix_misses(self):
        p, logs, ns = self.make()
        a, b = p.get_new_blocks(2)
        self.seed(p, ns, a, 0)
        self.seed(p, ns, b, 2)
        p.free_blocks([a, b])
        self.assertIsNone(p.get_cached_block(b'private-content-hash', [0, 1, 2]))
        row = p._flash_pool_snapshot()
        self.assertEqual((row['free'], row['free_cached'], row['referenced']), (11, 2, 0))
        self.assertEqual(row['groups'][0]['hits'], 1)
        self.assertEqual(row['groups'][1]['misses'], 1)
        self.assertEqual(row['groups'][2]['not_consulted'], 1)
        self.assertEqual(row['groups'][2]['misses'], 0)
        before = self.state(p)
        p.flash_pool_step()
        self.assertFalse(logs)
        p.flash_pool_step()
        self.assertEqual(self.state(p), before)
        self.assertEqual(json.loads(logs[0][1])['step'], 2)
        self.assertNotIn('private-content-hash', repr(logs))

    def test_aliases_count_one_evicted_block_and_two_hashes(self):
        p, _, ns = self.make()
        a = p.get_new_blocks(1)[0]
        self.seed(p, ns, a, 3)
        self.seed(p, ns, a, 3, b'another')
        self.assertEqual(p._flash_pool_snapshot()['referenced_cached'], 1)
        p.free_blocks([a])
        p.get_new_blocks(11)
        row = p._flash_pool_snapshot()['groups'][3]
        self.assertEqual((row['evictions'], row['evicted_hashes']), (1, 2))
        self.assertEqual(row['free_cached'], 0)

    def test_instrumentation_preserves_allocation_touch_demote_and_reuse(self):
        states = []
        for patched, stats in ((False, '0'), (True, '0'), (True, '1')):
            p, _, ns = self.make(stats, patched)
            trace = []
            for cycle in range(12):
                a, b = p.get_new_blocks(2)
                self.seed(p, ns, a, cycle % 4)
                p.free_blocks([a, b])
                got = p.get_cached_block(b'private-content-hash', [cycle % 4])
                p.touch(got)
                p.demote(got[0])
                p.free_blocks(got)
                if patched:
                    p.flash_pool_step()
                    p._flash_pool_snapshot()
                trace.append(self.state(p))
            states.append(trace)
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0], states[2])

    def test_step_hook_runs_once_then_coordinator(self):
        tree = ast.parse((CORE / 'kv_cache_manager.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KVCacheManager')
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'new_step_starts')
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[])), 'step', 'exec'), ns)
        calls = []
        ns['new_step_starts'](SimpleNamespace(block_pool=SimpleNamespace(flash_pool_step=lambda: calls.append('pool')),
                            coordinator=SimpleNamespace(new_step_starts=lambda: calls.append('coordinator'))))
        self.assertEqual(calls, ['pool', 'coordinator'])

    def test_disabled_and_invalid_stats(self):
        p, logs, ns = self.make('0')
        a = p.get_new_blocks(1)[0]
        self.seed(p, ns, a, 1)
        p.get_cached_block(b'private-content-hash', [1])
        p.flash_pool_step()
        self.assertEqual(p._flash_pool_counts, {})
        self.assertFalse(logs)
        for bad in ('-1', 'oops', '1.5'):
            with self.assertRaises(ValueError):
                self.make(bad)


class WarmColdTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i in range(8):
            row = {'index': i, 'prompt_sha256': 'a' * 64}
            for name in ('cold_a', 'cold_b', 'seed', 'warm'):
                row[name] = {'top': {str(j): -float(j) for j in range(20)}, 'first': '0',
                             'prompt_tokens': 120000 if name == 'seed' else 122048,
                             'cached': 119856 if name == 'warm' else 0,
                             't0': 3 if name == 'warm' else 1, 't1': 4 if name == 'warm' else 2}
            rows.append(row)
        return {'config': dict(CONFIG), 'rows': rows}

    def test_warm_cold_pass_requires_full_evidence(self):
        good = self.fixture()
        self.assertEqual(grade(good)['verdict'], 'PASS')
        for mutate in (lambda d: d['rows'].clear(),
                       lambda d: d['rows'][0]['warm'].update(cached=0),
                       lambda d: d['rows'][0]['cold_a'].update(cached=1),
                       lambda d: d['rows'][0]['warm'].update(cached=None),
                       lambda d: d['rows'][0]['warm'].update(t0=0),
                       lambda d: d['rows'][0]['warm']['top'].update({'0': float('nan')})):
            bad = copy.deepcopy(good)
            mutate(bad)
            self.assertEqual(grade(bad)['verdict'], 'INCONCLUSIVE')

    def test_warm_corruption_fails_and_noisy_reference_is_inconclusive(self):
        d = self.fixture()
        d['rows'][0]['warm']['top']['0'] -= .6
        self.assertEqual(grade(d)['verdict'], 'FAIL')
        d['rows'][0]['cold_b']['top']['0'] -= .6
        self.assertEqual(grade(d)['verdict'], 'INCONCLUSIVE')

    def test_baseline_requires_identical_tokens_and_regraded_results(self):
        baseline = self.fixture()
        candidate = self.fixture()
        self.assertEqual(grade(candidate, baseline)['verdict'], 'PASS')
        baseline['result'] = {'verdict': 'PASS'}
        baseline['rows'][0]['prompt_sha256'] = 'b' * 64
        self.assertEqual(grade(candidate, baseline)['verdict'], 'INCONCLUSIVE')
        baseline['rows'].clear()
        self.assertEqual(grade(candidate, baseline)['verdict'], 'INCONCLUSIVE')
        # An unrelated incomplete baseline must not conceal a proven warm/cold failure.
        candidate['rows'][0]['warm']['top']['0'] -= .6
        self.assertEqual(grade(candidate, baseline)['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
