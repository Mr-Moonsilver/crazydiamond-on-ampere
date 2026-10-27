"""CPU-only fixtures for the VM<vm-id> job12 sparse-selection qualification draft.

Nothing here imports a GPU library, opens the device or runs the queue.  The
grade fixture mirrors what ``vmid_job12.probe`` emits so the VM verdict can
be exercised offline; the queue is checked as text plus ``bash -n``.
"""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import arena_attention  # noqa: E402
import operation_source  # noqa: E402
import qsa_operation as qo  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import vmid_job12 as job  # noqa: E402
import vmid_qsa_dma as qd  # noqa: E402
import vmid_support as support  # noqa: E402

GENERATED_BYTES = qo.MAX_UNION_SCRATCH_BYTES - 1
CASE_INDEX = {case['name']: index for index, case in enumerate(job.CASES)}


def record_index(name, layer=0):
    return CASE_INDEX[name] * job.LAYERS + layer


def good():
    r = dict(
        schema='vm<vm-id>-job12/1', image='sha256:unused', source_sha256=qd.SOURCE_SHA,
        sparse_stage_sha256=operation_source.SPARSE_STAGE_SHA256,
        union_source_sha256=operation_source.UNION_SOURCE_SHA256,
        memory_fraction=.125, cases=[], cancellation=[], delayed=None,
        job4_regression=None, selection_bytes=job.SELECTION_BYTES,
        union_scratch_bytes=GENERATED_BYTES,
        geometry=dict(layers=job.LAYERS, query_rows=job.QUERY_ROWS,
                      query_heads=job.QUERY_HEADS, head_dim=job.HEAD_DIM,
                      topk=job.TOPK, page_rows=job.PAGE,
                      arena_pages=job.ARENA_PAGES,
                      piece_pages=shd.MAX_PIECE_PAGES,
                      descriptor_cap=shd.MAX_DESCRIPTORS,
                      capacity=job.CAPACITY, map_pages=job.MAP_PAGES),
        allocator_peak_allocated=400 * 1024 ** 2,
        allocator_peak_reserved=450 * 1024 ** 2,
        distinct_host_layers=True, cleanup_ok=True, failures=[], missing=[])
    for case in job.CASES:
        route = job.expected_route(case)
        plan = job.expected_plan(case)
        for layer in range(job.LAYERS):
            r['cases'].append(dict(
                name=case['name'], layer=layer, route=route,
                selection_count=len(job.selection_union(case)),
                selected_sorted=True if route == 'sparse_dma' else None,
                descriptors=plan.descriptors if route == 'sparse_dma' else None,
                buckets=job.expected_buckets(case), reference_calls=1,
                candidate_calls=1, bytes_exact=True, output_exact=True,
                output_finite=True, completed=True, hold_released=True,
                union_scratch_bytes=GENERATED_BYTES))
    r['delayed'] = dict(
        case='delayed_selection', layer=0, pending_selection_advances=1,
        first_advance_s=.002, selection_ready_s=1.4, event_pending=True,
        selection_count_read_refused=True, independent_submitted=True,
        independent_submit_s=.003, output_exact=True, completed=True)
    for layer in range(job.LAYERS):
        r['cancellation'].append(dict(
            layer=layer, planned_buckets=2, cancelled=True, bytes_exact=True,
            tail_poison=True, no_partial_publish=True, advance_refused=True,
            drain_ok=True, released=True, reuse_refused=True,
            refill_output_exact=True, refill_completed=True))
    r['job4_regression'] = dict(
        case='page_edges', reference_calls=1, candidate_calls=1, pieces=6,
        kernels=1, bytes_exact=True, tail_unchanged=True, completed=True,
        output_exact=True)
    return r


class SourcePinTests(unittest.TestCase):
    def test_fixed_capacity_kernels_are_pinned(self):
        path = ROOT / 'patches/sparse_stage/kernels.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         operation_source.SPARSE_STAGE_SHA256)
        text = path.read_text()
        for name in ('_s6_mark', '_s6_scan', '_s6_offsets', '_s6_pack',
                     '_s6_remap'):
            self.assertIn('def ' + name + '(', text)
        self.assertIn('_s6_gather', text)

    def test_generated_union_is_fixed_capacity_and_never_gathers(self):
        overlay = operation_source._union_overlay()
        self.assertEqual(hashlib.sha256(overlay.encode()).hexdigest(),
                         operation_source.UNION_SOURCE_SHA256)
        self.assertNotIn('_s6_gather', overlay)
        self.assertNotIn('torch.unique', overlay)
        # Buffers are the operation's explicit scratch, not shape-derived.
        self.assertIn('scratch.flags', overlay)
        self.assertIn('scratch.remapped', overlay)
        self.assertNotIn('torch.zeros((num_tokens,)', overlay)

    def test_real_chain_still_reproduces_the_pinned_union(self):
        source = (ROOT / 'patches/retain/ops_qsa.py.v8.1').read_text()
        e2 = arena_attention.generate(source)
        self.assertEqual(hashlib.sha256(e2.encode()).hexdigest(),
                         operation_source.E2_OPS_SHA256)
        generated = operation_source.generate_ops(e2)
        self.assertIn('flash_s5_sparse_arena_attention', generated)
        sparse = generated[generated.index('def flash_s5_sparse_arena_attention'):
                           generated.index('def _s6_mark')]
        self.assertNotIn('torch.unique', sparse)
        self.assertNotIn('unique[', sparse)
        self.assertIn('e3op.stage_sparse_selection', sparse)
        self.assertIn('while not e3op.sparse_selection_ready', sparse)
        self.assertIn('selected_count * 5 < capacity * 4', sparse)

    def test_operation_scratch_and_ticket_bounds_are_unchanged(self):
        self.assertEqual(qo.MAX_PAGES_PER_COPY, 4)
        self.assertEqual(qo.MAX_BUCKET_PAGES, 24)
        self.assertEqual(qo.MAX_OUTSTANDING_BUCKETS, 1)
        self.assertEqual(qo.MAX_LAYER_PAGES, 128)
        self.assertEqual(qo.MAX_UNION_SCRATCH_BYTES, 64 * 1024 ** 2)
        self.assertEqual(job.expected_route(
            dict(rows=(), mapping=job.IDENTITY)), 'empty')


class CaseMatrixTests(unittest.TestCase):
    def test_case_inventory_covers_the_requested_matrix(self):
        names = [case['name'] for case in job.CASES]
        self.assertEqual(names, [
            'all_padding', 'interleaved_padding', 'duplicate_rows',
            'page_edges', 'aliased_pages', 'threshold_sparse',
            'threshold_dense', 'arena_tail_cancel', 'delayed_selection'])

    def test_union_drops_padding_and_collapses_duplicates(self):
        self.assertEqual(job.selection_union(job.CASES[0]), ())
        self.assertEqual(
            job.selection_union(job.CASES[1]),
            (0, job.PAGE - 1, job.PAGE, job.PAGE + 1, 2 * job.PAGE - 1,
             2 * job.PAGE))
        self.assertEqual(job.selection_union(job.CASES[2]),
                         tuple(range(10)))
        for case in job.CASES:
            union = job.selection_union(case)
            self.assertEqual(union, tuple(sorted(set(union))))
            self.assertNotIn(-1, union)
            self.assertTrue(all(0 <= token < job.VALID_TOKENS
                                for token in union), case['name'])

    def test_threshold_is_the_pinned_integer_boundary(self):
        sparse = job.selection_union(job.CASES[5])
        dense = job.selection_union(job.CASES[6])
        self.assertEqual(len(sparse), 40 * job.PAGE - 1)
        self.assertEqual(len(dense), 40 * job.PAGE)
        self.assertEqual(job.expected_route(job.CASES[5]), 'sparse_dma')
        self.assertEqual(job.expected_route(job.CASES[6]), 'dense_dma')
        self.assertLess(len(sparse) * 5, job.CAPACITY * 4)
        self.assertEqual(len(dense) * 5, job.CAPACITY * 4)

    def test_real_plans_preserve_order_and_descriptor_bounds(self):
        for case in job.CASES:
            plan = job.expected_plan(case)
            self.assertEqual(plan.selected, job.selection_union(case))
            self.assertFalse(plan.refused, case['name'])
            if job.expected_route(case) == 'sparse_dma':
                self.assertEqual(len(plan.buckets), job.expected_buckets(case),
                                 case['name'])
            for bucket in plan.buckets:
                self.assertLessEqual(bucket.descriptors,
                                     shd.MAX_DESCRIPTORS)
                self.assertLessEqual(bucket.descriptors, 256)
                self.assertEqual(bucket.pieces[0].arena_row, 0)
                for piece in bucket.pieces:
                    self.assertLessEqual(piece.rows,
                                         shd.MAX_PIECE_PAGES * job.PAGE)
            if case['name'] in ('all_padding',):
                self.assertEqual(plan.buckets, ())

    def test_alias_and_page_edge_maps_are_the_real_fragmented_ones(self):
        edges = job.expected_plan(job.CASES[3])
        self.assertEqual(edges.selected,
                         (0, job.PAGE - 1, job.PAGE, job.PAGE + 1,
                          2 * job.PAGE - 1, 2 * job.PAGE,
                          3 * job.PAGE - 1))
        sources = [piece.source_row for piece in edges.buckets[0].pieces]
        self.assertEqual(sources, sorted(sources))
        aliased = job.CASES[4]['mapping']
        self.assertEqual(aliased[:2], (3, 3))
        plan = job.expected_plan(job.CASES[4])
        # Two logical pages alias one physical page; compact order is kept and
        # the duplicate physical row is copied twice, never deduplicated.
        self.assertEqual(sum(piece.rows for piece in plan.buckets[0].pieces),
                         len(plan.selected))

    def test_rows_matrix_pads_with_minus_one_and_carries_counts(self):
        flat, counts = job.rows_matrix(job.CASES[1]['rows'])
        self.assertEqual(len(flat), job.QUERY_ROWS * job.TOPK)
        self.assertEqual(len(counts), job.QUERY_ROWS)
        self.assertEqual(counts[:2], [3, 4])
        self.assertEqual(flat[:6], [0, -1, job.PAGE - 1, -1, job.PAGE, -1])
        self.assertEqual(flat[6:job.TOPK], [-1] * (job.TOPK - 6))
        second = flat[job.TOPK:2 * job.TOPK]
        self.assertEqual(second[:6],
                         [job.PAGE + 1, -1, 2 * job.PAGE - 1, 2 * job.PAGE,
                          -1, 0])
        self.assertEqual(second[6:], [-1] * (job.TOPK - 6))
        self.assertEqual(flat[2 * job.TOPK:], [-1] * (len(flat) - 2 * job.TOPK))
        self.assertEqual(counts[2:], [0] * (job.QUERY_ROWS - 2))
        empty, zero = job.rows_matrix(())
        self.assertEqual(set(empty), {-1})
        self.assertEqual(zero, [0] * job.QUERY_ROWS)

    def test_rows_matrix_refuses_bad_geometry(self):
        with self.assertRaises(ValueError):
            job.rows_matrix(((job.VALID_TOKENS,),))          # beyond frontier
        with self.assertRaises(ValueError):
            job.rows_matrix(((-2,),))                        # not padding
        with self.assertRaises(ValueError):
            job.rows_matrix(tuple(() for _ in range(job.QUERY_ROWS + 1)))
        with self.assertRaises(ValueError):
            job.rows_matrix(((0,) * (job.TOPK + 1),))
        with self.assertRaises(ValueError):
            job.rows_matrix(((True,),))
        with self.assertRaises(ValueError):
            job.chunks((2, 1))


class GradeTests(unittest.TestCase):
    def test_nonfinite_delay_timings_never_pass(self):
        for value in (float('nan'), float('inf'), -1):
            self.assertMissing(self.mutate(['delayed', 'first_advance_s'], value),
                               'delayed_first_advance_s')

    def test_good_record_passes(self):
        result = job.grade(good())
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['missing'], [])
        self.assertEqual(result['verdict'], 'PASS')

    def mutate(self, path, value):
        r = copy.deepcopy(good())
        cursor = r
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        return r

    def assertFailure(self, r, token):
        result = job.grade(r)
        self.assertEqual(result['verdict'], 'FAIL', result)
        self.assertIn(token, result['failures'], result)

    def assertMissing(self, r, token):
        result = job.grade(r)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE', result)
        self.assertIn(token, result['missing'], result)

    def test_pin_and_schema_mismatches_are_inconclusive(self):
        self.assertMissing(self.mutate(['schema'], 'vm<vm-id>-job4/1'), 'schema')
        self.assertMissing(self.mutate(['source_sha256'], '0' * 64),
                           'source_pin')
        self.assertMissing(self.mutate(['sparse_stage_sha256'], '0' * 64),
                           'sparse_stage_pin')
        self.assertMissing(self.mutate(['union_source_sha256'], '0' * 64),
                           'union_source_pin')
        self.assertFailure(self.mutate(['memory_fraction'], .25),
                           'memory_fraction')
        self.assertMissing(self.mutate(
            ['geometry', 'capacity'], job.CAPACITY + 1), 'geometry')

    def test_case_coverage_and_route_are_enforced(self):
        r = good()
        r['cases'] = r['cases'][:-1]
        self.assertMissing(r, 'case_layer_coverage')
        r = copy.deepcopy(good())
        r['cases'][0]['route'] = 'sparse_dma'
        self.assertFailure(r, 'all_padding:layer0:route')
        r = copy.deepcopy(good())
        r['cases'][0]['selected_sorted'] = True
        self.assertFailure(r, 'all_padding:layer0:dense_claims_plan')
        index = record_index('interleaved_padding')
        r = copy.deepcopy(good())
        r['cases'][index]['selected_sorted'] = False
        self.assertFailure(r, 'interleaved_padding:layer0:sorted_union')
        r = copy.deepcopy(good())
        r['cases'][index]['descriptors'] += 1
        self.assertFailure(r, 'interleaved_padding:layer0:descriptors')
        r = copy.deepcopy(good())
        r['cases'][index]['selection_count'] += 1
        self.assertFailure(r, 'interleaved_padding:layer0:selection_count')
        r = copy.deepcopy(good())
        r['cases'][index]['buckets'] += 1
        self.assertFailure(r, 'interleaved_padding:layer0:buckets')

    def test_per_case_values_must_be_proven_not_absent(self):
        index = record_index('interleaved_padding')
        for key in ('bytes_exact', 'output_exact', 'output_finite',
                    'completed', 'hold_released'):
            self.assertFailure(self.mutate(['cases', index, key], False),
                               'interleaved_padding:layer0:' + key)
            self.assertMissing(self.mutate(['cases', index, key], None),
                               'interleaved_padding:layer0:' + key)
        self.assertFailure(self.mutate(['cases', index, 'reference_calls'], 0),
                           'interleaved_padding:layer0:attention_calls')

    def test_delayed_selection_requires_a_real_pending_proof(self):
        self.assertMissing(self.mutate(['delayed'], None),
                           'delayed_selection')
        self.assertMissing(self.mutate(['delayed', 'case'], 'page_edges'),
                           'delayed_case')
        self.assertMissing(self.mutate(
            ['delayed', 'pending_selection_advances'], 0),
            'selection_pending_not_observed')
        self.assertFailure(self.mutate(['delayed', 'event_pending'], False),
                           'selection_event_not_pending')
        self.assertFailure(self.mutate(
            ['delayed', 'selection_count_read_refused'], False),
            'selection_count_read_early')
        self.assertFailure(self.mutate(
            ['delayed', 'independent_submitted'], False),
            'nonblocking_submit_missing')
        # A submit that only returns after the event is not nonblocking.
        self.assertFailure(self.mutate(['delayed', 'independent_submit_s'], .5),
                           'delayed_independent_submit_s_blocked')
        self.assertFailure(self.mutate(['delayed', 'first_advance_s'], .5),
                           'delayed_first_advance_s_blocked')
        # A selection that was already ready never exercised the pending path.
        self.assertFailure(self.mutate(['delayed', 'selection_ready_s'], .01),
                           'selection_not_delayed')
        self.assertMissing(self.mutate(['delayed', 'output_exact'], None),
                           'delayed_output_exact')

    def test_cancellation_requires_no_partial_publication(self):
        r = good()
        r['cancellation'] = r['cancellation'][:3]
        self.assertMissing(r, 'cancellation_layers')
        for key in ('cancelled', 'bytes_exact', 'tail_poison',
                    'no_partial_publish', 'advance_refused', 'drain_ok',
                    'released', 'reuse_refused', 'refill_output_exact',
                    'refill_completed'):
            self.assertFailure(self.mutate(['cancellation', 0, key], False),
                               'cancel0:' + key)
            self.assertMissing(self.mutate(['cancellation', 0, key], None),
                               'cancel0:' + key)
        self.assertFailure(self.mutate(['cancellation', 0, 'planned_buckets'],
                                       1), 'cancel0:planned_buckets')

    def test_job4_regression_is_required(self):
        self.assertMissing(self.mutate(['job4_regression'], None),
                           'job4_regression')
        self.assertMissing(self.mutate(['job4_regression', 'case'], 'empty'),
                           'job4_regression_case')
        self.assertFailure(self.mutate(
            ['job4_regression', 'output_exact'], False),
            'job4_regression:output_exact')
        self.assertFailure(self.mutate(
            ['job4_regression', 'candidate_calls'], 0),
            'job4_regression_calls')

    def test_bounds_and_cleanup(self):
        self.assertFailure(self.mutate(
            ['allocator_peak_allocated'], 2 * 1024 ** 3), 'allocator_growth')
        self.assertMissing(self.mutate(['allocator_peak_allocated'], None),
                           'allocator_peaks')
        self.assertMissing(self.mutate(['selection_bytes'],
                                       job.SELECTION_BYTES + 1),
                           'selection_scratch_bound')
        self.assertFailure(self.mutate(['union_scratch_bytes'],
                                       qo.MAX_UNION_SCRATCH_BYTES + 1),
                           'union_scratch_bound')
        self.assertMissing(self.mutate(['union_scratch_bytes'], 0),
                           'union_scratch_bytes')
        self.assertMissing(self.mutate(['distinct_host_layers'], False),
                           'independent_layers')
        self.assertFailure(self.mutate(['cleanup_ok'], False), 'cleanup')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'harness/runs/vmid_job12.sh'
        subprocess.run(['bash', '-n', str(self.path)], check=True)
        self.text = self.path.read_text()

    def test_output_directory_and_container_are_new(self):
        for token in ('Q=/opt/flashnext/runs/job12', 'flashnext-job12',
                      '.job.lock', 'flock -n 9',
                      '[[ ! -e $Q && ! -L $Q ]]'):
            self.assertIn(token, self.text)
        self.assertNotIn('runs/job4', self.text)
        self.assertNotIn('flashnext-job4', self.text)
        self.assertLess(self.text.index('[[ ! -e $Q'),
                        self.text.index('mkdir "$Q"'))

    def test_resource_bounds_and_fraction(self):
        for token in ('--gpus all', '--network none', '--cpus 4',
                      '--memory 4g', '--memory-swap 4g',
                      'timeout --signal=TERM --kill-after=15s 480s',
                      'VLLM_QSA_KVO_SPARSE_PREFILL=1', 'QSA_STAGE_TIMING=0',
                      'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'):
            self.assertIn(token, self.text)
        probe = (ROOT / 'harness/vmid_job12.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125, 0)', probe)
        self.assertNotIn('systemctl', self.text)
        self.assertNotIn('pkill', self.text)
        # Only this job's container may be stopped; never the engine or <gpu-tenant>.
        stopped = [line for line in self.text.splitlines()
                   if 'docker stop' in line]
        self.assertTrue(stopped)
        for line in stopped:
            self.assertIn('flashnext-job12', line)

    def test_monitor_starts_before_the_gpu_process(self):
        self.assertIn("'flashnext-job12'", (ROOT / 'harness/vmid_job12.py').read_text())
        self.assertLess(self.text.index('--mode monitor'),
                        self.text.index('docker run'))
        self.assertIn('touch "$Q/monitor.stop"', self.text)
        self.assertIn('vmid_support', (ROOT / 'harness/vmid_job12.py').read_text())

    def test_bundle_contains_every_reviewed_module(self):
        for token in ('vmid_job12.py', 'vmid_job4.py',
                      'vmid_support.py', 'vmid_qsa_dma.py',
                      'vmid_sparse_plan.py',
                      'vmid_sources/ops_qsa.py',
                      'vmid_sources/sparse_stage/kernels.py'):
            self.assertIn(token, self.text)
        for module in ('e3_types.py', 'sparse_history_dma.py',
                       'qsa_operation.py', 'operation_source.py',
                       'arena_attention.py', 'resident_attention.py'):
            self.assertIn(module, self.text)


class MonitorTests(unittest.TestCase):
    def test_job12_is_owned_and_long_running(self):
        self.assertIn('flashnext-job12', support.OWNED_CONTAINERS)
        self.assertIn('flashnext-job12', support.LONG_RUN_CONTAINERS)
        self.assertIn('flashnext-job4', support.OWNED_CONTAINERS)

    def test_unowned_container_is_refused_before_writing(self):
        with self.assertRaises(ValueError):
            support.monitor(Path('/nonexistent-job12-out'), 'flashnext-job13')
        with self.assertRaises(ValueError):
            support.monitor(Path('/nonexistent-job12-out'), 'flashnext-job1')
        with self.assertRaises(ValueError):
            support.monitor(Path('/nonexistent-job12-out'), 'vllm-flash')

    def test_probe_has_no_engine_path(self):
        source = (ROOT / 'harness/vmid_job12.py').read_text()
        for token in ('import vllm', 'import s5_segment_gpu', 'from vllm',
                      'distributed', 'HTTP', 'socket'):
            self.assertNotIn(token, source)
        self.assertIn('device.query', source)
        self.assertIn('sparse_selection_count', source)


class BundleLayoutTests(unittest.TestCase):
    def test_actual_bundle_supplies_both_generator_and_vm_union_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / 'bundle'
            subprocess.run([sys.executable, str(ROOT / 'tools/build_harness_bundle.py'),
                            str(bundle)], cwd=ROOT, capture_output=True, check=True)
            closure = json.loads((bundle / 'e3_build_closure.json').read_text())
            self.assertEqual(closure['external_source_pins']['sparse_stage/kernels.py'],
                             operation_source.SPARSE_STAGE_SHA256)
            for folder in ('e3_builder', 'vmid_sources/s5_engine'):
                code = ('import sys;sys.path.insert(0,sys.argv[1]);'
                        'import operation_source;'
                        'assert operation_source._union_overlay()')
                subprocess.run([sys.executable, '-I', '-c', code, str(bundle / folder)],
                               cwd=temp, capture_output=True, check=True)

    def test_union_overlay_resolves_in_repo_and_bundle_layouts(self):
        # The probe inserts vmid_sources/s5_engine when the bundle is staged;
        # operation_source then resolves sparse_stage/kernels.py one level up.
        self.assertTrue(operation_source._union_overlay())
        source = (ROOT / 'harness/vmid_job12.py').read_text()
        self.assertIn("HERE / 'vmid_sources' / 's5_engine'", source)
        self.assertIn("HERE / 'vmid_sources' / 'ops_qsa.py'", source)


class DelayedObservationTests(unittest.TestCase):
    def test_a_fast_retry_cannot_erase_a_blocked_first_submit(self):
        class Scratch:
            def zero_(self): pass
            def __getitem__(self, _): return self
            def copy_(self, *args, **kwargs): pass
        def refuse(_):
            raise qo.OperationError('selection_not_ready')
        scratch = Scratch()
        probe = SimpleNamespace(
            torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext())),
            device=SimpleNamespace(query=lambda _: False),
            scratch=scratch, scratch_pin=scratch, copy=object())
        operation = SimpleNamespace(
            _layers={'layer': SimpleNamespace(sparse=SimpleNamespace(
                selection=SimpleNamespace(event=object()), selection_token=object()))},
            sparse_selection_count=refuse, sparse_selection_ready=lambda _: False)
        stats = dict(pending_selection_advances=0, first_advance_s=None,
                     independent_submit_s=None, event_pending=False)
        with patch.object(job.time, 'perf_counter', side_effect=(10, 10.2, 20, 20.001)):
            job._delayed_checks(probe, operation, 'layer', .5, stats)
            job._delayed_checks(probe, operation, 'layer', .001, stats)
        self.assertEqual(stats['first_advance_s'], .5)
        self.assertAlmostEqual(stats['independent_submit_s'], .2)


if __name__ == '__main__':
    unittest.main()
