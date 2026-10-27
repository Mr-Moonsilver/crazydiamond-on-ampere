#!/usr/bin/env python3
"""CPU acceptance for vmid_job14 (native-source VM14 qualification).

Two kinds of evidence, deliberately kept apart:

* ``NativeSourceObjectTests`` drive the **real** ``NativeHostSourceAllocator``,
  ``LocalSourceRegistry``, ``SparseSourceHold`` and ``bind_local_sources``
  through the same fake tensors the sibling native-source tests use.  They are
  structural refusals, not GPU facts: no CUDA, engine or DMA is exercised.
* ``GradeTests`` feed hand-built records to the pure :func:`vmid_job14.grade`
  and require it to reject mutated/missing raw byte, lifetime, epoch, cancel
  and refusal evidence.  A green fixture proves the rule set is satisfiable;
  it is not native evidence and does not run the probe.

No tensor, stream, event or engine is constructed here.
"""
import copy
import ast
import subprocess
import tempfile
import types
from unittest.mock import patch
import sys
import unittest
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (str(HERE.parent / 'patches' / 's5_engine'), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import e3_types as et  # noqa: E402
import native_source as ns  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import test_e3_native_source_oracle as sibling  # noqa: E402
import vmid_job14 as job14  # noqa: E402


def lane(epoch=3):
    return et.LaneId(worker_rank=0, role=et.LaneRole.PREFILL,
                     storage_epoch=epoch, context_id=0)


def transaction(seq=1, chunk_end=4):
    return et.TransactionId(engine_epoch=1, request_uid='vm<vm-id>-job14-test',
                            request_generation=0, chunk_start=0,
                            chunk_end=chunk_end, txn_seq=seq)


def admission(state_epoch=2, seq=1, pages=(3,)):
    return sibling.make_admission(state_epoch=state_epoch, chunk_end=4,
                                  block_tables=(tuple(pages),),
                                  transaction=transaction(seq))


def bind(dispatcher, record):
    return ns.bind_local_sources(dispatcher, sibling.chunk_for(record),
                                 sibling.make_descriptor(record), rank=0)


def close(dispatcher, binding):
    for opened in binding.opened:
        if not opened.hold.released:
            opened.hold.release()
    binding.close(dispatcher.source_registry)


class NativeSourceObjectTests(unittest.TestCase):
    """The actual allocator/registry/hold objects, through the real binder."""

    def setUp(self):
        self.runner = sibling.SourceRunner(('qsa.0',), pool=8)
        self.dispatcher = sibling._Dispatcher(lane(), self.runner)
        self.first = admission()
        self.binding = bind(self.dispatcher, self.first)
        self.registry = self.dispatcher.source_registry
        self.allocator = self.dispatcher.source_allocator
        self.name = 'qsa.0'
        self.hold = self.binding.holds[self.name]

    def tearDown(self):
        close(self.dispatcher, self.binding)

    def test_binding_uses_the_real_objects_not_a_readiness_flag(self):
        self.assertIs(type(self.allocator), ns.NativeHostSourceAllocator)
        self.assertIs(type(self.hold), shd.SparseSourceHold)
        self.assertNotIsInstance(self.allocator, type(job14.job12c.ProbeAllocator()))
        self.assertIsNot(type(self.hold.reference), bool)
        self.assertEqual(self.allocator.inventory()['live'], 1)
        self.assertEqual(self.registry.lane, lane())

    def test_active_hold_refuses_rebind_and_names_the_open_lease(self):
        newer = admission(state_epoch=3, seq=2)
        with self.assertRaises(shd.SourceRefused) as caught:
            ns.bind_local_sources(
                self.dispatcher, sibling.chunk_for(newer),
                sibling.make_descriptor(newer), rank=0)
        self.assertIn('lease is open', str(caught.exception))

    def test_stale_descriptor_names_the_state_epoch(self):
        opened = self.binding.opened[0]
        group = self.binding.group_rows[0][1]
        with self.assertRaises(shd.SourceRefused) as caught:
            self.registry.open(
                replace(opened.descriptor, state_epoch=3), self.first,
                layer=self.name, group=group, allocator=self.allocator,
                registration=self.binding.registrations[self.name])
        self.assertIn('state epoch differs', str(caught.exception))

    def test_foreign_request_names_the_transaction(self):
        opened = self.binding.opened[0]
        group = self.binding.group_rows[0][1]
        foreign = transaction(seq=9)
        with self.assertRaises(shd.SourceRefused) as caught:
            self.registry.open(
                replace(opened.descriptor, transaction=foreign), self.first,
                layer=self.name, group=group, allocator=self.allocator,
                registration=self.binding.registrations[self.name])
        self.assertIn('transaction differs', str(caught.exception))

    def test_changed_physical_page_names_the_source_map(self):
        other = 7 if self.hold.owner.physical_pages[0] != 7 else 5
        with self.assertRaises(shd.SourceRefused) as caught:
            self.hold.validate(self.hold.owner, self.name, (other,))
        self.assertIn('physical source map differs', str(caught.exception))

    def test_premature_release_names_the_released_hold_and_reference(self):
        reference = self.hold.reference
        self.hold.release()
        with self.assertRaises(shd.SourceRefused) as caught:
            self.hold.validate(self.hold.owner, self.name,
                               self.hold.owner.physical_pages)
        self.assertIn('already released', str(caught.exception))
        with self.assertRaises(Exception):
            self.allocator.assert_retained(reference)
        self.assertEqual(self.allocator.inventory()['live'], 0)

    def test_two_epochs_keep_registration_and_advance_the_mapping_version(self):
        registry = self.registry
        first_registration = self.binding.registrations[self.name]
        first_epochs = dict(self.binding.source_epochs)
        first_allocations = self.allocator.inventory()['allocations']
        close(self.dispatcher, self.binding)
        # The same request/layers re-register at a newer chunk/storage epoch.
        self.dispatcher.operation.lane = lane(4)
        second = admission(state_epoch=3, seq=2)
        binding = bind(self.dispatcher, second)
        try:
            registration = binding.registrations[self.name]
            self.assertIs(self.dispatcher.source_registry, registry)
            self.assertEqual(registration.source_epoch,
                             first_registration.source_epoch)
            self.assertEqual(binding.source_epochs, first_epochs)
            self.assertGreater(registration.mapping_version,
                               first_registration.mapping_version)
            self.assertEqual(
                self.allocator.inventory()['allocations'], first_allocations)
            self.assertIsNot(binding.holds[self.name], self.hold)
        finally:
            close(self.dispatcher, binding)


# --- grading -----------------------------------------------------------------

def _digest(char):
    return char * 64


def _identity(case, layer):
    return dict(rank=0, worker_incarnation='vm<vm-id>-job14-worker',
                engine_epoch=1, route_generation=1, storage_epoch=2,
                source_epoch=1, state_epoch=2, mapping_version=1,
                layer='qsa.%d' % layer, operation='')


def green_record():
    pins = job14.expected_pins()
    cases = []
    for case in job14.CASES:
        layers = []
        for index in range(job14.LAYERS):
            route = job14.ROUTES[case]
            buckets = job14.expected_buckets(case)
            fills = job14.expected_fills(case)
            plan = job14.expected_plan(case)
            records = []
            for bucket in range(buckets):
                identity = _identity(case, index)
                identity['operation'] = (
                    'dense_page_bucket' if route == 'dense_dma'
                    else 'sparse_row_bucket')
                records.append(dict(
                    identity=identity, bucket=bucket, route=route,
                    rows=job14.PAGE, pieces=1,
                    compared_bytes=job14.PAGE * job14.ns.ROW_WIDTH * 2,
                    different_bytes=None, equal=True, finite=None,
                    digest=_digest('a')))
            output = _identity(case, index)
            output['operation'] = 'qsa_output_before_moe'
            records.append(dict(
                identity=output, bucket=-1, route=route, rows=job14.TOKENS,
                pieces=fills, compared_bytes=job14.TOKENS * 2,
                different_bytes=None, equal=True, finite=True,
                digest=_digest('b')))
            layers.append(dict(
                layer=index, route=route,
                selection_count=job14.SELECTION_COUNT[case],
                buckets=buckets,
                descriptors=(plan.descriptors if plan is not None else None),
                expected_fills=fills, reference_calls=1, candidate_calls=1,
                row1_records=records,
                row1_layers=[dict(passed=True, aborted=None)],
                row1_failures=[], row1_bucket_checks=buckets,
                row1_output_checks=1, arena_bytes_exact=True,
                output_exact=True, output_finite=True, completed=True,
                hold_released=True, allocator_live=0, uva_aliased=True,
                consumer_queried=buckets,
                probe_allocator_used=False))
        cases.append(dict(
            name=case, epoch=2, txn_seq=1, layers=layers,
            route=job14.ROUTES[case], locks=job14.LAYERS,
            holds_released=True, source_closed=True, registry_active=0,
            allocator_live=0, allocator_registrations=job14.LAYERS,
            allocator_allocations=job14.LAYERS, probe_allocator_used=False,
            distinct_host_layers=True, same_tensor_uva=True, uva_aliased=True))
    return dict(
        schema=job14.SCHEMA,
        source_sha256=job14.job12c.qd.SOURCE_SHA,
        layer_source_sha256=pins['layer_source'],
        sparse_stage_sha256=job14.operation_source.SPARSE_STAGE_SHA256,
        union_source_sha256=job14.operation_source.UNION_SOURCE_SHA256,
        generated_ops=dict(source_file='helpers/ops_qsa_native.py',
                           source_sha256=pins['ops']),
        generated_layer=dict(source_file='helpers/qsa_native.py',
                             source_sha256=pins['layer']),
        memory_fraction=job14.MEMORY_FRACTION, geometry=job14.geometry(),
        allocator='NativeHostSourceAllocator', probe_allocator_used=False,
        uva_aliased=True, distinct_host_layers=True, cases=cases,
        epochs=dict(
            request_same=True, contiguous_chunks=True, generation_reset=False,
            chunk_epochs=({}, {}),
            active_hold_rebind_refused=True,
            active_hold_rebind_cause=(
                'source lane changed while a local lease is open'),
            registry_same=True, lane_advanced=True,
            mapping_version_advanced=True, source_epoch_stable=True,
            allocation_retained=True, first_completed=True,
            first_output_exact=True, second_completed=True,
            second_output_exact=True, source_closed=True),
        cancellation=dict(
            name='cancel_partial_dense', planned_buckets=3,
            started_buckets=1, partial_fill=True, cancelled=True,
            advance_refused=True, release_refused_while_owned=True,
            release_refusal_cause='cannot release outstanding or quarantined '
                                  'operation', rebind_refused_while_owned=True,
            rebind_refusal_cause='source lane changed while a local lease is '
                                 'open', filling_at_cancel=0,
            publish_requires_all_pieces=True, no_partial_publish=True,
            drain_ok=True, released_after_drain=True,
            allocator_refs_zero=True, row1_aborted=True,
            refill_completed=True, refill_output_exact=True,
            refill_route='dense_dma'),
        refusals=[
            dict(case='stale_epoch', refused=True,
                 cause='descriptor or frame state epoch differs from '
                       'registration', error_type='SourceRefused'),
            dict(case='foreign_request', refused=True,
                 cause='descriptor or frame transaction differs from '
                       'registration', error_type='SourceRefused'),
            dict(case='changed_physical_page', refused=True,
                 cause='physical source map differs from the admission',
                 error_type='SourceRefused'),
            dict(case='premature_release', refused=True,
                 cause='source hold is already released',
                 error_type='SourceRefused'),
            dict(case='premature_reference', refused=True,
                 cause='allocator no longer retains that reference',
                 error_type='SourceRefused')],
        allocator_peak_allocated=700 << 20, allocator_peak_reserved=800 << 20,
        pinned_host_bytes=800 << 20, own_gpu_bytes=900 << 20,
        cleanup_ok=True, missing=[], failures=[])


class GradeTests(unittest.TestCase):
    """The verdict must move when the raw evidence is mutated or removed."""

    def setUp(self):
        self.record = green_record()

    def verdict(self, record):
        return job14.grade(record)

    def test_green_fixture_is_satisfiable(self):
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'PASS', result)
        self.assertEqual(result['rc'], 0)

    def test_grade_is_deterministic_and_frozen(self):
        first = self.verdict(copy.deepcopy(self.record))
        second = self.verdict(copy.deepcopy(self.record))
        self.assertEqual(first, second)
        self.assertIsNot(job14.grade(self.record), None)

    def test_missing_schema_is_inconclusive(self):
        del self.record['schema']
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')

    def test_mutated_bucket_bytes_fail(self):
        record = self.record['cases'][0]['layers'][0]['row1_records'][0]
        record['equal'] = False
        record['different_bytes'] = 64
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('raw_equal' in item for item in result['failures']))

    def test_missing_raw_output_record_is_inconclusive(self):
        records = self.record['cases'][0]['layers'][0]['row1_records']
        self.record['cases'][0]['layers'][0]['row1_records'] = [
            item for item in records
            if item['identity']['operation'] != 'qsa_output_before_moe']
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')

    def test_contradictory_equal_and_different_bytes_fail(self):
        record = self.record['cases'][0]['layers'][0]['row1_records'][0]
        record['equal'] = True
        record['different_bytes'] = 1
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('different_bytes' in item
                            for item in result['failures']))

    def test_missing_layer_is_inconclusive_not_pass(self):
        self.record['cases'][0]['layers'].pop()
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')

    def test_unreleased_hold_fails(self):
        self.record['cases'][0]['layers'][0]['hold_released'] = False
        self.record['cases'][0]['holds_released'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_consumer_never_queried_fails(self):
        layer = self.record['cases'][0]['layers'][0]
        layer['consumer_queried'] = layer['buckets'] - 1
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_probe_allocator_substitution_fails(self):
        self.record['allocator'] = 'ProbeAllocator'
        self.record['probe_allocator_used'] = True
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('probe_allocator_used', result['failures'])

    def test_active_hold_rebind_not_refused_fails(self):
        self.record['epochs']['active_hold_rebind_refused'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_mapping_version_not_advanced_fails(self):
        self.record['epochs']['mapping_version_advanced'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_uid_generation_reset_fails(self):
        self.record['epochs']['generation_reset'] = True
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_cancel_release_without_drain_fails(self):
        self.record['cancellation']['released_after_drain'] = False
        self.record['cancellation']['no_partial_publish'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_published_partial_bucket_fails(self):
        self.record['cancellation']['filling_at_cancel'] = 1
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_release_while_owned_not_refused_fails(self):
        self.record['cancellation']['release_refused_while_owned'] = False
        self.record['cancellation']['rebind_refused_while_owned'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_refill_mismatch_fails(self):
        self.record['cancellation']['refill_output_exact'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_wrong_refusal_cause_fails(self):
        self.record['refusals'][0]['cause'] = 'source hold is already released'
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('wrong_cause' in item for item in result['failures']))

    def test_missing_refusal_is_inconclusive(self):
        self.record['refusals'] = [item for item in self.record['refusals']
                                   if item['case'] != 'changed_physical_page']
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')

    def test_generated_ops_pin_mismatch_fails(self):
        self.record['generated_ops']['source_sha256'] = _digest('f')
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('generated_ops' in item for item in result['failures']))

    def test_missing_layer_source_pin_is_inconclusive(self):
        del self.record['layer_source_sha256']
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')

    def test_missing_native_uva_view_is_inconclusive(self):
        self.record['uva_aliased'] = False
        for case in self.record['cases']:
            for layer in case['layers']:
                layer['uva_aliased'] = False
        result = self.verdict(self.record)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('native_uva_view', result['missing'])

    def test_own_gpu_over_bound_fails(self):
        self.record['own_gpu_bytes'] = job14.OWN_GPU_BOUND + 1
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_pinned_host_over_bound_fails(self):
        self.record['pinned_host_bytes'] = job14.HOST_BOUND + 1
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_allocator_peak_over_bound_fails(self):
        self.record['allocator_peak_allocated'] = job14.OWN_GPU_BOUND + 1
        self.record['allocator_peak_reserved'] = job14.OWN_GPU_BOUND + 2
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_cleanup_failure_fails(self):
        self.record['cleanup_ok'] = False
        self.assertEqual(self.verdict(self.record)['verdict'], 'FAIL')

    def test_probe_exception_is_inconclusive(self):
        self.record['missing'].append('probe_exception')
        self.assertEqual(self.verdict(self.record)['verdict'], 'INCONCLUSIVE')


class ProbeContractTests(unittest.TestCase):
    """Static contract for the GPU-only entry points (never run here)."""

    def test_entry_points_exist_and_are_callable(self):
        import inspect
        self.assertEqual(list(inspect.signature(job14.probe).parameters),
                         ['modules', 'native', 'out'])
        self.assertEqual(list(inspect.signature(job14.grade).parameters),
                         ['record'])
        self.assertTrue(callable(job14.load_modules))
        self.assertTrue(callable(job14.load_native))

    def test_expected_routes_use_the_operation_threshold(self):
        self.assertEqual(job14.expected_route(job14.DENSE_CASE), 'dense_dma')
        self.assertEqual(job14.expected_route(job14.SPARSE_CASE), 'sparse_dma')
        self.assertEqual(job14.expected_buckets(job14.DENSE_CASE), 3)
        self.assertEqual(job14.expected_buckets(job14.SPARSE_CASE), 1)
        self.assertEqual(job14.expected_fills(job14.SPARSE_CASE), 0)

    def test_geometry_matches_the_pinned_arena(self):
        shape = job14.geometry()
        self.assertEqual(shape['page_rows'], 4048)
        self.assertEqual(shape['row_width'], 512)
        self.assertEqual(shape['arena_pages'], 24)
        self.assertEqual(shape['arena_rows'], 24 * 4048)
        self.assertEqual(shape['tokens'], job14.NUM_PAGES * 4048)


class RootIntegrationTests(unittest.TestCase):
    def test_actual_runner_binding_and_epoch_path_changes_real_lane(self):
        names = tuple('qsa.%d' % i for i in range(job14.LAYERS))
        sources = [sibling.T((job14.MAP_PAGES, 1, job14.PAGE, job14.ns.ROW_WIDTH),
                   device='cpu', dtype='torch.bfloat16', pinned=True)
                   for _ in names]
        runner = job14.NativeRunner(names, sources, [None] * len(names))
        probe = types.SimpleNamespace(runner=runner, route_generation=0,
                                      dispatcher=None)
        def complete_layer(probe, case, layer, binding, **kwargs):
            binding.holds[names[layer]].release()
            return dict(completed=True, output_exact=True)
        with patch.object(job14, 'run_layer', side_effect=complete_layer):
            result = job14.run_epochs(probe)
        self.assertTrue(result['active_hold_rebind_refused'], result)
        self.assertTrue(result['lane_advanced'], result)
        self.assertTrue(result['contiguous_chunks'], result)
        self.assertEqual(result['chunk_spans'][0][1],result['chunk_spans'][1][0])
        self.assertEqual(probe.dispatcher.source_registry.lane.storage_epoch, 4)
        self.assertEqual(result['chunk_epochs'][1]['storage_epoch'], 4)
        self.assertTrue(result['mapping_version_advanced'], result)
        self.assertTrue(result['source_epoch_stable'], result)
        self.assertTrue(result['source_closed'], result)

    def test_actual_runner_refusal_path_uses_live_group_rows_and_source_holds(self):
        names = tuple('qsa.%d' % i for i in range(job14.LAYERS))
        sources = [sibling.T((job14.MAP_PAGES, 1, job14.PAGE, job14.ns.ROW_WIDTH),
                   device='cpu', dtype='torch.bfloat16', pinned=True)
                   for _ in names]
        probe = types.SimpleNamespace(runner=job14.NativeRunner(
            names, sources, [None] * len(names)), route_generation=0, dispatcher=None)
        result = job14.run_refusals(probe)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(item['refused'] for item in result), result)
        for item in result:
            if item['case'] in job14.REFUSAL_CAUSES:
                self.assertIn(job14.REFUSAL_CAUSES[item['case']], item['cause'])
        self.assertEqual(probe.dispatcher.source_allocator.inventory()['live'], 0)

    def test_exact_generated_call_sites_load_without_model_owner_imports(self):
        text=job14.operation_source.generate_layer(
            job14.runner_hooks.qsa_staged(job14.LAYER_SOURCE.read_text()))
        tree=job14.helper_definitions(text,'qsa_native.py')
        wanted={job14.operation_source.LAYER_HELPER,
                job14.operation_source.SPARSE_LAYER_HELPER}
        self.assertEqual({item.name for item in tree.body},wanted)
        original=ast.parse(text)
        for item in tree.body:
            reference=next(n for n in original.body if isinstance(n,ast.FunctionDef)
                           and n.name==item.name)
            self.assertEqual(ast.dump(item),ast.dump(reference))
        namespace={}
        exec(compile(tree,'qsa_native.py','exec'),namespace)
        self.assertTrue(all(callable(namespace[name]) for name in wanted))
        with self.assertRaisesRegex(ValueError,'helper_definitions'):
            job14.helper_definitions('def wrong(): pass','wrong.py')

    def test_dense_checks_consult_actual_queried_operation_records(self):
        tree=ast.parse(Path(job14.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_layer')
        result=next(n.value for n in ast.walk(run) if isinstance(n,ast.Return))
        expression=next(k.value for k in result.keywords if k.arg=='consumer_queried')
        rows=[types.SimpleNamespace(event=object(),complete=True),
              types.SimpleNamespace(event=object(),complete=False),
              types.SimpleNamespace(event=None,complete=True)]
        value=eval(compile(ast.Expression(expression),'actual-consumer-check','eval'),
                   {'run':types.SimpleNamespace(consumed=rows)})
        self.assertEqual(value,1)
        source=Path(job14.__file__).read_text()
        cancel=source.split('def run_cancellation(probe):',1)[1].split('def _refusal',1)[0]
        self.assertIn('operation.release()',cancel)
        self.assertNotIn('hold.release()',cancel)
        self.assertIn('run.next_bucket == started_buckets',cancel)
        self.assertIn('ro._OUTPUT_OPERATION',cancel)

    def test_fixed_query_geometry_covers_exact_union_without_frontier_sized_query(self):
        self.assertLess(job14.QUERY_ROWS, 1024)
        for case in job14.CASES:
            flat, counts = job14.selection_matrix(case)
            self.assertEqual(len(flat), job14.QUERY_ROWS * (job14.TOPK + 1))
            selected = set()
            for row in range(job14.QUERY_ROWS):
                part=flat[row*(job14.TOPK+1):(row+1)*(job14.TOPK+1)]
                self.assertEqual(part[-1], counts[row])
                selected.update(x for x in part[:-1] if x != -1)
            self.assertEqual(selected, set(range(job14.SELECTION_COUNT[case])))

    def test_uva_view_checked_against_real_runtime_mapping_not_copy_flag(self):
        class View:
            shape=(51,1,4048,512);dtype='bf16';is_cuda=True;device='cuda:0'
            def data_ptr(self): return self.address
            def is_contiguous(self): return True
        source=View();source.address=100
        source.is_pinned=lambda:True
        view=View();view.address=200
        class NativeFn:
            def __call__(self, dest, host, flags):
                import ctypes
                ctypes.cast(dest,ctypes.POINTER(ctypes.c_void_p))[0]=200
                return 0
        utils=types.ModuleType('vllm.utils.torch_utils')
        utils.__file__='pinned.py'
        utils.get_accelerator_view_from_cpu_tensor=lambda tensor:view
        vllm=types.ModuleType('vllm');vp=types.ModuleType('vllm.utils')
        vllm.utils=vp;vp.torch_utils=utils
        with patch.dict(sys.modules,{'vllm':vllm,'vllm.utils':vp,
                                    'vllm.utils.torch_utils':utils}), \
             patch.object(job14.s,'sha',return_value=job14.TORCH_UTILS_SHA), \
             patch('ctypes.CDLL',return_value=types.SimpleNamespace(
                 cudaHostGetDevicePointer=NativeFn())):
            self.assertIs(job14.make_uva_view(source,'cuda:0'),view)
            view.address=300
            with self.assertRaisesRegex(RuntimeError,'same_mapped_storage'):
                job14.make_uva_view(source,'cuda:0')
            with patch.object(job14.s,'sha',return_value='0'*64):
                with self.assertRaisesRegex(RuntimeError,'source_pin'):
                    job14.make_uva_view(source,'cuda:0')

    def test_queue_owns_only_job14_and_uses_self_contained_pins(self):
        text=(HERE/'runs/vmid_job14.sh').read_text()
        for item in ('runs/job14','flashnext-job14','--network none',
                     '.job.lock','--memory 4g','--memory-swap 4g',
                     'e3_reference/models/qwen4_exp/nvidia/qsa.py',
                     'e3_builder/native_source.py'):
            # The last path is assembled by the bounded module loop.
            if item=='e3_builder/native_source.py':
                self.assertIn('native_source.py',text)
                self.assertIn('$H/e3_builder/$module',text)
            else:self.assertIn(item,text)
        subprocess.run(['bash','-n',str(HERE/'runs/vmid_job14.sh')],check=True)
        self.assertIn('flashnext-job14',job14.s.OWNED_CONTAINERS)
        self.assertIn('flashnext-job14',job14.s.LONG_RUN_CONTAINERS)
        self.assertTrue(job14.LAYER_SOURCE.is_file())
        tree=ast.parse(Path(job14.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        guarded=[n for n in ast.walk(main) if isinstance(n,ast.Try)]
        self.assertTrue(any(any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name)
                        and c.func.id=='load_modules' for c in ast.walk(t)) for t in guarded))
        self.assertIn('cudaHostGetDevicePointer',Path(job14.__file__).read_text())


if __name__ == '__main__':
    unittest.main()
