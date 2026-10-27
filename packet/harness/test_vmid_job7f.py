"""Job7c draft rules: prefill failure-before/pass-after and retention veto.

CPU only. No torch, no CUDA, no vLLM import: the run itself is an unarmed draft.
These tests prove the draft's grading is fail-closed and that the queue stays
fresh, bounded and explicitly unauthorized.
"""
import copy
import ctypes as C
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import vmid_job7f as j

PREFILL = 11
DECODE = 7
MAIN = 100
STOCK = (200, 201)
AUX = (300, 301)
DECODE_STREAMS = (500, 501)


def census_row(context, stream, graph=0, count=1):
    return {'context_id': context, 'stream_id': stream, 'graph_id': graph,
            'node_id': graph, 'count': count, 'kind': 'kernel'}


def census(*rows):
    return {'schema': 's5ro-activity-census/1', 'complete': True,
            'boundary_records': 0, 'invalid_records': 0, 'omitted_records': 0,
            'matched_records': sum(x['count'] for x in rows), 'rows': list(rows)}


def phase(contexts, predicate, **overrides):
    row = dict(census=census(*[census_row(c, s) for c, s in contexts]),
               predicate=predicate, output_exact=True, rule_failures=[],
               rule_missing=[])
    row.update(overrides)
    return row


@lru_cache(maxsize=None)
def native_window_fixture(variant="clean"):
    # Compile and exercise the actual native observer against CPU CUDA/CUPTI
    # stubs. The grade's positive fixture therefore has a canonical real schema.
    from test_e3_replay_observer import ObserverTestBase, STATIC_GRAPH
    if not hasattr(native_window_fixture, 'compiled'):
        class Fixture(ObserverTestBase):
            pass
        Fixture.setUpClass()
        native_window_fixture.compiled = Fixture
    Fixture = native_window_fixture.compiled
    fixture = Fixture('runTest')
    fixture.setUp()
    try:
        spec = copy.deepcopy(STATIC_GRAPH)
        if variant in ('mixed', 'foreign_copy'):
            spec['nodes'] += [(103, 1), (104, 2)]
        fixture.open_clean(specs=[spec])
        session = fixture.session
        fixture.ctl.stub_set_timestamp(1000)
        stamp = C.c_uint64()
        j.native_call({}, 'begin_decode_window', fixture.binding._lib.s5ro_begin_window,
                      session._token, 1, C.byref(stamp))
        for _ in range(2 if variant == 'missing_replay' else j.REPLAYS):
            fixture.emit_kernel(11, 999 if variant == 'foreign_node' else 101,
                                context_id=999 if variant == 'primary' else DECODE)
            fixture.emit_kernel(11, 102)
            if variant in ('mixed', 'foreign_copy'):
                fixture.ctl.stub_emit_copy(DECODE, 11,
                    999 if variant == 'foreign_copy' else 103, 1100, 1200)
                fixture.ctl.stub_emit_memset(DECODE, 11, 104, 1100, 1200)
        fixture.ctl.stub_set_timestamp(3000)
        raw = session._report_call(fixture.binding._lib.s5ro_end_window,
                                   (session._token, stamp.value, 1), retry=False)
        lib = fixture.binding._lib
        lib.s5ro_activity_census.argtypes = [C.c_uint64] * 3 + [
            C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64)]
        lib.s5ro_activity_census.restype = C.c_int
        rows = session._report_call(lib.s5ro_activity_census,
                                    (session._token, 1000, 3000))
        return dict(window=raw, census=rows)
    finally:
        fixture.doCleanups()


def good_case(family):
    fixture = copy.deepcopy(native_window_fixture())
    native = fixture['window']
    return dict(
        family=family, decode_sms=8,
        failures=[], missing=[],
        prefill_context_id=PREFILL, decode_context_id=DECODE,
        context_id_namespace='CUPTI', subscriber_started_before_split=True,
        prefill_binding_graph=dict(bound=True, slot=0, nodes=1,
                                   static_complete=True, replays_in_window=0),
        decode_binding_graph=dict(bound=True, slot=0, nodes=2, static_complete=True),
        context_queries=dict(
            prefill=dict(green_present=True, green_cupti_context_id=PREFILL),
            decode=dict(green_present=True, green_cupti_context_id=DECODE)),
        stream_inventory=dict(prefill_main=MAIN, stock_side=list(STOCK),
                              prefill_aux=list(AUX), required_stock_side=list(STOCK),
                              required_prefill_aux=list(AUX), decode_main=500,
                              decode_aux=list(DECODE_STREAMS)),
        source_pins_verified=True,
        helper=dict(native_symbols={n: True for n in
                                    j.PREFILL_SYMBOLS + j.DECODE_SYMBOLS}),
        discovery=dict(shared_owners=2, ple_owners=1, auxiliary_streams=2,
                       distinct_original_streams=2, basis='prefill-green',
                       sharing_preserved=True, original_owners_restored=True),
        prefill_aux_tokens=list(AUX), decode_aux_tokens=list(DECODE_STREAMS),
        pools_disjoint=True, role_checks=True, owners_restored=True,
        window_ended=True,
        phases=dict(
            before=phase([(PREFILL, MAIN), (0, STOCK[0]), (0, STOCK[1])], 'FAIL'),
            after=phase([(PREFILL, MAIN), (PREFILL, AUX[0]), (PREFILL, AUX[1])], 'PASS'),
            restored=phase([(PREFILL, MAIN), (0, STOCK[0]), (0, STOCK[1])], 'FAIL')),
        decode=dict(native_window=native, subscriber_started_before_split=True,
                    capture_ok=True, owners_restored=True, graph_retired=True,
                    output_exact=True, window_ended=True, predicate='PASS',
                    census=fixture['census'],
                    rule_failures=[], rule_missing=[]),
        native_lifetime=dict(native_release='process-exit', native_retained=True,
                             retained_prefill_auxiliary_streams=2),
        late_tensor_release_after_close=True, cleanup_ok=True, cleanup_errors=[])


def good_probe():
    return dict(cases=[good_case(f) for f in j.FAMILIES],
                regressions={n: {'verdict': 'PASS'} for n in ('job1', 'job2b_flat')},
                failures=[], missing=[])


class RuleTests(unittest.TestCase):
    def test_actual_native_typed_node_mapping_and_multiplicity_guards(self):
        for variant, expected in (('clean', 'PASS'), ('mixed', 'PASS'),
                                  ('primary', 'FAIL'), ('foreign_node', 'INCONCLUSIVE'),
                                  ('foreign_copy', 'INCONCLUSIVE'),
                                  ('missing_replay', 'INCONCLUSIVE')):
            with self.subTest(variant=variant):
                raw = native_window_fixture(variant)
                row = dict(native_window=raw['window'], census=raw['census'],
                           subscriber_started_before_split=True)
                f, m = j.decode_graph_rule(row, expected=DECODE, sms=8,
                    bound=dict(nodes=raw['window']['graph_slots'][0]['node_count']))
                self.assertEqual(j.s.verdict(f, m)['verdict'], expected)

    def test_raw_kernel_join_wrong_graph_or_missing_native_report_never_pass(self):
        row = copy.deepcopy(good_case('shared')['decode'])
        row['census']['rows'][0]['graph_id'] = 999
        f, m = j.decode_graph_rule(row, expected=DECODE, sms=8, bound=dict(nodes=2))
        self.assertIn('decode:unknown_graph', f)
        self.assertIn('decode:kernel_census_join', m)
        row = copy.deepcopy(good_case('shared')['decode'])
        del row['native_window']
        self.assertEqual(j.decode_graph_rule(row, expected=DECODE, sms=8,
                                            bound=dict(nodes=2)),
                         ([], ['decode:native_window']))

    def test_context_comparison_uses_cupti_id_not_driver_handle(self):
        facts = dict(green_present=True, green_cupti_context_id=11,
                     green_driver_id=987654, primary_cupti_context_id=1)
        self.assertEqual(j.cupti_context(facts), 11)
        with self.assertRaises(ValueError):
            j.cupti_context(dict(green_present=True, green_driver_id=987654))

    def test_native_refusal_retains_exact_operation_and_return_code(self):
        record = {}
        with self.assertRaises(j.NativeCallFailure) as caught:
            j.native_call(record, 'bind_prefill_context', lambda *_: 12, 1, 2, 38)
        self.assertEqual(caught.exception.rc, 12)
        self.assertEqual(record['native_calls'],
                         [dict(operation='bind_prefill_context', rc=12)])
        with self.assertRaises(ValueError):
            j.native_call(record, 'untrusted-message', lambda: 0)

    def test_subscription_precedes_context_creation_and_failure_stops_creation(self):
        from types import SimpleNamespace
        calls = []
        result = j.subscribed_split(SimpleNamespace(start=lambda: calls.append('start')),
                                    lambda: calls.append('split') or 42)
        self.assertEqual(result, 42)
        self.assertEqual(calls, ['start', 'split'])
        def refuse():
            raise RuntimeError('subscription_refused')
        calls.clear()
        with self.assertRaises(RuntimeError):
            j.subscribed_split(SimpleNamespace(start=refuse), lambda: calls.append('split'))
        self.assertEqual(calls, [])

    def test_expected_context_and_updated_side_stream_rules(self):
        row = {'census': census()}
        f, m = j.context_rule('after', row, expected=PREFILL,
                              allowed=[MAIN, *AUX], side=AUX,
                              required=list(AUX), want='expected')
        self.assertEqual(f, [])
        self.assertEqual(m, ['after:unobserved'])
        row = {'census': census(census_row(PREFILL, MAIN),
                                census_row(PREFILL, AUX[0]))}
        f, m = j.context_rule('after', row, expected=PREFILL,
                              allowed=[MAIN, *AUX], side=AUX,
                              required=list(AUX), want='expected')
        self.assertEqual(f, [])
        self.assertEqual(m, ['after:side_stream_unobserved'])

    def test_stock_side_stream_must_be_outside_the_green_context(self):
        row = {'census': census(census_row(PREFILL, MAIN),
                                census_row(0, STOCK[0]), census_row(0, STOCK[1]))}
        f, m = j.context_rule('before', row, expected=PREFILL,
                              allowed=[MAIN, *STOCK], side=STOCK,
                              required=list(STOCK), want='other')
        self.assertEqual((f, m), ([], []))
        green = {'census': census(census_row(PREFILL, MAIN),
                                  census_row(PREFILL, STOCK[0]))}
        f, m = j.context_rule('before', green, expected=PREFILL,
                              allowed=[MAIN, *STOCK], side=STOCK,
                              required=[STOCK[0]], want='other')
        self.assertIn('before:stock_side_stream_is_green', f)

    def test_incomplete_or_foreign_stream_census_never_passes(self):
        row = {'census': census(census_row(PREFILL, MAIN), census_row(PREFILL, 999))}
        f, m = j.context_rule('after', row, expected=PREFILL,
                              allowed=[MAIN, *AUX], side=AUX,
                              required=list(AUX), want='expected')
        self.assertIn('after:unexpected_side_stream', f)
        broken = {'census': dict(census(census_row(PREFILL, MAIN)), complete=False)}
        f, m = j.context_rule('after', broken, expected=PREFILL,
                              allowed=[MAIN, *AUX], side=AUX,
                              required=list(AUX), want='expected')
        self.assertEqual(f, [])
        self.assertEqual(m, ['after:census_incomplete'])


class GradeTests(unittest.TestCase):
    def test_real_graph_binding_is_required_for_both_windows(self):
        for role in ('prefill', 'decode'):
            raw = good_probe()
            raw['cases'][0][role + '_binding_graph']['nodes'] = 0
            self.assertEqual(j.grade(raw)['verdict'], 'INCONCLUSIVE')
        raw = good_probe()
        raw['cases'][0]['prefill_binding_graph']['replays_in_window'] = 1
        self.assertIn('shared:prefill:sentinel_replayed', j.grade(raw)['failures'])

    def test_wrong_namespace_or_query_binding_cannot_pass(self):
        raw = good_probe()
        raw['cases'][0]['context_id_namespace'] = 'CUgreenCtx'
        self.assertEqual(j.grade(raw)['verdict'], 'INCONCLUSIVE')
        raw = good_probe()
        raw['cases'][0]['prefill_context_id'] = 987654
        self.assertIn('shared:prefill:context_query_mismatch', j.grade(raw)['failures'])

    def test_raw_incomplete_census_overrides_cached_pass(self):
        probe = good_probe()
        probe['cases'][0]['phases']['after']['census']['complete'] = False
        result = j.grade(probe)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('shared:after:census_incomplete', result['missing'])

    def test_raw_wrong_context_overrides_cached_pass(self):
        probe = good_probe()
        probe['cases'][0]['phases']['after']['census']['rows'][0]['context_id'] = 987654
        result = j.grade(probe)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('shared:after:wrong_context', result['failures'])

    def test_graph_replay_stream_ids_may_differ_from_capture_ids(self):
        probe = good_probe()
        probe['cases'][0]['decode']['census']['rows'][1]['stream_id'] = 999
        probe['cases'][0]['decode']['native_window']['context_diagnostics']['executions'][1]['stream_id'] = 999
        result = j.grade(probe)
        self.assertEqual(result['verdict'], 'PASS')

    def test_complete_real_owner_evidence_passes(self):
        self.assertEqual(j.grade(good_probe())['verdict'], 'PASS')

    def test_owned_phase_must_be_green_and_unowned_phases_must_fail(self):
        probe = good_probe()
        probe['cases'][0]['phases']['after']['predicate'] = 'FAIL'
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')
        probe = good_probe()
        probe['cases'][0]['phases']['before']['predicate'] = 'PASS'
        self.assertEqual(j.grade(probe)['verdict'], 'INCONCLUSIVE')
        probe = good_probe()
        probe['cases'][0]['phases'].pop('restored')
        self.assertEqual(j.grade(probe)['verdict'], 'INCONCLUSIVE')

    def test_sharing_pools_roles_and_lifetime_are_required(self):
        for key, verdict in (('sharing_preserved', 'FAIL'),
                             ('basis', 'FAIL')):
            probe = good_probe()
            probe['cases'][0]['discovery'][key] = False
            self.assertEqual(j.grade(probe)['verdict'], verdict)
        probe = good_probe()
        probe['cases'][0]['pools_disjoint'] = False
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')
        probe = good_probe()
        probe['cases'][0]['role_checks'] = False
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')
        probe = good_probe()
        probe['cases'][0].pop('native_lifetime')
        self.assertEqual(j.grade(probe)['verdict'], 'INCONCLUSIVE')
        probe = good_probe()
        probe['cases'][0]['late_tensor_release_after_close'] = False
        self.assertEqual(j.grade(probe)['verdict'], 'INCONCLUSIVE')

    def test_decode_regression_and_cleanup_are_not_optional(self):
        probe = good_probe()
        probe['cases'][0]['decode']['predicate'] = 'FAIL'
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')
        probe = good_probe()
        probe['regressions']['job1']['verdict'] = 'FAIL'
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')
        probe = good_probe()
        probe['cases'][0]['cleanup_ok'] = False
        self.assertEqual(j.grade(probe)['verdict'], 'FAIL')


class FinalizeTests(unittest.TestCase):
    def write(self, root, probe, monitor):
        (root / 'probe.json').write_text(json.dumps(probe))
        (root / 'monitor.json').write_text(json.dumps(monitor))

    def test_clean_run_and_vetoes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.write(root, good_probe(), {'own_samples': 1, 'errors': 0,
                                            'own_peak_mib': 650})
            self.assertEqual(j.finalize(root, j.grade(good_probe())['rc']), 0)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.write(root, good_probe(), {'own_samples': 1, 'errors': 0,
                                            'own_peak_mib': 650})
            self.assertEqual(j.finalize(root, 139), 1)
            result = json.loads((root / 'result.json').read_text())
            self.assertIn('native_process_crash', result['failures'])
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.write(root, good_probe(), {'own_samples': 1, 'errors': 0,
                                            'own_peak_mib': 4000})
            self.assertEqual(j.finalize(root, 0), 1)
            result = json.loads((root / 'result.json').read_text())
            self.assertIn('resource_stop_or_cap', result['failures'])


class DraftTests(unittest.TestCase):
    def test_draft_queue_is_unarmed_fresh_and_bounded(self):
        root = Path(__file__).resolve().parent
        queue = root / 'runs/vmid_job7f.draft.sh'
        subprocess.run(['bash', '-n', str(queue)], check=True)
        text = queue.read_text()
        self.assertIn('DRAFT, UNARMED, NOT AUTHORIZED', text)
        for value in ('runs/job7f', 'flashnext-job7f', 'flock -n 9', '540s',
                      's5_prefill_aux_create', 's5_prefill_stream_check',
                      's5_prefill_aux_destroy', 's5_decode_aux_create'):
            self.assertIn(value, text)
        source = (root / 'vmid_job7f.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125, 0)', source)
        self.assertIn('> 3072', source)
        self.assertIn('late_tensor.record_stream(late_stream)', source)
        self.assertLess(source.index('split.close()'), source.index('del late_tensor'))
        self.assertIn('torch.cuda.empty_cache()', source)
        self.assertIn('image=s.IMAGE', source)
        self.assertLess(source.index('decode_session, prefill_session = prefill_session, None'),
                        source.index('graph = capture(split.decode_stream)'))


from test_e3_replay_observer import ObserverTestBase, EXPECTED_SMS, STATIC_GRAPH


class NativeWindowContractTests(ObserverTestBase):
    def test_real_native_window_refuses_context_only_then_accepts_bound_graph(self):
        import ctypes as c
        from types import SimpleNamespace
        self.make_bank([STATIC_GRAPH])
        session = self.new_session(start=True)
        lib = self.binding._lib
        record = {}
        j.native_call(record, 'bind_prefill_context', lib.s5ro_bind_context,
                      session._token, self.green, EXPECTED_SMS)
        stamp = c.c_uint64()
        self.ctl.stub_set_timestamp(1000)
        with self.assertRaises(j.NativeCallFailure) as caught:
            j.native_call(record, 'begin_prefill_window', lib.s5ro_begin_window,
                          session._token, 1, c.byref(stamp))
        self.assertEqual(caught.exception.rc, 23)
        graph = SimpleNamespace(raw_cuda_graph=lambda: self.bank.graph_handles[0])
        bound = j.bind_window_graph(record, 'bind_prefill_graph', session, graph)
        self.assertEqual(bound['nodes'], 2)
        self.assertTrue(bound['static_complete'])
        j.native_call(record, 'begin_prefill_window', lib.s5ro_begin_window,
                      session._token, 1, c.byref(stamp))
        self.assertGreater(stamp.value, 0)
        self.ctl.stub_set_timestamp(3000)
        session._report_call(lib.s5ro_end_window, (session._token, stamp.value, 1), retry=False)


if __name__ == '__main__':
    unittest.main()
