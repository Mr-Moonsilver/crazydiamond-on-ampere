"""Job7c draft rules: prefill failure-before/pass-after and retention veto.

CPU only. No torch, no CUDA, no vLLM import: the run itself is an unarmed draft.
These tests prove the draft's grading is fail-closed and that the queue stays
fresh, bounded and explicitly unauthorized.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import vmid_job7c as j

PREFILL = 11
DECODE = 12
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


def good_case(family):
    return dict(
        family=family,
        failures=[], missing=[],
        prefill_context_id=PREFILL, decode_context_id=DECODE,
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
        decode=dict(capture_ok=True, owners_restored=True, graph_retired=True,
                    output_exact=True, window_ended=True, predicate='PASS',
                    census=census(census_row(DECODE, 500, graph=7),
                                  census_row(DECODE, 501, graph=8)),
                    rule_failures=[], rule_missing=[]),
        native_lifetime=dict(native_release='process-exit', native_retained=True,
                             retained_prefill_auxiliary_streams=2),
        late_tensor_release_after_close=True, cleanup_ok=True, cleanup_errors=[])


def good_probe():
    return dict(cases=[good_case(f) for f in j.FAMILIES],
                regressions={n: {'verdict': 'PASS'} for n in ('job1', 'job2b_flat')},
                failures=[], missing=[])


class RuleTests(unittest.TestCase):
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

    def test_raw_decode_requires_every_declared_auxiliary(self):
        probe = good_probe()
        probe['cases'][0]['decode']['census']['rows'][1]['stream_id'] = 500
        result = j.grade(probe)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('shared:decode:side_stream_unobserved', result['missing'])

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
        queue = root / 'runs/vmid_job7c.draft.sh'
        subprocess.run(['bash', '-n', str(queue)], check=True)
        text = queue.read_text()
        self.assertIn('DRAFT, UNARMED, NOT AUTHORIZED', text)
        for value in ('runs/job7c', 'flashnext-job7c', 'flock -n 9', '540s',
                      's5_prefill_aux_create', 's5_prefill_stream_check',
                      's5_prefill_aux_destroy', 's5_decode_aux_create'):
            self.assertIn(value, text)
        source = (root / 'vmid_job7c.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125, 0)', source)
        self.assertIn('> 3072', source)
        self.assertIn('late_tensor.record_stream(late_stream)', source)
        self.assertLess(source.index('split.close()'), source.index('del late_tensor'))
        self.assertIn('torch.cuda.empty_cache()', source)
        self.assertIn('image=s.IMAGE', source)


if __name__ == '__main__':
    unittest.main()
