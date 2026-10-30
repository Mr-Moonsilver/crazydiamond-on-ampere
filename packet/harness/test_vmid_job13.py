"""Static/CPU contract tests for the VM<vm-id> job13 native-prefill job.

Nothing here runs CUDA, an engine or a container: the positive record is the
FakeBridge lane fixture, including sparse copy and PLE finalize submissions,
regraded by the real paired composer. Passing here proves the job's
grading, its queue text and its declared vocabulary only.  It is not native,
GPU, readiness or engine evidence.
"""
import copy
import subprocess
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import native_placement as np_  # noqa: E402
import native_prefill_offers as npo  # noqa: E402
import test_e3_native_placement as placement  # noqa: E402
import vmid_job13 as job  # noqa: E402

SCRIPT = ROOT / 'harness/runs/vmid_job13.sh'
REQUIRED = ('source_pins_verified', 'graph_bytes_stable', 'output_exact',
            'owners_restored', 'sharing_preserved', 'cleanup_ok',
            'contexts_disjoint', 'raw_unchanged', 'graph_output_exact')


def positive_record():
    fix = placement.positive(extra_copy_families=True)
    composed = placement.compose(fix)
    record: dict = {name: True for name in REQUIRED}
    record.update(
        mode='positive', failures=[], missing=[], copy_role_control=True,
        shadow_pins=[{'source_sha256': 'a' * 64}],
        composed=composed, prefill_receipt=fix['prefill_receipt'],
        native_report=fix['native'])
    return record


class GradeWiringTests(unittest.TestCase):
    def test_fixture_composition_is_a_real_pass(self):
        fix = placement.positive()
        composed = placement.compose(fix)
        self.assertEqual(composed['verdict'], 'PASS')
        self.assertTrue(composed['conservation']['proved'])
        self.assertEqual(composed['whole']['errors'], 0)

    def test_positive_record_passes(self):
        grade = job.case_grade(positive_record())
        self.assertEqual(grade['verdict'], 'PASS', grade)

    def test_missing_finalize_family_never_passes(self):
        record = copy.deepcopy(positive_record())
        del record['composed']['partitions']['prefill']['families']['ple'][
            'finalize_prefetch']
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'FAIL')
        self.assertIn('ple_finalize_unjoined', grade['failures'])

    def test_empty_evidence_is_inconclusive(self):
        grade = job.case_grade({})
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertFalse(grade['failures'])

    def test_whole_raw_loss_is_never_discharged(self):
        record = copy.deepcopy(positive_record())
        record['composed']['whole']['boundary_records'] = 2
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertIn('whole_raw_loss', grade['missing'])

    def test_incomplete_whole_report_is_inconclusive(self):
        record = copy.deepcopy(positive_record())
        record['composed']['whole']['complete'] = False
        record['native_report']['complete'] = False
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertIn('whole_raw_incomplete', grade['missing'])

    def test_rewritten_whole_report_fails(self):
        record = copy.deepcopy(positive_record())
        record['native_report']['activity_records'] += 1
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'FAIL')
        self.assertIn('whole_report_rewritten', grade['failures'])

    def test_partition_verdict_must_be_pass(self):
        record = copy.deepcopy(positive_record())
        record['composed']['partitions']['decode']['verdict'] = 'FAIL'
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertIn('decode_partition', grade['missing'])

    def test_unproved_conservation_fails(self):
        record = copy.deepcopy(positive_record())
        record['composed']['conservation']['proved'] = False
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'FAIL')
        self.assertIn('conservation_not_proved', grade['failures'])

    def test_prefill_receipt_must_be_pass(self):
        record = copy.deepcopy(positive_record())
        record['prefill_receipt'] = {'verdict': 'INCONCLUSIVE', 'missing':
                                     ['unsupported_api:cudaStreamIsCapturing']}
        grade = job.case_grade(record)
        self.assertNotEqual(grade['verdict'], 'PASS')
        self.assertIn('prefill_missing:unsupported_api:cudaStreamIsCapturing',
                      grade['missing'])

    def test_copy_role_control_required(self):
        record = copy.deepcopy(positive_record())
        record['copy_role_control'] = False
        self.assertIn('copy_role_control', job.case_grade(record)['failures'])

    def test_shadow_pins_must_be_hashes(self):
        record = copy.deepcopy(positive_record())
        record['shadow_pins'] = [{'source_sha256': 'short'}]
        self.assertIn('source_shadow_pin', job.case_grade(record)['missing'])


class NegativeControlTests(unittest.TestCase):
    def _record(self) -> dict:
        return dict(mode='ordinary', failures=[], missing=[],
                    source_pins_verified=True, graph_bytes_stable=True,
                    output_exact=True, owners_restored=True,
                    sharing_preserved=True, cleanup_ok=True,
                    contexts_disjoint=True, raw_unchanged=True,
                    graph_output_exact=True,
                    negative={'verdict': 'FAIL',
                              'failures': ['wrong_context', 'wrong_stream'],
                              'joined_records': 1, 'raw_complete': True})

    def test_wrong_stream_and_context_control_passes(self):
        self.assertEqual(job.case_grade(self._record())['verdict'], 'PASS')

    def test_unrejected_control_fails(self):
        record = self._record()
        record['negative']['verdict'] = 'PASS'
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'FAIL')
        self.assertIn('negative_not_rejected', grade['failures'])

    def test_missing_reason_fails(self):
        record = self._record()
        record['negative']['failures'] = ['wrong_stream']
        grade = job.case_grade(record)
        self.assertIn('negative_missing:wrong_context', grade['failures'])

    def test_vacuous_control_is_inconclusive(self):
        record = self._record()
        record['negative']['joined_records'] = 0
        grade = job.case_grade(record)
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertIn('negative_vacuous', grade['missing'])


class DeclaredVocabularyTests(unittest.TestCase):
    def test_receipt_schemas(self):
        self.assertEqual(npo.SCHEMA, 's5-native-prefill-accounting/1')
        self.assertEqual(np_.SCHEMA, 's5-native-overlap-accounting/1')
        self.assertEqual(job.SCHEMA, 'vm<vm-id>-job13/1')

    def test_graph_recipe_matches_the_pinned_probe(self):
        import vmid_job11 as job11
        self.assertEqual(job.MIX, dict(kernel=1386, memcpy=26, memset=7))
        self.assertEqual(job.MIX, job11.MIX)
        self.assertEqual(7 + job.MIX['kernel'], 1393)

    def test_owned_modules_exist(self):
        for name in job.MODULES:
            self.assertTrue((ROOT / 'patches/s5_engine' / name).is_file(), name)
        for name in job.PINNED_MODULES:
            self.assertTrue((ROOT / 'patches/s5_engine' / name).is_file(), name)

    def test_lane_families_and_roles_are_declared(self):
        for family in ('prefill_segment', 'qsa', 'shared_experts', 'ple'):
            self.assertIn(family, npo.FAMILIES)
        self.assertEqual(set(npo.ROLES), {'prefill_main',
                                          'shared_experts_aux', 'ple_aux',
                                          'copy'})
        self.assertEqual(npo.FAMILIES['qsa']['bucket_fill']['role'], 'copy')
        self.assertEqual(npo.FAMILIES['ple']['start_prefetch']['role'],
                         'ple_aux')

    def test_cannot_execute_is_declared(self):
        text = ' '.join(job.CANNOT_EXECUTE)
        for want in ('TP>1', 'PP>1', 'Qwen4ExpModel', 'overlap performance'):
            self.assertIn(want, text)
        self.assertIn('no engine', job.SCOPE)


class QueueScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = SCRIPT.read_text()

    def test_queue_bounds(self):
        for want in ('/opt/flashnext/runs/job13', 'flashnext-job13',
                     'flashnext-job13-build', '.job.lock', '--network none',
                     'FLASH_S5_ENGINE_OFFERS=1',
                     'FLASH_S5_NATIVE_PREFILL_OFFERS=1',
                     '--memory 4g', '--memory-swap 4g', '540s'):
            self.assertIn(want, self.text)

    def test_queue_uses_a_new_directory_and_no_gpu_build(self):
        self.assertIn('[[ ! -e $Q && ! -L $Q ]]', self.text)
        build = self.text.split('flashnext-job13-build')[1].split(
            'python3 "$H/vmid_job13.py" --mode monitor')[0]
        self.assertNotIn('--gpus', build)

    def test_queue_script_parses(self):
        completed = subprocess.run(['bash', '-n', str(SCRIPT)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout)


if __name__ == '__main__':
    unittest.main()
