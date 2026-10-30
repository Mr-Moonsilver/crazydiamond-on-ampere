"""Pinned host metadata APIs do zero GPU work; no generic query exemption."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import activity_accounting as aa
import native_placement as np_
import test_e3_native_placement as fixture
import test_vmid_job13 as old_fixture
import test_vmid_job13b as abi
import vmid_job13c as job

NAMES = ('cuFuncGetAttribute', 'cuKernelGetFunction', 'cuKernelGetName',
         'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', 'cuPointerGetAttribute',
         'cudaDeviceGetAttribute_v5000', 'cudaEventRecordWithFlags_v11010',
         'cudaStreamIsCapturing_v10000')


def extra_api(fix, name, *, offered=False, rc=0):
    native = fix['native']
    cid = max(row['correlation_id'] for row in native['apis']) + 1
    native['apis'].append(dict(correlation_id=cid, domain='runtime' if name.startswith('cuda') else 'driver',
        name=name, start=native['window']['start_ns'] + 1,
        end=native['window']['start_ns'] + 2, thread=1101, return_value=rc))
    if offered:
        oid = int(next(iter(fix['prefill_declared'])))
        thread = fix['prefill_declared'][str(oid)]['thread']
        native['apis'][-1]['thread'] = thread
        native['external'].append(dict(kind='custom0',correlation_id=cid,external_id=oid))
    return cid


class MetadataContractTests(unittest.TestCase):
    def test_exact_observed_names_are_host_only(self):
        for name in NAMES:
            self.assertEqual(aa.api_kind(name), 'host', name)
        self.assertIsNone(aa.api_kind('cuMadeUpMetadataQuery'))
        self.assertIsNone(aa.api_kind('cuKernelGetFunctionAndLaunch'))
        self.assertEqual(aa.api_kind('cuLaunchKernel'), 'kernel')
        self.assertEqual(aa.api_kind('cudaMemcpyAsync'), 'memcpy')

    def test_outside_scope_zero_work_queries_are_conserved(self):
        fix = fixture.positive()
        for name in NAMES:
            extra_api(fix, name)
        out = fixture.compose(fix)
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['conservation']['host_apis'], len(NAMES))
        self.assertEqual(out['whole']['activity_records'], fix['native']['activity_records'])

    def test_offer_scoped_metadata_queries_are_conserved(self):
        fix = fixture.positive()
        for name in NAMES:
            extra_api(fix, name, offered=True)
        out = fixture.compose(fix)
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['conservation']['host_apis'], 0)

    def test_failed_host_query_refuses_inside_and_outside_offer(self):
        for offered in (False, True):
            fix = fixture.positive()
            extra_api(fix, NAMES[0], offered=offered, rc=1)
            out = fixture.compose(fix)
            self.assertEqual(out['verdict'], 'FAIL', out)

    def test_host_name_cannot_hide_kernel_activity(self):
        fix = fixture.positive()
        cid = fix['native']['activities'][0]['correlation_id']
        next(row for row in fix['native']['apis'] if row['correlation_id'] == cid)['name'] = NAMES[0]
        out = fixture.compose(fix)
        self.assertEqual(out['verdict'], 'FAIL', out)

    def test_loss_flags_remain_binding(self):
        fix = fixture.positive()
        for name in NAMES:
            extra_api(fix, name)
        fix['native']['overflow'] = 1
        out = fixture.compose(fix)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
        self.assertIn('whole_raw_loss', out['missing'])

    def test_query_declarations_exist_in_pinned_headers(self):
        paths = (ROOT/'reference/cuda-13/cuda.h', ROOT/'reference/cuda-runtime-13/cuda_runtime_api.h')
        if not all(p.exists() for p in paths):
            self.skipTest('read-only pinned CUDA13 headers unavailable')
        text = '\n'.join(p.read_text() for p in paths)
        for name in NAMES:
            stem = name.split('_v')[0]
            self.assertIn(stem + '(', text)


class ScopedGradeTests(unittest.TestCase):
    def test_negative_requires_actual_applicable_evidence(self):
        record = old_fixture.NegativeControlTests()._record()
        for key in ('output_exact', 'owners_restored', 'sharing_preserved'):
            del record[key]
        self.assertEqual(job.case_grade(record)['verdict'], 'PASS')
        record['raw_unchanged'] = False
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')
        record['raw_unchanged'] = True
        record['negative']['failures'] = ['wrong_stream']
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')

    def test_positive_still_requires_outputs_and_restoration(self):
        for key in ('output_exact', 'owners_restored', 'sharing_preserved'):
            record = old_fixture.positive_record()
            del record[key]
            self.assertEqual(job.case_grade(record)['verdict'], 'INCONCLUSIVE')

    def test_actual_replay_sites_still_use_retained_owner(self):
        spy = abi.NativeABISpy()
        sites = abi.native_replay_sites(job)
        self.assertEqual(len(sites), 2)
        for site in sites:
            self.assertEqual(eval(compile(ast.Expression(body=site), job.__file__, 'eval'),
                dict(gl=types.SimpleNamespace(lib=spy), graph_owner=abi.OWNER,
                     graph_handle=abi.GRAPH, replay_owned_graph=job.replay_owned_graph)), 0)
        self.assertEqual(spy.replayed, [abi.OWNER, abi.OWNER])

    def test_fresh_queue_and_raw_metadata_retention(self):
        script = ROOT/'harness/runs/vmid_job13c.sh'
        text = script.read_text()
        for value in ('runs/job13c', 'flashnext-job13c', 'vmid_job13c.py', '/opt/flashnext/.job.lock', '--network none'):
            self.assertIn(value, text)
        subprocess.run(['bash', '-n', str(script)], check=True)
        source = Path(job.__file__).read_text()
        self.assertIn('decode_declared=', source)
        self.assertIn("r['raw_unchanged'] = native_before ==", source)
        self.assertIn('set_per_process_memory_fraction(.125, 0)', source)


if __name__ == '__main__':
    unittest.main()
