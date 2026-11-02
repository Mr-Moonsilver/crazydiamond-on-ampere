#!/usr/bin/env python3
"""tools/runner_staging_check.py over a runner in the frozen runner's shape: the staging requires found by AST and evaluated
against records, a missing case named, an unbound name a HARNESS failure, a lost reason a failure, sha(Path(__file__)) the runner's
own sha. The connected case is the real frozen runner over the exact delivered records (receipt named in delivery.json)."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_staging_check as rc  # noqa: E402

RUNNER = '''
BUNDLE='%s'
IMAGE='img@sha256:1'
def require(condition,reason):
    if not condition:raise SystemExit(reason)
def main(argv=None):
    a=parse(argv)
    if a.staging:
        require(a.auth_env_file is not None,'staging-auth-and-selfrun-records-required')
        result=json.loads(a.selfrun_record.read_text())
        require(result.get('bundle_manifest')==BUNDLE and result.get('tests_run',0)==351
                and result.get('frozen_runner_sha256')==sha(Path(__file__)),'exact-frozen-selfrun-required')
        engine=json.loads(a.engine_selfrun_record.read_text())
        require(engine.get('manifest')==BUNDLE and engine.get('image')==IMAGE
                and sorted(engine.get('cases') or {})==['a-case', 'b-case']
                and all((c or {}).get('verdict')=='PASS' for c in engine['cases'].values()),'exact-engine-selfrun-required')
        return 0
    require(stage.get('verdict')=='PASS','staging-receipt-required')
'''
BUNDLE = 'b' * 64


class Check(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()  # the unit fixture's own scratch, as the harness's tests use
        self.addCleanup(holder.cleanup)
        self.dir = Path(holder.name)

    def runner(self, text=None):
        path = self.dir / 'runner.py'
        path.write_text(text if text is not None else RUNNER % BUNDLE)
        return path

    def records(self, path, **engine_changes):
        result = dict(bundle_manifest=BUNDLE, tests_run=351, frozen_runner_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        engine = dict(manifest=BUNDLE, image='img@sha256:1', cases={'a-case': dict(verdict='PASS'), 'b-case': dict(verdict='PASS')})
        engine.update(engine_changes)
        return dict(result=result, engine=engine)

    def test_exact_records_pass(self):
        path = self.runner()
        out = rc.evaluate(path.read_text(), path, self.records(path))
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['reasons_found'], list(rc.EXPECTED_REASONS))

    def test_a_case_the_runner_pins_but_the_self_run_lacks_is_named(self):
        path = self.runner()
        cases = {'a-case': dict(verdict='PASS'), 'c-new': dict(verdict='PASS')}
        out = rc.evaluate(path.read_text(), path, self.records(path, cases=cases))
        engine = out['blocks'][1]
        self.assertEqual((out['verdict'], engine['verdict'], engine['classification']), ('FAIL', 'FAIL', 'HARNESS'))
        self.assertEqual(engine['case_lists'], [dict(expected=2, got=2, missing=['b-case'], extra=['c-new'])])
        self.assertEqual(len(engine['failed_clauses']), 1)
        self.assertIn("['a-case', 'b-case']", engine['failed_clauses'][0])

    def test_the_runner_sha_is_the_frozen_runners(self):
        path = self.runner()
        records = self.records(path)
        records['result']['frozen_runner_sha256'] = '0' * 64
        out = rc.evaluate(path.read_text(), path, records)
        self.assertEqual(out['blocks'][0]['failed_clauses'], ["result.get('frozen_runner_sha256') == sha(Path(__file__))"])

    def test_a_name_the_check_does_not_bind_is_a_harness_failure(self):
        path = self.runner((RUNNER % BUNDLE).replace("engine.get('image')==IMAGE", "engine.get('image')==NEW_PIN()"))
        out = rc.evaluate(path.read_text(), path, self.records(path))
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertEqual((out['blocks'][1]['unbound'], out['blocks'][1]['classification']), (['NEW_PIN'], 'HARNESS'))

    def test_a_lost_reason_fails(self):
        path = self.runner((RUNNER % BUNDLE).replace("'exact-engine-selfrun-required'", "'renamed'"))
        out = rc.evaluate(path.read_text(), path, self.records(path))
        self.assertEqual((out['verdict'], out['reasons_missing']), ('FAIL', ['exact-engine-selfrun-required']))

    def test_cli_receipt(self):
        path = self.runner()
        records = self.records(path)
        for name in ('result', 'engine'):
            (self.dir / (name + '.json')).write_text(json.dumps(records[name]))
        out = self.dir / 'receipt.json'
        code = rc.main(['--runner', str(path), '--selfrun', str(self.dir / 'result.json'), '--engine', str(self.dir / 'engine.json'),
                        '--out', str(out)])
        receipt = json.loads(out.read_text())
        self.assertEqual((code, receipt['rc'], receipt['verdict'], receipt['schema']), (0, 0, 'PASS', rc.SCHEMA))
        self.assertEqual(receipt['runner_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            rc.main(['--runner', str(path), '--selfrun', str(self.dir / 'result.json'), '--engine', str(self.dir / 'engine.json'),
                     '--out', str(out)])


if __name__ == '__main__':
    unittest.main()
