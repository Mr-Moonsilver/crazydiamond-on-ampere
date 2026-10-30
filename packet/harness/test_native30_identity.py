"""Native30-b real identity CLI/probe checks with Docker and elapsed-cost leaves."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import engine_identity as identity
import test_identity as fixtures

HERE=Path(__file__).resolve().parent
SETTINGS=HERE/'runs/q18a_native30_localize_d_settings.sh'
QUEUE=HERE/'runs/q18a_native30_localize_d.draft.sh'
IDENTITY_RECEIPTS=[]
OBSERVED_CAPTURE_S=170.5  # owner-reported; retained native28 queue also shows155.275s

class IdentityPath(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.IdentityTests('test_capture_before_after_through_qlib')
        with patch.object(fixtures,'TMP',Path(tempfile.gettempdir())):self.f.setUp()
        self.budget=int(re.search(r'^Q18A_IDENTITY_TIMEOUT_S=(\d+)$',SETTINGS.read_text(),re.M)[1])

    def test_actual_capture_check_and_active_file_mismatch(self):
        before=self.f.capture('native30-before',extra=('--timeout',str(self.budget)))
        after=self.f.capture('native30-after',extra=('--timeout',str(self.budget)))
        self.f.check(before,after)
        record=json.loads(before.read_text());self.assertTrue(record['complete'])
        self.assertGreater(record['io']['bytes_read'],0)
        self.f.file.write_bytes(b'changed active source\n')
        changed=self.f.capture('native30-changed',extra=('--timeout',str(self.budget)),want=1)
        self.f.check(before,changed,want=1)
        self.assertEqual(json.loads(changed.read_text())['check']['verdict'],'FAIL')
        IDENTITY_RECEIPTS.append(dict(case='real-capture-check',verdict='PASS',io=record['io'],active_mutation_verdict='FAIL',timeout_s=self.budget))

    def test_old_budget_timeout_rejected_with_safe_stage_and_new_budget_passes(self):
        real_run=subprocess.run;hash_calls=[]
        def modeled(args,*a,**kw):
            if args[:3]==['docker','exec','-i']:
                hash_calls.append(kw['timeout'])
                if kw['timeout']<OBSERVED_CAPTURE_S:
                    raise subprocess.TimeoutExpired(args,kw['timeout'],output=fixtures.DUMMY,stderr=fixtures.DUMMY)
            return real_run(args,*a,**kw)
        for label,budget,want in (('old120',120,2),('configured',self.budget,0)):
            target=self.f.root/(label+'.json');output=io.StringIO()
            argv=['engine_identity.py','capture',str(target),'--request',str(self.f.reqpath),'--defaults',str(self.f.defaults),'--timeout',str(budget)]
            with patch.dict(os.environ,self.f.env),patch.object(sys,'argv',argv),patch.object(identity.subprocess,'run',side_effect=modeled),contextlib.redirect_stdout(output):
                self.assertEqual(identity.main(),want)
            self.assertNotIn(fixtures.DUMMY,output.getvalue()+target.read_text())
            value=json.loads(target.read_text())
            IDENTITY_RECEIPTS.append(dict(case=label,verdict=value['check']['verdict'],diagnostic=value.get('diagnostic'),io=value.get('io'),modeled_hash_cost_s=OBSERVED_CAPTURE_S,timeout_s=budget))
            if want:
                self.assertFalse(value['complete']);self.assertEqual(value['diagnostic']['phase'],'mount-hash')
                self.assertEqual(value['diagnostic']['kind'],'timeout');self.assertEqual(value['diagnostic']['timeout_s'],120)
            else:self.assertTrue(value['complete']);self.f.check(target)
        self.assertEqual(hash_calls,[120,self.budget])

    def test_measured_budget_and_capture_outside_cleanup_margin(self):
        self.assertGreaterEqual(self.budget,2*OBSERVED_CAPTURE_S)
        source=QUEUE.read_text()
        self.assertIn('"$R/startup.json" --timeout "$Q18A_IDENTITY_TIMEOUT_S"',source)
        trap=source.split('on_exit() {',1)[1].split('\n}\n',1)[0]
        self.assertNotIn('capture_end ',trap)
        self.assertIn('capture_end "$Q18A_IDENTITY_TIMEOUT_S"',source)
        self.assertIn('--kill-after=420s 1200s',source)
        self.assertIn('EVIDENCE_DONE=1\nDONE=1\nexit "$rc"',source)
        original=(HERE/'runs/q18a_native28.draft.sh').read_text()
        for name in ('stop_engine','restore_launcher','restore_defaults','write_cleanup'):
            body=lambda s:s.split(name+'() {',1)[1].split('\n}\n',1)[0]
            self.assertEqual(body(source),body(original))
        subprocess.run(['bash','-n',str(QUEUE)],check=True)

if __name__=='__main__':unittest.main()
