"""Real owner argv/identity parser and full capture/check against retained CPU IO leaves."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import vision_fit_identity as helper
import vision_fit_finish as final
import test_identity as fixtures

class Identity(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='vision-fit-identity-'));self.packet=self.root/'packet';self.packet.mkdir()
        (self.packet/'harness').mkdir();(self.packet/'image-id.txt').write_text(fixtures.IMAGE+'\n')
        # Keep the existing identity fixture/probe; its legacy cleanup is deliberately NOT invoked.
        self.fixture=fixtures.IdentityTests();saved=fixtures.TMP;fixtures.TMP=self.root
        try:self.fixture.setUp()
        finally:fixtures.TMP=saved
        self.runroot=self.root/'root';(self.runroot/'etc/default').mkdir(parents=True)
        (self.runroot/'etc/default/vllm-flash').write_bytes(self.fixture.defaults.read_bytes())
        for name in ('engine_identity.py','identity_probe.py','identity_declare.py'):
            (self.packet/'harness'/name).write_bytes((ROOT/'harness'/name).read_bytes())
        shim=self.root/'pythonshim';shim.mkdir()
        # Only source declaration reads are synthetic. Actual capture hashes fixture mounts,
        # validates request/capture schemas, argv, default parsing, start/end equality and guards.
        (shim/'sitecustomize.py').write_text("import json,os,sys\nfrom pathlib import Path\nsys.path.insert(0,os.environ['IDENTITY_H'])\nimport identity_declare\ndef declare(a):\n Path(a.out).write_text(Path(os.environ['IDENTITY_FIXTURE']).read_text());return 0\nidentity_declare.run=declare\n")
        self.env=dict(self.fixture.env,R=str(self.runroot),PYTHONPATH=str(shim),IDENTITY_H=str(self.packet/'harness'),IDENTITY_FIXTURE=str(self.fixture.reqpath))
        self.out=self.root/'identity';self.out.mkdir()
    def invoke(self,phase,script=None):
        argv=[sys.executable,'-B',str(script or ROOT/'tools/vision_fit_identity.py'),'--packet',str(self.packet),'--out',str(self.out),'--phase',phase]
        (self.root/(phase+'.argv.json')).write_text(json.dumps(argv))
        result=subprocess.run(argv,env=self.env,capture_output=True,text=True,timeout=30)
        (self.root/(phase+'.stdout')).write_text(result.stdout);(self.root/(phase+'.stderr')).write_text(result.stderr)
        return result
    def test_actual_argv_start_end_capture_and_check(self):
        a=self.invoke('start');self.assertEqual(a.returncode,0,a.stderr)
        b=self.invoke('end');self.assertEqual(b.returncode,0,b.stderr)
        self.assertIn('VISION_FIT_IDENTITY PASS end',b.stdout)
        self.assertEqual(json.loads((self.out/'identity-end.json').read_text())['check']['verdict'],'PASS')
    def test_actual_end_identity_mutation_refuses(self):
        self.assertEqual(self.invoke('start').returncode,0)
        self.fixture.file.write_text('changed source\n')
        result=self.invoke('end');self.assertEqual(result.returncode,2)
        self.assertIn('full-identity-capture-end-rc=1',result.stderr)
    def test_old_check_argv_revert_mutant_is_rejected_by_real_parser(self):
        script=self.root/'bad-helper.py';script.write_text((ROOT/'tools/vision_fit_identity.py').read_text().replace("['check',str(start),str(request)]","['check',str(start),'--request',str(request)]"))
        self.env['PYTHONPATH']+=os.pathsep+str(ROOT/'harness')
        result=self.invoke('start',script);self.assertEqual(result.returncode,2)
        self.assertIn('full-identity-check-start-rc=2',result.stderr)

class Finish(unittest.TestCase):
    def root(self):
        root=Path(tempfile.mkdtemp(prefix='vision-fit-final-'));(root/'serve-engine').mkdir()
        events=dict(oom=0,oom_kill=0)
        (root/'input-preflight.json').write_text(json.dumps(dict(host=dict(memory_events_last=events))))
        (root/'fit-read.json').write_text(json.dumps(dict(verdict='FIT',reasons=[],raw=dict(host=dict(memory_events_last=events)))))
        (root/'owned-stop.log').write_text('SERVE_STOP_STATE READY\n')
        for phase in ('start','end'):(root/(phase+'-identity.stdout.log')).write_text('VISION_FIT_IDENTITY PASS '+phase+'\n')
        (root/'host-stop-intent.json').write_text(json.dumps(dict(wall_ns=1763323200000000000)))
        (root/'engine-stream.log').write_text('2025-11-16T19:50:00Z INFO Application startup complete\n2025-11-16T19:50:01Z INFO 127.0.0.1 GET /health HTTP/1.1 200 OK\n')
        return root
    def test_stop_boundary_keeps_real_death_and_ignores_owned_teardown(self):
        root=self.root()
        # Real scanner's health/startup proof, exact wall-time parsing, and death rule remain live.
        result=final.finish(root,0,0);self.assertEqual(result['verdict'],'FIT',result)
        with (root/'engine-stream.log').open('a') as f:f.write('2025-11-16T19:55:00Z ERROR EngineDeadError: fixture\n')
        self.assertEqual(final.finish(root,0,0)['verdict'],'FAIL')
    def test_nonzero_phase_and_missing_identity_cannot_fit(self):
        root=self.root();self.assertEqual(final.finish(root,2,0)['verdict'],'HOLD')
        (root/'end-identity.stdout.log').write_text('')
        self.assertIn('end-identity-unproven',final.finish(root,0,0)['reasons'])

    def test_owned_teardown_boundary_revert_mutant(self):
        root=self.root()
        with (root/'engine-stream.log').open('a') as f:f.write('2025-11-16T21:00:00Z ERROR FLASH_S5_WORKER_FAILURE: fixture\n')
        value=final.finish(root,0,0)
        self.assertEqual(value['verdict'],'FIT',value)
        self.assertEqual(value['death']['ignored_after_owned_stop']['count'],1)
        from fit_death import OriginalScanner
        scan=OriginalScanner([str(root/'engine-stream.log')],str(root/'host-stop-intent.json'));scan.poll(final=True)
        self.assertEqual(scan.decide()['verdict'],'DEAD')

    def test_startup_cgroup_oom_cannot_disappear_in_a_later_baseline(self):
        root=self.root();value=json.loads((root/'fit-read.json').read_text())
        value['raw']['host']['memory_events_last']['oom_kill']=1
        (root/'fit-read.json').write_text(json.dumps(value))
        self.assertEqual(final.finish(root,0,0)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
