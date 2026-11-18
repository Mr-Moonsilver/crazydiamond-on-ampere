"""Frozen full-G4 protocol and shared production engine, no device/service actions."""
import hashlib,importlib.util,json,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'harness/e3_builder')]
import test_vision_fit_8x200k as fixture
import test_vision_fit_owner_protocol as owner_fixture
import build_vision_headroom_w28 as build
PAIR=Path(os.environ.get('W28_PAIR','<hub-dir>/cargo-target/flash-next/vision-headroom-w28-20261118/build-1'))
PROD=PAIR/'production';FIT=PAIR/'g4-full'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check_pair(prod=PROD,fit=FIT):
    for parent in (prod,fit):
        for folder,name in ((parent,'PACKET-SHA256SUMS'),(parent/'harness','SHA256SUMS'),(parent/'harness/native30_overlay','SHA256SUMS')):
            r=subprocess.run(['sha256sum','--strict','--quiet','-c',name],cwd=folder,capture_output=True,text=True)
            if r.returncode:raise ValueError('packet-manifest:'+r.stderr)
    if sha(prod/'harness/native30_overlay/SHA256SUMS')!=build.OVERLAY_SHA or sha(fit/'harness/native30_overlay/SHA256SUMS')!=build.OVERLAY_SHA:raise ValueError('headroom-overlay-differs')
    if sha(prod/'harness/SHA256SUMS')!=sha(fit/'harness/SHA256SUMS'):raise ValueError('shared-harness-differs')
    for name in (*build.FULL_FILES,*('harness/'+n for n in build.FULL_CLIENTS),'owner.sh','seq.sh','tools/vision_fit_finish.py','tools/vision_fit_identity.py'):
        if sha(fit/name)!=sha(build.FIT/name):raise ValueError('full-G4-protocol-moved:'+name)
    for p in (prod,fit):
        env=json.loads((p/'vision-container-env.json').read_text())
        if env.get('FLASH_S5_G4_HEADROOM')!='1':raise ValueError('headroom-flag-missing')
        if 'FLASH_S5_G4_HEADROOM=1' not in (p/'serve-launch.sh').read_text() or "'FLASH_S5_G4_HEADROOM=1'" not in (p/'serve_readback.py').read_text():raise ValueError('flag-not-launched-and-read-back')
    return True

class Binding(unittest.TestCase):
    def test_same_overlay_full_workload_and_flags(self):self.assertTrue(check_pair())
    def test_new_production_namespaces_retain_approved_credentials_and_clients(self):
        for name in ('gateway-start.sh','gateway-stop.sh','production-pin.sh','production-rollback.sh','units/flashnext-vision-gateway.service','units/native-ready.conf','tools/vision_production_pin_guard.py'):
            source=(PROD/name).read_text();self.assertNotIn(build.OLD_CT,source)
        ready=(PROD/'harness/vision_production_ready.py').read_text();self.assertIn(build.OVERLAY,ready)
        self.assertNotIn('overlay-v16s5-vision-prod-w27-20261117a',ready)
        self.assertIn('/root/.flashnext-serve-vision-prod-w27-20261117a/vllm-api.env',(PROD/'gateway-start.sh').read_text())
        for name in ('vision_g5_probe.py','vision_w27_read.py','vision_g6_capture.py','vision_prod_a3.py','vision_public_gateway.py'):
            self.assertEqual((PROD/'harness'/name).read_bytes(),(build.PRODUCTION/'harness'/name).read_bytes())
        self.assertEqual((PROD/'vision-assets/g5-cases.json').read_bytes(),(build.PRODUCTION/'vision-assets/g5-cases.json').read_bytes())
        self.assertEqual((PROD/'G6-RECEIPT-CONTRACT-w27-20261117.md').read_bytes(),(build.PRODUCTION/'G6-RECEIPT-CONTRACT-w27-20261117.md').read_bytes())
    def test_protocol_and_overlay_revert_mutants_are_refused(self):
        real=sha
        for target in ('harness/vision_fit_8x200k.py','harness/native30_overlay/SHA256SUMS'):
            def changed(path):return '0'*64 if path==FIT/target else real(path)
            with patch(__name__+'.sha',side_effect=changed),self.assertRaises(ValueError):check_pair()
    def test_pin_guard_register_and_dry_run_use_the_actual_new_packet_paths(self):
        spec=importlib.util.spec_from_file_location('w28_frozen_pins',PROD/'tools/vision_production_pin_guard.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.PK,build.CT_PROD)
        import tempfile
        root=Path(tempfile.mkdtemp(prefix='w28-pins-'))
        for name in module.TARGETS:
            path=module.rooted(root,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU_PIN '+name)
        module.register(root);self.assertEqual(module.guard(root,dry=True),[])
        module.rooted(root,module.TARGETS[0]).write_text('MUTANT')
        self.assertEqual(module.guard(root,dry=True),[module.TARGETS[0]])

class FullProtocol(unittest.TestCase):
    def setUp(self):
        for name in ('ROOT','PACKET'):
            p=patch.object(fixture,name,FIT);p.start();self.addCleanup(p.stop)
        fixture.Protocol.setUp(self)
    tearDown=fixture.Protocol.tearDown
    def test_frozen_generated_owner_through_every_stage_to_post_stop_FIT(self):
        packet,run,env,argv=owner_fixture.OwnerProtocol.owner_fixture(self)
        result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180)
        (self.root/'owner.stdout').write_text(result.stdout);(self.root/'owner.stderr').write_text(result.stderr)
        detail='\n'.join(p.read_text()[-1000:] for p in run.glob('*.stderr.log'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+detail+' retained '+str(run))
        self.assertIn('VISION_FIT_OWNER FIT',result.stdout);record=json.loads((run/'fit-collect.json').read_text())
        self.assertEqual([s['prompt_tokens'] for s in record['stages']],[60000,120000,160000,190320])
        self.assertTrue(record['tail']['complete']);self.assertTrue(record['cold']['complete'])
        self.assertEqual(record['cold']['decoders_live_at_first'],7)
        self.assertTrue(record['reuse']['same_image_cached']>0);self.assertTrue((run/'host-stop-intent.json').exists())
        self.assertEqual(json.loads((run/'arm-result.json').read_text())['verdict'],'FIT')
        print('W28_FULL_G4_COMPLETE '+str(run))

if __name__=='__main__':unittest.main()
