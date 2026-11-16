"""Actual owner-copy argv -> both verified retained CT copies, with the PVE IO leaf local."""
import hashlib,json,os,shlex,subprocess,sys,tempfile,unittest
from pathlib import Path
from test_vision_fit_8x200k import CACHES
ROOT=Path(__file__).resolve().parents[1]
class Stage(unittest.TestCase):
    def test_whole_copy_protocol_and_wrong_remote_sha_refusal_mutant(self):
        root=Path(tempfile.mkdtemp(prefix='vision-fit-copy-'));(root/'data').mkdir()
        stub=root/'pve.py';stub.write_text("#!/usr/bin/env python3\nimport os,shlex,subprocess,sys\na=shlex.split(sys.argv[1]);assert a[:5]==['pct','exec','<ct-id>','--','python3'];raise SystemExit(subprocess.call(a[4:],env=os.environ))\n")
        env=dict(os.environ,R=str(root),PYTHONDONTWRITEBYTECODE='1')
        argv=[sys.executable,'-B',str(ROOT/'tools/vision_fit_stage_inputs.py'),'--private-cache-dir',str(CACHES),'--ct-run','/data/cpu-copy',
              '--out',str(root/'stage.json'),'--pve-cmd',shlex.join([sys.executable,str(stub)])]
        (root/'argv.json').write_text(json.dumps(argv))
        result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=30)
        (root/'stdout').write_text(result.stdout);(root/'stderr').write_text(result.stderr)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('VISION_FIT_STAGE_INPUTS PASS',result.stdout)
        for name in ('semantic','eight-unique'):
            copy=root/'data/cpu-copy/private'/(name+'.json')
            self.assertEqual(hashlib.sha256(copy.read_bytes()).digest(),hashlib.sha256((CACHES/copy.name).read_bytes()).digest())
            self.assertEqual(copy.stat().st_mode & 0o777,0o600)
        from vision_fit_stage_inputs import REMOTE
        command=[sys.executable,'-B','-c',REMOTE,'verify','/data/cpu-copy','0'*64,'0'*64]
        refused=subprocess.run(command,env=env,capture_output=True,text=True)
        self.assertNotEqual(refused.returncode,0);self.assertIn('CT-final-cache-SHA-mismatch',refused.stderr)
        mutant=command[:];mutant[3]=REMOTE.replace("if actual!=expected:","if False:")
        bypass=subprocess.run(mutant,env=env,capture_output=True,text=True)
        self.assertEqual(bypass.returncode,0);self.assertIn('VISION_FIT_CT_INPUTS PASS',bypass.stdout)
        again=subprocess.run(argv,env=env,capture_output=True,text=True)
        self.assertEqual(again.returncode,2)
