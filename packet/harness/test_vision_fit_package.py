"""Frozen-file pins, actual declaration argv semantics, and no-bank proof."""
import importlib.util,inspect,json,os,shlex,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKET=Path(os.environ.get('VISION_FIT_PACKET',ROOT))
class Package(unittest.TestCase):
    def identity(self):
        spec=importlib.util.spec_from_file_location('fit_frozen_identity',PACKET/'harness/engine_identity.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def test_root_harness_overlay_and_input_bindings_are_exact(self):
        for folder,manifest in ((PACKET,'PACKET-SHA256SUMS'),(PACKET/'harness','SHA256SUMS'),(PACKET/'harness/native30_overlay','SHA256SUMS')):
            result=subprocess.run(['sha256sum','--strict','--quiet','-c',manifest],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads((PACKET/'fit-plan.json').read_text());self.assertEqual(plan['native_workload_cap_s'],2700)
        env=json.loads((PACKET/'vision-container-env.json').read_text())
        self.assertEqual(env['FLASH_S5_FIT_MEMORY'],'e6')
        for key in ('FLASH_S5_DECODE_LAYERS','FLASH_S5_GRAPH_EVENTS'):
            self.assertIn(env.get(key,'0'),('0',None))
    def test_actual_vision_env_expansion_retains_json_and_unknown_flags_refuse(self):
        identity=self.identity()
        words='--enable-mm-embeds --limit-mm-per-prompt {"image":0,"video":0} --mm-processor-cache-gb 0 --skip-mm-profiling'
        cleaned=identity.clean_value('VISION_ARGS',words)
        self.assertEqual(cleaned[2],dict(image=0,video=0))
        self.assertEqual(cleaned[3:5],['--mm-processor-cache-gb','0'])
        value=identity.clean_args(['--unreviewed-new-flag','value'])[1]
        self.assertEqual(value.get('unclassified'),True)
        # A revert to shell-syntax parsing strips literal JSON quotes; the real launcher never evals this variable.
        with self.assertRaises(json.JSONDecodeError):identity.clean_args(shlex.split(words))
    def test_end_identity_has_a_separate_budget_and_keeps_model_hash(self):
        owner=(PACKET/'owner.sh').read_text()
        self.assertIn('1200s python3 -B "$PK/tools/vision_fit_identity.py"',owner)
        self.assertIn('2700s python3 -B "$PK/harness/vision_fit_8x200k.py"',owner)
        policy=json.loads((PACKET/'harness/e3_startup_mount_policy.json').read_text())
        self.assertNotIn('/model',policy)
        self.assertIn("['check',str(start),str(request),str(end)]",(PACKET/'tools/vision_fit_identity.py').read_text())
if __name__=='__main__':unittest.main()
