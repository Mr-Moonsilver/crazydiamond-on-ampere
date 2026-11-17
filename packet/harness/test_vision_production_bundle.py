import ast,json,re,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Templates(unittest.TestCase):
 def test_clients_actual_isolated_argv_parse(self):
  for name in ('vision_prod_a3','vision_g6_capture','vision_g5_probe','vision_w27_read','vision_public_gateway','vision_service_owner'):
   result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'harness'/(name+'.py')),'--help'],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,(name,result.stderr));self.assertIn('usage:',result.stdout)
 def test_service_alert_logging_manual_and_private_key_contract(self):
  text=(ROOT/'deploy/vision-production/flashnext-vision-encoder.service').read_text()
  for must in ('Restart=no','LogNamespace=flashnext-vision','ExecStopPost=+/usr/local/sbin/flashnext-encoder-alert','KillMode=mixed','TimeoutStopSec=90'):self.assertIn(must,text)
  gateway=(ROOT/'deploy/vision-production/gateway-start.sh').read_text()
  self.assertIn('--runtime runc',gateway);self.assertNotIn('--gpus',gateway);self.assertIn('600 0',gateway)
  unit=(ROOT/'deploy/vision-production/flashnext-vision-gateway.service').read_text();self.assertIn('ExecStartPre=/usr/bin/sha256sum --strict --quiet -c PACKET-SHA256SUMS',unit)
  self.assertIn('/root/.flashnext-',unit);self.assertNotIn('VLLM_API_KEY=',unit)
 def test_default_thinking_not_overridden_and_all_required_aliases_pinned(self):
  parent=Path('<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/serve-vision-fe-20261116a-build-7/harness/native30_overlay/run-flash-s5-test.sh').read_text()
  for name in ('qwen38-w8a8-tp4','cyankiwi/Qwen3.8-27B-AWQ-INT4','cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit','--reasoning-parser qwen3','--tool-call-parser qwen3_coder'):self.assertIn(name,parent)
  self.assertNotIn('--default-chat-template-kwargs',parent)
if __name__=='__main__':unittest.main()
