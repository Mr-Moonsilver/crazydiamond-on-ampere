"""The composition keeps prefetch execution/settings and adds only the j2 policies."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NEW='q18a_parity_n3s24q2024p_j2'

class Composition(unittest.TestCase):
    def test_real_settings_delta_is_only_j2_policies(self):
        import test_compose5_queue as prior
        values=[]
        for name in ('q18a_parity_n3s24q2024p',NEW):
            p=HERE/'runs'/(name+'_settings.sh')
            r=subprocess.run(['bash','-c','set -u; source "$1" || exit; printf "%s\\0" "${S5_SETTINGS[@]}"','bash',str(p)],
                env=dict(os.environ,**prior.ARGPARSE_VALUES),capture_output=True,check=True)
            settings=dict(v.split('=',1) for v in r.stdout.decode().split('\0') if v)
            extra=dict(v.split('=',1) for v in settings.pop('EXTRA_ENV').split())
            values.append((settings,extra))
        self.assertEqual(values[0][0],values[1][0])
        old,new=values[0][1],values[1][1]
        delta={k:v for k,v in new.items() if old.get(k)!=v}
        self.assertEqual(delta,dict(FLASH_S5_JUNCTION_YIELD='1',FLASH_S5_BULK_SHORTEST='1',FLASH_S5_APPEND_YIELD_PER_BULK='8'))
        self.assertEqual(set(new)-set(old),{'FLASH_S5_JUNCTION_YIELD','FLASH_S5_BULK_SHORTEST'})
        for k in old:
            if k!='FLASH_S5_APPEND_YIELD_PER_BULK':self.assertEqual(old[k],new[k],k)
        self.assertEqual(new['FLASH_S5_NEXT_LAYER_PREFETCH'],'1')
        self.assertEqual(new['FLASH_S5_FUSED_PHASE_EVERY'],'16')

    def test_actual_draft_grades_after_capture_and_keeps_prefetch_preflights(self):
        draft=(HERE/'runs'/(NEW+'.draft.sh')).read_text()
        self.assertLess(draft.rindex('capture_logs || true'),draft.index('timeout 120s python3 "$H/tools/n3j2_junction_rule.py"'))
        for suffix in ('.draft.sh','_declare.sh'):
            text=(HERE/'runs'/(NEW+suffix)).read_text()
            self.assertIn("assert env.get('FLASH_S5_NEXT_LAYER_PREFETCH') == '1'",text)
            self.assertIn("assert env.get('FLASH_S5_FUSED_PHASE_EVERY') == '16'",text)
        self.assertIn('N3J2_BUNDLE_MANIFEST',draft)

    def test_declared_runtime_inventory(self):
        import native_auto_arm,scheduler_hooks,engine_driver
        self.assertTrue(hasattr(native_auto_arm.AutoArmPolicy,'_shortest_bulk'))
        self.assertTrue(hasattr(scheduler_hooks.PrefillAdmissionController,'_yield_quota_receipts'))
        self.assertIn('controller._yield_quota_receipts',Path(engine_driver.__file__).read_text())
        source=ROOT/'patches/s5_engine/qsa_operation.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),'a4a9f3dbe6acd9b641ebbfc0e382d5bf3d43766f0a561c89a49586597546862a')

if __name__=='__main__':unittest.main()
