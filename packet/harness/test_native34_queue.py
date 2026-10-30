"""Fresh weekend gate and exact protected preconditioning/restore bodies."""
import datetime
from pathlib import Path
import subprocess
import unittest
import native34_gate as gate

ROOT=Path(__file__).resolve().parent


def normalize_queue(text,arm):
    text=text.replace('q18a-native34-'+arm.lower(),'q18a-native33').replace('q18a_native34_'+arm.lower(),'q18a_native33')
    text=text.replace('from native34_gate import','from native31_gate import')
    begin=text.index('  if [[ $name == row4 ]]; then\n',text.index('qualification_phase()'))
    end=text.index('  else\n',begin)+len('  else\n')
    text=text[:begin]+text[end:]
    text=text.replace('  fi\n  rc=$?\n','  rc=$?\n')
    old=(ROOT/'runs/q18a_native33.draft.sh').read_text()
    a=text.index('  python3 "$H/native34_grade.py"');b=text.index('  restore_launcher',a)
    oa=old.index('  python3 "$H/native33_grade.py"');ob=old.index('  restore_launcher',oa)
    return text[:a]+old[oa:ob]+text[b:]


class FrozenProtocol(unittest.TestCase):
    def test_each_role_keeps_preconditioning_restore_and_no_global_blocking(self):
        parent=(ROOT/'runs/q18a_native33.draft.sh').read_text()
        for arm in ('I','S','O'):
            q=ROOT/'runs'/('q18a_native34_'+arm.lower()+'.draft.sh')
            self.assertEqual(normalize_queue(q.read_text(),arm),parent)
            self.assertIn('--timeout-s 90 --arm '+arm,q.read_text())
            settings=q.with_name(q.name.replace('.draft.sh','_settings.sh')).read_text()
            self.assertIn('FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM='+arm,settings)
            self.assertNotIn('CUDA_LAUNCH_BLOCKING=1',settings)
            for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
                path=q.with_name('q18a_native34_'+arm.lower()+suffix)
                self.assertEqual(subprocess.run(['bash','-n',str(path)],capture_output=True).returncode,0)

    def test_weekend_bounds_keep_full_cleanup_margin(self):
        self.assertEqual(gate.HARD_END,datetime.datetime(2026,11,2,6,tzinfo=datetime.timezone.utc).timestamp())
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE-1));self.assertTrue(gate.window_ok(gate.NOT_BEFORE))
        self.assertTrue(gate.window_ok(gate.HARD_END-gate.ACTIVE_AND_CLEANUP))
        self.assertFalse(gate.window_ok(gate.HARD_END-gate.ACTIVE_AND_CLEANUP+1))
        self.assertEqual(gate.SNAPSHOT_CAP,512*1024**2)


if __name__=='__main__':unittest.main()
