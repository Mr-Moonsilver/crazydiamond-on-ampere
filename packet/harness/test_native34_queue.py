"""Fresh weekend gate and exact protected preconditioning/restore bodies."""
import datetime
import hashlib,json,os,sys

SLOT_RECEIPTS=[]
from pathlib import Path
import subprocess
import unittest
import native34_gate as gate

ROOT=Path(__file__).resolve().parent


def normalize_queue(text,arm):
    for revision in ('2',''):
        text=text.replace('q18a-native34-'+arm.lower()+revision,'q18a-native33').replace('q18a_native34_'+arm.lower()+revision,'q18a_native33')
    text=text.replace('from native34_gate import','from native31_gate import')
    begin=text.index('  if [[ $name == row4 ]]; then\n',text.index('qualification_phase()'))
    end=text.index('  else\n',begin)+len('  else\n')
    text=text[:begin]+text[end:]
    text=text.replace('  fi\n  rc=$?\n','  rc=$?\n')
    old=(ROOT/'runs/q18a_native33.draft.sh').read_text()
    a=text.index("python3 - <<'PY_SLOT'");b=text.index('\nPY_SLOT',a)+len('\nPY_SLOT')
    oa=old.index("python3 - <<'PY_SLOT'");ob=old.index('\nPY_SLOT',oa)+len('\nPY_SLOT')
    text=text[:a]+old[oa:ob]+text[b:]
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

    def test_every_draft_window_slot_executes_at_real_wall_clock(self):
        env=dict(os.environ,HARNESS_DIR=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
        for arm in ('i','s','o'):
            for revision in ('','2'):
                path=ROOT/'runs'/('q18a_native34_'+arm+revision+'.draft.sh')
                source=path.read_text();begin=source.index("python3 - <<'PY_SLOT'")
                end=source.index('\nPY_SLOT',begin)+len('\nPY_SLOT')
                actual=source[begin:end]
                self.assertIn('from native34_gate import window_ok',actual)
                self.assertNotIn('datetime',actual)
                # Only the entry slot runs: no timeout wrapper, engine path or
                # network command. Its real Python process reads the real clock.
                code='fail_early() { return 2; }\n'+actual
                completed=subprocess.run(['bash','-c',code],env=env,capture_output=True,text=True)
                self.assertEqual(completed.returncode,0,completed.stderr)
                SLOT_RECEIPTS.append(dict(draft=path.name,verdict='PASS',rc=0,
                    slot_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                    real_wall_clock=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    engine_contact=False))
        previous=(ROOT/'native34_window_before.txt').read_text()
        rejected=subprocess.run([sys.executable,'-B','-c',previous],env=env,capture_output=True,text=True)
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('queue worst case exceeds owner hard end',rejected.stderr)
        SLOT_RECEIPTS.append(dict(draft='frozen native34-I attempt1 inline slot',verdict='REFUSED',
            rc=rejected.returncode,real_wall_clock=True,engine_contact=False,
            source_sha256=hashlib.sha256(previous.encode()).hexdigest(),
            refusal='queue worst case exceeds owner hard end'))

    def test_weekend_bounds_keep_full_cleanup_margin(self):
        self.assertEqual(gate.HARD_END,datetime.datetime(2026,11,2,6,tzinfo=datetime.timezone.utc).timestamp())
        self.assertFalse(gate.window_ok(gate.NOT_BEFORE-1));self.assertTrue(gate.window_ok(gate.NOT_BEFORE))
        self.assertTrue(gate.window_ok(gate.HARD_END-gate.ACTIVE_AND_CLEANUP))
        self.assertFalse(gate.window_ok(gate.HARD_END-gate.ACTIVE_AND_CLEANUP+1))
        self.assertEqual(gate.SNAPSHOT_CAP,512*1024**2)


if __name__=='__main__':unittest.main()
