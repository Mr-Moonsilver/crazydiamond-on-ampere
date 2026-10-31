"""Native35 queues: exact derivation from frozen native34 O3/I4 and native28 texts, real-clock slots, unchanged weekend bounds."""
import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

import native34_gate
import native35_gate as gate

SLOT_RECEIPTS=[]
ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs'
LEDGER=('python3 "$H/native35_ledger_grade.py" --startup "$Q/startup-grade.json" --out "$Q/native35-ledger.json" > "$Q/native35-ledger.log" 2>&1\n'
        "record native35-ledger \"$?\" 'per-rank native-private builder bytes allocated once; fail-closed field classification; no headroom claim'\n")
O3_GRADE=('  python3 "$H/native34_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native34-grade.json" > "$Q/native34-grade.log" 2>&1\n'
          "  record native34-diagnostic \"$?\" 'bounded causal arm and indexer discriminator; no qualification claim'\n")
I4_GRADE=O3_GRADE.replace('"$H/native34_grade.py" --log','"$H/native34_grade.py" --require-readback --log')
O_GRADE=('  python3 "$H/native35_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
         "  record native35-diagnostic \"$?\" 'unfenced O-style arm; no qualification claim'\n")
R_GRADE=('  python3 "$H/native35_grade.py" --require-readback --require-native-writer --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
         "  record native35-diagnostic \"$?\" 'unfenced O-style arm with evidence-only readback and native-private writer catalogue; no qualification claim'\n")
L_GRADE=('  python3 "$H/native35_lifetime_grade.py" --queue "$Q" --out "$Q/native35-lifetime-grade.json" > "$Q/native35-lifetime-grade.log" 2>&1\n'
         "  record native35-lifetime \"$?\" 'native cancel/drain/reuse plus unchanged hygiene/leak oracles; no performance claim'\n")
TAIL_START='\n# The real phase producer has joined/stopped its owned clients before returning.\n'
L_ENDING=('stop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
    'stop_engine || true\nrestore_launcher || true\nrestore_defaults || true\narchive_owned_next || true\nwrite_cleanup\nDONE=1\n'
    '[[ $FAILS == 0 ]] || exit 1\n[[ $INCS == 0 ]] || exit 2\nexit 0\n')


def once(text,old,new):
    assert text.count(old)==1,old[:60];return text.replace(old,new)


def names(text,src,dst):
    return text.replace('q18a-native35-'+src,'q18a-native34-'+dst).replace('q18a_native35_'+src,'q18a_native34_'+dst)


def native28_blocks():
    text=(RUNS/'q18a_native28.draft.sh').read_text()
    g0=text.index('RULE4_CONTINUATION=0\n');g1=text.index('gate_rc=$?;step overlap-gates "$gate_rc"\n',g0);gates=text[g0:g1]
    gates=gates[:gates.index('qualification_ready=1\n')]+gates[gates.index("Q18A_PHASE_VERDICTS='{'\n"):]
    gates=once(gates,'for _gate in startup row1 rule4_controls rule4_candidate row4 w1_controls; do\n','for _gate in startup row1 rule4_controls rule4_candidate row4; do\n')
    l0=text.index('# Native cancellation and persistent reuse are measured separately from the\n');l1=text.index('    identity "$Q/canary.before.json"',l0)
    return gates+'gate_rc=$?;step lifetime-gates "$gate_rc"\n',text[l0:l1]


class Derivation(unittest.TestCase):
    def test_r_and_o_are_native34_o3_plus_ledger_grade_and_readback_flags(self):
        o3={s:(RUNS/('q18a_native34_o3'+s)).read_text() for s in ('.draft.sh','_settings.sh','_declare.sh')}
        for arm,grade in (('r',R_GRADE),('o',O_GRADE)):
            with self.subTest(arm=arm):
                draft=names((RUNS/('q18a_native35_'+arm+'.draft.sh')).read_text(),arm,'o3')
                self.assertIn('--timeout-s 90 --arm O',draft);self.assertIn('2100s',draft)
                draft=once(draft,grade,O3_GRADE);draft=once(draft,LEDGER,'')
                self.assertEqual(draft,o3['.draft.sh'])
                settings=names((RUNS/('q18a_native35_'+arm+'_settings.sh')).read_text(),arm,'o3')
                if arm=='r':
                    self.assertIn('FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM=O FLASH_NATIVE34_READBACK=1 ',settings)
                    settings=once(settings,' FLASH_NATIVE34_READBACK=1','')
                else:self.assertNotIn('READBACK',settings)
                self.assertEqual(settings,o3['_settings.sh'])
                self.assertEqual(names((RUNS/('q18a_native35_'+arm+'_declare.sh')).read_text(),arm,'o3'),o3['_declare.sh'])

    def test_l_is_native33_form_plus_native28_lifetime_and_leak_phases(self):
        i4=(RUNS/'q18a_native34_i4.draft.sh').read_text();draft=names((RUNS/'q18a_native35_l.draft.sh').read_text(),'l','i4')
        self.assertNotIn('e3_native34_causal.py',draft);self.assertNotIn('pp_canary_bridge.py prepare',draft);self.assertNotIn('e3_native_w1.py "$PROBE_URL"',draft)
        self.assertIn('python3 "$H/e3_q18a_qualification.py" --phase "$name"',draft)
        for tool in ('e3_q18a_lifecycle.py','leaktest.py" 8 3','pressure2.py','leak_partial_tail.py','pp_release_grade.py" leak','pp_release_grade.py" lifetime','pressure_wait.py'):
            self.assertIn(tool,draft)
        draft=once(draft,'# Native35-l: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> native lifetime -> hygiene/leak -> owned restore.\n'
            '# Inherited cleanup machinery below; W1/performance/canary phases are absent.\n',
            '# Native30-e: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> owned restore.\n'
            '# Inherited cleanup machinery below; later qualification/performance phases are absent.\n')
        draft=once(draft,'--kill-after=420s 7200s env','--kill-after=420s 2100s env')
        draft=once(draft,'# entry unless its 7200 s timeout plus 420 s kill margin fits that window.\n','# entry unless its 2100 s timeout plus 420 s kill margin fits that window.\n')
        draft=once(draft,'from native35_gate import lifetime_window_ok as window_ok\n','from native34_gate import window_ok\n')
        draft=once(draft,L_GRADE,I4_GRADE);draft=once(draft,LEDGER,'')
        # Row4 is the native33-form real placement qualification: I4 minus its causal branch.
        i4_norm=i4[:i4.index('  if [[ $name == row4 ]]; then\n',i4.index('qualification_phase()'))]+i4[i4.index('  else\n',i4.index('qualification_phase()'))+len('  else\n'):]
        i4_norm=once(i4_norm,'  fi\n  rc=$?\n','  rc=$?\n')
        self.assertEqual(draft[:draft.index(TAIL_START)],i4_norm[:i4_norm.index(TAIL_START)])
        rest=draft[draft.index(TAIL_START):];i4_rest=i4_norm[i4_norm.index(TAIL_START):]
        stop='# Stop after native28 row4; no W1/performance/lifecycle queue follows.\n'
        receipt=i4_rest[:i4_rest.index(stop)];self.assertTrue(rest.startswith(receipt))
        gates,lifetime=native28_blocks()
        expected=gates+lifetime+"  fi\nelse\n  record lifecycle 2 'native lifetime and hygiene NOT_RUN after missing gates or engine crash'\nfi\n\n"+L_ENDING
        self.assertEqual(rest[len(receipt):],expected)
        settings=names((RUNS/'q18a_native35_l_settings.sh').read_text(),'l','i4')
        self.assertIn(' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ',settings);self.assertNotIn('FLASH_NATIVE34_ARM',settings);self.assertNotIn('READBACK',settings)
        settings=once(settings,' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ',' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM=I FLASH_NATIVE34_READBACK=1 ')
        self.assertEqual(settings,(RUNS/'q18a_native34_i4_settings.sh').read_text())
        self.assertEqual(names((RUNS/'q18a_native35_l_declare.sh').read_text(),'l','i4'),(RUNS/'q18a_native34_i4_declare.sh').read_text())

    def test_every_script_parses_and_no_global_blocking(self):
        for arm in ('r','o','l'):
            for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
                path=RUNS/('q18a_native35_'+arm+suffix)
                self.assertEqual(subprocess.run(['bash','-n',str(path)],capture_output=True).returncode,0,path.name)
                self.assertNotIn('CUDA_LAUNCH_BLOCKING=1',path.read_text())


class Window(unittest.TestCase):
    def test_weekend_bounds_unchanged_and_lifetime_budget_fits_same_hard_end(self):
        self.assertEqual(native34_gate.NOT_BEFORE,datetime.datetime(2026,10,30,22,1,tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual(native34_gate.HARD_END,datetime.datetime(2026,11,2,6,tzinfo=datetime.timezone.utc).timestamp())
        self.assertIs(gate.window_ok,native34_gate.window_ok);self.assertEqual((gate.NOT_BEFORE,gate.HARD_END),(native34_gate.NOT_BEFORE,native34_gate.HARD_END))
        self.assertEqual((gate.LIFETIME_ACTIVE_S,gate.CLEANUP_MARGIN_S,gate.LIFETIME_ACTIVE_AND_CLEANUP),(7200,420,7620))
        self.assertFalse(gate.lifetime_window_ok(gate.NOT_BEFORE-1));self.assertTrue(gate.lifetime_window_ok(gate.NOT_BEFORE))
        self.assertTrue(gate.lifetime_window_ok(gate.HARD_END-7620));self.assertFalse(gate.lifetime_window_ok(gate.HARD_END-7620+1))
        self.assertGreater(gate.LIFETIME_ACTIVE_AND_CLEANUP,gate.ACTIVE_AND_CLEANUP);self.assertEqual(gate.SNAPSHOT_CAP,512*1024**2)

    def test_every_native35_draft_window_slot_executes_at_real_wall_clock(self):
        env=dict(os.environ,HARNESS_DIR=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
        for arm in ('r','o','l'):
            path=RUNS/('q18a_native35_'+arm+'.draft.sh');source=path.read_text()
            begin=source.index("python3 - <<'PY_SLOT'");end=source.index('\nPY_SLOT',begin)+len('\nPY_SLOT');actual=source[begin:end]
            self.assertIn('from native35_gate import lifetime_window_ok as window_ok' if arm=='l' else 'from native34_gate import window_ok',actual)
            self.assertNotIn('datetime',actual)
            code='fail_early() { return 2; }\n'+actual
            completed=subprocess.run(['bash','-c',code],env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            SLOT_RECEIPTS.append(dict(draft=path.name,verdict='PASS',rc=0,slot_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                real_wall_clock=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),engine_contact=False,
                active_s=7200 if arm=='l' else 2100,cleanup_margin_s=420))
        previous=(ROOT/'native34_window_before.txt').read_text()
        rejected=subprocess.run([sys.executable,'-B','-c',previous],env=env,capture_output=True,text=True)
        self.assertNotEqual(rejected.returncode,0);self.assertIn('queue worst case exceeds owner hard end',rejected.stderr)
        SLOT_RECEIPTS.append(dict(draft='frozen native34-I attempt1 inline slot',verdict='REFUSED',rc=rejected.returncode,real_wall_clock=True,
            engine_contact=False,source_sha256=hashlib.sha256(previous.encode()).hexdigest(),refusal='queue worst case exceeds owner hard end'))


if __name__=='__main__':unittest.main()
