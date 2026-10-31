"""Native35 P queue: exact derivation from frozen native35 O and native28 texts, the declared speed
window at real wall clock, the extracted measurement tail over fake phase executables with the real
aggregate grade, and queue-grade.json.measured_performance from the real PY_FINAL. CPU only.

Fixture leaves: the W1/W2/memory phase executables are short fakes (their real code paths have
their own suites); systemctl/health/crash probes are shell functions. The tail text, the gates
producer, record/step and the aggregate grade are the queue's own bytes.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import native34_gate
import native35p_gate as gate

SLOT_RECEIPTS=[]
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs'
P_DRAFT=RUNS/'q18a_native35_p.draft.sh'
PB_DRAFT=RUNS/'q18a_native35_p_b.draft.sh'  # the candidate; executed by the window and tail tests
TAIL_START='\n# The real phase producer has joined/stopped its owned clients before returning.\n'
O_STOP_HEAD="# Stop after native28 row4; no W1/performance/lifecycle queue follows.\n"
GATES_END='gate_rc=$?;step overlap-gates "$gate_rc"\n'
L_ENDING=('stop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
    'stop_engine || true\nrestore_launcher || true\nrestore_defaults || true\narchive_owned_next || true\nwrite_cleanup\nDONE=1\n'
    '[[ $FAILS == 0 ]] || exit 1\n[[ $INCS == 0 ]] || exit 2\nexit 0\n')
# Every declared difference of the P draft head from frozen O (P text, O text).
HEAD_DIFFERENCES=(
    ('# Native35-p: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> W1 -> two retained W2 rounds -> owned restore.\n'
     '# Inherited cleanup machinery below; lifecycle/leak (L) and canary phases are absent.\n',
     '# Native30-e: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> owned restore.\n'
     '# Inherited cleanup machinery below; later qualification/performance phases are absent.\n'),
    ('--kill-after=900s 14400s env','--kill-after=420s 2100s env'),
    ('# entry unless its 14400 s timeout plus 900 s kill margin fits that window (native35p_gate derivation).\n',
     '# entry unless its 2100 s timeout plus 420 s kill margin fits that window.\n'),
    ('from native35p_gate import speed_window_ok as window_ok\n','from native34_gate import window_ok\n'),
    ('  pressure2.py pressure_wait.py leak_partial_tail.py leaktest.py \\\n  e3_native35_performance.py native35_performance_grade.py native35p_gate.py; do\n',
     '  pressure2.py pressure_wait.py leak_partial_tail.py leaktest.py; do\n'),
    ("  metrics={'verdict':'NOT_RUN','missing':['native35-performance-grade-absent'],'target_is_gate':False}\n"
     "  perf=p.parent/'native35-performance-grade.json'\n",
     "  metrics=None\n  perf=p.parent/'overlap.json'\n"),
    ("   try:metrics=json.load(open(perf))\n"
     "   except (OSError,ValueError):metrics={'verdict':'INCONCLUSIVE','missing':['performance-artifact-unreadable'],'target_is_gate':False}\n",
     "   try:metrics=json.load(open(perf)).get('grade')\n"
     "   except (OSError,ValueError):metrics={'verdict':'INCONCLUSIVE','missing':['performance-artifact-unreadable']}\n"),
)


def once(text,old,new):
    assert text.count(old)==1,(old[:70],text.count(old));return text.replace(old,new)


def as_o(text):
    return text.replace('q18a-native35-p','q18a-native35-o').replace('q18a_native35_p','q18a_native35_o')


def without_o_row4_branch(o):
    """O's queue minus its causal --arm O row4 branch: the native33-form real placement (as L)."""
    q=o.index('qualification_phase()');b=o.index('  if [[ $name == row4 ]]; then\n',q);e=o.index('  else\n',b)+len('  else\n')
    return once(o[:b]+o[e:],'  fi\n  rc=$?\n','  rc=$?\n')


def native28_gates_block():
    text=(RUNS/'q18a_native28.draft.sh').read_text()
    start=text.index('RULE4_CONTINUATION=0\n');end=text.index(GATES_END,start)+len(GATES_END)
    return text[start:end]


def p_parts(draft=P_DRAFT):
    p=draft.read_text();tail=p[p.index(TAIL_START):]
    receipt=tail[:tail.index('RULE4_CONTINUATION=0\n')]
    gates=native28_gates_block();assert tail.count(gates)==1
    measurement=tail[len(receipt)+len(gates):len(tail)-len(L_ENDING)]
    assert tail.endswith(L_ENDING)
    return p[:p.index(TAIL_START)],receipt,gates,measurement


class Derivation(unittest.TestCase):
    def test_p_head_is_o_minus_causal_row4_with_only_the_declared_differences(self):
        o=without_o_row4_branch((RUNS/'q18a_native35_o.draft.sh').read_text())
        head,receipt,gates,measurement=p_parts()
        head=as_o(head)
        for new,old in HEAD_DIFFERENCES:head=once(head,new,old)
        self.assertEqual(head,o[:o.index(TAIL_START)])
        o_tail=o[o.index(TAIL_START):]
        self.assertEqual(as_o(receipt),o_tail[:o_tail.index(O_STOP_HEAD)])
        self.assertNotIn('e3_native34_causal.py',P_DRAFT.read_text())
        self.assertIn('python3 "$H/e3_q18a_qualification.py" --phase "$name"',head)

    def test_tail_is_native28_gates_w1_block_then_two_rounds_then_the_l_ending(self):
        self.assertEqual(p_parts(PB_DRAFT)[1:],p_parts(P_DRAFT)[1:])  # P-b tail = P-a tail, byte for byte
        _,_,gates,measurement=p_parts()
        self.assertIn('python3 "$H/e3_native_w1.py" "$PROBE_URL" "$MODEL" "$Q/w1.json"',gates)
        self.assertIn('for _gate in startup row1 rule4_controls rule4_candidate row4 w1_controls; do',gates)
        self.assertIn('  for _round in 0 1; do\n',measurement)
        self.assertEqual(measurement.count('python3 "$H/e3_native35_performance.py"'),1)
        self.assertEqual(measurement.count('python3 "$H/e3_native_memory_grade.py"'),1)
        self.assertNotIn('e3_native_performance.py"',measurement)  # frozen collector runs only through the round CLI
        code='\n'.join(line for line in measurement.splitlines() if not line.lstrip().startswith('#'))
        for absent in ('e3_q18a_lifecycle.py','leaktest.py"','pressure2.py"','pp_canary_bridge.py','sleep','retry','restart "$SERVICE"'):
            self.assertNotIn(absent,code)
        self.assertTrue(measurement.rstrip().endswith("80/80 is a marker, never a gate'"))

    def test_settings_are_o_with_diagnostics_off_and_the_native28_read_bound(self):
        p=as_o((RUNS/'q18a_native35_p_settings.sh').read_text());o=(RUNS/'q18a_native35_o_settings.sh').read_text()
        self.assertIn(' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 FLASH_NATIVE30_SYNC=',p);self.assertNotIn('FLASH_NATIVE34_ARM',p)
        self.assertIn('FLASH_NATIVE30_LOCALIZE=1',p)  # inherited, unchanged
        p=once(p,' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ',' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM=O ')
        p=once(p,'Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\n','Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=480\n')
        self.assertEqual(p,o)
        # The EXTRA_ENV fragment is exactly L's (the other DIAG=0 arm); native28 declared the same 1200 s bound.
        l=(RUNS/'q18a_native35_l_settings.sh').read_text().replace('q18a-native35-l','q18a-native35-p')
        self.assertEqual(once(l,'Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=480\n','Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\n'),(RUNS/'q18a_native35_p_settings.sh').read_text())
        self.assertIn('Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\nexport Q18A_FIRST_ADVANCE_READ_TIMEOUT_S\n',(RUNS/'q18a_native28_settings.sh').read_text())

    def test_declare_pins_the_three_new_harness_files_and_nothing_else_changes(self):
        p=as_o((RUNS/'q18a_native35_p_declare.sh').read_text())
        p=once(p,'      e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n','')
        self.assertEqual(p,(RUNS/'q18a_native35_o_declare.sh').read_text())

    def test_every_p_script_parses_and_no_global_blocking(self):
        for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
            path=RUNS/('q18a_native35_p'+suffix)
            self.assertEqual(subprocess.run(['bash','-n',str(path)],capture_output=True).returncode,0,path.name)
            self.assertNotIn('CUDA_LAUNCH_BLOCKING=1',path.read_text())


class Window(unittest.TestCase):
    def test_bounds_unchanged_and_budget_derivation_fits_the_declared_limits(self):
        self.assertEqual((gate.NOT_BEFORE,gate.HARD_END),(native34_gate.NOT_BEFORE,native34_gate.HARD_END))
        self.assertEqual(gate.HARD_END,datetime.datetime(2026,11,2,6,tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual((gate.SPEED_ACTIVE_S,gate.SPEED_CLEANUP_MARGIN_S,gate.SPEED_ACTIVE_AND_CLEANUP),(14400,900,15300))
        self.assertLessEqual(sum(s for _,s,_ in gate.ACTIVE_DERIVATION),gate.SPEED_ACTIVE_S)
        self.assertLessEqual(sum(s for _,s,_ in gate.CLEANUP_DERIVATION),gate.SPEED_CLEANUP_MARGIN_S)
        for draft in (P_DRAFT,PB_DRAFT):self.assertIn('--kill-after=%ds %ds env'%(gate.SPEED_CLEANUP_MARGIN_S,gate.SPEED_ACTIVE_S),draft.read_text())
        self.assertFalse(gate.speed_window_ok(gate.NOT_BEFORE-1));self.assertTrue(gate.speed_window_ok(gate.NOT_BEFORE))
        self.assertTrue(gate.speed_window_ok(gate.HARD_END-15300));self.assertFalse(gate.speed_window_ok(gate.HARD_END-15300+1))
        self.assertLess(gate.EXPECTED_RUNTIME_S['conservative'],gate.SPEED_ACTIVE_S)

    def test_p_draft_window_slot_executes_at_real_wall_clock(self):
        env=dict(os.environ,HARNESS_DIR=str(HERE),PYTHONDONTWRITEBYTECODE='1')
        for draft in (P_DRAFT,PB_DRAFT):
            source=draft.read_text();begin=source.index("python3 - <<'PY_SLOT'");end=source.index('\nPY_SLOT',begin)+len('\nPY_SLOT');actual=source[begin:end]
            self.assertIn('from native35p_gate import speed_window_ok as window_ok',actual);self.assertNotIn('datetime',actual)
            completed=subprocess.run(['bash','-c','fail_early() { return 2; }\n'+actual],env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            SLOT_RECEIPTS.append(dict(draft=draft.name,verdict='PASS',rc=0,slot_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                real_wall_clock=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),engine_contact=False,
                active_s=gate.SPEED_ACTIVE_S,cleanup_margin_s=gate.SPEED_CLEANUP_MARGIN_S))


FAKE_PHASE=r'''
import json,os,sys
from pathlib import Path
fake=Path(os.environ['FAKE']);name=Path(__file__).stem
with open(fake/'calls.jsonl','a') as f:f.write(json.dumps({'name':name,'argv':sys.argv[1:],
    'env':{k:os.environ.get(k) for k in ('WORKLOAD','READ_TIMEOUT_S','RUN_ID','READINESS','GEN','S5_OVERLAY','PACKAGE_MANIFEST','GENERATION')}})+'\n')
def arg(key):return sys.argv[sys.argv.index(key)+1]
if name=='e3_native_w1':
    Path(sys.argv[3]).write_text(json.dumps({'schema':2,'summary':{'verdict':'COMPLETE'}}));sys.exit(0)
if name=='e3_native35_performance':
    r=int(arg('--round'));verdict=(fake/('round-%d'%r)).read_text().strip()
    Path(arg('--out')).write_text(json.dumps({'grade':{'verdict':verdict},'native35_round':{'round':r}}))
    if (fake/('unhealthy-after-%d'%r)).exists():(fake/'unhealthy').touch()
    sys.exit({'PASS':0,'FAIL':1}.get(verdict,2))
if name=='e3_native_memory_grade':
    Path(arg('--out')).write_text(json.dumps({'verdict':'PASS'}));sys.exit(0)
sys.exit(2)
'''


class MeasurementTail(unittest.TestCase):
    """The queue's own tail bytes, from the rule4 continuation through the aggregate grade."""

    def run_tail(self,fake,verdicts,rounds=('INCONCLUSIVE','PASS'),unhealthy_after=None):
        p=PB_DRAFT.read_text()
        functions=''.join(p[p.index(name+'() {\n'):p.index('\n}\n',p.index(name+'() {\n'))+3] for name in ('step','record'))
        _,_,gates,measurement=p_parts(PB_DRAFT)
        h=fake/'h';h.mkdir()
        for name in ('e3_native_w1','e3_native35_performance','e3_native_memory_grade','e3_native28_rule4_continuation'):
            (h/(name+'.py')).write_text(FAKE_PHASE)
        shutil.copyfile(HERE/'native35_performance_grade.py',h/'native35_performance_grade.py')  # the real grade
        for index,verdict in enumerate(rounds):(fake/('round-%d'%index)).write_text(verdict)
        if unhealthy_after is not None:(fake/('unhealthy-after-%d'%unhealthy_after)).touch()
        q=fake/'q';q.mkdir()
        preset=''.join('PHASE_VERDICT[%s]=%s\n'%item for item in verdicts.items())
        script=('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=0; N0=4; SERVICE=vllm-flash\ndeclare -A PHASE_VERDICT=()\n'+functions+
            'health_ok() { [[ ! -e $FAKE/unhealthy ]]; }\nhost_crash_detected() { return 1; }\nsystemctl() { echo 4; }\n'+preset+
            gates+measurement+'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\necho "TOTAL FAILS=$FAILS INCS=$INCS ENGINE_CRASHED=$ENGINE_CRASHED"\n')
        env=dict(os.environ,FAKE=str(fake),H=str(h),Q=str(q),PROBE_URL='http://127.0.0.1:1',MODEL='m',RUN_ID='q18a-native35-p-cpu-1',EPOCH='7',
                 Q18A_FIRST_ADVANCE_READ_TIMEOUT_S='1200',Q18A_READINESS='/overlay/e3/native-overlap-readiness.json',S5_OVERLAY='/overlay',
                 S5_LAUNCHER_NAME='run-flash-s5-test.sh',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT/'patches/s5_engine'))))
        done=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True,timeout=300,cwd=str(fake))
        phases=dict(line[3:].split('=',1) for line in done.stdout.splitlines() if line.startswith('PV '))
        calls=[json.loads(l) for l in (fake/'calls.jsonl').read_text().splitlines()] if (fake/'calls.jsonl').exists() else []
        return done,phases,calls,q

    QUALIFIED={'startup':'PASS','row1':'PASS','rule4_controls':'PASS','rule4_candidate':'PASS','row4':'PASS'}

    def test_second_round_runs_after_an_inconclusive_first_and_both_feed_the_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            done,phases,calls,q=self.run_tail(Path(d),self.QUALIFIED)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertEqual([c['name'] for c in calls],['e3_native_w1','e3_native35_performance','e3_native_memory_grade','e3_native35_performance','e3_native_memory_grade'])
            w1,r0,m0,r1,m1=calls
            self.assertEqual((w1['env']['READ_TIMEOUT_S'],w1['env']['RUN_ID']),('1200','q18a-native35-p-cpu-1'))
            for index,(run,memory) in enumerate(((r0,m0),(r1,m1))):
                a=run['argv'];self.assertEqual(a[a.index('--round')+1],str(index));self.assertEqual(a[a.index('--run-id')+1],'q18a-native35-p-cpu-1')
                self.assertEqual(a[a.index('--out')+1],str(q/('overlap-round-%d.json'%index)))
                self.assertEqual(memory['argv'][memory['argv'].index('--performance')+1],str(q/('overlap-round-%d.json'%index)))
                self.assertEqual(memory['argv'][memory['argv'].index('--out')+1],str(q/('memory-grade-round-%d.json'%index)))
                self.assertEqual(run['env']['READINESS'],'/overlay/e3/native-overlap-readiness.json')
            self.assertEqual({k:phases[k] for k in ('w1_controls','overlap-round-0','overlap-round-1','memory-round-0','memory-round-1','overlap-engine','native35-performance')},
                {'w1_controls':'PASS','overlap-round-0':'INCONCLUSIVE','overlap-round-1':'PASS','memory-round-0':'PASS','memory-round-1':'PASS',
                 'overlap-engine':'PASS','native35-performance':'INCONCLUSIVE'})
            graded=json.loads((q/'native35-performance-grade.json').read_text())
            self.assertEqual([r['round'] for r in graded['rounds']],[0,1]);self.assertIn('round-1:lanes-absent',graded['missing'])
            self.assertIn('TOTAL FAILS=0 INCS=2 ENGINE_CRASHED=0',done.stdout)
            self.final_grade(q,graded)

    def test_unhealthy_engine_after_round_zero_withholds_round_one_without_restart(self):
        with tempfile.TemporaryDirectory() as d:
            done,phases,calls,q=self.run_tail(Path(d),self.QUALIFIED,rounds=('PASS','PASS'),unhealthy_after=0)
            self.assertEqual([c['name'] for c in calls],['e3_native_w1','e3_native35_performance','e3_native_memory_grade'])
            self.assertEqual((phases['overlap-round-0'],phases['overlap-round-1'],phases['overlap-engine']),('PASS','INCONCLUSIVE','FAIL'))
            self.assertNotIn('memory-round-1',phases)
            graded=json.loads((q/'native35-performance-grade.json').read_text())
            self.assertEqual(graded['rounds'][1]['verdict'],'NOT_RUN');self.assertIn('ENGINE_CRASHED=1',done.stdout)

    def test_unqualified_row4_withholds_w1_and_both_rounds_but_the_aggregate_still_records(self):
        with tempfile.TemporaryDirectory() as d:
            done,phases,calls,q=self.run_tail(Path(d),dict(self.QUALIFIED,row4='FAIL'))
            self.assertEqual(calls,[])
            self.assertEqual((phases['w1_controls'],phases['overlap'],phases['native35-performance']),('INCONCLUSIVE','INCONCLUSIVE','INCONCLUSIVE'))
            self.assertEqual(json.loads((q/'gates.json').read_text())['outcomes']['row4'],'FAIL')
            graded=json.loads((q/'native35-performance-grade.json').read_text())
            self.assertEqual([r['verdict'] for r in graded['rounds']],['NOT_RUN','NOT_RUN'])
            self.final_grade(q,graded)

    def final_grade(self,q,graded):
        """The queue's PY_FINAL writes measured_performance from the aggregate, or an explicit NOT_RUN."""
        p=PB_DRAFT.read_text();begin=p.index("<<'PY_FINAL' 2>/dev/null || true\n")
        code=p[p.index('\n',begin)+1:p.index('\nPY_FINAL\n',begin)]
        env=dict(os.environ,Q18A_TERMINAL_PHASES=json.dumps({'row4':'PASS'}),PYTHONDONTWRITEBYTECODE='1')
        for directory,expected in ((q,graded),(q.parent/'empty',{'verdict':'NOT_RUN','missing':['native35-performance-grade-absent'],'target_is_gate':False})):
            directory.mkdir(exist_ok=True)
            done=subprocess.run([sys.executable,'-B','-',str(directory),'INCONCLUSIVE','2','0','1','stopped','0',str(HERE),''],input=code,text=True,env=env,capture_output=True)
            self.assertEqual(done.returncode,0,done.stderr)
            final=json.loads((directory/'queue-grade.json').read_text())
            self.assertEqual(final['measured_performance'],expected);self.assertFalse(final['measured_performance']['target_is_gate'])


if __name__=='__main__':unittest.main()
