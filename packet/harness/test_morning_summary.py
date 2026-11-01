"""Offline report fixtures. Retained scratch evidence; no GPU or box access."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import morning_summary as ms
from test_s5_followup import make_q11
from vision_tower_probe import MODEL_COMMIT, TOWER_SHA, embedding_budget

HERE = Path(__file__).resolve().parent
SECRET = 'DO_NOT_EMIT_PAYLOAD_OR_CREDENTIAL_SENTINEL'


def write(p, d):
    p.write_text(json.dumps(d)+'\n')


def log(p, steps, finish):
    p.write_text('\n'.join('STEP '+s+' PASS '+SECRET for s in steps)+'\n'+finish+' DONE 05:00:00 fails=0 inconclusive=0\n')


def replay_file(n):
    cfg = dict(AGENTS=n, DURATION=1500, WARMUP=300, SEED=2, START_MIN=60000,
               START_MAX=150000, APPEND_MIN=500, APPEND_MAX=5000,
               REPLY_MIN=200, REPLY_MAX=1000, TOOL_MIN=1, TOOL_MAX=5, ROLL_AT=200000)
    turns = [dict(agent=a, kind=kind, t=t, seconds=10, ttft=2, completion_tokens=200+100*a,
                  prompt_tokens=60000, cached_tokens=59900 if kind=='append' else 0, complete=True, text=SECRET)
             for a in range(n) for kind, t in (('fresh', 100), ('append', 350), ('append', 400))]
    return dict(config=cfg, turns=turns, summary={'output_tokens_per_s':999999, 'text':SECRET})


def w2(gain=1):
    def window(scale):
        streams=[dict(tps=(20+i)*scale, gap_ms={'p99':20}) for i in range(8)]
        return dict(agg_tps=sum(x['tps'] for x in streams), per_stream=streams,
                    worst_stream_gap_ms={'p99':20})
    return dict(verdict='PASS', fails=[], streams=8, stream_prompt=153537, long_prompt=200000,
                before=window(2), during=window(gain), newcomer=dict(cached_tokens=0,client_ttft_s=25), payload=SECRET)


class MorningTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='morning-summary-', dir=HERE.parent.parent/'tmp'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q10 = self.root/'q10-20261025T010000Z'; self.q10.mkdir()
        self.q11 = self.root/'q11-20261025T030000Z'; self.q11.mkdir(); make_q11(self.q11)
        self.q12 = self.root/'q12b-gate'; self.q12.mkdir()
        self.q13 = self.root/'q13-s6b'; self.q13.mkdir()
        self.q14 = self.root/'q14b-vision'; self.q14.mkdir()
        steps = ['overlay','launcher']
        for n in (4,6):
            write(self.q10/f'on-{n}.json', replay_file(n))
            steps += [f'on-{n}.restart', f'on-{n}', f'on-{n}.layout']
        log(self.root/'q10.log', steps+['launcher-restore','restart-final'], 'Q10')
        log(self.root/'q11.log', ['s5-prio','s5-green24','restart-final'], 'Q11')
        log(self.root/'q12b.log', [s for s in ms.release_steps() if s not in ('engine-stop','gpu-handoff','launcher-restore','defaults-restore')]
            + ['engine-stop','gpu-handoff','launcher-restore','defaults-restore'], 'QN-PP-RELEASE')
        for name in ms.RELEASE_MEASURES:
            d = {'verdict':'PASS', 'payload':SECRET}
            if name=='canary':
                d.update(transcript_requests=20,relay_completed_messages=24,engine_request_delta=24)
                write(self.q12/'canary.result', {'release_grade':d,'traffic':SECRET})
            elif name.endswith('lifetime'):
                d.update(state_evictions=20,evictions=25); write(self.q12/(name+'.coverage.json'),d)
            else: write(self.q12/(name+'.grade.json'),d)
        log(self.root/'q13.log', [s for s in ms.s6_steps() if s not in ('stop','gpu-handoff','launcher-restore','defaults-restore')]
            + ['stop','gpu-handoff','launcher-restore','defaults-restore'], 'EXPERIMENT-LEFT-STOPPED')
        for name,gain in (('baseline-w2',1),('candidate-w2',1.1),('baseline-last-w2',1)):
            write(self.q13/(name+'.json'), w2(gain))
        write(self.q13/'baseline-w2.before.json', {'io':{'wall_seconds':100,'bytes_read':170000000000},
                                                  'defaults':SECRET,'env':SECRET})
        p=self.q14/'vision-probe-q14'; p.mkdir()
        d=dict(mode='tower_only', phase='complete', source_commit=MODEL_COMMIT,tower_sha256=TOWER_SHA,
               lm_loaded=False,tp=1,pp=1,warmups=1,repetitions=1,weights={'bytes':900000000,'tensors':333},images=[],payload=SECRET)
        for s in (512,1024,1536):
            budget=embedding_budget([1,s//16,s//16])
            d['images'].append(dict(budget,input_square=s,base64_payload_bytes=budget['raw_base64_bytes_lower_bound']+100,
                                    encode_cuda_event_ms=12,encode_wall_ms=13,peak_allocated_bytes=1100000000))
        write(p/'probe.json',d)
        (self.root/'q14b.log').write_text('STEP vision-probe PASS rc=0 42 s gpu-idle-after=0\nQ14B DONE 06:00:00 rc=0 (engine untouched)\n')

    def summary(self): return ms.build(self.root)
    def rows(self,q): return [r for r in self.summary()['rows'] if r['queue']==q]

    def test_complete_fixture_and_scope(self):
        d=self.summary()
        self.assertEqual({r['verdict'] for r in d['rows']}, {'PASS'}, d)
        self.assertEqual(d['live_engine_state'],'UNVERIFIED_OFFLINE')
        self.assertIn('liveness ONLY',self.rows('q11')[0]['case'])
        self.assertEqual(self.rows('q10')[0]['engine_end'],'TEST_ENGINE_RESTORED')
        self.assertEqual(self.rows('q12b')[-1]['engine_end'],'STOPPED_GPU_IDLE_RECORDED')
        self.assertEqual(self.rows('q13')[-1]['metrics']['capture_bytes_read'],170000000000)

    def test_projection_never_repeats_payloads_or_secrets(self):
        d=self.summary()
        self.assertNotIn(SECRET,json.dumps(d))
        self.assertNotIn(SECRET,ms.markdown(d))
        p=self.q11/'s5-20261025T030000Z-prio/pair-0-2.json.rank0.progress.json'
        write(p,{'rank':0,'phase':SECRET})
        self.assertNotIn(SECRET,json.dumps(self.summary()))

    def test_replay_uses_raw_turns_and_individual_agents(self):
        r=self.rows('q10')[0]
        self.assertAlmostEqual(r['aggregate']['output_tokens_per_s'],round(2*(200+300+400+500)/1200,1))
        self.assertNotEqual(r['streams'][0]['decode_tps_p50'],r['streams'][1]['decode_tps_p50'])
        self.assertAlmostEqual(r['streams'][0]['decode_tps_p50'],199/8)
        # Finishing after the window does not increase completed-turn throughput.
        p=self.q10/'on-4.json'; d=json.loads(p.read_text()); d['turns'].append(dict(d['turns'][-1],t=1495))
        write(p,d)
        self.assertEqual(self.rows('q10')[0]['aggregate'],r['aggregate'])

    def test_missing_agent_and_truncated_turn_never_pass(self):
        p=self.q10/'on-4.json'; d=json.loads(p.read_text()); d['turns']=[t for t in d['turns'] if t['agent']!=3]; write(p,d)
        self.assertEqual(self.rows('q10')[0]['verdict'],'INCONCLUSIVE')
        d=replay_file(4); d['turns'][-1]['complete']=False; write(p,d)
        self.assertEqual(self.rows('q10')[0]['verdict'],'FAIL')

    def test_failed_second_replay_does_not_relabel_first(self):
        with (self.root/'q10.log').open('a') as f: f.write('STEP on-6 FAIL rc=1\n')
        rows=self.rows('q10')
        self.assertEqual([r['verdict'] for r in rows],['PASS','FAIL','FAIL'])

    def test_q11_hang_phase_and_fatal_error_override(self):
        base=self.q11/'s5-20261025T030000Z-prio/pair-0-2.json.rank0'
        write(Path(str(base)+'.progress.json'),{'rank':0,'phase':'prio:measurement'})
        write(Path(str(base)+'.error.json'),{'watchdog':True,'error':SECRET})
        r=self.rows('q11')[0]; self.assertEqual(r['verdict'],'INCONCLUSIVE')
        self.assertEqual(r['streams'][0]['phase'],'prio:measurement')
        write(Path(str(base)+'.error.json'),{'fatal_correctness':True})
        self.assertEqual(self.rows('q11')[0]['verdict'],'FAIL')

    def test_ambiguous_root_requires_explicit_selection(self):
        (self.root/'q10-20261025T020000Z').mkdir()
        self.assertEqual(self.rows('q10')[0]['verdict'],'INCONCLUSIVE')
        d=ms.build(self.root,{'q10':self.q10})
        self.assertEqual(d['rows'][0]['verdict'],'PASS')

    def test_completed_prio_not_hidden_by_absent_green_stage(self):
        (self.q11/'s5-20261025T030000Z-green24').rename(self.q11/'retained-green24')
        rows=self.rows('q11')
        self.assertEqual(rows[0]['verdict'],'PASS')
        self.assertEqual(rows[1]['verdict'],'INCONCLUSIVE')
        self.assertEqual(rows[2]['verdict'],'INCONCLUSIVE')
        self.assertFalse(rows[0]['metrics']['measurement_allowed'])

    def test_done_is_not_enough_and_late_failure_wins(self):
        (self.root/'q12b.log').write_text('Q12B DONE 05:00:00 gate-rc=0 (engine left STOPPED by design)\n')
        r=self.rows('q12b')[-1]
        self.assertEqual(r['verdict'],'INCONCLUSIVE'); self.assertEqual(r['engine_end'],'UNVERIFIED')
        with (self.root/'q13.log').open('a') as f: f.write('STEP defaults-restore FAIL rc=1 '+SECRET+'\n')
        self.assertEqual(self.rows('q13')[-1]['verdict'],'FAIL')
        with (self.root/'q13.log').open('a') as f: f.write('STEP gpu-handoff FAIL rc=1\n')
        self.assertEqual(self.rows('q13')[-1]['engine_end'],'UNVERIFIED')

    def test_missing_capture_or_identity_check_prevents_release_pass(self):
        p=self.root/'q12b.log'; p.write_text(p.read_text().replace('STEP canary.identity-after PASS','UNRECORDED canary.identity-after PASS'))
        self.assertEqual(self.rows('q12b')[-1]['verdict'],'INCONCLUSIVE')
        # The report's identity evidence is the queue's checks, not re-hashing models.
        self.assertEqual(self.rows('q12b')[0]['verdict'],'PASS')

    def test_aba_reports_each_stream_and_no_gain_fails(self):
        rows=self.rows('q13'); g=next(r for r in rows if r['case']=='ABA measured gain')
        self.assertAlmostEqual(g['metrics']['W2_candidate_pct_vs_mean'],10)
        self.assertEqual(len(g['streams']),16)
        write(self.q13/'baseline-last-w2.json',w2(1.11))
        self.assertEqual(self.rows('q13')[-1]['verdict'],'FAIL')

    def test_partial_or_nonfinite_metrics_never_pass(self):
        p=self.q13/'candidate-w2.json'; d=w2(1.1); d['during']['per_stream']=[]; write(p,d)
        self.assertEqual(self.rows('q13')[-1]['verdict'],'INCONCLUSIVE')
        d=w2(1.1); d['before']['agg_tps']=float('nan'); write(p,d)
        self.assertEqual(self.rows('q13')[-1]['verdict'],'INCONCLUSIVE')
        # Finite inputs can still overflow the derived percentage.
        for name,gain in (('baseline-w2',1e-305),('candidate-w2',1e305),('baseline-last-w2',1e-305)):
            write(self.q13/(name+'.json'),w2(gain))
        gain=next(r for r in self.rows('q13') if r['case']=='ABA measured gain')
        self.assertEqual(gain['verdict'],'INCONCLUSIVE')

    def test_vision_skip_cleanup_and_error_after_success(self):
        p=self.root/'q14b.log'; good=p.read_text()
        p.write_text('STEP preflight SKIP '+SECRET+'\nQ14B DONE 06:00:00 skipped=1\n')
        self.assertEqual(self.rows('q14b')[0]['verdict'],'INCONCLUSIVE')
        p.write_text(good.replace('PASS rc=0','FAIL cleanup: GPU not idle after probe rc=0').replace('gpu-idle-after=0','gpu-idle-after=1'))
        self.assertEqual(self.rows('q14b')[0]['verdict'],'FAIL')
        self.assertEqual(self.rows('q14b')[0]['engine_end'],'GPU_HANDOFF_FAILED')
        p.write_text(good); write(self.q14/'vision-probe-q14/error.json',{'error_type':SECRET})
        self.assertEqual(self.rows('q14b')[0]['verdict'],'INCONCLUSIVE')

    def test_superseded_q14_is_not_selected(self):
        self.q14.rename(self.root/'q14-vision')
        self.assertEqual(self.rows('q14b')[0]['verdict'],'INCONCLUSIVE')

    def test_vision_null_measurement_cannot_pass(self):
        p=self.q14/'vision-probe-q14/probe.json'; d=json.loads(p.read_text())
        d['images'][0]['encode_wall_ms']=None; write(p,d)
        self.assertEqual(self.rows('q14b')[0]['verdict'],'INCONCLUSIVE')

    def test_cli_no_overwrite_and_safe_outputs(self):
        out=self.root/'summary.json'
        command=[sys.executable,str(HERE/'morning_summary.py'),'--root',str(self.root),'--out',str(out)]
        r=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
        self.assertNotIn(SECRET,r.stdout+r.stderr+out.read_text())
        before=out.read_bytes(); r=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(r.returncode,2); self.assertEqual(before,out.read_bytes())


if __name__=='__main__': unittest.main()
