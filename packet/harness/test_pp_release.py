"""Offline release false-pass regressions and handshake checks. No SSH or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import canary_grade
from pp_release_grade import canary, lifetime, leak
from pp_canary_bridge import status
from canary_hub_runner import safe_path

HERE=Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    def temp(self): return Path(tempfile.mkdtemp(prefix='pp-release-',dir=os.environ.get('TMPDIR',str(HERE.parents[1]/'tmp'))))

    def canary_fixture(self):
        return dict(verdict='PASS',expected=2,partial_hits=1,sessions=[
            dict(round='1',session=str(i),verdict='OK',request_ids=[f'id{i}']) for i in (1,2)])

    def traffic(self):
        return dict(active=0,requests=[dict(id=f'http{i}',session=f'run-r1-s{i}',
            kind='messages',status=200,outcome='completed') for i in (1,2)])

    def test_background_call_without_transcript_is_attributable(self):
        t=self.traffic()
        t['requests'].append(dict(id='background',session='run-r1-s1',kind='messages',status=200,outcome='completed'))
        self.assertEqual(canary(self.canary_fixture(),2,1,3,t,'run')['verdict'],'PASS')
        self.assertEqual(canary(self.canary_fixture(),2,1,4,t,'run')['verdict'],'INCONCLUSIVE')
        t['requests'][-1]['outcome']='client_disconnect'
        self.assertEqual(canary(self.canary_fixture(),2,1,2,t,'run')['verdict'],'INCONCLUSIVE')

    def test_empty_saved_canary_cannot_pass(self):
        self.assertEqual(canary(dict(verdict='PASS',sessions=[],expected=0),10,2,0)['verdict'],'FAIL')
        out=self.temp();(out/'expected.txt').write_text('')
        self.assertEqual(canary_grade.grade(out)['verdict'],'FAIL')

    def test_traffic_and_transcripts_bound_to_exact_count(self):
        d=self.canary_fixture()
        self.assertEqual(canary(d,2,1,2,self.traffic(),'run')['verdict'],'PASS')
        self.assertEqual(canary(d,2,1,3,self.traffic(),'run')['verdict'],'INCONCLUSIVE')
        d['sessions'][1]['request_ids']=['id1']
        self.assertEqual(canary(d,2,1,2,self.traffic(),'run')['verdict'],'INCONCLUSIVE')

    def test_missing_or_duplicate_round_cannot_pass(self):
        d=self.canary_fixture();d['sessions'][1]['session']='1'
        self.assertEqual(canary(d,2,1,2,self.traffic(),'run')['verdict'],'FAIL')

    def test_old_round_transcript_still_graded(self):
        out=self.temp(); wd='/tmp/canary-test'
        (out/'run-start.json').write_text(json.dumps({'started_wall':10}))
        (out/'expected.txt').write_text('1 1 CODE-1 NOTE-1 '+wd+'\n')
        (out/'r1-s1.out').write_text('CODE=CODE-1 NOTE=NOTE-1\nrc=0\n')
        projects=self.temp();p=projects/'-tmp-canary-test';p.mkdir()
        transcript=p/'s.jsonl'
        transcript.write_text(json.dumps({'type':'assistant','message':{'id':'msg1','usage':{'cache_read_input_tokens':7568},
            'content':[{'type':'text','text':'CODE-1 NOTE-1'}]+[{'type':'tool_use','name':t} for t in ['Read','Bash','Bash']]}})+'\n')
        os.utime(transcript,(20,20));os.utime(out/'expected.txt',(30,30))
        self.assertEqual(canary_grade.grade(out,projects)['verdict'],'PASS')

    def test_pressure_success_without_tokens_is_not_coverage(self):
        d=dict(streams=2,errors=[],completed_requests=[0,0],first_tokens=[None,None],last_tokens=[None,None])
        self.assertEqual(lifetime('','',d,[1,2],False)['verdict'],'INCONCLUSIVE')

    def pool(self, ev):
        return 'FLASH_PP_LAYOUT '+json.dumps({'groups':[{'id':2,'kind':'MambaSpec'}]})+'\nFLASH_POOL_STATS '+json.dumps({'groups':{'2':{'evictions':ev}}})

    def test_pressure_must_cover_interval_and_evict_states(self):
        d=dict(streams=2,errors=[],completed_requests=[1,1],first_tokens=[1,1],last_tokens=[10,10])
        self.assertEqual(lifetime(self.pool(1),self.pool(2),d,[2,9],False)['verdict'],'PASS')
        self.assertEqual(lifetime(self.pool(2),self.pool(2),d,[2,9],False)['verdict'],'INCONCLUSIVE')
        d['last_tokens'][0]=8
        self.assertEqual(lifetime(self.pool(1),self.pool(2),d,[2,9],False)['verdict'],'INCONCLUSIVE')

    def test_byte_mode_cannot_pass_with_missing_rank_invariants(self):
        d=dict(streams=2,errors=[],completed_requests=[1,1],first_tokens=[1,1],last_tokens=[10,10])
        self.assertEqual(lifetime(self.pool(1),self.pool(2),d,[2,9],True)['verdict'],'FAIL')

    def test_saved_leak_pass_with_empty_results_fails(self):
        self.assertEqual(leak({'run':{'verdict':'PASS'}},10,4,7568)['verdict'],'FAIL')

    def test_raw_leak_rescored_not_saved_label(self):
        rows=[]
        for i in (0,1):
            rows.append(dict(round=0,session=i,code=f'CODE{i}',locker=f'L{i}',gen1='',answer=f'CODE{i}',cold=None,
                             cached=7568,answer3=f'L{i}',cold3=None,cached3=7568))
        d=dict(N=2,ROUNDS=1,B=7568,MODE='inside',results=rows,errors=[],identity_start='x',identity_end='x')
        self.assertEqual(leak(d,2,1,7568)['verdict'],'PASS')
        rows[0]['answer']='CODE0 CODE1'
        self.assertEqual(leak(d,2,1,7568)['verdict'],'FAIL')

    def test_paths_and_request_expiry_fail_closed(self):
        for path in ('/etc/passwd','/home/<service-user>/s0/../x','/home/<service-user>/s0/x;touch'):
            with self.assertRaises(ValueError): safe_path(path,'/home/<service-user>/s0')
        q=self.temp();(q/'canary.request').write_text(json.dumps({'deadline_wall':0}))
        with self.assertRaises(ValueError): status(q)

    def test_bridge_rejects_changed_engine_before_metrics(self):
        q=self.temp();(q/'canary.request').write_text(json.dumps({'deadline_wall':1e20,'engine':['wanted','start']}))
        with patch('pp_canary_bridge.engine',return_value=('other','start',True)),patch('pp_canary_bridge.metrics') as m:
            with self.assertRaises(ValueError): status(q)
            m.assert_not_called()

    def test_variant_launcher_umask_and_unreadable_stage(self):
        # Extract only the real installer, never source the engine lifecycle.
        source=(HERE/'variant_queue.sh').read_text()
        fn='install_variant_launcher() {'+source.split('install_variant_launcher() {',1)[1].split('\n}\n',1)[0]+'\n}\n'
        self.assertEqual(source.count('install_variant_launcher "'),2)  # install AND restore
        root=self.temp();src=root/'source';src.write_text('#!/bin/bash\ntrue\n');src.chmod(0o600)
        target=root/'launcher';target.write_text('old launcher');target.chmod(0o755)
        stage=root/'staged'
        stub='''sudo() {
          [[ $1 == -n && $2 == -u && $3 == <service-user> ]] || return 99
          [[ ${DENY:-0} == 0 ]] || return 1
          shift 3; "$@"
        }
'''
        script='umask 077\n'+stub+fn+'install_variant_launcher "$1" "$2" "$3"\n'
        args=['bash','-c',script,'test',str(src),str(stage),str(target)]
        out=subprocess.run(args,capture_output=True,text=True,env=dict(os.environ,DENY='1'))
        self.assertNotEqual(out.returncode,0);self.assertEqual(target.read_text(),'old launcher')
        out=subprocess.run(args,capture_output=True,text=True,env=dict(os.environ,DENY='0'))
        self.assertEqual(out.returncode,0,out.stderr)
        self.assertEqual(target.read_bytes(),src.read_bytes());self.assertEqual(target.stat().st_mode&0o777,0o755)
        self.assertEqual(src.stat().st_mode&0o777,0o600)  # private source stays private

    def test_shell_syntax(self):
        for p in ('runs/qN_pp_grouping.draft.sh','variant_queue.sh','canary_hub_runner.sh','flash_canary.sh'):
            r=subprocess.run(['bash','-n',str(HERE/p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__': unittest.main()
