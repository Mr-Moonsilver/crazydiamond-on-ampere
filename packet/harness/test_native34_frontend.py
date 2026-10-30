"""Real queue argv/validation, bounded client protocol and causal reading."""
import http.client
import json
import os
from pathlib import Path
import shlex
from string import Template
import tempfile
import unittest
from unittest.mock import patch

import e3_native34_causal as causal
from native34_grade import grade

HERE=Path(__file__).resolve().parent
CLI_RECEIPTS=[]


def queue_argv(directory,arm='i'):
    text=(HERE/f'runs/q18a_native34_{arm}.draft.sh').read_text()
    invocation=text.split('python3 "$H/e3_native34_causal.py"',1)[1].split('> "$Q/qual-row4.log"',1)[0]
    values=dict(H=str(HERE),Q=str(directory),PROBE_URL='http://127.0.0.1:1',MODEL='qwen38-w8a8-tp4',
        RUN_ID='native34-'+arm+'-cpu',EPOCH='3',SNAPSHOT_MAX_BYTES=str(512*1024**2),Q18A_STORE=str(directory/'private/store'))
    return [Template(token).substitute(values) for token in shlex.split(invocation.replace('\\\n',' '))]


class Frontend(unittest.TestCase):
    def test_each_exact_queue_main_validates_real_files_before_first_http(self):
        for arm in ('i','s','o'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);calls=[];fixture_key='native34-cpu-fixture'
                def refuse(connection,method,url,body=None,headers=None,**kwargs):
                    command=json.loads(body);calls.append(command['action'])
                    self.assertEqual((connection.host,connection.port),('127.0.0.1',1))
                    self.assertEqual(headers['Authorization'],'Bearer '+fixture_key)
                    self.assertEqual((command['action'],command['target_route'],command['engine_epoch']),('prepare','full',3))
                    raise ConnectionRefusedError('CPU-network-boundary')
                with patch.dict(os.environ,VLLM_API_KEY=fixture_key),patch.object(http.client.HTTPConnection,'request',refuse):
                    self.assertEqual(causal.main(queue_argv(root,arm)),2)
                record=json.loads((root/'qual-row4.json').read_text())
                self.assertEqual(calls,['prepare'])
                self.assertEqual((record['arm'],record['failure_class']),(arm.upper(),'HARNESS'))
                self.assertFalse(record['release_qualified']);self.assertNotIn(fixture_key,json.dumps(record))
                CLI_RECEIPTS.append(dict(arm=arm.upper(),verdict='PASS',first_network_call='prepare-full',network_contact=False,
                    workload='q17b-correctness-precontrols.json',prefill_plan='q18a-native-placement-prefill.json'))

    def test_bad_plan_is_refused_on_cpu_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);argv=queue_argv(root)
            bad=root/'bad-plan.json';bad.write_text('{"schema":"bad"}')
            argv[argv.index('--prefill-plan')+1]=str(bad)
            with patch.dict(os.environ,VLLM_API_KEY='***'),patch.object(http.client.HTTPConnection,'request') as send:
                self.assertEqual(causal.main(argv),2);send.assert_not_called()
            record=json.loads((root/'qual-row4.json').read_text())
            self.assertEqual(record['reason'],'native-prefill-plan-schema-differs')

    def test_prior_error_and_bracketed_target_error_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log=root/'engine.err';arm=root/'arm.json'
            arm.write_text(json.dumps(dict(arm='I',verdict='INCONCLUSIVE')))
            log.write_text('NATIVE30 '+json.dumps(dict(event='native34-indexer-prior-work-exception',rank=1,monotonic_ns=10))+'\n')
            r=grade([log],arm);self.assertEqual((r['verdict'],r['attribution']),('FAIL','PRIOR_RANK1_WORK_BEFORE_TARGET'))
            other=root/'target.err';other.write_text('\n'.join('NATIVE30 '+json.dumps(row) for row in [
                dict(event='native34-indexer-prior-work-drain-complete',rank=1,monotonic_ns=10),
                dict(event='native34-indexer-kernel-exception',rank=1,monotonic_ns=20)])+'\n')
            r=grade([other],arm);self.assertEqual(r['attribution'],'BRACKETED_INDEXER_LAUNCH_OR_FENCE')
            self.assertFalse(r['release_qualified'])
            other.write_text(other.read_text().replace('\"rank\": 1, \"monotonic_ns\": 20','\"rank\": 0, \"monotonic_ns\": 20'))
            self.assertEqual(grade([other],arm)['attribution'],'OVERLAPPED_PHASE_OR_BOUNDARY')

    def test_missing_discriminator_or_record_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log=root/'empty.log';log.write_text('')
            self.assertEqual(grade([log])['verdict'],'INCONCLUSIVE')
            arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='I',verdict='PASS')))
            self.assertEqual(grade([log],arm)['attribution'],'I_DISCRIMINATOR_NOT_PROVED')


if __name__=='__main__':unittest.main()
