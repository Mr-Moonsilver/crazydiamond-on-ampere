"""Exact short-queue CLI through real host validation, stopped at HTTP send."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
from string import Template
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
QUEUE='q18a_native30_localize_d3.draft.sh'
CLI_RECEIPTS=[]

def queue_argv(directory):
    text=(HERE/'runs'/QUEUE).read_text()
    invocation=text.split('480s python3 "$H/e3_native30_row4.py"',1)[1].split('> "$Q/localize-row4.log"',1)[0]
    values=dict(H=str(HERE),Q=str(directory),PROBE_URL='http://127.0.0.1:1',
                MODEL='qwen38-w8a8-tp4',RUN_ID=QUEUE.removesuffix('.draft.sh')+'-cpu',
                EPOCH='7',SNAPSHOT_MAX_BYTES=str(512*1024**2))
    argv=[str(HERE/'e3_native30_row4.py')]+[Template(t).substitute(values) for t in shlex.split(invocation.replace('\\\n',' '))]
    return argv,values

class QueueCLI(unittest.TestCase):
    def test_exact_queue_main_reaches_first_network_after_real_validation(self):
        import e3_native30_row4 as short
        calls=[]
        fixture_key='native30-cpu-transport-fixture'
        def refuse_request(connection,method,url,body=None,headers=None,**kwargs):
            command=json.loads(body)
            self.assertEqual((connection.host,connection.port,connection.timeout),('127.0.0.1',1,90.0))
            self.assertEqual((method,url),('POST','/v1/flash_s5_native_overlap'))
            self.assertEqual(headers['Authorization'],'Bearer '+fixture_key)
            self.assertEqual((command['action'],command['target_route'],command['engine_epoch'],command['command_seq']),('prepare','full',7,1))
            calls.append(command['action'])
            raise ConnectionRefusedError('declared CPU boundary')
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);argv,values=queue_argv(directory)
            # No constructors/validators/placement producers are replaced.
            with patch.object(sys,'argv',argv),patch.dict(os.environ,VLLM_API_KEY=fixture_key),patch('http.client.HTTPConnection.request',refuse_request):
                self.assertEqual(short.main(),2)
            record=json.loads((directory/'localize-row4.json').read_text())
            self.assertEqual(calls,['prepare'])
            self.assertEqual((record['verdict'],record['failure_class'],record['reason']),('INCONCLUSIVE','HARNESS','transport-failed:ConnectionRefusedError'))
            self.assertEqual((record['run_id'],record['engine_epoch']),(values['RUN_ID'],7))
            self.assertTrue((directory/'private/localization-store').is_dir())
            partial=record['evidence']['prefill_request']
            self.assertEqual(partial['streams'],1)
            self.assertEqual(len(partial['request_sha256']),1)
            self.assertEqual(len(partial['plan_sha256']),64)
            self.assertNotIn(fixture_key,json.dumps(record))
            paths=[HERE/'workloads/q17b-correctness-precontrols.json',HERE/'workloads/q18a-native-placement-prefill.json']
            CLI_RECEIPTS.append(dict(case='exact-queue-main',verdict='PASS',first_network_call='prepare-full',network_calls=len(calls),network_contact=False,
                expected_refusal=record['reason'],epoch_type='int',snapshot_cap_type='int',timeout_type='float',
                workload_and_plan_shas={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))

    def test_old_path_boundary_and_bad_plan_still_refuse_before_network(self):
        import e3_native30_row4 as short
        from e3_q18a_qualification import _workload_object
        with self.assertRaisesRegex(ValueError,'workload-manifest-invalid:manifest_not_object'):
            _workload_object(str(HERE/'workloads/q17b-correctness-precontrols.json'))
        fixture_key='native30-cpu-transport-fixture'
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);argv,_=queue_argv(directory)
            bad=directory/'bad-plan.json';bad.write_text('{"schema":"invalid"}')
            argv[argv.index('--prefill-plan')+1]=str(bad)
            with patch.object(sys,'argv',argv),patch.dict(os.environ,VLLM_API_KEY=fixture_key),patch('http.client.HTTPConnection.request') as request:
                self.assertEqual(short.main(),2)
                request.assert_not_called()
            record=json.loads((directory/'localize-row4.json').read_text())
            self.assertEqual(record['reason'],'native-prefill-plan-schema-differs')
            CLI_RECEIPTS.append(dict(case='negative-boundaries',verdict='PASS',parent_path_refusal='workload-manifest-invalid:manifest_not_object',bad_plan_refusal=record['reason'],network_calls=0))

if __name__=='__main__':unittest.main()
