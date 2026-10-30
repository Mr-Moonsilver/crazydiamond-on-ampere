"""Public owner runner contracts only; no hardware/engine/process execution."""
import datetime
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owner_runner',HERE/'q18a_native3_owner_run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class OwnerRunner(unittest.TestCase):
    def test_stopped_proof_preserves_failed_flag_but_requires_both_pids_zero(self):
        for active,sub in (('inactive','dead'),('failed','failed')):
            raw=f'ActiveState={active}\nSubState={sub}\nMainPID=0\nControlPID=0'
            self.assertEqual(runner.stopped_unit(raw)['ActiveState'],active)
            for field in ('MainPID','ControlPID'):
                with self.assertRaises(runner.Refused):
                    runner.stopped_unit(raw.replace(field+'=0',field+'=42'))
        for active,sub in (('active','running'),('activating','start'),
                           ('deactivating','stop'),('failed','running')):
            with self.assertRaises(runner.Refused):
                runner.stopped_unit(f'ActiveState={active}\nSubState={sub}\nMainPID=0\nControlPID=0')
        for raw in ('ActiveState=inactive',
                    'ActiveState=failed\nSubState=failed\nMainPID=0\nControlPID=0\nMainPID=0',
                    'ActiveState=failed\nSubState=failed\nMainPID=0\nOther=0'):
            with self.assertRaises(runner.Refused):runner.stopped_unit(raw)

    def test_file_value_overrides_old_env_and_cannot_enter_argv(self):
        key='a'*64
        env=runner.environment(key,'model',64*1024**2,
            {'VLLM_API_KEY':'old','Q17_TEST_OFFLINE':'1','Q18A_OVERLAP_BOUNDED':'1',
             'S5_LAUNCHER_SHA':'foreign','OTHER':'keep'})
        self.assertEqual(env['VLLM_API_KEY'],key)
        self.assertEqual(env['OTHER'],'keep')
        self.assertNotIn('Q17_TEST_OFFLINE',env)
        self.assertNotIn('Q18A_OVERLAP_BOUNDED',env)
        self.assertNotIn('S5_LAUNCHER_SHA',env)
        self.assertEqual(env['S5_GREEN_LIB_SHA256'],runner.GREEN)
        self.assertEqual(env['PROBE_URL'],'http://127.0.0.1:8000')
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.subprocess,'run',return_value=types.SimpleNamespace(returncode=2)) as run:
            rc=runner.phase('q18a_native3_declare.sh',Path(tmp)/'declare.log',env)
        self.assertEqual(rc,2)
        self.assertNotIn(key,repr(run.call_args.args))
        self.assertEqual(run.call_args.kwargs['env']['VLLM_API_KEY'],key)

    def test_invalid_key_model_and_cap_refused(self):
        for key in ('old-token','A'*64,'a'*63,'a'*65):
            with self.subTest(key_length=len(key)),self.assertRaises(runner.Refused):
                runner.environment(key,'model',2*1024**3,{})
        for cap in (True,1,2*1024**3+1):
            with self.assertRaises(runner.Refused):runner.environment('a'*64,'model',cap,{})
        with self.assertRaises(runner.Refused):runner.environment('a'*64,'model\nchanged',2*1024**3,{})

    def test_cutoff_is_0245_cet_and_refuses_before_any_process(self):
        limit=datetime.datetime(2026,10,28,1,45,tzinfo=datetime.timezone.utc).timestamp()
        self.assertEqual(runner.LATEST_QUEUE_START,limit)
        runner.check_slot(limit)
        with patch.object(runner.time,'time',return_value=limit+1),patch.object(runner.subprocess,'run') as run:
            with self.assertRaisesRegex(runner.Refused,'cutoff'):
                runner.preflight(2*1024**3)
            run.assert_not_called()

    def test_fresh_logs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'declare.log';log.write_text('prior evidence')
            with patch.object(runner.subprocess,'run') as run:
                with self.assertRaises(FileExistsError):runner.phase('q18a_native3_declare.sh',log,{})
                run.assert_not_called()
            self.assertEqual(log.read_text(),'prior evidence')

    def test_inner_identity_log_and_outer_chain_log_do_not_share_a_file(self):
        # Reproduce both writers using a real, CPU-only shell child.  The old
        # runner opened the very path its child truncated, losing the refusal.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runs').mkdir()
            identity=root/'q18a-native3-declare.log'
            chain=root/'q18a-native3-declare-chain.log'
            script=root/'runs/q18a_native3_declare.sh'
            message='identity declaration INCONCLUSIVE: '+('public-site '*40)
            script.write_text('printf "%s\\n" "$PUBLIC_MESSAGE" > "$IDENTITY_LOG"\n'
                              'echo "STEP declare FAIL rc=2"\nexit 1\n')
            with patch.object(runner,'H',root):
                self.assertEqual(runner.phase(script.name,chain,
                    {'PUBLIC_MESSAGE':message,'IDENTITY_LOG':str(identity)}),1)
            self.assertEqual(identity.read_text(),message+'\n')
            self.assertEqual(chain.read_text(),'STEP declare FAIL rc=2\n')
            text=(HERE/'q18a_native3_owner_run.py').read_text()
            self.assertIn("phase('q18a_native3_declare.sh',S0/(QNAME+'-declare-chain.log'),env)",text)
            self.assertIn("'-declare-chain.log'",text)

    def run_main_fixture(self, declare_rc):
        # The real owner main -> phase -> real shell children. Only its OS/GPU
        # preflight is stubbed; both child scripts are inert fixture programs.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h=root/'harness';(h/'runs').mkdir(parents=True)
            s0=root/'out';s0.mkdir()
            script=h/'runs/q18a_native3_declare.sh'
            script.write_text('printf "identity fixture diagnostic\\n" > "$S0/$QNAME-declare.log"\n'
                              'echo "declare fixture chain"\nexit '+str(declare_rc)+'\n')
            (h/'runs/q18a_native3.draft.sh').write_text('echo "queue fixture ran"\n')
            private=root/'private';private.mkdir(mode=0o700)
            sys.path.insert(0,str(HERE.parent/'harness'))
            try:
                from e3_private_auth import prepare
                auth=private/'auth.env';prepare(auth,{'VLLM_API_KEY':'c'*64})
                output=io.StringIO()
                with patch.object(runner,'H',h),patch.object(runner,'S0',s0), \
                        patch.object(runner,'preflight'), \
                        patch.object(runner.time,'time',return_value=runner.LATEST_QUEUE_START-300), \
                        contextlib.redirect_stdout(output):
                    rc=runner.main(['--auth-env-file',str(auth)])
                logs={p.name:p.read_text() for p in s0.iterdir()}
                self.assertNotIn('c'*64,output.getvalue()+repr(logs))
                return rc,output.getvalue(),logs
            finally:
                sys.path.pop(0)

    def test_main_declare_refusal_preserves_both_logs_and_withholds_queue(self):
        rc,output,logs=self.run_main_fixture(1)
        self.assertEqual(rc,1)
        self.assertIn('OWNER_CHAIN DONE rc=1 queue_started=false',output)
        self.assertEqual(logs['q18a-native3-declare.log'],'identity fixture diagnostic\n')
        self.assertEqual(logs['q18a-native3-declare-chain.log'],'declare fixture chain\n')
        self.assertNotIn('q18a-native3.log',logs)

    def test_main_declare_clearance_rechecks_slot_then_runs_queue(self):
        rc,output,logs=self.run_main_fixture(0)
        self.assertEqual(rc,0)
        self.assertIn('OWNER_CHAIN DONE rc=0 queue_started=true',output)
        self.assertEqual(logs['q18a-native3-declare.log'],'identity fixture diagnostic\n')
        self.assertEqual(logs['q18a-native3-declare-chain.log'],'declare fixture chain\n')
        self.assertEqual(logs['q18a-native3.log'],'queue fixture ran\n')

    def test_bounded_exception_reason_and_location_without_key_or_input_bytes(self):
        key='f'*64;runner.environment(key,'model',2*1024**3,{})
        try:raise ValueError('refused '+key+'\n'+('x'*1000))
        except ValueError as exc:result=runner.diagnostic(exc)
        self.assertNotIn(key,repr(result))
        self.assertIn('***',result['error_message'])
        self.assertLessEqual(len(result['error_message']),512)
        self.assertEqual(result['error_stack'][-1]['file'],Path(__file__).name)
        bad=UnicodeDecodeError('utf-8',b'credential\xff',10,11,'invalid')
        self.assertEqual(runner.diagnostic(bad)['error_message'],'input-encoding-invalid')


if __name__=='__main__':unittest.main()
