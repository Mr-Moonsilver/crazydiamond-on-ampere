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
spec=importlib.util.spec_from_file_location('owner_runner',HERE/'q18a_native12_owner_run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class OwnerRunner(unittest.TestCase):
    def setUp(self):
        self.green=patch.object(runner,'GREEN','b'*64);self.green.start();self.addCleanup(self.green.stop)

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
            rc=runner.phase('q18a_native12_declare.sh',Path(tmp)/'declare.log',env)
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

    def test_operator_removed_absolute_clock_refusal_before_and_after_declare(self):
        self.assertFalse(hasattr(runner,'LATEST_QUEUE_START'))
        self.assertFalse(hasattr(runner,'check_slot'))
        late=datetime.datetime(2026,10,29,12,tzinfo=datetime.timezone.utc).timestamp()
        with patch('time.time',return_value=late):
            rc,output,logs=self.run_main_fixture(0)
        self.assertEqual(rc,0)
        self.assertIn('OWNER_CHAIN DONE rc=0 queue_started=true',output)
        self.assertEqual(logs['q18a-native12.log'],'queue fixture ran\n')
        self.assertIn('--kill-after=900s 14400s',
            (HERE.parent/'harness/runs/q18a_native12.draft.sh').read_text())

    def test_fresh_logs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'declare.log';log.write_text('prior evidence')
            with patch.object(runner.subprocess,'run') as run:
                with self.assertRaises(FileExistsError):runner.phase('q18a_native12_declare.sh',log,{})
                run.assert_not_called()
            self.assertEqual(log.read_text(),'prior evidence')

    def test_inner_identity_log_and_outer_chain_log_do_not_share_a_file(self):
        # Reproduce both writers using a real, CPU-only shell child.  The old
        # runner opened the very path its child truncated, losing the refusal.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runs').mkdir()
            identity=root/'q18a-native12-declare.log'
            chain=root/'q18a-native12-declare-chain.log'
            script=root/'runs/q18a_native12_declare.sh'
            message='identity declaration INCONCLUSIVE: '+('public-site '*40)
            script.write_text('printf "%s\\n" "$PUBLIC_MESSAGE" > "$IDENTITY_LOG"\n'
                              'echo "STEP declare FAIL rc=2"\nexit 1\n')
            with patch.object(runner,'H',root):
                self.assertEqual(runner.phase(script.name,chain,
                    {'PUBLIC_MESSAGE':message,'IDENTITY_LOG':str(identity)}),1)
            self.assertEqual(identity.read_text(),message+'\n')
            self.assertEqual(chain.read_text(),'STEP declare FAIL rc=2\n')
            text=(HERE/'q18a_native12_owner_run.py').read_text()
            self.assertIn("phase('q18a_native12_declare.sh',S0/(QNAME+'-declare-chain.log'),env)",text)
            self.assertIn("'-declare-chain.log'",text)

    def run_main_fixture(self, declare_rc):
        # The real owner main -> phase -> real shell children. Only its OS/GPU
        # preflight is stubbed; both child scripts are inert fixture programs.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h=root/'harness';(h/'runs').mkdir(parents=True)
            s0=root/'out';s0.mkdir();(s0/'q18a-native12-build-logs').mkdir()
            script=h/'runs/q18a_native12_declare.sh'
            script.write_text('printf "identity fixture diagnostic\\n" > "$S0/$QNAME-declare.log"\n'
                              'echo "declare fixture chain"\nexit '+str(declare_rc)+'\n')
            (h/'runs/q18a_native12.draft.sh').write_text('echo "queue fixture ran"\n')
            private=root/'private';private.mkdir(mode=0o700)
            sys.path.insert(0,str(HERE.parent/'harness'))
            try:
                from e3_private_auth import prepare
                auth=private/'auth.env';prepare(auth,{'VLLM_API_KEY':'c'*64})
                output=io.StringIO()
                with patch.object(runner,'H',h),patch.object(runner,'S0',s0), \
                        patch.object(runner,'preflight'), \
                        patch.object(runner,'cpu_build',return_value=('b'*64,'d'*64,{'CPU_fixture':True})), \
                        patch.object(runner,'BUILD_LOGS',s0/'q18a-native12-build-logs'), \
                        patch.object(runner,'live_guards'),patch.object(runner,'verify_sources'), \
                        patch.object(runner,'sha',return_value='d'*64),patch.object(runner,'manifest'), \
                        contextlib.redirect_stdout(output):
                    rc=runner.main(['--auth-env-file',str(auth)])
                logs={p.name:p.read_text() for p in s0.iterdir() if p.is_file()}
                self.assertNotIn('c'*64,output.getvalue()+repr(logs))
                return rc,output.getvalue(),logs
            finally:
                sys.path.pop(0)

    def test_main_declare_refusal_preserves_both_logs_and_withholds_queue(self):
        rc,output,logs=self.run_main_fixture(1)
        self.assertEqual(rc,1)
        self.assertIn('OWNER_CHAIN DONE rc=1 queue_started=false',output)
        self.assertEqual(logs['q18a-native12-declare.log'],'identity fixture diagnostic\n')
        self.assertEqual(logs['q18a-native12-declare-chain.log'],'declare fixture chain\n')
        self.assertNotIn('q18a-native12.log',logs)

    def test_main_declare_clearance_runs_bounded_queue(self):
        rc,output,logs=self.run_main_fixture(0)
        self.assertEqual(rc,0)
        self.assertIn('OWNER_CHAIN DONE rc=0 queue_started=true',output)
        self.assertEqual(logs['q18a-native12-declare.log'],'identity fixture diagnostic\n')
        self.assertEqual(logs['q18a-native12-declare-chain.log'],'declare fixture chain\n')
        self.assertEqual(logs['q18a-native12.log'],'queue fixture ran\n')

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


class PredeclaredSourceProof(unittest.TestCase):
    def fixture(self,root):
        # The real finalizer and verifier run on explicitly synthetic CPU ELF
        # fixture bytes. No device or Docker process executes.
        sys.path[:0]=[str(HERE.parent/'harness'),str(HERE.parent/'patches/s5_engine')]
        from test_e3_native_readiness import source_package
        import native_readiness as ready
        import e3_finalize_package as finalizer
        source,closure,receipt=source_package(root)
        for name in ('core','vision'):
            (source/name).mkdir();(source/name/'SHA256SUMS').write_text('nested manifest bytes\n')
        generation=source/'S5-OVERLAP-GENERATION.json'
        import json
        row=json.loads(generation.read_text());row['base_manifest_sha256']=runner.BASE_MANIFEST
        generation.write_text(json.dumps(row))
        # Original source manifest intentionally excludes exactly built extras.
        excluded={'SHA256SUMS','e3/native/flash_s5_green.so','e3/native/flash_s5_replay_observer.so','e3/native/q17-helper-build.json'}
        (source/'SHA256SUMS').write_text(''.join(finalizer.sha(p)+'  '+p.relative_to(source).as_posix()+'\n'
            for p in sorted(source.rglob('*')) if p.is_file() and p.relative_to(source).as_posix() not in excluded))
        expected={p.relative_to(source).as_posix():runner.sha(p) for p in source.rglob('*')
                  if p.is_file() and p.relative_to(source).as_posix() not in excluded|{'S5-OVERLAP-GENERATION.json'}}
        final=root/'final';finalizer.finalize(source,final)
        return source,final,expected,closure,receipt

    def test_actual_source_map_finalizer_and_verifier_bind_produced_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,final,expected,closure,receipt=self.fixture(Path(tmp))
            with patch.object(runner,'EXPECTED_SOURCES',expected), \
                    patch.object(runner,'REVIEW',closure['review_sha256']), \
                    patch.object(runner,'LAUNCHER',expected['run-flash-s5-test.sh']), \
                    patch.object(runner,'OBSERVER',receipt['files']['flash_s5_replay_observer.so']), \
                    patch.object(runner,'OVERLAY',final):
                runner.verify_sources(source,built=True)
                green,manifest,proof=runner.verify_final_build()
                self.assertEqual(green,receipt['files']['flash_s5_green.so'])
                self.assertEqual(manifest,runner.sha(final/'SHA256SUMS'))
                self.assertFalse(proof['hardware_qualified'])
                # Self-consistent manifest replacement cannot alter the pinned source.
                file=final/'e3/native_backend.py';file.write_text('# foreign runtime')
                import e3_finalize_package as f
                (final/'SHA256SUMS').write_text(''.join(f.sha(p)+'  '+p.relative_to(final).as_posix()+'\n'
                    for p in sorted(final.rglob('*')) if p.is_file() and p.relative_to(final).as_posix()!='SHA256SUMS'))
                with self.assertRaisesRegex(runner.Refused,'predeclared-source-pin'):
                    runner.verify_final_build()

    def test_omitted_extra_and_symlink_payloads_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,final,expected,closure,receipt=self.fixture(Path(tmp))
            with patch.object(runner,'EXPECTED_SOURCES',expected), \
                    patch.object(runner,'REVIEW',closure['review_sha256']), \
                    patch.object(runner,'LAUNCHER',expected['run-flash-s5-test.sh']):
                runner.verify_sources(source,built=True)
                extra=source/'foreign';extra.write_text('extra')
                with self.assertRaises(runner.Refused):runner.verify_sources(source,built=True)
                extra.rename(source.parent/'retained-extra')
                file=source/'e3/native_backend.py';file.rename(source.parent/'retained-backend.py')
                file.symlink_to(source.parent/'retained-backend.py')
                with self.assertRaises(runner.Refused):runner.verify_sources(source,built=True)

    def test_cpu_build_steps_are_pinned_and_no_gpu_flags_or_credentials_enter_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);logs=root/'logs'
            with patch.object(runner,'BUILD_LOGS',logs),patch.object(runner,'verify_sources') as verify, \
                    patch.object(runner,'verify_final_build',return_value=('green','manifest',{})), \
                    patch.object(runner.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)) as run, \
                    patch.dict(runner.os.environ,{'VLLM_API_KEY':'fixture-secret','Q17_TEST_OFFLINE':'1'}):
                self.assertEqual(runner.cpu_build()[:2],('green','manifest'))
                self.assertEqual(run.call_count,3)
                self.assertEqual(verify.call_count,2)
                commands=[c.args[0] for c in run.call_args_list]
                self.assertEqual(commands[1],['bash',str(runner.H/'e3_builder/native/build_q17_helpers.sh'),runner.IMAGE,str(runner.RAW/'e3/native')])
                for call in run.call_args_list:
                    self.assertNotIn('fixture-secret',repr(call.args))
                    self.assertNotIn('VLLM_API_KEY',call.kwargs['env'])
                    self.assertNotIn('Q17_TEST_OFFLINE',call.kwargs['env'])
                    self.assertNotIn('--gpus',call.args[0])
                self.assertIn('--network none', (HERE.parent/'patches/s5_engine/native/build_q17_helpers.sh').read_text())

    def test_failed_cpu_child_refuses_before_declare(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.subprocess,'run',return_value=types.SimpleNamespace(returncode=7)):
                with self.assertRaisesRegex(runner.Refused,'CPU-build-step-refused'):
                    runner.build_step(['false'],Path(tmp)/'step.log')

if __name__=='__main__':unittest.main()
