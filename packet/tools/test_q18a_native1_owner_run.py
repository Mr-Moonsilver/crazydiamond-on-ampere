"""Public owner runner contracts only; no hardware/engine/process execution."""
import datetime
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owner_runner',HERE/'q18a_native1_owner_run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class OwnerRunner(unittest.TestCase):
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
            rc=runner.phase('q18a_native1_declare.sh',Path(tmp)/'declare.log',env)
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
                with self.assertRaises(FileExistsError):runner.phase('q18a_native1_declare.sh',log,{})
                run.assert_not_called()
            self.assertEqual(log.read_text(),'prior evidence')


if __name__=='__main__':unittest.main()
