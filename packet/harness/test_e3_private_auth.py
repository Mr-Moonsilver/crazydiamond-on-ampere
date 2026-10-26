"""Private q17b auth transport: local temporary files and fake declaration only."""
import contextlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import e3_private_auth as auth

VALUE = 'offline-' + 'auth-fixture-' * 3  # Deliberate fixture, not a credential.


class PrivateAuthTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='q17-auth-'))
        self.root.chmod(0o700)
        self.path = self.root / 'api.env'
        self.env = {auth.KEY: VALUE}

    def test_exclusive_private_file_and_in_memory_check(self):
        auth.prepare(self.path, self.env)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(auth.read_private(self.path), VALUE)
        auth.check(self.path, self.env)
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            auth.prepare(self.path, self.env)
        self.assertEqual(self.path.read_bytes(), original)

    def test_cli_no_key_in_argv_or_output(self):
        command = [sys.executable, str(Path(auth.__file__)), 'prepare', str(self.path)]
        self.assertFalse(any(VALUE in word for word in command))
        proc = subprocess.run(command, env=dict(os.environ, **self.env), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn(VALUE, proc.stdout + proc.stderr)
        proc = subprocess.run(command, env=dict(os.environ, **self.env), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn(VALUE, proc.stdout + proc.stderr)

    def test_missing_malformed_or_changed_key_refused(self):
        for value in (None, '', 'short', 'with a space', 'line\nnext', 'hash#value', '\x00' * 9):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(auth.PrivateAuthError):
                    auth.prepare(self.path, {auth.KEY: value})
                self.assertFalse(self.path.exists())
        auth.prepare(self.path, self.env)
        with self.assertRaises(auth.PrivateAuthError):
            auth.check(self.path, {auth.KEY: VALUE + '-different'})

    def test_broad_mode_symlink_and_hardlink_refused(self):
        auth.prepare(self.path, self.env)
        self.path.chmod(0o644)
        with self.assertRaises(auth.PrivateAuthError): auth.read_private(self.path)
        self.path.chmod(0o600)
        alias = self.root / 'alias.env'; alias.symlink_to(self.path)
        with self.assertRaises(auth.PrivateAuthError): auth.read_private(alias)
        link = self.root / 'hard.env'; os.link(self.path, link)
        with self.assertRaises(auth.PrivateAuthError): auth.read_private(self.path)

    def test_parent_must_be_private_and_no_follow(self):
        self.root.chmod(0o755)
        with self.assertRaises(auth.PrivateAuthError): auth.prepare(self.path, self.env)
        self.root.chmod(0o700)
        alias = self.root / 'alias'; alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(auth.PrivateAuthError): auth.prepare(alias / 'api.env', self.env)

    def test_only_one_exact_env_assignment(self):
        for raw in ('OTHER=' + VALUE + '\n', 'VLLM_API_KEY=' + VALUE + '\nOTHER=1\n',
                    'VLLM_API_KEY=' + VALUE, '# ignored\nVLLM_API_KEY=' + VALUE + '\n'):
            path = self.root / ('shape-' + str(len(list(self.root.iterdir()))) + '.env')
            path.write_text(raw); path.chmod(0o600)
            with self.assertRaises(auth.PrivateAuthError): auth.read_private(path)

    def test_file_changed_during_read_refused(self):
        auth.prepare(self.path, self.env)
        original = os.read
        def racing(fd, size):
            result = original(fd, size)
            self.path.write_text('VLLM_API_KEY=' + VALUE + '-changed\n')
            return result
        with patch.object(auth.os, 'read', racing), self.assertRaises(auth.PrivateAuthError):
            auth.read_private(self.path)

    def test_declaration_envfile_binding_and_duplicate_keys_stay_strict(self):
        import identity_declare as declaration
        import engine_identity
        auth.prepare(self.path, self.env)
        image = {'Id': 'sha256:' + 'a' * 64, 'Config': {'Entrypoint': ['vllm', 'serve'], 'Env': []}}
        prefix = ['run', '--rm', '--init', '--name', 'qwen-flash-prod',
                  '--label', 'qwen.flash-next-production=true', '--gpus', 'all',
                  '--ipc=host', '--network=host']
        suffix = ['--env-file', str(self.path), image['Id']]
        spec = declaration.docker_spec(prefix + suffix, image, private_envfile=str(self.path))
        self.assertEqual(spec['env']['VLLM_API_KEY'], {'redacted_length': len(VALUE)})
        for argv, bound in ((prefix + suffix, None),
                            (prefix + suffix, str(self.root / 'other.env')),
                            (prefix + ['-e', 'VLLM_API_KEY=duplicate'] + suffix, str(self.path)),
                            (prefix + suffix[:-1] + ['-e', 'VLLM_API_KEY=duplicate', image['Id']], str(self.path)),
                            (prefix + suffix[:-1] + suffix, str(self.path)),
                            (prefix + ['-e', 'VLLM_API_KEY', image['Id']], None)):
            with self.subTest(bound_present=bound is not None):
                with self.assertRaises(engine_identity.EvidenceError):
                    declaration.docker_spec(argv, image, private_envfile=bound)

    def test_error_never_echoes_exception_or_key(self):
        out = io.StringIO()
        with patch.object(auth, 'check', side_effect=RuntimeError(VALUE)), contextlib.redirect_stdout(out):
            self.assertEqual(auth.main(['check', str(self.path)]), 2)
        self.assertNotIn(VALUE, out.getvalue())


if __name__ == '__main__': unittest.main()
