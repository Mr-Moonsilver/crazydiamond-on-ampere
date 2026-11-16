"""CPU-only extraction/staging boundary checks; fixtures are retained on sdc."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
import vision_encoder_bundle as bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='vision-bundle-'))
        self.source = self.root / 'source'
        self.source.mkdir()
        self.config = b'{"vision_config":{"depth":27}}'
        self.sha = hashlib.sha256(self.config).hexdigest()
        self.cfg = patch.object(bundle, 'CONFIG_SHA', self.sha)
        self.cfg.start()
        self.addCleanup(self.cfg.stop)
        for name in bundle.REQUIRED:
            (self.source / name).write_bytes(self.config if name == 'config.json' else b'{}')
        (self.source / 'mixed.safetensors').write_bytes(b'stub shard; no GPU')
        self.mapping = {'visual.p%03d' % n: 'mixed.safetensors' for n in range(333)}
        self.mapping['language_model.secret'] = 'mixed.safetensors'
        self.write_index()
        self.read = []
        read = self.read

        class Tensor:
            dtype = 'bf16'
            nbytes = (856 * 2**20) // 333
            def contiguous(self): return self

        class Handle:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get_tensor(self, name):
                if not name.startswith('visual.'):
                    raise AssertionError('language tensor read')
                read.append(name)
                return Tensor()

        self.modules = {'torch': types.SimpleNamespace(bfloat16='bf16'),
            'safetensors': types.SimpleNamespace(safe_open=lambda *a, **k: Handle()),
            'safetensors.torch': types.SimpleNamespace(save=lambda ts: json.dumps(sorted(ts)).encode())}

    def write_index(self):
        (self.source / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=self.mapping)))

    def archive(self):
        stream = io.BytesIO()
        with patch.dict(sys.modules, self.modules):
            bundle.pack(self.source, stream)
        path = self.root / 'vision.tar'
        path.write_bytes(stream.getvalue())
        return path

    def test_extract_only_vision_and_stage_verify(self):
        archive = self.archive()
        self.assertEqual(len(self.read), 333)
        self.assertFalse(any(name.startswith('language') for name in self.read))
        staged, digest = bundle.unpack(archive, self.root / 'models')
        self.assertEqual(staged.name, 'flash-next-vision-' + digest[:8])
        self.assertEqual(bundle.verify(staged), digest)
        self.assertEqual(len(json.loads((staged / 'model.safetensors.index.json').read_text())['weight_map']), 333)

    def test_changed_file_refuses(self):
        staged, _ = bundle.unpack(self.archive(), self.root / 'models')
        (staged / 'vision.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            bundle.verify(staged)

    def test_missing_tensor_refuses_before_data_read(self):
        self.mapping.pop('visual.p000'); self.write_index()
        with self.assertRaisesRegex(ValueError, '333'):
            self.archive()
        self.assertEqual(self.read, [])

    def test_foreign_shard_refuses(self):
        outside = self.root / 'foreign.safetensors'; outside.write_bytes(b'no')
        self.mapping['visual.p000'] = '../foreign.safetensors'; self.write_index()
        with self.assertRaisesRegex(ValueError, 'outside checkpoint'):
            self.archive()
        self.assertEqual(self.read, [])

    def test_changed_config_refuses(self):
        (self.source / 'config.json').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'config differs'):
            self.archive()

    def test_tar_traversal_refuses_without_creating_destination(self):
        archive = self.root / 'bad.tar'
        with tarfile.open(archive, 'w') as tar:
            entry = tarfile.TarInfo('../foreign'); entry.size = 1
            tar.addfile(entry, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'unsafe vision archive'):
            bundle.unpack(archive, self.root / 'models')
        self.assertFalse((self.root / 'models').exists())

    def test_existing_bundle_is_retained(self):
        archive = self.archive(); target = self.root / 'models'
        staged, digest = bundle.unpack(archive, target)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            bundle.unpack(archive, target)
        self.assertEqual(bundle.verify(staged), digest)
        self.assertTrue((target / ('incoming-' + digest[:12])).is_dir())

    def test_whole_cli_protocol_uses_the_real_parser(self):
        binary = io.BytesIO()
        stdout = types.SimpleNamespace(buffer=binary)
        with patch.dict(sys.modules, self.modules), patch.object(sys, 'stdout', stdout), \
                patch.object(sys, 'stderr', io.StringIO()), \
                patch.object(sys, 'argv', ['vision_encoder_bundle.py', 'pack', '--source', str(self.source)]):
            bundle.main()
        archive = self.root / 'cli.tar'; archive.write_bytes(binary.getvalue())
        output = io.StringIO()
        with patch.object(sys, 'stdout', output), patch.object(sys, 'argv', ['vision_encoder_bundle.py',
                'unpack', '--archive', str(archive), '--parent', str(self.root / 'cli-models')]):
            bundle.main()
        result = json.loads(output.getvalue()); output = io.StringIO()
        with patch.object(sys, 'stdout', output), patch.object(sys, 'argv', ['vision_encoder_bundle.py',
                'verify', '--model', result['model']]):
            bundle.main()
        self.assertEqual(json.loads(output.getvalue()), dict(verdict='PASS', manifest_sha256=result['manifest_sha256']))


if __name__ == '__main__':
    unittest.main()
