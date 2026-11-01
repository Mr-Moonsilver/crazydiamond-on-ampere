"""CPU fixtures for the immutable pre-declaration native package closure."""
import copy
import json
import os
import shutil
import stat
from pathlib import Path
import tempfile
import unittest

import e3_finalize_package as f


class FinalizeTests(unittest.TestCase):
    def fixture(self):
        root = Path(tempfile.mkdtemp(prefix='q17-finalize-'))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        source = root/'source'
        native = source/'e3/native'
        native.mkdir(parents=True)
        for name in f.SOURCES:
            (native/name).write_text('// CPU fixture only\n')
        (source/f.GENERATION).write_text(json.dumps({'armable': False, 'generated_files': []}))
        (source/'SHA256SUMS').write_text(''.join(f.sha(p)+'  '+p.relative_to(source).as_posix()+'\n'
                                       for p in sorted(source.rglob('*')) if p.is_file()))
        for name in f.LIBS:
            (native/name).write_bytes(b'\x7fELF CPU fixture; never loaded')
        receipt = dict(schema=1, image=f.IMAGE, gpu_work=False, cupti_library_sha256=f.CUPTI,
                       declared_abi={'green': 1, 'observer': 1},
                       files={name:f.sha(native/name) for name in f.LIBS+f.SOURCES},
                       needed={f.LIBS[0]:['libcuda.so.1'], f.LIBS[1]:['libcuda.so.1','libcupti.so.13']})
        (source/f.BUILD).write_text(json.dumps(receipt))
        return source, root/'final', receipt

    def test_new_output_preserves_source_and_binds_every_binary(self):
        source,out,_ = self.fixture()
        before={p.relative_to(source).as_posix():f.sha(p) for p in source.rglob('*') if p.is_file()}
        result=f.finalize(source,out)
        self.assertFalse(result['armable'])
        self.assertEqual(before,{p.relative_to(source).as_posix():f.sha(p) for p in source.rglob('*') if p.is_file()})
        declared=f.manifest(out)
        self.assertEqual(set(declared),{p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}-{'SHA256SUMS'})
        self.assertEqual(json.loads((out/f.GENERATION).read_text())['native_build']['receipt_sha256'],f.sha(source/f.BUILD))
        with self.assertRaises(ValueError):f.finalize(source,out)

    def test_native_mutations_refused_before_output(self):
        for mutate in (
            lambda s,r:(s/'extra').write_text('unexpected'),
            lambda s,r:(s/'e3/native'/f.SOURCES[0]).write_text('drift'),
            lambda s,r:(s/'e3/native'/f.LIBS[0]).write_bytes(b'changed'),
            lambda s,r:r.update(image='wrong'),
            lambda s,r:r.update(cupti_library_sha256='0'*64),
            lambda s,r:r['needed'][f.LIBS[1]].append('libcupti.so.12'),
            lambda s,r:r.update(gpu_work=True),
        ):
            source,out,receipt=self.fixture()
            mutate(source,receipt)
            (source/f.BUILD).write_text(json.dumps(receipt))
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):f.finalize(source,out)
            self.assertFalse(out.exists())

    def test_private_queue_umask_does_not_make_public_overlay_unreadable(self):
        source,out,_=self.fixture()
        old=os.umask(0o077)
        try:
            f.finalize(source,out)
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(out.stat().st_mode),0o755)
        for path in out.rglob('*'):
            mode=stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode & 0o555,0o555 if path.is_dir() else mode & 0o111 | 0o444)

    def test_output_cannot_be_nested_in_source(self):
        source,_,_=self.fixture()
        out=source/'nested-final'
        before=set(source.rglob('*'))
        with self.assertRaises(ValueError):f.finalize(source,out)
        self.assertEqual(before,set(source.rglob('*')))

    def test_symlink_refused(self):
        source,out,_=self.fixture()
        (source/'unexpected-link').symlink_to(source/f.GENERATION)
        with self.assertRaises(ValueError):f.finalize(source,out)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
