"""Hub tests for tools/attn_meta_vm_input_b.py and tools/attn_meta_vm_run_gpu_b.py (vm-b input, supersedes vm-a's).

The wrapper runs probe b and only differs from the vm-a wrapper in the entry file, schema and docstring; the input
ships probe b plus probe a (imported) and the engine bytes; check-bind passes on equal bytes and fails on any change.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import attn_meta_vm_input_b as IB  # noqa: E402


class InputB(unittest.TestCase):
    def test_files(self):
        self.assertEqual(IB.FILES['w/tools/attn_meta_vm_probe_b.py'], 'tools/attn_meta_vm_probe_b.py')
        self.assertEqual(IB.FILES['w/tools/attn_meta_vm_probe.py'], 'tools/attn_meta_vm_probe.py')
        self.assertEqual(sorted(k for k in IB.FILES if k.startswith('w/patches/')),
                         ['w/patches/s5_engine/attn_meta_batch.py', 'w/patches/s5_engine/e3_types.py',
                          'w/patches/s5_engine/lane_chunk.py'])

    def test_wrapper_b_differs_only_in_entry_schema_doc(self):
        a = (ROOT / 'tools/attn_meta_vm_run_gpu.py').read_text().split('"""', 2)[2]
        b = (ROOT / 'tools/attn_meta_vm_run_gpu_b.py').read_text().split('"""', 2)[2]
        a = a.replace("'/w/tools/attn_meta_vm_probe.py'", "'/w/tools/attn_meta_vm_probe_b.py'")
        a = a.replace("schema='attn-meta-vm-owner/1'", "schema='attn-meta-vm-owner-b/1'")
        self.assertEqual(a, b)

    def test_name_rule(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            IB.build('attn-meta-vm-a2', Path(d), Path('/dev/null'))

    def test_check_bind(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            inp, eng = d / 'in', d / 'eng'
            inp.mkdir(), eng.mkdir()
            for rel in ('attn_meta_batch.py', 'lane_chunk.py', 'e3_types.py'):
                (eng / rel).write_bytes((ROOT / 'patches/s5_engine' / rel).read_bytes())
            prov = dict(engine_bytes={'patches/s5_engine/' + r: IB.sha(ROOT / 'patches/s5_engine' / r)
                                      for r in ('attn_meta_batch.py', 'lane_chunk.py', 'e3_types.py')})
            (inp / 'provenance.json').write_text(json.dumps(prov))
            (inp / 'SHA256SUMS').write_text('x\n')
            self.assertEqual(IB.check_bind(inp, eng), 0)
            (eng / 'attn_meta_batch.py').write_bytes((eng / 'attn_meta_batch.py').read_bytes() + b'\n')
            self.assertEqual(IB.check_bind(inp, eng), 1)
            (eng / 'attn_meta_batch.py').unlink()
            self.assertEqual(IB.check_bind(inp, eng), 1)


if __name__ == '__main__':
    unittest.main()
