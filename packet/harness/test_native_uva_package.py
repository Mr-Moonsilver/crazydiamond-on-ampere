"""The mounted package must include and relatively bind the new shared owner."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'patches/e2_arena')]
import make_s5_overlap as make
import decode_probe


class Package(unittest.TestCase):
    def test_owner_is_generated_measured_and_exempt_from_launcher_name_changes(self):
        self.assertIn('native_uva', make.E3_MODULES)
        self.assertIn('native_uva', decode_probe.GENERATED_SOURCES)
        self.assertIn('native_uva', make.LAUNCHER_LIST_EXEMPT)
        for name in ('lane_chunk',):
            output = make.package((ROOT / 'patches/s5_engine' / (name + '.py')).read_text())
            imports = [n for n in ast.walk(ast.parse(output))
                       if isinstance(n, ast.ImportFrom) and n.module == 'native_uva']
            self.assertEqual(len(imports), 1)
            self.assertEqual(imports[0].level, 1)
        compile(make.package((ROOT / 'patches/s5_engine/native_uva.py').read_text()), 'native_uva', 'exec')
