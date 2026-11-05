#!/usr/bin/env python3
"""Hub only (needs git history, so it isn't in the container's suites): with the yield's settings absent, the process-level
scenario over this tree equals the same scenario over the six engine files of a1-trunk's base."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_append_yield as ay

BASE_COMMIT = 'a9b5817'     # claude/inwait, the tree this package ports the yield onto


class OffIsFrozen(unittest.TestCase):
    def test_settings_absent_matches_the_base_engine(self):
        self.assertIsNotNone(shutil.which('git'))
        with tempfile.TemporaryDirectory(prefix='append-yield-frozen-') as frozen:
            for name in ay.ENGINE_FILES:
                blob = subprocess.run(['git', '-C', str(ay.ROOT), 'show', '%s:patches/s5_engine/%s' % (BASE_COMMIT, name)],
                                      capture_output=True, check=True).stdout
                Path(frozen, name).write_bytes(blob)
                self.assertNotEqual(blob, (ay.ROOT / 'patches/s5_engine' / name).read_bytes(), name)
            before = ay.run_scenario(frozen)
        after = ay.run_scenario()
        self.assertEqual(after, before)
        self.assertEqual(before['holds'], [True, True, True, False])
        self.assertNotIn('append_yield', before['journal'])


if __name__ == '__main__':
    unittest.main()
