import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent.parent / 'tools/stage_vision_encoder_weights.sh'


class Stage(unittest.TestCase):
    def run_stage(self, fail=False):
        root = Path(tempfile.mkdtemp(prefix='vision-stage-'))
        (root / 'archive.tar').write_bytes(b'CPU_STAGE_FIXTURE')
        bindir = root / 'bin'; bindir.mkdir()
        stub = '''#!/usr/bin/python3
import os,json,sys
with open(os.environ['STAGE_STUB_LOG'],'a') as out: out.write(json.dumps(sys.argv)+'\\n')
if os.environ.get('STAGE_STUB_FAIL')=='1' and 'sha256sum -c' in sys.argv[-1]:
 print('STUB_ERROR ValueError: transferred SHA mismatch at stage-stub:5',file=sys.stderr)
 sys.exit(41)
'''
        for name in ('ssh', 'scp'):
            path = bindir / name; path.write_text(stub); path.chmod(0o755)
        env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'],
            STAGE_STUB_LOG=str(root / 'calls.jsonl'), STAGE_STUB_FAIL='1' if fail else '0')
        result = subprocess.run(['bash', str(SCRIPT), str(root / 'archive.tar')], env=env, capture_output=True, text=True)
        (root / 'stdout').write_text(result.stdout); (root / 'stderr').write_text(result.stderr)
        calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
        return result, calls

    def test_actual_copy_argv_and_remote_verification(self):
        result, calls = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 4)
        self.assertIn('sha256sum -c -', calls[-1][-1])
        self.assertIn('unpack --archive vision.tar --parent /opt/flashnext/models', calls[-1][-1])
        self.assertIn('GPU_NOT_USED', result.stdout)

    def test_remote_sha_failure_stops_copy(self):
        result, _ = self.run_stage(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('VISION_COPY PASS', result.stdout)
        self.assertIn('SHA mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
