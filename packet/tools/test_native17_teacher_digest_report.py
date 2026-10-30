"""Execute the queue's retained-artifact digest extraction, no engine action."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TeacherDigest(unittest.TestCase):
    def test_missing_malformed_and_valid_keep_clearance_semantics(self):
        def body(name):
            text=(ROOT/'harness/runs'/name).read_text()
            return text.split("<<'PY_TEACHER_SHA'\n",1)[1].split('\nPY_TEACHER_SHA',1)[0]
        old,new=body('q18a_native16.draft.sh'),body('q18a_native17.draft.sh')
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp)/'controls.json'
            for value in (None,123,'x'*64,'a'*63,'a'*64):
                artifact.write_text(json.dumps({'teacher_artifact_sha256':value}))
                a=subprocess.run(['python3','-',str(artifact)],input=old,text=True,capture_output=True)
                b=subprocess.run(['python3','-',str(artifact)],input=new,text=True,capture_output=True)
                self.assertEqual((a.returncode,a.stdout),(b.returncode,b.stdout))
                self.assertNotIn('Traceback',b.stderr)
                if b.returncode:self.assertIn('teacher artifact digest unavailable',b.stderr)
                else:self.assertEqual(b.stdout,'a'*64+'\n')

if __name__=='__main__':unittest.main()
