"""Actual held queue stdin snippets: diagnostic change preserves clearance."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ArtifactDigestReports(unittest.TestCase):
    def test_row1_and_teacher_exit_and_stdout_unchanged_without_traceback(self):
        def snippet(number,marker):
            text=(ROOT/f'harness/runs/q18a_native{number}.draft.sh').read_text()
            return text.split("<<'"+marker+"'\n",1)[1].split('\n'+marker,1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp)/'retained.json'
            for marker,key in (('PY_ROW1_SHA','row1_sha256'),('PY_TEACHER_SHA','teacher_artifact_sha256')):
                old,new=snippet(17,marker),snippet(18,marker)
                for value in (None,123,'x'*64,'a'*63,'a'*64):
                    artifact.write_text(json.dumps({key:value}))
                    a=subprocess.run(['python3','-',str(artifact)],input=old,text=True,capture_output=True)
                    b=subprocess.run(['python3','-',str(artifact)],input=new,text=True,capture_output=True)
                    self.assertEqual((a.returncode,a.stdout),(b.returncode,b.stdout))
                    self.assertNotIn('Traceback',b.stderr)
                    if b.returncode:self.assertIn('digest unavailable',b.stderr)
                    else:self.assertEqual(b.stdout,'a'*64+'\n')
        # Aside from names and the row1 stderr snippet, queue bytes are equal.
        old=(ROOT/'harness/runs/q18a_native17.draft.sh').read_text()
        new=(ROOT/'harness/runs/q18a_native18.draft.sh').read_text().replace('native18','native17')
        marker="<<'PY_ROW1_SHA'";a=old.index(marker);b=old.index('\nPY_ROW1_SHA',a)
        x=new.index(marker);y=new.index('\nPY_ROW1_SHA',x)
        self.assertEqual(old[:a],new[:x]);self.assertEqual(old[b:],new[y:])

if __name__=='__main__':unittest.main()
