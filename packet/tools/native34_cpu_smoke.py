"""Provisional pinned-image CPU execution, explicitly not frozen-input proof."""
from pathlib import Path
import json
import os
import sys
import unittest

if '--inside' not in sys.argv:
    stub=Path('/tmp/native34-libcuda');stub.mkdir();(stub/'libcuda.so.1').symlink_to('/usr/local/cuda/lib64/stubs/libcuda.so')
    env=dict(os.environ,LD_LIBRARY_PATH=str(stub)+':'+os.environ.get('LD_LIBRARY_PATH',''))
    os.execvpe('python3',['python3','-B',__file__,'--inside'],env)

sys.path[:0]=['/source/harness','/source/harness/e3_builder','/source/tools']
from vllm.plugins import load_general_plugins
load_general_plugins()
import native34_diag as diagnostic
import torch
assert diagnostic.INSTALLED and not torch.cuda.is_initialized()
from triton.runtime.jit import JITFunction
assert getattr(JITFunction.run,'_native34_wrapped',False)
suite=unittest.TestSuite()
for pattern in ('test_native30*.py','test_native31.py','test_native32.py','test_native33.py','test_native34_*.py'):
    suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
import native31_cpu_check
from unittest.mock import patch
before=native31_cpu_check.run()
with patch.object(__import__('lane_comm'),'register_selection_op',lambda:torch.ops.vllm.flash_s5_selection):
    after=native31_cpu_check.run()
assert before['output_sha256']==after['output_sha256']
record=dict(verdict='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
    tests_run=result.testsRun,skips=len(result.skipped),cuda_initialized=torch.cuda.is_initialized(),
    provisional_only=True,final_input_proof=False,ordinary_exactness='PASS',steps=32,rows=8)
Path('/out/result.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
raise SystemExit(0 if record['verdict']=='PASS' else 1)
