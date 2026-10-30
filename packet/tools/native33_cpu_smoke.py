"""Early CPU-only compatibility check; not an exact frozen-input receipt."""
from pathlib import Path
import json
import os
import sys
if '--inside' not in sys.argv:
    stub=Path('/tmp/native33-libcuda');stub.mkdir();(stub/'libcuda.so.1').symlink_to('/usr/local/cuda/lib64/stubs/libcuda.so')
    env=dict(os.environ,LD_LIBRARY_PATH=str(stub)+':'+os.environ.get('LD_LIBRARY_PATH',''))
    os.execvpe('python3',['python3','-B',__file__,'--inside'],env)
sys.path[:0]=['/current-harness','/source/harness/e3_builder','/tools']
import native33_diag as d
d.install()
import native31_cpu_check
from unittest.mock import patch
import torch
before=native31_cpu_check.run();events=[];fences=[]
prior=getattr(d.host._TLS,'context',None)
d.host._TLS.context=dict(state=dict(rank=1,active=True),fields=dict(rank=1,label='DECODE_STEP',step=1))
try:
    with patch.object(d,'fence',lambda:fences.append(1)),patch.object(d,'emit',lambda event,**fields:events.append(event)),patch.object(__import__('lane_comm'),'register_selection_op',lambda:torch.ops.vllm.flash_s5_selection):
        after=native31_cpu_check.run()
finally:d.host._TLS.context=prior
assert before['output_sha256']==after['output_sha256'] and fences and not torch.cuda.is_initialized()
record=dict(verdict='PASS',provisional_only=True,final_input_proof=False,ordinary_exactness='PASS',
            steps=32,rows=8,fences=len(fences),events=len(events),cuda_initialized=False)
Path('/out/result.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
