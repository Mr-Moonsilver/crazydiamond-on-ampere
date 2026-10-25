"""Narrow, fail-closed Marlin cap diagnostic. No GPU import until install()."""
import ctypes
import hashlib
import os
from pathlib import Path
import threading
from s5_clock import Inconclusive


def checked_result(cap,hits,original):
    if cap>0 and (hits<1 or original<cap):
        raise Inconclusive('Marlin SM query not intercepted or invalid cap',
                           dict(cap=cap,hits=hits,original=original))


def install(progress):
    import torch
    import vllm._custom_ops as ops
    path=Path(os.environ['S5_MARLIN_CAP_LIBRARY'])
    lib=ctypes.CDLL(str(path))
    lib.s5_marlin_begin.argtypes=[ctypes.c_uint64,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    lib.s5_marlin_end.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
    lib.s5_marlin_begin.restype=lib.s5_marlin_end.restype=ctypes.c_int
    original_op=ops.moe_wna16_marlin_gemm;counts={};lock=threading.Lock()
    def wrapped(*args,**kwargs):
        stream=torch.cuda.current_stream();cap=ctypes.c_int();hits=ctypes.c_int();whole=ctypes.c_int()
        rc=lib.s5_marlin_begin(stream.cuda_stream,torch.cuda.current_device(),ctypes.byref(cap))
        if rc:raise Inconclusive('Marlin guard begin failed',dict(rc=rc))
        try:out=original_op(*args,**kwargs)
        finally:
            endrc=lib.s5_marlin_end(ctypes.byref(hits),ctypes.byref(whole))
            if endrc:raise Inconclusive('Marlin guard end failed',dict(rc=endrc))
        checked_result(cap.value,hits.value,whole.value)
        key=(stream.cuda_stream,cap.value)
        with lock:
            first=key not in counts
            row=counts.setdefault(key,dict(stream=key[0],cap=key[1],calls=0,intercepted_queries=0,whole_sms=whole.value))
            row['calls']+=1;row['intercepted_queries']+=hits.value
        if first:progress.event('marlin','cap-observed',**row)
        return out
    ops.moe_wna16_marlin_gemm=wrapped
    def evidence():
        with lock:rows=[dict(x) for x in counts.values()]
        return dict(library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),streams=rows,
                    scope='exact Python Marlin op host call only; per-thread guard; graph capture fixes launch grids')
    return evidence
