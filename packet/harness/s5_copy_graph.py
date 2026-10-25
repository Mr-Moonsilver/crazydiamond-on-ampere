"""Opt-in clone with copy kernels; source graph remains alive for its buffers."""
import ctypes
from s5_clock import Inconclusive

class KernelCopyGraph:
    def __init__(self,original,stream,library):
        self.original=original;self.lib=ctypes.CDLL(str(library));self.handle=ctypes.c_void_p();self.raw=ctypes.c_uint64()
        self.lib.s5_copy_graph.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)]
        self.lib.s5_copy_launch.argtypes=[ctypes.c_void_p,ctypes.c_uint64]
        self.lib.s5_copy_close.argtypes=[ctypes.c_void_p]
        copies,sets=ctypes.c_uint(),ctypes.c_uint()
        rc=self.lib.s5_copy_graph(original.graph.raw_cuda_graph(),stream.cuda_stream,ctypes.byref(self.handle),ctypes.byref(self.raw),ctypes.byref(copies),ctypes.byref(sets))
        self.counts=dict(memcpy=copies.value,memset=sets.value)
        if rc:raise Inconclusive('copy graph conversion unavailable',dict(rc=rc,**self.counts))
        self.graph=self
    def raw_cuda_graph(self):return self.raw.value
    def replay(self):
        import torch
        rc=self.lib.s5_copy_launch(self.handle,torch.cuda.current_stream().cuda_stream)
        if rc:raise Inconclusive('copy graph launch failed',dict(rc=rc))
        self.original.lane.last_output=self.original.output
    def reset(self):
        rc=self.lib.s5_copy_close(self.handle)
        if rc:raise Inconclusive('copy graph cleanup failed',dict(rc=rc))
