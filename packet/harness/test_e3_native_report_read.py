"""Actual native JSON read metadata; CPU ctypes leaves, no GPU/bank-file read."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from replay_observer import ObserverSession, ObserverLimits, ReplayObserverError
from activity_accounting import CorrelationBridge, CORRELATION_REPORT_MAX_BYTES

class Export:
    __name__='s5ro_correlation_report'
    def __init__(self,report,padding=0):self.raw=b' '*padding+json.dumps(report).encode()
    def __call__(self,*args):
        buffer,capacity,length=args[-3:];length._obj.value=len(self.raw)
        if capacity<len(self.raw):return 1
        ctypes.memmove(buffer,self.raw,len(self.raw));return 0

def report(value=1):
    return dict(schema='s5ro-correlation/2',thread_id_type='default-pthread-low32',
        complete=True,overflow=0,errors=0,open_scopes=0,boundary_records=0,
        activity_records=0,scopes=[],apis=[],external=[],activities=[],fixture=value)

def session():
    s=ObserverSession.__new__(ObserverSession)
    s.binding=types.SimpleNamespace(_c=ctypes);s.limits=ObserverLimits();s.state='ended'
    s._native_read_metadata=threading.local();return s

def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class NativeReadTests(unittest.TestCase):
    def test_real_read_records_capacity_and_actual_size_not_64mib_of_data(self):
        s=session();r=report();export=Export(r)
        result=s._report_call(export,(),max_bytes=CORRELATION_REPORT_MAX_BYTES)
        m=s.native_read_metadata()
        self.assertEqual(result,r);self.assertTrue(m['completed'])
        self.assertEqual(m['function'],'s5ro_correlation_report')
        self.assertEqual(m['limit_bytes'],64*1024**2)
        self.assertEqual(m['capacity_bytes'],64*1024**2)
        self.assertEqual(m['required_bytes'],len(export.raw))
        self.assertLess(m['required_bytes'],1024)
        self.assertEqual(m['decoded_report_sha256'],digest(r))
        m['completed']=False;self.assertTrue(s.native_read_metadata()['completed'])
    def test_failed_read_cannot_reuse_previous_success(self):
        s=session();s._report_call(Export(report()),())
        with self.assertRaises(ReplayObserverError):s._report_call(Export(report(),2048),(),max_bytes=1024)
        m=s.native_read_metadata();self.assertFalse(m['completed'])
        self.assertEqual(m['error'],'receipt_too_large')
        self.assertEqual(m['capacity_bytes'],1024)
        self.assertGreater(m['required_bytes'],1024)
    def test_two_threads_cannot_borrow_each_others_report_identity(self):
        s=session();barrier=threading.Barrier(2);got={};errors=[]
        def run(n):
            try:
                r=report(n);s._report_call(Export(r),());barrier.wait(timeout=5)
                got[n]=s.native_read_metadata()['decoded_report_sha256']
            except BaseException as exc:errors.append(type(exc).__name__)
        threads=[threading.Thread(target=run,args=(n,)) for n in (1,2)]
        for t in threads:t.start()
        for t in threads:t.join(6)
        self.assertEqual(errors,[]);self.assertEqual(got,{n:digest(report(n)) for n in (1,2)})
        self.assertIsNone(s.native_read_metadata())
    def test_bridge_exposes_actual_report_read_separately(self):
        s=session();b=CorrelationBridge.__new__(CorrelationBridge)
        b.session=s;b.lib=types.SimpleNamespace(s5ro_correlation_report=Export(report()))
        b.token=123;b.offers=[];b.epoch=1;b.generation=1
        result=b.report()
        self.assertEqual(result['native'],report())
        self.assertEqual(result['native_report_read']['decoded_report_sha256'],digest(result['native']))
        self.assertEqual(result['native_report_read']['limit_bytes'],CORRELATION_REPORT_MAX_BYTES)
        self.assertNotIn('retained-split-bank',json.dumps(result))

if __name__=='__main__':unittest.main()
