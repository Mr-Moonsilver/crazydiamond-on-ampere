"""Receipt-only policy for e; the two d3 diagnostic modules remain unchanged."""
import contextlib
import os
import threading
import time
import native30_host as host
import native30_localize as localize

BYTES_PER_RANK=1024**3
RECORDS_PER_RANK=2*1024**2
TAIL_S=30
LEGACY_BOUNDARY=host.boundary
LEGACY_QUALIFICATION=host.qualification
_INSTALLED=False

def window(s):
    return s.setdefault('native30_e',dict(armed=False,started=False,joined=None,end=None,finished=False))

def finish(s):
    w=window(s)
    if w['end'] is None or time.monotonic()<w['end'] or w['finished']:return
    w['finished']=True
    host.safe(host.emit,s,'host-policy-end',seconds_after_join=time.monotonic()-w['joined'],
              bytes_emitted=s['bytes'],records_emitted=s['records'],receipt_limit=s['limit'])
    s['active']=False

def qualification(original,runner,command):
    # Invoke the real qualification handler once. Only diagnostic state changes.
    result=original(runner,command)
    def observe():
        s=host.state(runner);w=window(s);action=command.get('action')
        if action=='observer_begin':w['armed']=True
        if action=='observer_end' and w['started'] and w['joined'] is None:
            w['joined']=time.monotonic();w['end']=w['joined']+TAIL_S
            host.emit(s,'host-policy-join',tail_s=TAIL_S,source='observer_end after both client joins')
            timer=threading.Timer(TAIL_S,lambda:host.safe(finish,s));timer.daemon=True;timer.start()
        # teacher_configure during row1/rule4 never activates detailed receipts,
        # and its post-row4 disable never truncates the diagnostic join tail.
    host.safe(observe)
    return result

@contextlib.contextmanager
def boundary(obj,fields):
    runner=getattr(obj,'model_runner',None) or getattr(obj,'runner',None)
    def observe():
        s=host.state(runner,fields.get('rank'))
        if s is None:return
        w=window(s)
        if (fields.get('label')=='FIRST_ADMISSION' and fields.get('chunk_start')==0
                and w['armed'] and not w['started']):
            w['started']=True;s['active']=True
            host.emit(s,'host-policy-admission',bytes_per_rank=BYTES_PER_RANK,records_per_rank=RECORDS_PER_RANK,
                      tail_s=TAIL_S,counters='global since startup; never reset',
                      activation='first native admission after row4 observer_begin')
        finish(s)
    host.safe(observe)
    with LEGACY_BOUNDARY(obj,fields):yield

def install():
    global _INSTALLED
    if _INSTALLED:return
    if os.environ.get('FLASH_NATIVE30_LOCALIZE')!='1':return
    host.MAX_BYTES_PER_RANK=BYTES_PER_RANK;host.MAX_RECORDS_PER_RANK=RECORDS_PER_RANK
    host.qualification=qualification;host.boundary=boundary
    localize.install();_INSTALLED=True
    localize.emit('host-policy-installed',bytes_per_rank=BYTES_PER_RANK,records_per_rank=RECORDS_PER_RANK,
                  max_line_bytes=host.MAX_LINE_BYTES,tail_s=TAIL_S,activation='row4-first-admission',sync_mode=localize.mode())
