#!/usr/bin/env python3
"""Fit-only prospective stop-intent boundary; retained deaths are never erased."""
import json
from pathlib import Path
import re
import sys

import native43_engine_death_h as base

OriginalScanner=base.Scanner
PROCESS=re.compile(rb'\([^)]*pid=\d+\)')


class Scanner(OriginalScanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.trace_start={}

    def _line(self,index,f,line):
        process=PROCESS.search(line)
        key=(index,process.group(0) if process else None)
        start=self.trace_start.get(key)
        if b'AsyncLLM output_handler failed' in line or b'Traceback (most recent call last)' in line:
            if start is None:
                self.trace_start[key]=dict(wall_ns=base.wall_ns(line),line=f['lines']+1)
        elif b'ERROR' not in line and process is not None:
            self.trace_start.pop(key,None)
        before=len(self.events)
        super()._line(index,f,line)
        if len(self.events)==before and (b'OutOfMemoryError' in line or b'CUDA out of memory' in line):
            self.events.append(dict(family='hard',kind='allocator_oom',file=index,line=f['lines'],
                wall_ns=base.wall_ns(line),text=line[:base.TEXT_MAX].decode('utf-8','replace')))
        if len(self.events)>before and key in self.trace_start:
            self.events[-1]['first_line']=dict(self.trace_start[key])

    def decide(self):
        intent=base.read_stop_intent(self.stop_intent_path)
        saved=self.events;ignored=[];kept=[]
        for event in saved:
            first=event.get('first_line',event)
            if (event['family']!='orderly' and type(intent['wall_ns']) is int
                    and type(first.get('wall_ns')) is int and first['wall_ns']>intent['wall_ns']):
                ignored.append(event)
            else:kept.append(event)
        self.events=kept
        try:result=super().decide()
        finally:self.events=saved
        result['ignored_after_owned_stop']=dict(count=len(ignored),events=ignored[:16])
        result['rules']['fit_stop_boundary']='Only first lines strictly after a valid host-stop-intent; unstamped/earlier retained'
        return result


def host_crash(summary,stop_intent):
    row=json.loads(Path(summary).read_text()).get('shutdown',{}).get('crash')
    if not row:return False
    stamp=base.wall_ns((str(row.get('timestamp',''))+' ').encode())
    intent=base.read_stop_intent(stop_intent)['wall_ns']
    return not (type(stamp) is int and type(intent) is int and stamp>intent)


def main(argv=None):
    base.Scanner=Scanner
    return base.main(argv)


if __name__=='__main__':
    from packet_client_contract import report_exception
    try:raise SystemExit(main())
    except Exception as exc:
        report_exception(exc);raise SystemExit(2)
