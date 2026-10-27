#!/usr/bin/env python3
"""Run an unchanged owner tool with bearer auth only on its declared loopback.

The credential stays in memory/environment. No headers, payloads, environment
or credential-bearing command line is written. The legacy body and workload
remain unchanged; both urllib and direct http.client use the same narrow hook.
"""
from contextlib import contextmanager
import http.client
import os
from pathlib import Path
import runpy
import sys
from urllib.parse import urlsplit
from e3_private_auth import _value
from e3_probe_rpc import target


@contextmanager
def authentication(base,key):
    scheme,host,port=target(base)
    key=_value(key)
    original=http.client.HTTPConnection.request
    def request(connection,method,url,body=None,headers=None,*,encode_chunked=False):
        supplied=dict(headers or {})
        if connection.host==host and connection.port==port:
            # All forwarded requests still use their original exact body.
            supplied={name:value for name,value in supplied.items() if name.lower()!='authorization'}
            supplied['Authorization']='Bearer '+key
        return original(connection,method,url,body,supplied,encode_chunked=encode_chunked)
    http.client.HTTPConnection.request=request
    try:yield
    finally:http.client.HTTPConnection.request=original


def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if len(args)<3:return 2
    base,script,*arguments=args
    path=Path(script).resolve()
    allowed={'leaktest.py','leak_partial_tail.py','pressure2.py','pp_canary_bridge.py'}
    if path.parent!=Path(__file__).resolve().parent or path.name not in allowed:
        return 2
    original_argv=sys.argv
    try:
        with authentication(base,os.environ.get('VLLM_API_KEY')):
            sys.argv=[str(path)]+arguments
            runpy.run_path(str(path),run_name='__main__')
    finally:sys.argv=original_argv
    return 0


if __name__=='__main__':raise SystemExit(main())
