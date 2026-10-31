"""Pinned-image only: actual FastAPI/Starlette health middleware with CPU owner."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

async def probe():
    from fastapi import FastAPI
    from vllm.entrypoints.serve.instrumentator import health as stock_health
    import inspect,hashlib
    import torch
    from vllm.v1.flash_e3.probe_control import install_owner_health
    from s5_startup import bootstrap_full_startup
    from test_native38_plain_bind import OwnerStartup
    assert not torch.cuda.is_initialized()
    stack,core=OwnerStartup().stack()
    app=FastAPI()
    calls=[]
    async def utility(method,command):
        calls.append(method)
        return stack.driver.submit_control(command)
    health_calls=[]
    async def check_health():health_calls.append(True)
    app.state.engine_client=SimpleNamespace(errored=False,check_health=check_health,
        engine_core=SimpleNamespace(call_utility_async=utility))
    app.include_router(stock_health.router)
    install_owner_health(app)
    async def request():
        records=[]
        first=True
        async def receive():
            nonlocal first
            if first:
                first=False
                return {'type':'http.request','body':b'','more_body':False}
            await asyncio.Event().wait()
        async def send(message):records.append(message)
        await app({'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},
            'http_version':'1.1','method':'GET','scheme':'http','path':'/health',
            'raw_path':b'/health','query_string':b'','headers':[],
            'client':('127.0.0.1',1),'server':('fixture',80)},receive,send)
        return next(r['status'] for r in records if r['type']=='http.response.start')
    before=await request();assert before==503
    assert not stack.driver._bootstrapped
    bootstrap_full_startup(core,stack.driver,timeout_s=2)
    after=await request();assert after==200
    n=len(calls);assert await request()==200 and len(calls)==n
    app.state.engine_client.errored=True
    fatal=await request();assert fatal==503
    assert len(health_calls)==2
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS',http_statuses=[before,after,fatal],
        actual_fastapi=True,actual_stock_health_route=True,actual_mounted_middleware=True,stock_health_calls=len(health_calls),
        stock_health_source_sha256=hashlib.sha256(Path(inspect.getfile(stock_health)).read_bytes()).hexdigest(),
        cached_health_no_extra_rpc=True,cuda_initialized=False,
        fixture_leaves='native38_fixture_leaves.json')

def run():
    result=asyncio.run(probe())
    with Path('/out/health-probe.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
    print('NATIVE38_ASGI_HEALTH PASS')
    return result
