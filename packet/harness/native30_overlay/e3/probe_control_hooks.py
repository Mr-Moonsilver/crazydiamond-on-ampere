"""Pinned frontend registration for the decode diagnostic, with exact inverse."""
import ast
import hashlib

PATH = 'entrypoints/launchers/api_server/routers.py'
PINS = {
    'v1/core/sched/output.py': '3a66c8efaefe10c572f6f33cf1f8aa4610758c945130cad62eb9f7331005677c',
    PATH: 'c2f1bb2708f44f0fe24f3d8ab8f884c7262e551c6557ebc606263abc4d7b1db1',
    'entrypoints/serve/middleware/authenticate.py': '0633540cb788738dfee5a69075fbdc30eb98a510e26f69a0b43c57c50d9ef5ab',
    'v1/engine/async_llm.py': '32ab1c7b43bfa7e5fcada335f49627d3c540a43bb8376b4845006792c3415fb4',
    'v1/engine/core_client.py': 'fd6ca0cb7499b3086eef2fc57674b7ace56ef1c648f09e92e144b255b79018f8',
}
OLD = '    register_vllm_serve_api_routers(app)\n'
NEW = OLD + ('\n    from vllm.v1.flash_e3.probe_control import attach_router as _flash_s5_probe_router\n'
             '    _flash_s5_probe_router(args, app)\n')


def generate(source):
    if hashlib.sha256(source.encode()).hexdigest() != PINS[PATH] or source.count(OLD) != 1:
        raise ValueError('probe frontend source drift')
    result = source.replace(OLD, NEW, 1)
    ast.parse(result)
    return result


def revert(source):
    if source.count(NEW) != 1:
        raise ValueError('probe frontend inverse anchor differs')
    result = source.replace(NEW, OLD, 1)
    if hashlib.sha256(result.encode()).hexdigest() != PINS[PATH]:
        raise ValueError('probe frontend inverse drift')
    return result
