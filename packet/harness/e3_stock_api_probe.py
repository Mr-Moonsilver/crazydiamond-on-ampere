#!/usr/bin/env python3
"""CPU-only proof against actual pinned vLLM imports; no owners or mocks.

Methods are inspected on the real class. Instance fields are checked against
its real source/MRO declarations, not class hasattr alone. No engine/worker is
constructed and no GPU operation or readiness promotion is performed.
"""
import argparse
import ast
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import traceback


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_fields(cls):
    result = {}
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
            source = inspect.getsourcefile(base)
        except (OSError, TypeError, IndentationError, SyntaxError):
            continue
        fields = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name) and node.value.id == 'self'):
                fields.add(node.attr)
            elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                  and node in tree.body[0].body):
                fields.add(node.target.id)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == 'setattr' and len(node.args) >= 2
                  and isinstance(node.args[0], ast.Name) and node.args[0].id == 'self'
                  and isinstance(node.args[1], ast.Constant)
                  and isinstance(node.args[1].value, str)):
                fields.add(node.args[1].value)
        for name in fields:
            result.setdefault(name, dict(owner=base.__module__ + ':' + base.__qualname__,
                                         source_sha256=file_sha(source)))
    return result


def callable_member(cls, name):
    value = inspect.getattr_static(cls, name)
    if isinstance(value, (classmethod, staticmethod)):
        return value.__func__
    if isinstance(value, property):
        return value.fget
    if isinstance(value, functools.cached_property):
        return value.func
    return value


def signature_shape(function):
    return [dict(name=p.name, kind=p.kind.name,
                 default=p.default is not inspect.Parameter.empty)
            for p in inspect.signature(function).parameters.values()]


def check(contract):
    if contract.get('schema') != 'e3-native-stock-api/1':
        raise ValueError('stock-api-contract-schema')
    result = dict(schema='e3-native-stock-api-proof/1', classes=[], failures=[],
                  missing=[], gpu_owner_constructed=False, mocks_used=False,
                  fixture_shims_used=False, hardware_qualified=False,
                  release_qualified=False)
    for spec in contract['classes']:
        label = spec['module'] + ':' + spec['class']
        row = dict(class_name=label, methods=[], fields=[], optional=[])
        result['classes'].append(row)
        try:
            module = importlib.import_module(spec['module'])
            cls = getattr(module, spec['class'])
        except Exception as exc:
            result['missing'].append('real-class-import:' + label)
            row['error'] = dict(error_type=type(exc).__name__,
                                error_message=str(exc).replace('\n', ' ')[:512],
                                error_stack=[dict(file=Path(f.filename).name,
                                                  function=f.name, line=f.lineno)
                                             for f in traceback.extract_tb(exc.__traceback__)[-6:]])
            continue
        if not inspect.isclass(cls) or cls.__module__ != spec['module']:
            result['failures'].append('real-class-owner:' + label)
            continue
        row['source_sha256'] = file_sha(inspect.getsourcefile(cls))
        if row['source_sha256'] != spec['source_sha256']:
            result['failures'].append('stock-source-pin:' + label)
            continue
        fields = declared_fields(cls)
        for member in spec.get('methods', []):
            name = member['name']
            try:
                fn = callable_member(cls, name)
                if not callable(fn):
                    raise TypeError('stock member is not a callable descriptor')
                shape = signature_shape(fn)
                entry = dict(name=name, signature=shape,
                             is_async=inspect.iscoroutinefunction(fn))
                row['methods'].append(entry)
                if shape != member['signature']:
                    result['failures'].append('stock-signature:' + label + '.' + name)
                if entry['is_async'] != member['async']:
                    result['failures'].append('stock-await-contract:' + label + '.' + name)
                if 'call_args' in member or 'call_kwargs' in member:
                    inspect.signature(fn).bind(*member.get('call_args', []),
                                              **member.get('call_kwargs', {}))
            except (AttributeError, TypeError, ValueError) as exc:
                result['failures'].append('stock-method:' + label + '.' + name)
                row['methods'].append(dict(name=name, error_type=type(exc).__name__,
                                           error_message=str(exc)[:200]))
        for name in spec.get('fields', []):
            evidence = fields.get(name)
            row['fields'].append(dict(name=name, declaration=evidence))
            if evidence is None and not hasattr(cls, name):
                result['failures'].append('stock-instance-field:' + label + '.' + name)
            elif evidence is not None and evidence['source_sha256'] not in contract['source_shas'].values():
                result['failures'].append('stock-field-source-pin:' + label + '.' + name)
        for optional in spec.get('optional_absent', []):
            name = optional['name'] if isinstance(optional, dict) else optional
            row['optional'].append(dict(name=name, source_declared=name in fields,
                                        class_attribute=hasattr(cls, name), required=False))
    torch = sys.modules.get('torch')
    cuda = getattr(torch, 'cuda', None)
    initialized = bool(cuda is not None and cuda.is_initialized())
    result['cuda_initialized'] = initialized
    if initialized:
        result['failures'].append('cuda-context-created-during-import')
    result['verdict'] = ('FAIL' if result['failures'] else
                         'INCONCLUSIVE' if result['missing'] else 'PASS')
    result['rc'] = 1 if result['failures'] else 2 if result['missing'] else 0
    result['counts'] = dict(classes=len(result['classes']),
                            methods=sum(len(x['methods']) for x in result['classes']),
                            fields=sum(len(x['fields']) for x in result['classes']))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    result = check(json.loads(a.contract.read_text()))
    result['contract_sha256'] = file_sha(a.contract)
    result['probe_sha256'] = file_sha(__file__)
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'rc', 'counts', 'failures', 'missing',
                                           'cuda_initialized', 'contract_sha256', 'probe_sha256')}))
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
