#!/usr/bin/env python3
"""Index every Python attribute/call in E3 sources and join identifiable stock APIs.

CPU/source inspection only. Unknown receiver types remain explicitly unproved.
The real-image import/inspect test is separate; this tool never imports vLLM.
"""
import argparse
import ast
import hashlib
import inspect
from functools import lru_cache
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = {'getattr', 'hasattr', 'setattr', 'delattr'}
FRONT = {'native_overlap_control', 'probe_control', 'native_qualification_frontend'}
CORE = {'engine_driver', 'executor_hooks', 's5_startup', 'startup'}
RUNNERS = {'native_startup', 'model_hooks', 'worker_hooks', 'graph_hooks',
           'decode_probe', 'observer_runtime', 'native_overlap_worker',
           'native_qualification', 'native_observation', 'teacher_decode',
           'lane_private', 'lane_hooks', 'capture_streams', 'native_source',
           'engine_offers', 'native_prefill_offers'}
EDGES = {
    ('AsyncLLM', 'engine_core'): 'AsyncMPClient',
    ('AsyncLLM', 'input_processor'): 'InputProcessor',
    ('AsyncLLM', 'output_processor'): 'OutputProcessor',
    ('EngineCore', 'scheduler'): 'Scheduler',
    ('EngineCoreProc', 'scheduler'): 'Scheduler',
    ('EngineCore', 'model_executor'): 'MultiprocExecutor',
    ('EngineCoreProc', 'model_executor'): 'MultiprocExecutor',
    ('Worker', 'model_runner'): 'GPUModelRunner',
    ('WorkerWrapperBase', 'model_runner'): 'GPUModelRunner',
    ('MessageQueue', 'buffer'): 'ShmRingBuffer',
    ('Scheduler', 'kv_cache_manager'): 'KVCacheManager',
    ('KVCacheManager', 'block_pool'): 'BlockPool',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=256)
def _lines(text):
    return text.encode().splitlines(keepends=True)


def segment(text, node):
    if not hasattr(node, 'end_lineno') or node.end_lineno is None:
        return ast.unparse(node)
    lines = _lines(text)
    if node.lineno == node.end_lineno:
        raw = lines[node.lineno-1][node.col_offset:node.end_col_offset]
    else:
        raw = (lines[node.lineno-1][node.col_offset:] +
               b''.join(lines[node.lineno:node.end_lineno-1]) +
               lines[node.end_lineno-1][:node.end_col_offset])
    return raw.decode()


def const_name(node):
    return node.value if isinstance(node, ast.Constant) and type(node.value) is str else None


def contract_classes(value):
    classes = value.get('classes', [])
    if isinstance(classes, dict):
        classes = [dict(row, **{'class': row.get('class', name)})
                   for name, row in classes.items()]
    return {row['class']: row for row in classes}


def methods(row):
    result = row.get('methods', [])
    if isinstance(result, dict):
        return {name: dict(value if isinstance(value, dict) else {}, name=name)
                for name, value in result.items()}
    return {x if isinstance(x, str) else x['name']:
            {'name': x} if isinstance(x, str) else x for x in result}


def fields(row):
    result = row.get('fields', [])
    return set(result) if isinstance(result, dict) else {
        x if isinstance(x, str) else x['name'] for x in result}


def optional(row):
    result = row.get('optional_absent', [])
    if isinstance(result, dict):
        return result
    return {x if isinstance(x, str) else x['name']:
            None if isinstance(x, str) else x.get('reason') for x in result}


def signature_from_ast(source):
    node = ast.parse('def _contract(' + source + '):\n pass').body[0]
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    offset = len(positional) - len(args.defaults)
    params = []
    for index, item in enumerate(positional):
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = None if index >= offset else inspect.Parameter.empty
        params.append(inspect.Parameter(item.arg, kind, default=default))
    if args.vararg:
        params.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for item, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(inspect.Parameter(item.arg, inspect.Parameter.KEYWORD_ONLY,
            default=None if default is not None else inspect.Parameter.empty))
    if args.kwarg:
        params.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params)


class Index(ast.NodeVisitor):
    def __init__(self, path, text, contract, *, embedded=None, line=0):
        self.path, self.text, self.contract = path, text, contract
        self.module = Path(path).stem
        self.embedded, self.anchor = embedded, line
        self.scopes, self.aliases, self.sites = [], [{}], []
        self.parent = None
        self.domains = [{}]
        try:
            for item in ast.parse(text).body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)) and item.value is not None:
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    try: value = ast.literal_eval(item.value)
                    except (ValueError, TypeError, SyntaxError): continue
                    if type(value) is str: value = [value]
                    if type(value) in (tuple, list) and value and all(type(x) is str for x in value):
                        for target in targets:
                            if isinstance(target, ast.Name): self.domains[0][target.id] = list(value)
        except SyntaxError: pass

    def names(self, node):
        value = const_name(node)
        if value is not None: return [value]
        if isinstance(node, ast.Name):
            for domain in reversed(self.domains):
                if node.id in domain: return domain[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [const_name(x) for x in node.elts]
            if values and all(x is not None for x in values): return values
        return []

    def scope(self):
        return '.'.join(self.scopes) or '<module>'

    def role(self, node):
        if isinstance(node, ast.Name):
            for aliases in reversed(self.aliases):
                if node.id in aliases:
                    return aliases[node.id]
            name = node.id
            if name == 'engine':
                if self.module in FRONT:
                    return 'AsyncLLM'
                if self.module in CORE:
                    return 'EngineCore'
            if name in ('executor',) and self.module in CORE:
                return 'MultiprocExecutor'
            if name == 'scheduler' and self.module in CORE | {'scheduler_hooks', 'admission_metadata'}:
                return 'Scheduler'
            if name == 'core' and self.module in FRONT:
                return 'AsyncMPClient'
            if name == 'runner' and self.module in RUNNERS:
                return 'GPUModelRunner'
            if name == 'worker' and self.module in {'worker_hooks', 'native_startup',
                    'observer_runtime', 'native_overlap_worker', 'executor_hooks'}:
                if self.module == 'executor_hooks' and 'retain_mq' in self.scopes:
                    return 'WorkerProcHandle'
                return 'Worker'
            if name in ('ctl', 'control') and self.module in {'probe_session',
                    'native_qualification', 'native_qualification_frontend'}:
                return 'e3-control'
            if name in self.contract:
                return name
            return None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr == 'engine':
                    if self.module in FRONT | {'native_overlap_session'}:
                        return 'AsyncLLM'
                    if self.module in CORE:
                        return 'EngineCore'
                if node.attr == 'core' and self.module == 'native_overlap_control':
                    return 'AsyncMPClient'
                if node.attr == 'executor' and self.module == 'engine_driver':
                    return 'MultiprocExecutor'
                if node.attr == 'scheduler' and self.module in {'engine_driver', 'scheduler_hooks', 'admission_metadata'}:
                    return 'Scheduler'
                if node.attr == 'runner' and self.module in RUNNERS:
                    return 'GPUModelRunner'
                if node.attr == 'queue' and self.module == 'executor_hooks' and 'BoundedSender' in self.scopes:
                    return 'MessageQueue'
            base = self.role(node.value)
            if base == 'e3-control' and node.attr == 'engine':
                return 'AsyncLLM'
            return EDGES.get((base, node.attr))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) >= 2:
                base, name = self.role(node.args[0]), const_name(node.args[1])
                return EDGES.get((base, name))
            if isinstance(node.func, ast.Name) and node.func.id in self.contract:
                return node.func.id
        return None

    def binding(self, role, name, node, *, dynamic=False, fallback=None):
        if not role or role == 'e3-control':
            return {'status': 'unproved-python-e3-or-data-receiver', 'stock_proven': False}
        if name and name.startswith(('_flash_', 'flash_s5_', '_qsa_uid')):
            return {'status': 'e3-installed-extension', 'stock_proven': False,
                    'required_installer_proof': True,
                    'fallback_reason': 'E3-installed member; never represented as a stock class capability'}
        if name == '__dict__':
            return {'status': 'python-instance-dictionary-protocol', 'stock_proven': False,
                    'proof_scope': 'ordinary non-slotted Python instance storage; does not prove dictionary content or owner lifecycle'}
        cls = self.contract.get(role)
        if cls is None:
            return {'status': 'stock-role-not-in-contract', 'stock_proven': False,
                    'required': fallback is None, 'reason': 'identified nested stock class needs contract/source review'}
        if dynamic or name is None:
            return {'status': 'dynamic-stock-name-unresolved', 'stock_proven': False,
                    'required': fallback is None, 'fallback_expression': fallback}
        member = methods(cls).get(name)
        if member is not None:
            result = {'status': 'contract-method', 'stock_proven': False,
                      'proof_scope': 'source contract join only; real-image imports/signatures are separate',
                      'defining_class': member.get('defining_class', role),
                      'defining_source': member.get('defining_source', cls.get('source')),
                      'definition_line': member.get('line')}
            signature_source = member.get('signature_ast') or member.get('source_signature')
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and signature_source):
                if any(isinstance(x, ast.Starred) for x in node.args) or any(x.arg is None for x in node.keywords):
                    result['call_shape'] = 'dynamic-star-arguments-unresolved'
                else:
                    try:
                        sig = signature_from_ast(signature_source)
                        # Receiver is an instance; methods are stored unbound.
                        args = [None] + [None] * len(node.args)
                        sig.bind(*args, **{x.arg: None for x in node.keywords})
                        result['call_shape'] = 'source-signature-bind-pass'
                    except (ValueError, TypeError, SyntaxError) as exc:
                        result.update(status='required-call-signature-mismatch', required=True,
                                      call_shape='REFUSED', reason=str(exc))
            return result
        if name in fields(cls):
            return {'status': 'contract-instance-or-class-field', 'stock_proven': False,
                    'proof_scope': 'real source declaration/MRO; does not construct an instance or prove runtime value'}
        absent = optional(cls)
        if name in absent and fallback is not None:
            reason = absent[name]
            return {'status': 'reviewed-optional-absence' if reason else 'optional-absence-reason-unresolved',
                    'stock_proven': False, 'fallback_expression': fallback,
                    'fallback_reason': reason or 'contract lists optional absence but requires explicit review reason'}
        return {'status': 'required-stock-member-missing' if fallback is None else 'guarded-stock-absence-unresolved',
                'stock_proven': False, 'required': fallback is None,
                'fallback_expression': fallback,
                'fallback_reason': (None if fallback is None else
                    'source fallback retained verbatim; downstream fallback safety is not inferred by this index')}

    def add(self, kind, node, receiver=None, name=None, *, fallback=None, dynamic=False):
        role = self.role(receiver) if receiver is not None else None
        source = segment(self.text, node)
        line = self.anchor if self.embedded else node.lineno
        site = {'file': self.path, 'line': line, 'column': node.col_offset,
                'scope': self.scope(), 'kind': kind, 'expression': source,
                'receiver': ast.unparse(receiver) if receiver is not None else None,
                'member': name, 'stock_role': role,
                'route_scope': ('native-and-legacy-diagnostic' if self.module in {'probe_control','probe_session','executor_hooks','observer_runtime','decode_probe','teacher_decode'}
                    else 'native' if self.module.startswith('native_') else 'shared-runtime-or-builder'),
                'binding': self.binding(role, name, node, dynamic=dynamic, fallback=fallback)}
        if self.embedded:
            site.update(embedded=True, literal_line=self.anchor,
                        embedded_line=node.lineno, literal_sha256=self.embedded)
        else:
            site['embedded'] = False
        self.sites.append(site)

    def visit_ClassDef(self, node):
        self.scopes.append(node.name); self.aliases.append({})
        self.generic_visit(node)
        self.aliases.pop(); self.scopes.pop()

    def visit_FunctionDef(self, node):
        self.scopes.append(node.name); self.aliases.append({})
        self.generic_visit(node)
        self.aliases.pop(); self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        role = self.role(node.value)
        if role:
            for target in node.targets:
                if isinstance(target, ast.Name): self.aliases[-1][target.id] = role
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            role = self.role(node.value)
            if role and isinstance(node.target, ast.Name): self.aliases[-1][node.target.id] = role
        self.generic_visit(node)

    def visit_For(self, node):
        domain = {}
        values = self.names(node.iter)
        if values and isinstance(node.target, ast.Name): domain[node.target.id] = values
        self.domains.append(domain)
        self.generic_visit(node)
        self.domains.pop()

    visit_AsyncFor = visit_For

    def visit_Attribute(self, node):
        self.add('attribute-' + type(node.ctx).__name__.lower(), node, node.value, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in SPECIAL and len(node.args) >= 2:
            names = self.names(node.args[1])
            name = names[0] if len(names) == 1 else None
            fallback = (ast.unparse(node.args[2]) if len(node.args) > 2 and node.func.id == 'getattr'
                        else 'presence predicate' if node.func.id == 'hasattr' else None)
            for member in names or [None]:
                self.add(node.func.id, node, node.args[0], member,
                         fallback=fallback, dynamic=member is None)
                if names and not isinstance(node.args[1], ast.Constant):
                    self.sites[-1]['member_resolution'] = {'basis': 'exact source literal constant/loop', 'expression': ast.unparse(node.args[1]), 'domain': names}
        elif isinstance(node.func, ast.Attribute):
            self.add('method-call', node, node.func.value, node.func.attr)
        else:
            self.add('call', node)
        self.generic_visit(node)


def build(root=ROOT, contract_path=None):
    root = Path(root)
    contract_path = Path(contract_path or root/'harness/e3_stock_api_contract.json')
    contract_raw = contract_path.read_bytes() if contract_path.is_file() else None
    contract = contract_classes(json.loads(contract_raw)) if contract_raw else {}
    source_files = sorted((root/'patches/s5_engine').glob('*.py'))
    sites, embedded_unparsed, hashes, parse_errors = [], [], {}, []
    for path in source_files:
        raw = path.read_bytes(); text = raw.decode(); rel = path.relative_to(root).as_posix(); hashes[rel] = sha(raw)
        try: tree = ast.parse(text)
        except SyntaxError as exc:
            parse_errors.append({'file':rel,'line':exc.lineno,'error':str(exc)});continue
        index = Index(rel, text, contract); index.visit(tree); sites.extend(index.sites)
        # Generated transforms/kernel bodies are Python string literals. Index
        # parseable bodies separately, and retain every unparseable candidate
        # rather than quietly losing a partial replacement fragment.
        for literal in ast.walk(tree):
            if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str): continue
            value = literal.value
            if '\n' not in value or not any(x in value for x in ('.', 'getattr(', 'hasattr(', 'setattr(', 'delattr(')): continue
            decoded = textwrap.dedent(value)
            digest = sha(value.encode())
            try: nested = ast.parse(decoded)
            except SyntaxError:
                embedded_unparsed.append({'file':rel,'line':literal.lineno,
                    'literal_sha256':digest,'bytes':len(value.encode()),
                    'status':'unparseable-fragment-or-documentation-not-counted-as-verified-python',
                    'text':value})
                continue
            inner = Index(rel, decoded, contract, embedded=digest, line=literal.lineno)
            inner.scopes = ['<literal@%d>' % literal.lineno]; inner.visit(nested); sites.extend(inner.sites)
    # Both source layers retain distinct exact hashes; historic candidate base
    # overlays never masquerade as pristine pinned image files.
    reference_hashes = {p.relative_to(root).as_posix():sha(p.read_bytes())
        for tree in ('vllm-src','vllm-stock-ad067db5')
        for p in sorted((root/'reference'/tree).rglob('*.py')) if p.is_file()}

    required, guarded, nested, dynamic = [], [], [], []
    for i, site in enumerate(sites):
        site['site_id'] = i
        site['source_sha256'] = hashes[site['file']]
        status = site['binding']['status']
        if status in ('required-stock-member-missing','required-call-signature-mismatch'):
            required.append(i)
        elif status in ('guarded-stock-absence-unresolved','optional-absence-reason-unresolved'):
            guarded.append(i)
        elif status == 'stock-role-not-in-contract': nested.append(i)
        elif status == 'dynamic-stock-name-unresolved': dynamic.append(i)
    verdict = 'FAIL' if required or parse_errors else 'INCONCLUSIVE' if not contract_raw or guarded or nested or dynamic else 'PASS'
    return {'schema':'native7-stock-api-sites/1',
        'scope':'all patches/s5_engine Python modules (superset of actual E3_MODULES and related hooks); every AST attribute and call, dynamic attr builtins, plus parseable embedded bodies; unparsed literal fragments retained explicitly',
        'basis':'CPU source index; no stock receiver is called proven from naming or a contract join; real-image class import/signature test is independently required',
        'contract_path':str(contract_path.relative_to(root)) if contract_path.is_relative_to(root) else str(contract_path),
        'contract_sha256':sha(contract_raw) if contract_raw else None,
        'source_hashes':hashes,'reference_source_hashes':reference_hashes,
        'class_source_hashes':{row.get('source','?'):row.get('source_sha256') for row in contract.values()},
        'reference_layers': {'reference/vllm-src': 'reviewed historic Flash-Next candidate baseline; three derived files are not pristine-image identities', 'reference/vllm-stock-ad067db5': 'new independently extracted exact pristine image files; identities remain distinct'},
        'verdict':verdict,'verdict_scope':'source stock-contract joins and full-module AST integrity only; not an engine or real-import result',
        'embedded_fragment_status':'retained data/documentation/partial transform snippets, not executable module proof; actual whole-module AST sites are indexed independently',
        'file_count':len(source_files),'site_count':len(sites),
        'required_missing_or_signature':required,'guarded_absence_unresolved':guarded,
        'nested_stock_contract_unresolved':nested,'dynamic_stock_name_unresolved':dynamic,
        'parse_errors':parse_errors,'embedded_unparsed':embedded_unparsed,'sites':sites,
        'limitations':['Receiver aliases inferred only at explicit source roles/assignments; all other Python/E3/data calls stay unproved, not silently promoted to stock.',
          'A source field declaration does not prove its runtime value/lifetime. Installed E3 extension names require installer proof and are never accepted as pristine stock attributes.',
          'Unparsed embedded fragments/documentation are retained with exact literal anchor/hash/text; no claim that incomplete fragments are verified generated Python.',
          'This index is not engine, GPU, numerical or performance evidence.']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--contract',type=Path)
    p.add_argument('--out',type=Path)
    args = p.parse_args(argv)
    result = build(args.root,args.contract)
    if args.out:
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:result[key] for key in ('verdict','file_count','site_count','contract_sha256')},sort_keys=True))
    return 1 if result['verdict']=='FAIL' else 2 if result['verdict']=='INCONCLUSIVE' else 0


if __name__ == '__main__':raise SystemExit(main())
