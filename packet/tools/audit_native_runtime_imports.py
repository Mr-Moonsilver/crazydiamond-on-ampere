#!/usr/bin/env python3
"""Source-only import ownership graph for the actual reviewed launcher mounts.

No vLLM import, Docker, GPU, network or engine action is performed. The graph
separates runtime ownership from real-image import/invocation qualification.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

TRUSTED_LAUNCHER = '83ca0128798fe184b87c006149800994c825979ecedae24a72387b4d7a7085a4'
SITE = '/usr/local/lib/python3.12/dist-packages/'
VARS = {'V': SITE+'vllm/v1', 'C': SITE+'vllm/v1/core',
        'D': SITE+'vllm/models/qwen4_exp/nvidia'}
BUILDER_ONLY = {'vllm.v1.flash_e3.operation_source'}


def sha(data): return hashlib.sha256(data).hexdigest()


def expand(value, substitutions):
    for name, replacement in substitutions.items():
        value = value.replace('${'+name+'}', replacement)
        value = re.sub(r'\$'+re.escape(name)+r'\b', lambda _: replacement, value)
    return value


def module_name(destination):
    if not destination.startswith(SITE) or not destination.endswith('.py'):
        return None
    result = destination[len(SITE):-3].replace('/', '.')
    return result[:-9] if result.endswith('.__init__') else result


def mounts(overlay, text):
    result, missing = [], []
    pattern = re.compile(r'-v\s+"(\$OVERLAY[^":]*):([^"\n]+):ro"')
    for match in pattern.finditer(text):
        source, target = match.groups()
        values = ('kv_cache_utils.py', 'kv_cache_manager.py') if '$f' in source else (None,)
        for item in values:
            substitutions = dict(VARS, OVERLAY=str(overlay))
            if item is not None: substitutions['f'] = item
            path = Path(expand(source, substitutions))
            destination = expand(target, substitutions)
            rel = path.relative_to(overlay).as_posix()
            branch = ('optional-vision:VISION=1' if rel.startswith('vision/') else
                      'optional-file-existence' if item is not None or rel=='ngram_embedding.py' else
                      'candidate-required')
            active = branch != 'optional-vision:VISION=1'
            if not path.exists():
                if branch == 'candidate-required': missing.append(rel)
                result.append({'source':rel,'destination':destination,'exists':False,
                    'profile':branch,'active':False,'launcher_line':text[:match.start()].count('\n')+1})
                continue
            result.append({'source':rel,'destination':destination,'exists':True,
                'profile':branch,'active':active,'launcher_line':text[:match.start()].count('\n')+1})
    return result, missing


def symbols(tree):
    result = set()
    def visit(body):
        for node in body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                result.add(node.name)
            elif isinstance(node,ast.Import):
                result.update(x.asname or x.name.split('.')[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom):
                result.update(x.asname or x.name for x in node.names)
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    result.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name))
            elif isinstance(node,(ast.If,ast.Try)):
                visit(node.body);visit(node.orelse)
                if isinstance(node,ast.Try):
                    visit(node.finalbody)
                    for handler in node.handlers:visit(handler.body)
    visit(tree.body)
    return result


def owner(target, e3, e2, builders, harness):
    if target.startswith('vllm.v1.flash_e3'):
        return 'E3-package'
    if target.startswith('vllm.v1.flash_e2'):
        return 'E2-package'
    if target=='vllm' or target.startswith('vllm.'):
        return 'external-vllm-image-or-mounted-source'
    first=target.split('.')[0]
    if first in e3: return 'unowned-bare-E3'
    if first in e2: return 'unowned-bare-E2'
    if first in builders: return 'unowned-builder-namespace'
    if first in harness: return 'unowned-harness-namespace'
    if first in sys.stdlib_module_names: return 'stdlib'
    return 'external-thirdparty-image'


class Edges(ast.NodeVisitor):
    def __init__(self, record, text, modules):
        self.record,self.text,self.modules=record,text,modules
        self.module=record['module'];self.scopes=[];self.conditions=[];self.context=[];self.edges=[]
        self.strict_packaged_peer=False
        for item in ast.parse(text).body:
            if isinstance(item,ast.FunctionDef) and item.name=='_peer_module':
                self.strict_packaged_peer=any(isinstance(x,ast.If) and isinstance(x.test,ast.Name)
                    and x.test.id=='__package__' and len(x.body)==1 and isinstance(x.body[0],ast.Return)
                    and ast.unparse(x.body[0].value)=="importlib.import_module('.' + name, __package__)"
                    for x in item.body)

    def profile(self):
        scope='.'.join(self.scopes)
        guards=' '.join(x['expression'] for x in self.conditions)
        if self.module in BUILDER_ONLY: return 'builder-source-only'
        if any(x['branch']=='true' and x['expression'] in ('TYPE_CHECKING','typing.TYPE_CHECKING') for x in self.conditions):
            return 'typing-only-not-runtime'
        if self.module.endswith('.probe_session') or scope.startswith('ProbeController.'):
            return 'optional-legacy-diagnostic'
        if ('install_decode_probe' in self.scopes or
                any('DECODE_ONLY_PROBE' in x['expression'] for x in self.conditions)):
            return 'conditional-legacy-diagnostic'
        if ('EngineCoreActorMixin' in self.scopes or any('elastic' in name.lower() for name in self.scopes)
                or 'VLLM_ELASTIC_EP' in guards or 'data_parallel_backend' in guards):
            return 'optional-ray-or-elastic-or-DP'
        if self.module=='vllm.v1.worker.gpu_worker' and self.scopes==['Worker','init_device']:
            return 'conditional-model-runner-version'
        return 'candidate-runtime-possible'

    def add(self,node,target,form,fromlist=(),**extra):
        item={'source':self.record['source'],'module':self.module,
              'source_sha256':self.record['sha256'],'line':node.lineno,
              'column':node.col_offset,'scope':'.'.join(self.scopes) or '<module>',
              'depth':len(self.scopes),'deferred':bool(self.scopes),
              'conditional':list(self.conditions),'contexts':list(self.context),
              'form':form,'qualified_target':target,'fromlist':list(fromlist),
              'profile':self.profile(),'expression':ast.get_source_segment(self.text,node)}
        item.update(extra);self.edges.append(item)

    def visit_ClassDef(self,node):
        self.scopes.append(node.name);self.generic_visit(node);self.scopes.pop()

    def visit_FunctionDef(self,node):
        self.scopes.append(node.name);self.generic_visit(node);self.scopes.pop()

    visit_AsyncFunctionDef=visit_FunctionDef

    def visit_If(self,node):
        self.visit(node.test)
        for branch,body in [('true',node.body),('false',node.orelse)]:
            self.conditions.append({'line':node.lineno,'branch':branch,'expression':ast.unparse(node.test)})
            for child in body:self.visit(child)
            self.conditions.pop()

    def visit_Try(self,node):
        self.context.append({'kind':'try','line':node.lineno})
        for child in node.body:self.visit(child)
        self.context.pop()
        for handler in node.handlers:
            self.context.append({'kind':'except','line':handler.lineno,
                'exception':ast.unparse(handler.type) if handler.type else None})
            for child in handler.body:self.visit(child)
            self.context.pop()
        for branch,body in [('try-else',node.orelse),('finally',node.finalbody)]:
            if body:
                self.context.append({'kind':branch,'line':node.lineno})
                for child in body:self.visit(child)
                self.context.pop()

    def visit_With(self,node):
        self.context.append({'kind':'with','line':node.lineno,
            'expression':', '.join(ast.unparse(x.context_expr) for x in node.items)})
        self.generic_visit(node);self.context.pop()

    visit_AsyncWith=visit_With

    def visit_Import(self,node):
        for alias in node.names:self.add(node,alias.name,'import',alias=alias.asname)

    def visit_ImportFrom(self,node):
        if node.level:
            package=(self.module if self.record['package'] else self.module.rsplit('.',1)[0])
            target=importlib.util.resolve_name('.'*node.level+(node.module or ''),package)
        else:target=node.module or ''
        if node.module is None:
            for alias in node.names:
                possible=target+'.'+alias.name
                resolved=possible if possible in self.modules else target
                self.add(node,resolved,'from',[alias.name],level=node.level,
                    from_base=target,module_fromlist=resolved==possible)
        else:
            self.add(node,target,'from',[x.name for x in node.names],level=node.level,
                from_base=target,module_fromlist=False)

    def visit_Call(self,node):
        func=ast.unparse(node.func)
        if func in ('importlib.import_module','__import__'):
            literal=(node.args[0].value if node.args and isinstance(node.args[0],ast.Constant)
                     and type(node.args[0].value) is str else None)
            if literal:
                if literal.startswith('.'):
                    package=self.module.rsplit('.',1)[0]
                    literal=importlib.util.resolve_name(literal,package)
                self.add(node,literal,'dynamic-literal',dynamic=True,resolution='exact literal')
            elif self.module=='vllm.v1.flash_e3.native_qualification_frontend' and '_peer_module' in self.scopes:
                # All callers are statically finite literals; preserve the real
                # relative and bare fallback branches separately, never waive it.
                calls=[]
                for item in ast.walk(ast.parse(self.text)):
                    if (isinstance(item,ast.Call) and isinstance(item.func,ast.Name)
                            and item.func.id=='_peer_module' and item.args
                            and isinstance(item.args[0],ast.Constant)
                            and type(item.args[0].value) is str):calls.append(item.args[0].value)
                relative=(ast.unparse(node.args[0])=="'.' + name" if node.args else False)
                for name in sorted(set(calls)):
                    target=self.module.rsplit('.',1)[0]+'.'+name if relative else name
                    self.add(node,target,'dynamic-peer',dynamic=True,
                        resolution='finite literal caller domain',peer_name=name,
                        package_relative=relative)
                    if not relative and self.strict_packaged_peer:
                        self.edges[-1]['profile']='source-mode-only-not-runtime'
                        self.edges[-1]['inactive_reason']='nonempty deployed __package__ directly returns relative module first; bare import is only reachable in unpackaged source mode'
            else:
                self.add(node,None,'dynamic-unresolved',dynamic=True,
                    resolution='unresolved; cannot prove runtime import owner')
        elif (func.endswith('spec_from_file_location') or func.endswith('SourceFileLoader')):
            self.add(node,None,'dynamic-file-loader',dynamic=True,
                resolution='file-loader path needs separate ownership proof')
        self.generic_visit(node)


def audit(overlay, root=None):
    overlay=Path(overlay).resolve();root=Path(root or Path(__file__).resolve().parents[1])
    launcher=overlay/'run-flash-s5-test.sh';raw=launcher.read_bytes()
    if sha(raw)!=TRUSTED_LAUNCHER:raise ValueError('launcher differs from trusted concrete mount layout')
    mount_rows,missing=mounts(overlay,raw.decode())
    records=[]
    for mount in mount_rows:
        if not mount['exists']:continue
        path=overlay/mount['source']
        files=sorted(path.rglob('*.py')) if path.is_dir() else [path]
        for file in files:
            if file.suffix!='.py':continue
            destination=(mount['destination']+'/'+file.relative_to(path).as_posix()
                         if path.is_dir() else mount['destination'])
            module=module_name(destination)
            if module:
                records.append({'source':file.relative_to(overlay).as_posix(),
                    'module':module,'destination':destination,'package':file.stem=='__init__',
                    'sha256':sha(file.read_bytes()),'mount_profile':mount['profile'],
                    'active_mount':mount['active']})
    modules={x['module']:x for x in records if x['active_mount']}
    e3={p.stem for p in (overlay/'e3').glob('*.py')}
    e2={p.stem for p in (overlay/'e2').glob('*.py')}
    builders={p.stem for p in (root/'patches/s5_engine').glob('*.py')}
    harness={p.stem for p in (root/'harness').glob('*.py')}
    exports={x['module']:symbols(ast.parse((overlay/x['source']).read_text()))
             for x in records if x['active_mount']}
    edges=[]
    for record in records:
        visitor=Edges(record,(overlay/record['source']).read_text(),modules)
        visitor.visit(ast.parse(visitor.text));edges.extend(visitor.edges)
    adjacency={m:set() for m in modules}
    for item in edges:
        target=item['qualified_target']
        item['owner']=owner(target,e3,e2,builders,harness) if target else 'unresolved-dynamic'
        if target in modules:adjacency[item['module']].add(target)
        if (item['form']=='from' and target in exports and not item.get('module_fromlist')):
            item['missing_symbols']=[name for name in item['fromlist']
                if name!='*' and name not in exports[target] and target+'.'+name not in modules]
    roots=sorted(m for m in modules if m not in BUILDER_ONLY)
    reached=set();pending=list(roots)
    while pending:
        module=pending.pop()
        if module in reached:continue
        reached.add(module);pending.extend(adjacency.get(module,()))
    builder_evidence=[]
    for module in sorted(BUILDER_ONLY & set(modules)):
        incoming=[x for x in edges if x['qualified_target']==module]
        builder_evidence.append({'module':module,'source':modules[module]['source'],
            'incoming_import_edges':len(incoming),'incoming_sources':sorted({x['module'] for x in incoming}),
            'reachable_from_other_mounted_runtime_modules':module in reached,
            'reason':'CPU generator source retained for package closure; no engine runtime import edge references it; never silently imported by image proof'})
    errors=[];unresolved=[];builder_edges=[]
    for i,item in enumerate(edges):
        item['edge_id']=i
        if item['module'] in BUILDER_ONLY and item['module'] not in reached:
            item['runtime_required']=False
            if item['owner'].startswith('unowned'):builder_edges.append(i)
        else:item['runtime_required']=item['profile']=='candidate-runtime-possible'
        exempt_builder=(item['module'] in BUILDER_ONLY and item['module'] not in reached)
        inactive_source=item['profile'] in ('source-mode-only-not-runtime','typing-only-not-runtime')
        if item['owner'].startswith('unowned') and not exempt_builder and not inactive_source:
            errors.append(i)
        if item['owner'] in ('E3-package','E2-package') and item['qualified_target'] not in modules and not inactive_source:
            item['missing_owned_module']=True
            errors.append(i)
        if item.get('missing_symbols'):errors.append(i)
        if item['owner']=='unresolved-dynamic':unresolved.append(i)
    result={'schema':'native8-mounted-runtime-imports/1','overlay':str(overlay),
        'launcher_sha256':sha(raw),'profile':'q18a native performance; S5=1 CORE_OVERLAY=1 VISION=0, MP executor DP1 TP2 PP2, model-runner-v2; no setting/workload change',
        'scope':'all concrete Python mounts derived from the trusted launcher, including directory files, conditional paths and every static/deferred/dynamic AST import; source graph only',
        'verdict':'FAIL' if errors or missing else 'INCONCLUSIVE' if unresolved else 'PASS',
        'verdict_scope':'source import ownership/exports only; actual-image imports and pure provider invocation separately required',
        'mounts':mount_rows,'missing_required_mount_sources':missing,'modules':records,
        'module_count':len(records),'edge_count':len(edges),
        'deferred_edge_count':sum(x['deferred'] for x in edges),
        'runtime_roots':roots,'builder_only_proof':builder_evidence,
        'runtime_ownership_errors':sorted(set(errors)),'unresolved_dynamic_edges':unresolved,
        'unowned_builder_only_edges':builder_edges,
        'source_mode_only_edges':[i for i,x in enumerate(edges) if x['profile']=='source-mode-only-not-runtime'],
        'typing_only_edges':[i for i,x in enumerate(edges) if x['profile']=='typing-only-not-runtime'],
        'edges':edges,
        'source_hashes':{x['source']:x['sha256'] for x in records},
        'adjacency':{k:sorted(v) for k,v in sorted(adjacency.items())},
        'limitations':['External vLLM/thirdparty names denote image ownership, not a successful import or invocation.',
            'Conditional guards/scopes are retained. Optional Ray/elastic/legacy/runner-version paths are not silently promoted to tested runtime paths.',
            'Builder-only exemption requires no incoming runtime edge and no unresolved dynamic target; only operation_source is named, with evidence retained.',
            'No top-level E3/E2/harness namespace is added to engine sys.path. CPU proof must use these package mounts and withhold all host builder/source paths.']}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);result=audit(a.overlay,a.root)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('verdict','module_count','edge_count','deferred_edge_count','launcher_sha256')}))
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[result['verdict']]


if __name__=='__main__':raise SystemExit(main())
