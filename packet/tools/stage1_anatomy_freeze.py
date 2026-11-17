#!/usr/bin/env python3
"""Fail-closed default-off proof of the stage-1 handler anatomy against claude/step-b 5f46c07.

Step A's method (tools/step_a_freeze.py: project, function, removals; tools/native37_robust_freeze.py: changed, prove), over
this build's declarations (tools/stage1_anatomy_declarations.json). With ``FLASH_S5_HANDLER_ANATOMY`` unset:

* every pre-existing function this build changes in executor_hooks.py projects back to 5f46c07's exact AST once the declared
  guarded statements are removed; each removal is named and counted, and an undeclared statement makes the proof refuse;
* the one new function on the off path, ``_anatomy``, reduces to "set ``_ANATOMY`` False, return None" (NEW_OFF), and the
  only import of ``handler_anatomy`` anywhere in the engine sources sits in its env-guarded else branch (imports);
* ``_anatomy``, ``_ANATOMY`` and ``handler_anatomy`` are referenced nowhere else in the engine sources (reach);
* the module statements and the two packaging constants are named (MODULE_STATEMENTS); the launcher's mount-check list,
  the only launcher input they feed, is identical to 5f46c07's (launcher);
* with --overlay/--parent: every e3 module other than the declared ones is byte-identical to the parent overlay, the
  declared ones equal make_s5_overlap.package(committed source), and e3/handler_anatomy.py is the one added file.

The runtime half (nothing installed, no attribute set, no import, no handler line, progress lines byte-identical to
5f46c07's off and on) is harness/test_stage1_handler_anatomy.Off. CPU only, no box action.

    python3 tools/stage1_anatomy_freeze.py --out FILE [--overlay DIR --parent DIR]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from native37_robust_freeze import changed, prove  # noqa: E402
from step_a_freeze import function, project, removals, _units  # noqa: E402

PARENT_SHORT = '5f46c07'
DECLARATION = ROOT / 'tools/stage1_anatomy_declarations.json'
ENGINE = ROOT / 'patches/s5_engine'
NEW_MODULE = 'handler_anatomy.py'
ENV_TEST = "if os.environ.get('FLASH_S5_HANDLER_ANATOMY', '') != '1':"
PROGRESS_GUARD = 'if (anatomy := _anatomy(proc)) is not None:'

# Default-off projections (step_a_freeze kinds). With the switch unset, _anatomy(proc) returns None (NEW_OFF), so each test
# below is false; _ANATOMY is None (unread) or False, so ``if _ANATOMY:`` is false.
PROJECTIONS = {
    'executor_hooks.py': {
        'worker_rpc_begin': dict(off={PROGRESS_GUARD: 1,
                                      'if state is not None and (anatomy := _anatomy(proc)) is not None:': 1}),
        'worker_rpc_returned': dict(off={'if _ANATOMY:': 1}),
        'worker_handle_output': dict(off={'if rpc is not None and (anatomy := _anatomy(proc)) is not None:': 1}),
        'worker_enqueue_reply': dict(off={PROGRESS_GUARD: 1}),
    },
}

# The one new function the default-off path calls, and its off form: the env test holds, so the else branch (the import and
# install) is never taken.
NEW_OFF = {
    ('executor_hooks.py', '_anatomy'): (
        dict(subs=((
            "if os.environ.get('FLASH_S5_HANDLER_ANATOMY', '') != '1':\n"
            "                    _ANATOMY = False\n"
            "                else:\n"
            "                    try:\n"
            "                        import handler_anatomy\n"
            "                        _ANATOMY = handler_anatomy.install(proc)\n"
            "                    except Exception as error:\n"
            "                        _ANATOMY = False\n"
            "                        _progress_logger().warning('flash-s5 handler install refused: %s', type(error).__name__)",
            '_ANATOMY = False'),)),
        'def f(proc):\n'
        '    global _ANATOMY\n'
        '    state = _ANATOMY\n'
        '    if state is None:\n'
        '        with _WORKER_PROGRESS_LOCK:\n'
        '            if _ANATOMY is None:\n'
        '                _ANATOMY = False\n'
        '            state = _ANATOMY\n'
        '    return state or None'),
}

NEW_HELPERS = {
    'executor_hooks.py': {'_anatomy': 'off form: sets _ANATOMY False, returns None (NEW_OFF)'},
}

MODULE_STATEMENTS = {
    'executor_hooks.py': {'module:_ANATOMY': 'one module global, None at import; read only by _anatomy and worker_rpc_returned'},
    'decode_probe.py': {'module:GENERATED_SOURCES': "adds 'handler_anatomy' to the generated-module inventory the decode "
                                                    'probe evidence lists (diagnostic decode-probe mode and the build)'},
    'make_s5_overlap.py': {'module:E3_MODULES': "adds 'handler_anatomy': the overlay holds e3/handler_anatomy.py and package() "
                                                'binds its import to the e3 package',
                           'module:LAUNCHER_LIST_EXEMPT': "adds 'handler_anatomy', so the launcher's name-only mount list and "
                                                          'the reviewed launcher text stay unchanged (launcher)'},
}

# Names that exist only for the anatomy; outside the projected functions and _anatomy itself they must not be referenced.
ANATOMY_NAMES = {'_anatomy', '_ANATOMY', 'handler_anatomy'}

NOT_PROJECTED = [
    dict(what='module global', detail='_ANATOMY = None is created at import; with the switch unset it becomes False at the first '
         'stock RPC carrying progress meta on a worker whose progress state is live, and is otherwise never read.'),
    dict(what='declared overhead', detail='Default-off, with progress on: one _anatomy(proc) call per stock RPC begin, '
         'handle_output and ordered reply (a global read after the first), one global read per RPC return, and one per '
         'release. With progress unset: none of the guards is reached (they sit inside the progress-state branches). Not '
         'measured; no host wait, no device call, no import.'),
    dict(what='overlay file set', detail='e3/handler_anatomy.py is added to the overlay and its checksum list; the launcher '
         'mounts the whole e3 directory (unchanged), so the file is present but never imported default-off (imports).'),
    dict(what='decode probe evidence', detail='GENERATED_SOURCES and the generated decode-probe-evidence.json list one more '
         'module; the decode-only diagnostic mode verifies its sha like every other module.'),
    dict(what='FLASH_S5_HANDLER_ANATOMY set to other values', detail="Only '1' switches it on; any other value behaves as unset "
         '(no refusal: a measurement switch never stops the engine).'),
]


def show(path, commit=PARENT_SHORT):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, text=True)


def parent():
    return subprocess.check_output(['git', 'rev-parse', PARENT_SHORT + '^{commit}'], cwd=ROOT, text=True).strip()


def paths():
    return sorted(subprocess.check_output(['git', 'diff', '--name-status', PARENT_SHORT, '--', 'patches/s5_engine'], cwd=ROOT,
                                          text=True).splitlines())


def declaration_draft():
    rows = paths()
    definitions = {}
    for row in rows:
        status, name = row.split('\t', 1)
        if status == 'M' and name.endswith('.py'):
            definitions[name] = changed(show(name), (ROOT / name).read_text())
    return dict(schema='stage1-anatomy-freeze-declaration/1', parent=PARENT_SHORT, paths=rows, definitions=definitions)


def default_off():
    result, table = {}, {}
    for file, specs in PROJECTIONS.items():
        old, new = show('patches/s5_engine/' + file), (ENGINE / file).read_text()
        for name, spec in specs.items():
            result[file + ':' + name] = project(function(new, name), function(old, name), spec)
            table[file + ':' + name] = removals(spec)
    for (file, name), (spec, target) in NEW_OFF.items():
        new = (ENGINE / file).read_text()
        result[file + ':' + name + ' (off path)'] = project(function(new, name), target, spec)
        table[file + ':' + name + ' (off path)'] = removals(spec)
    d = json.loads(DECLARATION.read_text())
    wanted = set()
    for path, names in d['definitions'].items():
        file = path.rsplit('/', 1)[1]
        old_names = {q for q, _ in _units(show(path), file)}
        wanted |= {(file, n) for n in names if n in old_names and not n.startswith(('class:', 'module:'))}
    projected = {(f, n) for f, specs in PROJECTIONS.items() for n in specs}
    assert wanted == projected, sorted(wanted ^ projected)
    return dict(verdict='PASS', legacy_function_bodies=result, functions=len(result), removals=table,
                removal_count=sum(row['count'] for rows in table.values() for row in rows),
                meaning='source projection with FLASH_S5_HANDLER_ANATOMY unset: every 5f46c07 function this build changes is '
                        'its AST once the declared statements are removed; _anatomy on that path sets False and returns None.')


def _guards(tree):
    """(node, guards) for every node; guards are (test text, 'body'|'else') of the enclosing ifs, innermost last."""
    out = []

    def walk(node, guards):
        out.append((node, guards))
        if isinstance(node, ast.If):
            walk(node.test, guards)
            head = 'if ' + ast.unparse(node.test) + ':'
            for child in node.body:
                walk(child, guards + ((head, 'body'),))
            for child in node.orelse:
                walk(child, guards + ((head, 'else'),))
            return
        for child in ast.iter_child_nodes(node):
            walk(child, guards)
    walk(tree, ())
    return out


def imports(texts=None):
    """The only import of handler_anatomy in the engine sources: executor_hooks._anatomy, under the env test's else."""
    texts = texts or {}
    sites = []
    for path in sorted(ENGINE.glob('*.py')):
        if path.name == NEW_MODULE:
            continue
        text = texts.get(path.name) or path.read_text()
        if 'handler_anatomy' not in text:
            continue
        for qual, unit in _units(text, path.name):
            for node, guards in _guards(unit):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ''] + [a.name for a in node.names]
                if any(n == 'handler_anatomy' or n.endswith('.handler_anatomy') for n in names):
                    sites.append((path.name, qual, guards))
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and 'handler_anatomy' in ast.unparse(node):
                sites.append((path.name, 'module', ()))
    assert len(sites) == 1, 'handler_anatomy import sites: %r' % (sites,)
    file, qual, guards = sites[0]
    assert (file, qual) == ('executor_hooks.py', '_anatomy'), sites
    assert (ENV_TEST, 'else') in guards, 'handler_anatomy imported outside the env guard: %r' % (guards,)
    return dict(verdict='PASS', site='%s:%s' % (file, qual), guard=ENV_TEST + ' else')


def reach(texts=None):
    """ANATOMY_NAMES appear only in the projected functions, _anatomy and the module statement."""
    texts = texts or {}
    allowed = {('executor_hooks.py', n) for n in PROJECTIONS['executor_hooks.py']} | {('executor_hooks.py', '_anatomy')}
    hits, checked = [], 0
    for path in sorted(ENGINE.glob('*.py')):
        if path.name == NEW_MODULE:
            continue
        text = texts.get(path.name) or path.read_text()
        for qual, unit in _units(text, path.name):
            checked += 1
            if isinstance(unit, ast.Assign) and any(
                    'module:' + ast.unparse(target) in MODULE_STATEMENTS.get(path.name, {}) for target in unit.targets):
                continue  # a declared module statement (MODULE_STATEMENTS)
            for node in ast.walk(unit):
                name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                        else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None)
                if name in ANATOMY_NAMES and (path.name, qual) not in allowed:
                    hits.append('%s:%s -> %s' % (path.name, qual, name))
    assert not hits, 'anatomy name referenced outside its declared sites: %r' % hits
    return dict(verdict='PASS', units_checked=checked, allowed=sorted('%s:%s' % x for x in allowed))


def _literal(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == [name]:
            return ast.literal_eval(node.value)
    raise AssertionError('no module constant ' + name)


def launcher():
    """The launcher's name-only mount list is the only launcher input E3_MODULES feeds: it is 5f46c07's list."""
    def listed(text):
        modules, exempt = _literal(text, 'E3_MODULES'), _literal(text, 'LAUNCHER_LIST_EXEMPT')
        return [name for name in modules if name not in exempt], modules, exempt
    old, new = show('patches/s5_engine/make_s5_overlap.py'), (ENGINE / 'make_s5_overlap.py').read_text()
    (a, old_modules, old_exempt), (b, new_modules, new_exempt) = listed(old), listed(new)
    assert a == b, 'launcher mount list changed'
    assert new_modules == old_modules + ('handler_anatomy',), 'E3_MODULES changed beyond the one module'
    assert new_exempt == old_exempt + ('handler_anatomy',), 'LAUNCHER_LIST_EXEMPT changed beyond the one module'
    gen_old = _literal(show('patches/s5_engine/decode_probe.py'), 'GENERATED_SOURCES')
    gen_new = _literal((ENGINE / 'decode_probe.py').read_text(), 'GENERATED_SOURCES')
    assert gen_new == gen_old + ('handler_anatomy',), 'decode_probe inventory changed beyond the one module'
    return dict(verdict='PASS', mount_list_entries=len(a), mount_list_sha256=hashlib.sha256(' '.join(a).encode()).hexdigest(),
                added_module='handler_anatomy', exempt=list(new_exempt))


def source_level():
    d = json.loads(DECLARATION.read_text())
    rows = paths()
    assert rows == sorted(d['paths']), sorted(set(rows) ^ set(d['paths']))
    modified = {row.split('\t', 1)[1] for row in rows if row.startswith('M\t')}
    added = {row.split('\t', 1)[1] for row in rows if row.startswith('A\t')}
    assert added == {'patches/s5_engine/' + NEW_MODULE}, sorted(added)
    assert modified == set(d['definitions']), sorted(modified ^ set(d['definitions']))
    defs = {p: prove(show(p), (ROOT / p).read_text(), p, names) for p, names in d['definitions'].items()}
    for name in sorted(modified | added):
        compile((ROOT / name).read_bytes(), name, 'exec')
    news = []
    for path, names in d['definitions'].items():
        file = path.rsplit('/', 1)[1]
        old = {q for q, _ in _units(show(path), file)}
        news += [(file, n) for n in names if not n.startswith('module:') and n not in old]
    assert set(news) == {(f, n) for f, names in NEW_HELPERS.items() for n in names}, sorted(news)
    modules = {(p.rsplit('/', 1)[1], n) for p, names in d['definitions'].items() for n in names if n.startswith('module:')}
    assert modules == {(f, n) for f, names in MODULE_STATEMENTS.items() for n in names}, sorted(modules)
    return dict(default_off=default_off(), imports=imports(), reach=reach(), launcher=launcher(), parent=parent(),
                paths=rows, definitions=defs, new_helpers={f + ':' + n: v for f, names in NEW_HELPERS.items()
                                                           for n, v in names.items()},
                module_statements={f + ':' + n: v for f, names in MODULE_STATEMENTS.items() for n, v in names.items()},
                not_projected=NOT_PROJECTED, declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())


def overlay_level(parent_overlay, candidate):
    sys.path.insert(0, str(ENGINE))
    import make_s5_overlap
    from build_native30_held import verify
    before, after = verify(parent_overlay), verify(candidate)
    assert set(after) - set(before) == {'e3/' + NEW_MODULE}, sorted(set(after) ^ set(before))
    assert set(before) <= set(after)
    changed_files = sorted(n for n in before if before[n] != after[n])
    declared = {'e3/executor_hooks.py', 'e3/decode_probe.py'}
    meta = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
            'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json'}
    assert set(changed_files) <= declared | meta, sorted(set(changed_files) - declared - meta)
    for rel in sorted(declared | {'e3/' + NEW_MODULE}):
        name = rel.split('/', 1)[1]
        assert (candidate / rel).read_text() == make_s5_overlap.package((ENGINE / name).read_text()), rel
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent_overlay / 'run-flash-s5-test.sh').read_bytes()
    return dict(verdict='PASS', parent=str(parent_overlay), candidate=str(candidate), changed=changed_files,
                added=['e3/' + NEW_MODULE], generated_sources_byte_identical=len(generated), launcher_byte_identical=True)


def negative_controls():
    d = json.loads(DECLARATION.read_text())
    hooks = (ENGINE / 'executor_hooks.py').read_text()
    old = show('patches/s5_engine/executor_hooks.py')
    p = 'patches/s5_engine/executor_hooks.py'

    def refused(fn):
        try:
            fn()
        except AssertionError:
            return 'REFUSED'
        raise AssertionError('undeclared change accepted')

    begin = function(hooks, 'worker_rpc_begin')
    begin.body.insert(0, ast.parse('proc._flash_s5_seen = True').body[0])
    spec = PROJECTIONS['executor_hooks.py']['worker_rpc_begin']
    outside = hooks.replace('def worker_rpc_begin(proc, method, kwargs):\n',
                            'def worker_rpc_begin(proc, method, kwargs):\n    import handler_anatomy\n', 1)
    top = hooks.replace('import hashlib\n', 'import hashlib\nimport handler_anatomy\n', 1)
    unguarded = hooks.replace('def worker_rpc_returned(proc, rpc, output_rank):\n',
                              'def worker_rpc_returned(proc, rpc, output_rank):\n    _anatomy(proc)\n', 1)
    elsewhere = (ENGINE / 'engine_driver.py').read_text() + '\n\ndef _stage1_probe():\n    return _ANATOMY\n'
    return {
        'undeclared-statement': refused(lambda: project(begin, function(old, 'worker_rpc_begin'), spec)),
        'undeclared-function': refused(lambda: prove(old, hooks + '\ndef stage1_undeclared():\n    pass\n', p,
                                                     d['definitions'][p])),
        'undeclared-path': refused(lambda: (_ for _ in ()).throw(AssertionError('path')) if sorted(
            paths() + ['A\tpatches/s5_engine/stage1_undeclared.py']) != sorted(d['paths']) else None),
        'import-outside-guard': refused(lambda: imports({'executor_hooks.py': outside})),
        'import-at-module-top': refused(lambda: imports({'executor_hooks.py': top})),
        'miscounted-removal': refused(lambda: project(function(hooks, 'worker_rpc_begin'), function(old, 'worker_rpc_begin'),
                                                      dict(off={PROGRESS_GUARD: 2, 'if state is not None and (anatomy := '
                                                                '_anatomy(proc)) is not None:': 1}))),
        'missing-removal': refused(lambda: project(function(hooks, 'worker_rpc_begin'), function(old, 'worker_rpc_begin'),
                                                   dict(off={PROGRESS_GUARD: 1}))),
        'unguarded-call-site': refused(lambda: project(function(unguarded, 'worker_rpc_returned'),
                                                       function(old, 'worker_rpc_returned'),
                                                       PROJECTIONS['executor_hooks.py']['worker_rpc_returned'])),
        'anatomy-name-elsewhere': refused(lambda: reach({'engine_driver.py': elsewhere})),
        'wrong-off-form': refused(lambda: project(function(hooks, '_anatomy'), 'def f(proc):\n    return None',
                                                  NEW_OFF[('executor_hooks.py', '_anatomy')][0])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--draft-declaration', action='store_true')
    a = parser.parse_args()
    if a.draft_declaration:
        print(json.dumps(declaration_draft(), indent=1, sort_keys=True))
        return 0
    if a.out is None:
        parser.error('--out is required')
    r = dict(verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.overlay:
        r['overlay'] = overlay_level(a.parent, a.overlay)
    r['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
    print('STAGE1_ANATOMY_FREEZE ' + r['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
