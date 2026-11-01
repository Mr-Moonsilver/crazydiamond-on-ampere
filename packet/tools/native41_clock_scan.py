#!/usr/bin/env python3
"""Native41 client-clock gate, derived from L2's native43 gate (claude/native43-packaging e527ba9): no client-side
comparison of the client's own clock identity in the native41 queue, its children or the owner runner.

On the box the client runs in the CT (lxcfs boot_id, CT time namespace) and the engine in its container, so a
client-side equality with the client's own identity refuses although CLOCK_MONOTONIC is shared (native40 run 1:
e3_native40_unarmed_box.py:113-118).  Native41's unarmed client (e3_native41_unarmed_box) is the one declared
client-side clock site: it compares the engine status clock with the CONTAINER identity read from the CT side
(docker inspect init pid -> /proc/<pid>/ns/time and timens_offsets; docker exec cat boot_id; zero offsets on
both sides) and requires the engine sample inside the bracket (prove_clock_domain).  Differences from native43:
  * shell texts: the native41 draft, declare and settings plus qlib.sh (no pair wrapper or policies);
  * CONTAINER_PROOF: in e3_native41_unarmed_box only prove_clock_domain may appear, its expected identity must
    come from container_identity, it must never call clock_identity, the boot_id path may appear only in the
    read-only docker exec argv, and every docker argv must be one of the two read-only forms (DOCKER_ARGV);
  * the native43 boundary-client check is replaced by: e3_native41_unarmed_box is a queue entry run with
    --container-id, and no client unit or client root sends overlap_evidence/overlap_begin/overlap_end or
    calls native_window (the auto arms read status only);
  * two more negative controls: the unarmed client comparing clock_identity(), and a docker exec that is not
    the read-only cat.

Static gate over the frozen bundle (or a source tree), no box action:
  * client texts = the pair wrapper, both per-arm drafts, both declares (staging), both settings files, qlib.sh
    (sourced) and, when given, the owner runner.  Every `source`/`bash` child must resolve inside that set.
  * every `python3` invocation in them is classified: $DIR/x.py entry, -m, -c code, `-` heredoc, or the
    box-resident $SETCFG (scanned through its repo reference).  An unclassified invocation refuses.
  * entries, heredoc bodies and -c code are client code; their imports, plus harness scripts spawned through
    sys.executable/'python3' inside the closure, give the transitive client closure.
  * refuse if: a client code unit names prove_clock_domain, clock_identity, native_window(, NativeControlClient,
    e3_native_performance or e3_native40_unarmed_box; one of the last two modules is in the closure; a call of
    prove_clock_domain()/clock_identity() sits outside the engine-side allowlist; NativeControlClient( or
    NativeOverlapSession( is constructed outside the declared engine-attach functions; a closure module reads
    /proc/sys/kernel/random/boot_id or /proc/self/ns/time outside the allowlist; or native43_boundary_client does
    not import native43_boundary_clock.prove.
  * negative controls (each must be flagged): an extra queue line running e3_native40_unarmed_box.py, a synthetic
    entry calling prove_clock_domain(clock_identity()), a synthetic entry comparing boot_id inline, an
    unclassified `$PY x.py`-style python3 form, the unarmed client calling clock_identity(), a non-read-only
    docker exec, and an arm sending overlap_evidence.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

POLICIES = ()
DRAFT = 'q18a_native41_autoarm.draft.sh'
SHELL = (DRAFT, 'q18a_native41_autoarm_declare.sh', 'q18a_native41_autoarm_settings.sh')
# The one declared client-side clock site and what it may contain.
CONTAINER_PROOF = 'e3_native41_unarmed_box'
CONTAINER_PROOF_BASIS = ('prove_clock_domain against the engine container identity read from the CT side (docker inspect '
                         'init pid -> /proc/<pid>/ns/time and timens_offsets, docker exec cat boot_id), zero offsets on '
                         'both sides, engine sample inside the client send/recv bracket')
BOOT_ID = '/proc/sys/kernel/random/boot_id'
DOCKER_ARGV = (('inspect', '--format', '{{.State.Pid}}', '<container_id>'), ('exec', '<container_id>', 'cat', BOOT_ID))
NATIVE_WINDOW = re.compile(r"native_window\(|['\"](?:overlap_evidence|overlap_begin|overlap_end|auto_window_begin|auto_window_end)['\"]")
NAMES = re.compile(r'prove_clock_domain|clock_identity|native_window\(|NativeControlClient|e3_native_performance|e3_native40_unarmed_box')
CALLS = re.compile(r'\b(prove_clock_domain|clock_identity)\(')
PROC = re.compile(r'/proc/sys/kernel/random/boot_id|/proc/self/ns/time')
FORBIDDEN_MODULES = ('e3_native_performance', 'e3_native40_unarmed_box')
# Engine-side modules whose calls of the identity functions are allowed (they compare the engine's own identity
# with replies produced in the same process/view): definitions, the in-process controller, the worker.
ENGINE_CALLS = {'native_overlap_metrics': 'definitions', 'native_overlap_control': 'NativeControlClient methods (in-process EngineCore controller)',
                'native_overlap_worker': 'worker-side identity of its own replies'}
# The only places that may construct the in-process controller/session: functions that receive the engine object.
ENGINE_CONSTRUCTORS = {('NativeControlClient', 'native_overlap_session', 'NativeOverlapSession.__init__'),
                       ('NativeControlClient', 'native_qualification_frontend', 'attach_router._default_frontend_factory'),
                       ('NativeControlClient', 'native_qualification_frontend', 'attach_router._default_worker_factory'),
                       ('NativeControlClient', 'probe_control', 'attach_native_router.endpoint'),
                       ('NativeOverlapSession', 'probe_control', 'attach_native_router.endpoint')}
PROC_READERS = {'native43_boundary_clock': 'provenance only (identity_equality_required=False)',
                'native_overlap_metrics': 'clock_identity definition', 'engine_identity': 'descriptive identity capture',
                'engine_driver': "engine status clock record (_clock_record: the engine's own view, proc='engine')"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imports(source):
    out = set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import):
            out |= {a.name.split('.')[0] for a in n.names}
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split('.')[0])
        if (isinstance(n, ast.Call) and getattr(n.func, 'attr', getattr(n.func, 'id', '')) in ('import_module', '__import__')
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add(str(n.args[0].value).split('.')[0])
    return out


def constructions(source):
    found = []

    def walk(node, path):
        for child in ast.iter_child_nodes(node):
            sub = path + [child.name] if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else path
            if isinstance(child, ast.Call):
                name = getattr(child.func, 'id', getattr(child.func, 'attr', ''))
                if name in ('NativeControlClient', 'NativeOverlapSession'):
                    found.append((name, '.'.join(path) or '<module>', child.lineno))
            walk(child, sub)
    walk(ast.parse(source), [])
    return found


def heredocs(text):
    """(tag, body, start, end) for every PY* heredoc; the tag line may carry redirections after the tag."""
    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        m = re.search(r"<<-?\s*['\"]?(PY[A-Z_0-9]*)['\"]?", lines[i])
        if m and not lines[i].lstrip().startswith('#'):
            tag, j = m.group(1), i + 1
            while j < len(lines) and lines[j].strip() != tag:
                j += 1
            out.append((tag, '\n'.join(lines[i + 1:j]), i, j))
            i = j + 1
        else:
            i += 1
    return out


def load(bundle=None, tree=None, extra_modules=None):
    if bundle:
        dirs, runs, qlib = (bundle, bundle / 'e3_builder'), bundle / 'runs', bundle / 'qlib.sh'
    else:
        dirs, runs, qlib = (tree / 'harness', tree / 'patches' / 's5_engine'), tree / 'harness' / 'runs', tree / 'harness' / 'qlib.sh'
    top = bundle or tree
    sources = {}
    for d in dirs:
        for p in sorted(d.glob('*.py')):
            sources.setdefault(p.stem, (str(p.relative_to(top)), p.read_text()))
    for name, text in (extra_modules or {}).items():
        sources[name] = ('<synthetic>/' + name + '.py', text)
    texts = {q: (runs / q).read_text() for q in SHELL}
    texts['qlib.sh'] = qlib.read_text()
    return sources, texts


def functions(text):
    """{line (1-based): function name} for the bodies of top-level shell functions."""
    out, lines, i = {}, text.split('\n'), 0
    while i < len(lines):
        m = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{', lines[i])
        if m:
            j = i
            if not lines[i].rstrip().endswith('}'):
                j = i + 1
                while j < len(lines) and not lines[j].startswith('}'):
                    j += 1
            for k in range(i, j + 1):
                out[k + 1] = m.group(1)
            i = j + 1
        else:
            i += 1
    return out


def called(fn, texts):
    for name, text in texts.items():
        body = functions(text)
        for i, line in enumerate(text.split('\n'), 1):
            if body.get(i) == fn and re.match(r'%s\(\)' % fn, line) or line.lstrip().startswith('#'):
                continue
            if re.search(r'(?<![\w-])%s(?![\w(-])' % re.escape(fn), line):
                return True
    return False


def scan(sources, texts, runner=None, setcfg=None):
    problems, units, entries, invocations, children, dead = [], {}, {}, [], [], []
    roots = set()
    live = {}
    for name, text in texts.items():
        for line, fn in functions(text).items():
            if fn not in live:
                live[fn] = called(fn, texts)
    for name, text in texts.items():
        fnat = functions(text)
        docs = heredocs(text)
        skip = set()
        for tag, body, start, end in docs:
            skip |= set(range(start + 1, end + 1))
            units['%s:%s@%d' % (name, tag, start + 1)] = body
        logical, buf, first = [], '', None
        for i, line in enumerate(text.split('\n')):
            if i in skip:
                continue
            first = i if not buf else first
            if line.endswith('\\'):
                buf += line[:-1] + ' '
                continue
            logical.append((first + 1, buf + line))
            buf = ''
        for lineno, line in logical:
            code = line.split(' #')[0] if not line.lstrip().startswith('#') else ''
            fn = fnat.get(lineno)
            if fn and not live.get(fn) and re.search(r'\bpython3\b|(?:^|\s)(?:source|\.|bash)\s', code):
                dead.append(dict(at='%s:%d' % (name, lineno), function=fn, line=code.strip()[:140]))
                continue
            for m in re.finditer(r'(?:^|[\s;&|(`$])(?:source|\.|bash)\s+("?[^\s;|&)]+"?)', code):
                target = m.group(1).strip('"')
                if target in ('-', '-c') or target.startswith('-'):
                    continue
                resolved = target.replace('$H/runs/', '').replace('$H/', '').replace('${H}/', '')
                if target == '$0':
                    resolved = name
                cands = [resolved.replace('${policy}', p) for p in POLICIES] if '${policy}' in resolved else [resolved]
                ok = all(c in texts for c in cands)
                children.append(dict(at='%s:%d' % (name, lineno), target=target, resolved=cands, scanned=ok))
                if not ok:
                    problems.append('unscanned-shell-child:%s:%d:%s' % (name, lineno, target))
            for m in re.finditer(r'\bpython3(?![.\w])((?:\s+-(?![cm](?=\s))[A-Za-z]+(?=\s))*)\s+(\S+)(?:\s+(\S+))?', code):
                flags, arg, nxt = m.group(1), m.group(2).strip('"'), (m.group(3) or '').strip('"')
                at = '%s:%d' % (name, lineno)
                e = re.fullmatch(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/([A-Za-z0-9_]+)\.py', arg)
                if e and e.group(1) in sources:
                    kind, target = 'entry', e.group(1)
                    roots.add(target)
                    entries.setdefault(target, []).append(at)
                elif arg == '-m' and nxt:
                    kind, target = 'module', nxt
                    roots.add(nxt)
                    entries.setdefault(nxt, []).append(at)
                elif arg == '-c':
                    c = re.search(r"python3[^\n]*?-c\s+'([^']*)'", code[m.start():])
                    kind, target = 'inline', c.group(1) if c else None
                    if c:
                        units[at + ':-c'] = c.group(1)
                    else:
                        problems.append('unparsed-python-c:' + at)
                elif arg == '-' and re.search(r"<<-?\s*['\"]?PY", code):
                    kind, target = 'heredoc', None
                elif arg in ('$SETCFG', '${SETCFG}'):
                    kind, target = 'box-resident-setcfg', 'deploy/setcfg.py reference'
                    if setcfg is None:
                        problems.append('setcfg-reference-missing:' + at)
                    else:
                        units[at + ':setcfg-reference'] = setcfg
                else:
                    kind, target = 'unclassified', arg
                    problems.append('unclassified-python-invocation:%s:%s' % (at, arg))
                invocations.append(dict(at=at, kind=kind, target=target, flags=flags.split()))
    if runner is not None:
        units['owner-runner'] = runner
    for unit, code in units.items():
        try:
            roots |= {d for d in imports(code) if d in sources}
        except SyntaxError:
            problems.append('unit-not-python:' + unit)
    closure, stack, spawned = set(), sorted(roots), {}
    while stack:
        m = stack.pop()
        if m in closure:
            continue
        closure.add(m)
        src = sources[m][1]
        nxt = {d for d in imports(src) if d in sources}
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if re.search(r"sys\.executable|['\"]python3['\"]", line):
                for name in re.findall(r'\b([A-Za-z0-9_]+)\.py\b', ' '.join(lines[i:i + 4])):
                    if name in sources and name != m:
                        nxt.add(name)
                        spawned.setdefault(name, []).append('%s:%d' % (sources[m][0], i + 1))
        stack += sorted(nxt - closure)
    direct, declared, windows = [], [], []
    for unit, code in sorted(units.items()):
        for i, line in enumerate(code.splitlines(), 1):
            if NAMES.search(line):
                direct.append('%s+%d: %s' % (unit, i, line.strip()[:140]))
            if NATIVE_WINDOW.search(line):
                windows.append('%s+%d: %s' % (unit, i, line.strip()[:140]))
    for m in sorted(roots):
        for i, line in enumerate(sources[m][1].splitlines(), 1):
            hit = NAMES.search(line)
            # The declared site: prove_clock_domain only, nothing else from NAMES.
            if hit and m == CONTAINER_PROOF and set(NAMES.findall(line)) == {'prove_clock_domain'}:
                declared.append('%s:%d: %s' % (sources[m][0], i, line.strip()[:140]))
            elif hit:
                direct.append('%s:%d: %s' % (sources[m][0], i, line.strip()[:140]))
            if NATIVE_WINDOW.search(line):
                windows.append('%s:%d: %s' % (sources[m][0], i, line.strip()[:140]))
    if direct:
        problems.append('client-entry-references-identity-check')
    if windows:
        problems.append('client-sends-window-or-evidence-action')
    for m in FORBIDDEN_MODULES:
        if m in closure:
            problems.append('forbidden-module-in-client-closure:' + m)
    calls, ctors, readers = [], [], []
    for m in sorted(closure):
        path, src = sources[m]
        for i, line in enumerate(src.splitlines(), 1):
            s = line.lstrip()
            if s.startswith(('#', 'def ')):
                continue
            if CALLS.search(line):
                site = m == CONTAINER_PROOF and set(CALLS.findall(line)) == {'prove_clock_domain'}
                calls.append(dict(at='%s:%d' % (path, i), module=m,
                                  allowed=ENGINE_CALLS.get(m) or (CONTAINER_PROOF_BASIS if site else None)))
                if m not in ENGINE_CALLS and not site:
                    problems.append('identity-call-outside-engine-side:%s:%d' % (path, i))
            if PROC.search(line):
                docker_exec = m == CONTAINER_PROOF and BOOT_ID in line and re.search(r"\['exec', \w+, 'cat', '%s'\]" % re.escape(BOOT_ID), line)
                docstring_or_const = m == CONTAINER_PROOF and not re.search(r'open\(|read_text|readlink|\.read\(', line)
                readers.append(dict(at='%s:%d' % (path, i), module=m, allowed=PROC_READERS.get(m) or (
                    'read-only docker exec argv in the engine container' if docker_exec else
                    'text only (docstring/constant), no read' if docstring_or_const else None)))
                if m not in PROC_READERS and not (docker_exec or docstring_or_const):
                    problems.append('clock-identity-read-outside-allowlist:%s:%d' % (path, i))
        for cls, fn, line in constructions(src):
            ok = (cls, m, fn) in ENGINE_CONSTRUCTORS
            ctors.append(dict(at='%s:%d' % (path, line), cls=cls, function=fn, engine_attach=ok))
            if not ok:
                problems.append('engine-controller-constructed-outside-engine-attach:%s:%d:%s' % (path, line, fn))
    site = container_site(sources.get(CONTAINER_PROOF, ('', ''))[1])
    entry = any(i['kind'] == 'entry' and i['target'] == CONTAINER_PROOF for i in invocations)
    container_arg = any(CONTAINER_PROOF + '.py' in line and '--container-id' in line
                        for text in texts.values() for line in re.sub(r'\\\n\s*', ' ', text).split('\n'))
    site.update(queue_entry=entry, queue_passes_container_id=container_arg)
    if not (entry and container_arg):
        problems.append('unarmed-container-proof-not-run-by-the-queue')
    problems += ['container-proof-site:' + r for r in site['reasons']]
    bracket = not site['reasons'] and entry and container_arg
    return dict(uncalled_function_lines=dead, client_units=sorted(units), entries={k: sorted(set(v)) for k, v in sorted(entries.items())},
                invocations=invocations, shell_children=children, roots=sorted(roots), spawned=spawned,
                closure=sorted(closure), direct_references=direct, reachable_identity_calls=calls,
                controller_constructions=ctors, clock_identity_readers=readers, declared_container_site=declared,
                window_or_evidence_references=windows, container_proof=site,
                unarmed_client_uses_container_bracket=bracket, problems=problems)


def container_site(src):
    """The declared site's exact shape: expected identity from container_identity, never clock_identity,
    every docker argv one of DOCKER_ARGV, docker only through docker_cli."""
    out = dict(reasons=[], docker_argv=[], prove_calls=[])
    if not src:
        out['reasons'].append('module-absent')
        return out
    tree = ast.parse(src)
    if 'clock_identity' in src:
        out['reasons'].append('names-clock-identity')
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant) \
                and node.elts[0].value in ('inspect', 'exec'):
            form = tuple(e.value if isinstance(e, ast.Constant) else '<container_id>' for e in node.elts)
            out['docker_argv'].append(list(form))
            if form not in DOCKER_ARGV:
                out['reasons'].append('docker-argv-not-read-only:' + ' '.join(map(str, form)))
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'prove_clock_domain':
            out['prove_calls'].append(node.lineno)
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', '')) in (
                'run', 'Popen', 'call', 'check_call', 'check_output') and getattr(node.func, 'value', None) is not None \
                and getattr(node.func.value, 'id', '') == 'subprocess':
            first = node.args[0] if node.args else None
            ok = (isinstance(first, ast.BinOp) and isinstance(first.left, ast.List)
                  and [getattr(e, 'value', None) for e in first.left.elts] == ['docker'])
            if not ok:
                out['reasons'].append('subprocess-not-the-docker-cli:%d' % node.lineno)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clock_proof'), None)
    body = ast.get_source_segment(src, fn) if fn else ''
    if len(out['prove_calls']) != 1 or not fn or not any(fn.lineno <= n <= fn.end_lineno for n in out['prove_calls']):
        out['reasons'].append('prove-call-not-the-one-in-clock_proof')
    if not re.search(r'container = container_identity\(', body) \
            or not re.search(r"expected = dict\(proved=not reasons, boot_id=container\['boot_id'\],\s*time_namespace=container\['time_namespace'\]\)", body) \
            or not re.search(r'prove_clock_domain\(engine, expected, send, recv\)', body):
        out['reasons'].append('expected-identity-not-the-container')
    return out


def main(a):
    bundle = a.bundle.resolve() if a.bundle else None
    tree = a.tree.resolve() if a.tree else None
    sources, texts = load(bundle, tree)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    result = scan(sources, texts, runner, setcfg)
    draft = DRAFT
    site_src = sources[CONTAINER_PROOF][1]
    controls = {
        'queue-runs-native40-unarmed-box': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/e3_native40_unarmed_box.py" --out "$Q/x.json"\n'}),
                                            {'e3_native40_unarmed_box': 'from native_overlap_metrics import clock_identity\n'}),
        'entry-calls-prove-clock-domain': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/native43_synthetic_client.py"\n'}),
                                           {'native43_synthetic_client': 'from native_overlap_metrics import prove_clock_domain, clock_identity\n'
                                                                         'prove_clock_domain({}, clock_identity(), 1, 2)\n'}),
        'entry-compares-boot-id-inline': (dict(texts, **{draft: texts[draft] + '\npython3 -B "$H/native43_synthetic_inline.py"\n'}),
                                          {'native43_synthetic_inline': "from pathlib import Path\nok = Path('/proc/sys/kernel/random/boot_id').read_text() == 'x'\n"}),
        'unclassified-python-form': (dict(texts, **{draft: texts[draft] + '\npython3 "$X_DIR/../e3_native41_unarmed_box.py"\n'}), {}),
        'unarmed-client-compares-client-identity': (texts, {CONTAINER_PROOF: site_src.replace(
            'domain = prove_clock_domain(engine, expected, send, recv)',
            'from native_overlap_metrics import clock_identity\n    domain = prove_clock_domain(engine, clock_identity(), send, recv)')}),
        'docker-exec-not-read-only': (texts, {CONTAINER_PROOF: site_src.replace(
            "['exec', container_id, 'cat', '/proc/sys/kernel/random/boot_id']",
            "['exec', container_id, 'sh', '-c', 'cat /proc/sys/kernel/random/boot_id']")}),
        'arm-sends-overlap-evidence': (texts, {'e3_native41_autoarm_box': sources['e3_native41_autoarm_box'][1]
                                               + "\n\ndef _x(client):\n    return client.call('overlap_evidence')\n"}),
    }
    negatives = {}
    for label, (t, extra) in controls.items():
        s2 = dict(sources, **{k: ('<synthetic>/' + k + '.py', v) for k, v in extra.items()})
        negatives[label] = scan(s2, t, runner, setcfg)['problems']
    ok = not result['problems'] and all(negatives.values())
    origin = dict(bundle=str(bundle), bundle_manifest=sha(bundle / 'SHA256SUMS')) if bundle else dict(tree=str(tree))
    record = dict(schema='native41-clock-scan/1', derived_from='claude/native43-packaging e527ba9 tools/native43_clock_scan.py', verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS',
                  **origin, shell_shas={q: hashlib.sha256(t.encode()).hexdigest() for q, t in sorted(texts.items())},
                  runner=dict(path=str(a.runner), sha256=sha(a.runner)) if a.runner else None,
                  setcfg_reference=dict(path=str(a.setcfg_reference), sha256=sha(a.setcfg_reference),
                                        note='box-resident /tmp/setcfg.py is not in the bundle; its repo reference is scanned')
                  if a.setcfg_reference else None,
                  scan=result, negative_controls=negatives, static_only=True, box_action=False)
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], problems=result['problems'], units=len(result['client_units']),
                          roots=len(result['roots']), closure=len(result['closure']), invocations=len(result['invocations']),
                          negatives={k: bool(v) for k, v in negatives.items()})))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
