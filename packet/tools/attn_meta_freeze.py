#!/usr/bin/env python3
"""attn-meta freeze proof: attn-meta off is c1, and that reaches c1's own freeze (CPU, git; no box action).

Method: tools/attn_meta_default_off.py's projection, unchanged in its code (its files_check and project), run over the tree.
  attn_meta_off_is_c1   files_check(c1 delivery commit, the declared rows): every engine file is byte-identical to the c1
                        delivery's except the two added modules (attn_meta_batch.py, attn_meta_select.py, recorded by sha256),
                        the review record, and the eight declared files whose rows give the base bytes back exactly.
  both_off              the projected tree equals c1's engine at the pinned build source (tools/attn_meta_parent.json 'source',
                        the commit c1's build bundled and its alpha-next-c1-freeze.json proved) byte for byte, the review pin
                        aside; c1's freeze record is the pinned one (sha256), verdict PASS, source.tip == the build source,
                        candidate_manifest == the pinned manifest. c1's freeze carries the rest (c1 off is alpha-next, and on).
  negative_controls     mutated copies of the engine tree, each must be refused by the same files_check: an undeclared byte
                        in a file without rows, a byte outside the rows of a declared file, a declared row's current text gone,
                        a third added module, a removed base file.
  overlay_level         the built overlay against the regenerated c1 overlay (build OUT/parent): exactly the two modules
                        added; each changed runtime module and each new one is make_s5_overlap.package(committed source); each
                        parent module is package(c1 source); launcher, generated vllm sources, plugins and native libraries
                        byte-identical.
The switch-unset refusal tests are the default-off tool's own (a chain step, tools/attn_meta_default_off.py); nothing is claimed
equal with the switch on: the engine self-run, the VM records and the box compare that.

    python3 tools/attn_meta_freeze.py --work NEW_DIR [--controls]
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_native30_held import verify  # noqa: E402
import attn_meta_default_off as off  # noqa: E402
import attn_meta_parent_pin as parent_pin  # noqa: E402

PIN = parent_pin.load()
PARENT_SOURCE = PIN['source']
ENGINE = off.ENGINE + '/'
ROWS = ROOT / 'tools/attn_meta_default_off_rows.json'
DOC = json.loads(ROWS.read_text())
NEW = {'e3/' + name for name in DOC['added']}
CHANGED = {'e3/' + name for name in sorted({row['file'] for row in DOC['rows']})}
DECLARED_META = set(DOC['exempt'])
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}


def git(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout
    return out if binary else out.decode()


def base_commit():
    """The c1 delivery commit the rows project onto (the pin's, which is the default-off tool's default base)."""
    full = git('rev-parse', PIN.get('delivery_commit') or DOC['base']).strip()
    assert full == git('rev-parse', DOC['base']).strip(), 'the rows file names another base than the pinned c1 delivery'
    return full


@contextlib.contextmanager
def tree(root):
    """attn_meta_default_off read over ROOT-like ``root`` (its engine copy), git still answered by this repository."""
    saved = off.ROOT, off.git
    off.ROOT = Path(root)
    off.git = lambda *args: subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout
    try:
        yield
    finally:
        off.ROOT, off.git = saved


def projected_texts(root, base):
    """name -> bytes of ``root``'s engine with the declared rows applied (the added modules left out)."""
    by_file = {}
    for row in DOC['rows']:
        by_file.setdefault(row['file'], []).append(row)
    engine = Path(root) / off.ENGINE
    texts = {}
    for path in sorted(p for p in engine.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        name = str(path.relative_to(engine))
        if name in DOC['added']:
            continue
        data = path.read_bytes()
        texts[name] = off.project(data.decode(), by_file[name]).encode() if name in by_file else data
    return texts


def both_off(texts):
    record_path = Path(PIN['freeze'])
    faults = []
    differ = sorted(n for n, data in texts.items() if n not in DECLARED_META
                    and data != git('show', '%s:%s%s' % (PARENT_SOURCE, ENGINE, n), binary=True))
    if differ:
        faults.append('projected tree differs from the c1 engine bytes: %r' % (differ,))
    if hashlib.sha256(record_path.read_bytes()).hexdigest() != PIN['freeze_sha256']:
        faults.append('c1 freeze record changed')
    record = json.loads(record_path.read_text())
    if record.get('verdict') != 'PASS':
        faults.append('c1 freeze not PASS')
    if (record.get('source') or {}).get('tip') != PARENT_SOURCE:
        faults.append('c1 freeze proved another commit')
    if record.get('candidate_manifest') != PIN['manifest']:
        faults.append('c1 freeze names another bundle')
    return dict(projected_files=len(texts), projected_byte_identical=not differ, declared_meta=sorted(DECLARED_META),
                c1_freeze=str(record_path), c1_freeze_sha256=PIN['freeze_sha256'], c1_freeze_verdict=record.get('verdict'),
                proved_commit=(record.get('source') or {}).get('tip'), parent_source=PARENT_SOURCE,
                verdict='FAIL' if faults else 'PASS', faults=faults)


def prove(root, work):
    """Both receipts over ``root``'s engine; the projected tree is written to WORK (a new directory, kept)."""
    base = base_commit()
    with tree(root):
        problems, rows = off.files_check(base, DOC)
    texts = projected_texts(root, base) if not problems else {}
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    for name, data in texts.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_bytes(data)
    link = both_off(texts) if texts else dict(verdict='FAIL', faults=['no projection (attn-meta off is not c1)'])
    faults = ['attn-meta-off-vs-c1: ' + f for f in problems] + ['both-off: ' + f for f in link['faults']]
    engine = Path(root) / off.ENGINE
    added = {n: hashlib.sha256((engine / n).read_bytes()).hexdigest() for n in DOC['added'] if (engine / n).is_file()}
    return dict(schema='attn-meta-default-off-freeze/1', verdict='FAIL' if faults else 'PASS', tip=git('rev-parse', 'HEAD').strip(),
                dirty=bool(git('status', '--porcelain', '--', off.ENGINE).strip()), work=str(work), base=base,
                method='attn_meta_default_off.files_check over the declared rows (tools/attn_meta_default_off_rows.json)',
                rows_sha256=hashlib.sha256(ROWS.read_bytes()).hexdigest(), added_modules_sha256=added,
                attn_meta_off_is_c1=dict(verdict='FAIL' if problems else 'PASS', files=rows, problems=problems),
                both_off_reaches_c1_freeze=link, faults=faults, dev_parent=bool(PIN.get('dev')))


def source_level(work):
    receipt = prove(ROOT, work)
    assert receipt['verdict'] == 'PASS', receipt['faults']
    return receipt


def _mutants():
    """name -> function(engine dir) mutating a copy; each copy must be refused."""
    plain = sorted(n for n in git('ls-tree', '-r', '--name-only', 'HEAD', off.ENGINE).split()
                   if n.endswith('.py') and n.rsplit('/', 1)[1] not in {r['file'] for r in DOC['rows']} | set(DOC['added']))[0]
    declared = DOC['rows'][0]

    def undeclared_byte(engine):
        with (engine / Path(plain).name).open('a') as f:
            f.write('\n# undeclared\n')

    def byte_outside_rows(engine):
        with (engine / declared['file']).open('a') as f:
            f.write('\n# outside the rows\n')

    def row_text_gone(engine):
        path = engine / declared['file']
        path.write_text(path.read_text().replace(declared['current'], declared['current'] + '#gone', 1))

    def third_module(engine):
        (engine / 'attn_meta_extra.py').write_text('X = 1\n')

    def removed_base_file(engine):
        (engine / Path(plain).name).unlink()

    return dict(undeclared_byte=undeclared_byte, byte_outside_rows=byte_outside_rows, row_text_gone=row_text_gone,
                third_added_module=third_module, removed_base_file=removed_base_file)


def negative_controls():
    rows = {}
    for name, mutate in _mutants().items():
        with tempfile.TemporaryDirectory(prefix='attn-meta-freeze-neg-') as tmp:
            root = Path(tmp) / 'root'
            shutil.copytree(ROOT / off.ENGINE, root / off.ENGINE, ignore=shutil.ignore_patterns('__pycache__'))
            mutate(root / off.ENGINE)
            receipt = prove(root, Path(tmp) / 'work')
            rows[name] = 'REFUSED' if receipt['verdict'] == 'FAIL' else 'ACCEPTED'
    assert set(rows.values()) == {'REFUSED'}, rows
    return rows


def _packaged(source):
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_s5_overlap as maker
    return maker.package(source)


def overlay_level(parent, candidate):
    before, after = verify(parent), verify(candidate)
    assert set(after) - set(before) == NEW, ('overlay files added', sorted(set(after) - set(before)))
    assert not set(before) - set(after), 'overlay file removed'
    changed = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'}
    runtime = {n for n in CHANGED if n in after}
    assert changed <= runtime | META, sorted(changed - (runtime | META))
    for path in sorted(runtime | NEW):
        name = path.split('/', 1)[1]
        assert (candidate / path).read_text() == _packaged((ROOT / ENGINE / name).read_text()), \
            'overlay module differs from the packaged committed source: ' + path
    for path in sorted(runtime):
        name = path.split('/', 1)[1]
        assert (parent / path).read_text() == _packaged(git('show', PARENT_SOURCE + ':' + ENGINE + name)), \
            'parent overlay module differs from the packaged c1 source: ' + path
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), parent_commit=PARENT_SOURCE, changed=sorted(changed), added=sorted(NEW),
                generated_sources_byte_identical=len(generated), model_and_plugin_byte_identical=True,
                native_libraries_byte_identical=True, launcher_byte_identical=True,
                modules_equal_packaged_committed_source=sorted(runtime | NEW),
                parent_modules_equal_packaged_c1_source=sorted(runtime))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--work', type=Path, required=True, help='new directory for the projected engine tree (kept)')
    p.add_argument('--controls', action='store_true')
    a = p.parse_args()
    record = prove(ROOT, a.work)
    if a.controls:
        record['negative_controls'] = negative_controls()
    print(json.dumps(dict(verdict=record['verdict'], faults=record['faults'], added_modules_sha256=record['added_modules_sha256'],
                          negative_controls=record.get('negative_controls')), indent=1, sort_keys=True))
    raise SystemExit(0 if record['verdict'] == 'PASS' else 1)
