#!/usr/bin/env python3
"""alpha-next-c1 freeze proof: C1 off is alpha-next, and both off reaches alpha-next's own freeze (CPU, git; no box action).

Chosen method (the smaller honest one): tools/stage_default_off_c1.py's projection, unchanged in its code, re-pointed at the
pinned alpha-next parent's engine commit (tools/alpha_next_c1_parent.json 'source', the commit alpha-next's build bundled and
its alpha-next-freeze.json proved) instead of C0's. Not alpha-next's freeze extended: that proof is git-composition-specific
to alpha-next's own steps and would have to be re-derived; C1's projection is already the exact C1 delta.

  c1_off_is_alpha_next  stage_default_off_c1.project(engine, parent source) with COMPOSED_PROJECTIONS: C1's own rows, of which
                        exactly two (decode_probe's GENERATED_SOURCES tail and make_s5_overlap's module list and launcher
                        exemption) are re-based on alpha-next's text, where 'handler_anatomy' sits before 'stage_overlap'
                        (COMPOSED_SWAPS, each applied exactly once to C1's row), preceded in s5_startup, native_overlap_session
                        and native_overlap_control by the per-arm deferred bind's rows (DEFER_ROWS: the OTHERS of
                        tools/alpha_next_c1_rebind_default_off.py, reused). Every other engine file is byte-identical to
                        the parent's; each declared file, after its projection, has the parent's exact AST; the one new module
                        is stage_overlap.py; the new names appear outside it only on the declared lines (REFERENCES, the counts
                        the merged tree holds). stage_overlap.py's sha256 is recorded, never pinned here (it is still edited).
  both_off              the projected tree equals alpha-next's engine at the parent source byte for byte (the declared review
                        pin aside; C1's seams are pure insertions and swaps, so the projection gives back the bytes), and that
                        commit is exactly the one alpha-next's own freeze proved: alpha-next-freeze.json, pinned by sha256 in
                        the parent file, verdict PASS, source.head == parent source, candidate_manifest == parent manifest.
                        alpha-next's freeze carries the rest (alpha-next off is Step B, through its composition and features).
  negative_controls     stage_default_off_c1.negative_controls (its ten mutated copies, C0's last one included) over the same
                        re-pointed projection; each must be refused.
  overlay_level         the built overlay against the regenerated alpha-next overlay (build OUT/parent): exactly
                        e3/stage_overlap.py added; each changed module and the new one is make_s5_overlap.package(committed
                        source); each parent module is package(alpha-next source); launcher, generated vllm sources, plugins
                        and native libraries byte-identical.
With the overlap switch on nothing is claimed equal here: the engine self-run's pairs, the suites and the box compare that.

    python3 tools/alpha_next_c1_freeze.py --work NEW_DIR [--controls]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_native30_held import verify  # noqa: E402
import alpha_next_c1_parent_pin as parent_pin  # noqa: E402
import stage_default_off_c1 as off  # noqa: E402
import alpha_next_c1_rebind_default_off as defer_off  # noqa: E402  (its OTHERS: the defer_bind rows, reused unchanged)

PIN = parent_pin.load()
PARENT_SOURCE = PIN['source']
ENGINE = off.ENGINE
NEW = {'e3/' + name for name in off.NEW_MODULES}
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}

# C1's rows re-based on alpha-next's text (stage-1 anatomy's 'handler_anatomy' precedes C1's 'stage_overlap'); applied to both
# sides of C1's row, each exactly once.
COMPOSED_SWAPS = {
    'decode_probe.py': (("'stage_wire', 'stage_progress', 'stage_overlap')\n",
                         "'stage_wire', 'stage_progress', 'handler_anatomy', 'stage_overlap')\n"),
                        ("'stage_wire', 'stage_progress')\n", "'stage_wire', 'stage_progress', 'handler_anatomy')\n")),
    'make_s5_overlap.py': (("              'stage_wire', 'stage_progress',\n"
                            "              # C1: two chunks in flight (FLASH_S5_STAGE_OVERLAP).\n"
                            "              'stage_overlap')\n",
                            "              'stage_wire', 'stage_progress',\n"
                            "              # Stage-1 handler anatomy (FLASH_S5_HANDLER_ANATOMY, default off): imported by executor_hooks"
                            " only under the switch.\n"
                            "              'handler_anatomy',\n"
                            "              # C1: two chunks in flight (FLASH_S5_STAGE_OVERLAP).\n"
                            "              'stage_overlap')\n"),
                           ("              'stage_wire', 'stage_progress')\n",
                            "              'stage_wire', 'stage_progress',\n"
                            "              # Stage-1 handler anatomy (FLASH_S5_HANDLER_ANATOMY, default off): imported by executor_hooks"
                            " only under the switch.\n"
                            "              'handler_anatomy')\n"),
                           ("'native_auto_arm', 'stage_wire', 'stage_progress', 'stage_overlap')\n",
                            "'native_auto_arm', 'stage_wire', 'stage_progress', 'handler_anatomy', 'stage_overlap')\n"),
                           ("'native_auto_arm', 'stage_wire', 'stage_progress')\n",
                            "'native_auto_arm', 'stage_wire', 'stage_progress', 'handler_anatomy')\n")),
}
# The per-arm deferred bind (e656ee81) in the three non-new files that carry it: tools/alpha_next_c1_rebind_default_off.py's
# OTHERS, reused unchanged. Each row takes the file back to C1's text before the defer_bind work (its default-off proof checks
# that against 844dfe8d), so they run first and C1's own rows follow. stage_overlap.py needs none here: it is the new module,
# recorded by sha256, never projected (its PER_ARM rows are that tool's own proof).
DEFER_ROWS = {name: [tuple(row) for row in rows] for name, rows in defer_off.OTHERS.items()}
# The merged tree's counts of lines naming the C1 surface (no alpha-next line names it); the defer_bind rows add lines naming
# FLASH_S5_STAGE_OVERLAP_* and native_stage_overlap in s5_startup (+3), native_overlap_control (+7) and native_overlap_session (+4).
REFERENCES = dict(off.REFERENCES, **{'s5_startup.py': 12, 'native_overlap_control.py': 15, 'native_overlap_session.py': 6})


def composed_projections(base=None):
    """C1's PROJECTIONS with COMPOSED_SWAPS applied; every swap must change exactly one string of its file's rows."""
    base = off.PROJECTIONS if base is None else base
    out = {}
    for name, rows in base.items():
        swaps = COMPOSED_SWAPS.get(name, ())
        new_rows, used = [], [0] * len(swaps)
        for old, new in rows:
            pair = []
            for text in (old, new):
                for i, (a, b) in enumerate(swaps):
                    if text.count(a) == 1 and (b not in text):
                        text = text.replace(a, b)
                        used[i] += 1
                        break
                pair.append(text)
            new_rows.append(tuple(pair))
        assert used == [1] * len(swaps), ('composed swap not applied exactly once', name, used)
        out[name] = new_rows
    assert set(COMPOSED_SWAPS) <= set(out), 'a composed swap names a file C1 does not project'
    return out


def with_defer(projections):
    """DEFER_ROWS ahead of each file's composed rows; every defer file must already be one C1 projects."""
    assert set(DEFER_ROWS) <= set(projections), 'a defer row names a file C1 does not project'
    return {name: DEFER_ROWS.get(name, []) + list(rows) for name, rows in projections.items()}


COMPOSED_PROJECTIONS = with_defer(composed_projections())
CHANGED = {'e3/' + name for name in COMPOSED_PROJECTIONS}   # the overlay tamper control's declared set


class _Pointed:
    """stage_default_off_c1's module constants re-pointed at the composition for one call; restored on exit."""

    def __enter__(self):
        self.saved = off.PROJECTIONS, off.REFERENCES
        off.PROJECTIONS, off.REFERENCES = COMPOSED_PROJECTIONS, REFERENCES
        return self

    def __exit__(self, *exc):
        off.PROJECTIONS, off.REFERENCES = self.saved
        return False


def both_off(receipt, texts):
    """The projected tree is alpha-next's engine at the parent source byte for byte (the declared review pin aside), and
    alpha-next's own freeze record proved exactly that commit."""
    record_path = Path(PIN['freeze'])
    faults = []
    differ = sorted(name for name, data in texts.items() if name not in off.DECLARED_META
                    and data != off.git('show', '%s:%s%s' % (PARENT_SOURCE, ENGINE, name), binary=True))
    if differ:
        faults.append('projected tree differs from the alpha-next engine bytes: %r' % (differ,))
    if off.sha(record_path.read_bytes()) != PIN['freeze_sha256']:
        faults.append('alpha-next freeze record changed')
    record = json.loads(record_path.read_text())
    if record.get('verdict') != 'PASS':
        faults.append('alpha-next freeze not PASS')
    if (record.get('source') or {}).get('head') != PARENT_SOURCE:
        faults.append('alpha-next freeze proved another commit')
    if record.get('candidate_manifest') != PIN['manifest']:
        faults.append('alpha-next freeze names another bundle')
    if receipt['verdict'] != 'PASS':
        faults.append('C1 off is not alpha-next, so both off cannot reach its freeze')
    return dict(projected_files=len(texts), projected_byte_identical=not differ, declared_meta=sorted(off.DECLARED_META),
                alpha_next_freeze=str(record_path), alpha_next_freeze_sha256=PIN['freeze_sha256'],
                alpha_next_freeze_verdict=record.get('verdict'), proved_commit=(record.get('source') or {}).get('head'),
                parent_source=PARENT_SOURCE, verdict='FAIL' if faults else 'PASS', faults=faults)


def prove(engine, work):
    """Both receipts over ENGINE; the projected tree is written to WORK (a new directory, kept)."""
    with _Pointed():
        receipt, texts = off.project(engine, PARENT_SOURCE)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    for name, data in texts.items():
        (work / name).write_bytes(data)
    link = both_off(receipt, texts)
    faults = ['c1-vs-alpha-next: ' + f for f in receipt['faults']] + ['both-off: ' + f for f in link['faults']]
    receipt = dict(receipt, schema='alpha-next-c1-default-off/1', parent_source=PARENT_SOURCE)
    return dict(schema='alpha-next-c1-default-off/1', verdict='FAIL' if faults else 'PASS', tip=off.git('rev-parse', 'HEAD').strip(),
                dirty=bool(off.git('status', '--porcelain', '--', ENGINE).strip()), work=str(work),
                method='stage_default_off_c1.project re-pointed at the alpha-next parent source with COMPOSED_PROJECTIONS',
                composed_swaps_sha256=hashlib.sha256(json.dumps(COMPOSED_SWAPS, sort_keys=True).encode()).hexdigest(),
                stage_overlap_sha256=receipt['new_modules'].get('stage_overlap.py'),
                c1_off_is_alpha_next=receipt, both_off_reaches_alpha_next_freeze=link, faults=faults, dev_parent=bool(PIN.get('dev')))


def source_level(work):
    receipt = prove(ROOT / ENGINE, work)
    assert receipt['verdict'] == 'PASS', receipt['faults']
    return receipt


def negative_controls():
    """stage_default_off_c1's ten mutated copies, proved by this module's prove (its own prove is swapped for the call)."""
    saved = off.prove
    off.prove = lambda engine, work, _rev=None: prove(engine, work)
    try:
        rows = off.negative_controls(PARENT_SOURCE)
    finally:
        off.prove = saved
    assert set(rows.values()) == {'REFUSED'}, rows
    return rows


def _packaged(source):
    """make_s5_overlap's rewriting of bare imports, as the package step does it."""
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_s5_overlap as maker
    return maker.package(source)


def show(path):
    return subprocess.check_output(['git', 'show', PARENT_SOURCE + ':' + path], cwd=ROOT, text=True)


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
        assert (parent / path).read_text() == _packaged(show(ENGINE + name)), \
            'parent overlay module differs from the packaged alpha-next source: ' + path
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), parent_commit=PARENT_SOURCE, changed=sorted(changed), added=sorted(NEW),
                generated_sources_byte_identical=len(generated), model_and_plugin_byte_identical=True,
                native_libraries_byte_identical=True, launcher_byte_identical=True,
                launcher_sha256=hashlib.sha256((candidate / 'run-flash-s5-test.sh').read_bytes()).hexdigest(),
                modules_equal_packaged_committed_source=sorted(runtime | NEW),
                parent_modules_equal_packaged_alpha_next_source=sorted(runtime))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--work', type=Path, required=True, help='new directory for the C1-projected engine tree (kept)')
    p.add_argument('--controls', action='store_true')
    a = p.parse_args()
    record = prove(ROOT / ENGINE, a.work)
    if a.controls:
        record['negative_controls'] = negative_controls()
    print(json.dumps(dict(verdict=record['verdict'], faults=record['faults'], references=record['c1_off_is_alpha_next']['references'],
                          stage_overlap_sha256=record['stage_overlap_sha256'],
                          negative_controls=record.get('negative_controls')), indent=1, sort_keys=True))
    raise SystemExit(0 if record['verdict'] == 'PASS' else 1)
