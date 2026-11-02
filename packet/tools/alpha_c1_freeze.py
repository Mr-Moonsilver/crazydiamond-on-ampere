#!/usr/bin/env python3
"""C1's freeze proof against C0 l's delivered engine (and, through C0's own tool, the frozen trunk c3f1f64).

source_level is tools/stage_default_off_c1.py over the committed engine sources against C0 l's build commit: C1 off is C0
(every undeclared file byte-identical, each declared projection AST-identical, one new module), and the C1-projected tree
handed to C0's unchanged stage_default_off.prove is c3f1f64. negative_controls runs that proof over copies with one undeclared
change each and every one must be refused. overlay_level proves the built overlay against C0 l's overlay: each changed module
and the new module is make_s5_overlap.package(committed source) byte for byte; the parent's modules are package(C0 source);
the launcher, the generated vllm sources, plugins and native libraries are byte-identical to the parent's; nothing else
moves but the declared metadata. With the switch on, C1 is not claimed equal here: the probe, the suites and the box compare
that. CPU only, no box action.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_native30_held import verify  # noqa: E402
import stage_default_off_c1 as off  # noqa: E402

# C0 l's delivered build: its review commit, the bundle's provenance source (bundle a1526e066c6a).
C0_COMMIT = 'cf0e3016825e4a96d8a09166fdd982cd7429f976'
NEW = {'e3/' + name for name in off.NEW_MODULES}
CHANGED = {'e3/' + name for name in off.PROJECTIONS}
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}


def _packaged(source):
    """make_s5_overlap's rewriting of bare imports, as the package step does it."""
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_s5_overlap as maker
    return maker.package(source)


def show(path):
    return subprocess.check_output(['git', 'show', C0_COMMIT + ':' + path], cwd=ROOT, text=True)


def source_level(work):
    receipt = off.prove(ROOT / off.ENGINE, work, C0_COMMIT)
    assert receipt['verdict'] == 'PASS', receipt['faults']
    return receipt


def negative_controls():
    rows = off.negative_controls(C0_COMMIT)
    assert set(rows.values()) == {'REFUSED'}, rows
    return rows


def overlay_level(parent, candidate):
    before, after = verify(parent), verify(candidate)
    assert set(after) - set(before) == NEW, ('overlay files added', sorted(set(after) - set(before)))
    assert not set(before) - set(after), 'overlay file removed'
    changed = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'}
    runtime = {n for n in CHANGED if n in after}
    # An engine source's projection may leave its packaged form unchanged only if it is not packaged at all.
    assert changed <= runtime | META, sorted(changed - (runtime | META))
    for path in sorted(runtime | NEW):
        name = path.split('/', 1)[1]
        assert (candidate / path).read_text() == _packaged((ROOT / off.ENGINE / name).read_text()), \
            'overlay module differs from the packaged committed source: ' + path
    for path in sorted(runtime):
        name = path.split('/', 1)[1]
        assert (parent / path).read_text() == _packaged(show(off.ENGINE + name)), \
            'parent overlay module differs from the packaged C0 l source: ' + path
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), parent_commit=C0_COMMIT, changed=sorted(changed), added=sorted(NEW),
                generated_sources_byte_identical=len(generated), model_and_plugin_byte_identical=True,
                native_libraries_byte_identical=True, launcher_byte_identical=True,
                launcher_sha256=hashlib.sha256((candidate / 'run-flash-s5-test.sh').read_bytes()).hexdigest(),
                modules_equal_packaged_committed_source=sorted(runtime | NEW),
                parent_modules_equal_packaged_c0_source=sorted(runtime))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--work', type=Path, required=True, help='new directory for the C1-projected engine tree (kept)')
    a = p.parse_args()
    print(json.dumps(dict(source=source_level(a.work)['verdict'], negative_controls=negative_controls()), indent=1,
                     sort_keys=True))
