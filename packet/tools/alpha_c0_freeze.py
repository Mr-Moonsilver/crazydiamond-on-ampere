#!/usr/bin/env python3
"""C0's freeze proof against the composed trunk c3f1f64: source, negative controls and generated overlay.

source_level is tools/stage_default_off.py over the committed engine sources. negative_controls runs the same proof over
copies with one undeclared change each and every one must be refused. overlay_level proves the built overlay: each changed
module and each new module is make_s5_overlap.package(committed source) byte for byte; the parent's modules are
package(c3f1f64 source); the launcher, the generated vllm sources, plugins and native libraries are byte-identical to the
parent's; nothing else moves but the declared metadata. With the switch on, C0 is not claimed equal here: the identity tool,
the suites and the box compare that. CPU only, no box action.
"""
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
import stage_default_off as off  # noqa: E402

PARENT = off.frozen_commit()
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
    return subprocess.check_output(['git', 'show', off.FROZEN + ':' + path], cwd=ROOT, text=True)


def source_level():
    receipt = off.prove(ROOT / off.ENGINE)
    assert receipt['verdict'] == 'PASS', receipt['faults']
    return receipt


def negative_controls():
    rows = {}

    def control(name, mutate):
        with tempfile.TemporaryDirectory(prefix='alpha-c0-freeze-') as held:
            engine = Path(held) / 'engine'
            shutil.copytree(ROOT / off.ENGINE, engine)
            mutate(engine)
            receipt = off.prove(engine)
        rows[name] = 'REFUSED' if receipt['verdict'] == 'FAIL' and receipt['faults'] else 'ACCEPTED'

    def append(name, text):
        def mutate(engine):
            with (engine / name).open('a') as f:
                f.write(text)
        return mutate

    def swap(name, old, new):
        def mutate(engine):
            data = (engine / name).read_text()
            assert data.count(old) == 1, old
            (engine / name).write_text(data.replace(old, new))
        return mutate
    control('undeclared-byte-in-a-frozen-file', append('engine_driver.py', '\n# undeclared\n'))
    control('undeclared-statement-in-a-declared-file', append('s5_startup.py', '\nUNDECLARED = 1\n'))
    control('undeclared-module', lambda engine: (engine / 'stage_undeclared.py').write_text('X = 1\n'))
    control('default-flipped-on', swap('s5_startup.py', "env.get('FLASH_S5_STAGE_LOCAL', '0') != '0'",
                                       "env.get('FLASH_S5_STAGE_LOCAL', '1') != '0'"))
    control('driver-class-left-staged', swap('s5_startup.py', '    driver_class = NativeSegmentDriver\n',
                                             '    driver_class = None\n'))
    control('control-exposed-without-the-switch', swap(
        'probe_control.py', "    if auto_actions and env.get('FLASH_S5_STAGE_LOCAL', '0') == '1':\n",
        "    if auto_actions:\n"))
    control('wire-record-built-in-a-new-module', lambda engine: (engine / 'stage_undeclared.py').write_text(
        'from stage_wire import StageScoped\nX = StageScoped(None, (0,))\n'))
    control('new-surface-named-in-another-file', append('lane_dispatch.py', '\n# FLASH_S5_STAGE_LOCAL\n'))
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
            'parent overlay module differs from the packaged c3f1f64 source: ' + path
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), changed=sorted(changed), added=sorted(NEW), generated_sources_byte_identical=len(generated),
                model_and_plugin_byte_identical=True, native_libraries_byte_identical=True,
                launcher_byte_identical=True, launcher_sha256=hashlib.sha256((candidate / 'run-flash-s5-test.sh').read_bytes()).hexdigest(),
                modules_equal_packaged_committed_source=sorted(runtime | NEW),
                parent_modules_equal_packaged_c3f1f64_source=sorted(runtime))


if __name__ == '__main__':
    print(json.dumps(dict(source=source_level()['verdict'], negative_controls=negative_controls()), indent=1, sort_keys=True))
