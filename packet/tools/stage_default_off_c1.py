#!/usr/bin/env python3
"""Default-off proof for FLASH_S5_STAGE_OVERLAP (C1), composed with C0's unchanged proof.

Two receipts over the engine directory, both required:

1. **C1 off is C0.** Against C0's delivered commit (``--c0``): every file not declared here is
   byte-identical; each declared file, after its declared projection (exact text, exact count),
   has the same AST as C0's file; the only new module is ``stage_overlap.py``; the new names
   appear outside it only on the declared lines.  So with the overlap switch unset, every
   statement on the engine's path is C0's.
2. **Both off is the frozen trunk.** The C1-projected tree (written to ``--work``, a new
   directory, and kept) is handed to C0's own tool, ``stage_default_off.prove``, unchanged,
   which proves it against c3f1f64 the way C0's delivery did.

Prints a JSON receipt; exit 0 only when both pass.
usage: stage_default_off_c1.py --work NEW_DIR [--c0 REV] [--out FILE]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import stage_default_off as c0_off  # noqa: E402

ENGINE = c0_off.ENGINE
C0_REV = 'cea3024'
C0_MODULES = c0_off.NEW_MODULES
NEW_MODULES = ('stage_overlap.py',)
DECLARED_META = c0_off.DECLARED_META
PROJECTIONS = {
    'engine_driver.py': [
        ("    # The exact controller type this driver accepts (FLASH_S5_STAGE_OVERLAP names its subclass).\n"
         "    CONTROLLER_CLASS = PrefillAdmissionController\n", ''),
        ("        if type(controller) is not self.CONTROLLER_CLASS:\n",
         "        if type(controller) is not PrefillAdmissionController:\n"),
    ],
    's5_startup.py': [
        ("    # FLASH_S5_STAGE_OVERLAP: unset or 0 constructs the frozen controller, unchanged.\n"
         "    controller_class = PrefillAdmissionController\n"
         "    overlap = False\n"
         "    if env.get('FLASH_S5_STAGE_OVERLAP', '0') != '0':\n"
         "        from stage_overlap import StagedAdmissionController, stage_overlap_enabled\n"
         "        overlap = stage_overlap_enabled(env)\n"
         "        if overlap:\n"
         "            controller_class = StagedAdmissionController\n"
         "    controller = controller_class(\n", "    controller = PrefillAdmissionController(\n"),
        ("    if overlap:\n"
         "        from stage_overlap import OverlapSegmentDriver\n"
         "        driver_class = OverlapSegmentDriver\n", ''),
    ],
    'decode_probe.py': [
        ("'stage_wire', 'stage_progress', 'stage_overlap')\n", "'stage_wire', 'stage_progress')\n"),
    ],
    'make_s5_overlap.py': [
        ("              'stage_wire', 'stage_progress',\n"
         "              # C1: two chunks in flight (FLASH_S5_STAGE_OVERLAP).\n"
         "              'stage_overlap')\n", "              'stage_wire', 'stage_progress')\n"),
        ("LAUNCHER_LIST_EXEMPT = ('native_auto_arm', 'stage_wire', 'stage_progress', 'stage_overlap')\n",
         "LAUNCHER_LIST_EXEMPT = ('native_auto_arm', 'stage_wire', 'stage_progress')\n"),
    ],
    'native_overlap_control.py': [
        ('def _stage_overlap():\n'
         '    """C1\'s field is exposed only where the engine installed it beside C0 (FLASH_S5_STAGE_OVERLAP)."""\n'
         "    return _stage_local() and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1'\n\n\n", ''),
        ('    async def native_stage_overlap(self, local, overlap):\n'
         '        """C1 per-arm switch (installed only): C0\'s command with ``overlap``; the engine re-checks idle FULL."""\n'
         '        if not _stage_overlap():\n'
         "            raise NativeControlError('native-control-stage-overlap-not-installed')\n"
         '        if type(local) is not bool:\n'
         "            raise NativeControlError('native-control-stage-local-invalid')\n"
         '        if type(overlap) is not bool:\n'
         "            raise NativeControlError('native-control-stage-overlap-invalid')\n"
         '        if self.engine_epoch is None:\n'
         '            await self.status()\n'
         "        reply = await self._call({'action': 'native_stage_local', 'local': local, 'overlap': overlap})\n"
         "        if reply.get('local') is not local or reply.get('overlap') is not overlap:\n"
         "            raise NativeControlError('native-control-stage-overlap-not-set')\n"
         '        return reply\n\n', ''),
    ],
    'native_overlap_session.py': [
        ("        if 'overlap' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1':\n"
         "            # C1 only; with the switch off the command is C0's, field and all, as before.\n"
         "            return await self.control.native_stage_overlap(command.get('local'), command['overlap'])\n",
         ''),
    ],
}
NAMES = re.compile(r'stage_overlap|StagedAdmissionController|OverlapSegmentDriver|FLASH_S5_STAGE_OVERLAP'
                   r'|CONTROLLER_CLASS|native_stage_overlap|stage-overlap|controller_class')
REFERENCES = {'decode_probe.py': 1, 'engine_driver.py': 3, 'make_s5_overlap.py': 3,
              'native_overlap_control.py': 8, 'native_overlap_session.py': 2, 's5_startup.py': 9}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, binary=False):
    out = subprocess.check_output(['git', *args], cwd=ROOT)
    return out if binary else out.decode()


def project(engine, c0_rev):
    """C1 off against C0: the receipt and the projected file texts."""
    faults, texts = [], {}
    engine = Path(engine)
    names = sorted(p.name for p in engine.iterdir() if p.is_file())
    base_names = sorted(line[len(ENGINE):] for line in
                        git('ls-tree', '-r', '--name-only', c0_rev, ENGINE).splitlines()
                        if '/' not in line[len(ENGINE):])
    added = sorted(set(names) - set(base_names))
    removed = sorted(set(base_names) - set(names))
    if added != sorted(NEW_MODULES):
        faults.append('new engine files differ: %r' % (added,))
    if removed:
        faults.append('engine files removed: %r' % (removed,))
    identical, projected = [], {}
    for name in base_names:
        if name in removed:
            continue
        tip = (engine / name).read_bytes()
        base = git('show', '%s:%s%s' % (c0_rev, ENGINE, name), binary=True)
        texts[name] = tip
        if tip == base:
            identical.append(name)
            if name in PROJECTIONS:
                faults.append('declared projection on an unchanged file: ' + name)
            continue
        if name in DECLARED_META:
            projected[name] = dict(declared=DECLARED_META[name], tip=sha(tip), c0=sha(base))
            continue
        rows = PROJECTIONS.get(name)
        if rows is None:
            faults.append('undeclared change: ' + name)
            continue
        text = tip.decode()
        for old, new in rows:
            if text.count(old) != 1:
                faults.append('%s: projection row not found exactly once: %r' % (name, old[:60]))
            text = text.replace(old, new)
        same = ast.dump(ast.parse(text)) == ast.dump(ast.parse(base.decode()))
        if not same:
            faults.append(name + ': AST differs from C0 after projection')
        texts[name] = text.encode()
        projected[name] = dict(tip=sha(tip), c0=sha(base), projected=sha(texts[name]), rows=len(rows),
                               ast_identical=same)
    references = {}
    for path in sorted(engine.glob('*.py')):
        if path.name in NEW_MODULES:
            continue
        lines = [line for line in path.read_text().splitlines() if NAMES.search(line)]
        if lines:
            references[path.name] = len(lines)
        # A row may start mid-line (decode_probe's tuple tail): its lines are fragments of file lines.
        declared = [part for old, _ in PROJECTIONS.get(path.name, ()) for part in old.splitlines() if part]
        for line in lines:
            if not any(NAMES.search(part) and part in line for part in declared):
                faults.append('%s: a new name outside the declared lines: %r' % (path.name, line[:80]))
    if references != REFERENCES:
        faults.append('references to the new surface differ: %r' % (references,))
    # The overlap controller and driver are constructed only on the switch's line in s5_startup.
    for path in sorted(engine.glob('*.py')):
        if path.name in NEW_MODULES or path.name == 's5_startup.py':
            continue
        if re.search(r'\b(StagedAdmissionController|OverlapSegmentDriver)\b', path.read_text()):
            faults.append('the overlap classes are named in ' + path.name)
    receipt = dict(schema='stage-default-off-c1/1', verdict='FAIL' if faults else 'PASS',
                   c0=git('rev-parse', c0_rev).strip(), identical_files=len(identical),
                   new_modules={name: sha((engine / name).read_bytes()) for name in added},
                   projected=projected, references=references, faults=faults)
    return receipt, texts


def prove(engine, work, c0_rev=C0_REV):
    receipt, texts = project(engine, c0_rev)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    for name, data in texts.items():
        (work / name).write_bytes(data)
    frozen = c0_off.prove(work)
    faults = ['c1-vs-c0: ' + f for f in receipt['faults']] + ['both-off-vs-frozen: ' + f for f in frozen['faults']]
    return dict(schema='stage-default-off-c1/1', verdict='FAIL' if faults else 'PASS',
                tip=git('rev-parse', 'HEAD').strip(),
                dirty=bool(git('status', '--porcelain', '--', ENGINE).strip()),
                work=str(work), c1_off_is_c0=receipt, both_off_is_frozen=frozen, faults=faults)


def negative_controls(c0_rev=C0_REV):
    """Mutated copies of the engine directory; each must be refused (C0's freeze does the same)."""
    rows = {}

    def control(name, mutate):
        with tempfile.TemporaryDirectory(prefix='stage-default-off-c1-') as held:
            engine = Path(held) / 'engine'
            shutil.copytree(ROOT / ENGINE, engine)
            mutate(engine)
            receipt = prove(engine, Path(held) / 'work', c0_rev)
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
    control('undeclared-byte-in-a-c0-file', append('engine_driver.py', '\n# undeclared\n'))
    control('undeclared-statement-in-a-declared-file', append('s5_startup.py', '\nUNDECLARED = 1\n'))
    control('undeclared-module', lambda engine: (engine / 'stage_undeclared.py').write_text('X = 1\n'))
    control('overlap-module-missing', lambda engine: (engine / 'stage_overlap.py').unlink())
    control('default-flipped-on', swap('s5_startup.py', "env.get('FLASH_S5_STAGE_OVERLAP', '0') != '0'",
                                       "env.get('FLASH_S5_STAGE_OVERLAP', '1') != '0'"))
    control('controller-class-left-staged', swap('s5_startup.py', '    controller_class = PrefillAdmissionController\n',
                                                 '    controller_class = StagedAdmissionController\n'))
    control('driver-accepts-the-subclass-by-default', swap(
        'engine_driver.py', '    CONTROLLER_CLASS = PrefillAdmissionController\n',
        '    CONTROLLER_CLASS = object\n'))
    control('session-field-without-the-switch', swap(
        'native_overlap_session.py',
        "        if 'overlap' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1':\n",
        "        if 'overlap' in command:\n"))
    control('overlap-class-named-in-another-file', append('lane_dispatch.py', '\n# OverlapSegmentDriver\n'))
    control('c0-default-flipped-on', swap('s5_startup.py', "env.get('FLASH_S5_STAGE_LOCAL', '0') != '0'",
                                          "env.get('FLASH_S5_STAGE_LOCAL', '1') != '0'"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True, help='new directory for the C1-projected engine tree (kept)')
    parser.add_argument('--c0', default=C0_REV)
    parser.add_argument('--out')
    parser.add_argument('--controls', action='store_true', help='also run the negative controls')
    options = parser.parse_args()
    receipt = prove(ROOT / ENGINE, options.work, options.c0)
    if options.controls:
        receipt['negative_controls'] = rows = negative_controls(options.c0)
        if set(rows.values()) != {'REFUSED'}:
            receipt['faults'].append('a negative control was accepted: %r' % (rows,))
            receipt['verdict'] = 'FAIL'
    text = json.dumps(receipt, indent=1, sort_keys=True)
    print(text)
    if options.out:
        Path(options.out).write_text(text + '\n')
    return 0 if receipt['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
