#!/usr/bin/env python3
"""Default-off proof for FLASH_S5_STAGE_LOCAL (and later FLASH_S5_STAGE_OVERLAP).

Against the frozen composed build, on the engine directory:

* every file not declared here is byte-identical to the frozen build;
* each declared file, after its declared projection (exact text, exact count), has the
  same AST as the frozen file: the switch-off path is the frozen code, statement by
  statement, plus these declared lines only;
* every reference to the new names outside the new modules is one of the declared lines,
  so the new record type and the staged driver are unreachable with the switch off.

Prints a JSON receipt; exit 0 only on PASS.  usage: stage_default_off.py [--out FILE]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = 'c3f1f64'
ENGINE = 'patches/s5_engine/'
NEW_MODULES = ('stage_wire.py', 'stage_progress.py')
# Generated and review metadata, proven by their own tools.
DECLARED_META = {'native_candidate_review.json': 'review pins: source and harness shas, later the CPU receipts; no runtime behaviour'}
PROJECTIONS = {
    'executor_hooks.py': [
        ('from stage_wire import STAGE_SKIP, StageScoped, stage_rpc\n', ''),
        ("    if type(message) is StageScoped:\n"
         "        # FLASH_S5_STAGE_LOCAL only: the engine never sends this record otherwise.\n"
         "        if _wire_method(message.message) != name:\n"
         "            raise E3ContractError('S5 RPC method and wire record disagree')\n"
         "        return stage_rpc(_wire_bridge(worker), name, message,\n"
         "                         lambda inner: worker_rpc(worker, name, inner))\n", ''),
        ("    if output is STAGE_SKIP:\n"
         "        # A stage-scoped command for other ranks: no reply is expected or written.\n"
         "        state = None if rpc is None else _worker_progress(proc.rank)\n"
         "        if state is not None:\n"
         "            state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], None, int(rpc[4]), 0)\n"
         "        return\n", ''),
    ],
    's5_startup.py': [
        ("    # FLASH_S5_STAGE_LOCAL: unset or 0 constructs the frozen driver, unchanged.\n"
         "    driver_class = NativeSegmentDriver\n"
         "    if env.get('FLASH_S5_STAGE_LOCAL', '0') != '0':\n"
         "        from stage_progress import StagedSegmentDriver, stage_local_enabled\n"
         "        if stage_local_enabled(env):\n"
         "            driver_class = StagedSegmentDriver\n"
         "    driver = driver_class(\n", "    driver = NativeSegmentDriver(\n"),
    ],
    'decode_probe.py': [
        ("'native_readiness', 'stage_wire', 'stage_progress')\n", "'native_readiness')\n"),
    ],
    'native_overlap_session.py': [
        ("        if action == 'native_stage_local' and not (\n"
         "                os.environ.get('FLASH_S5_STAGE_LOCAL', '0') == '1'\n"
         "                and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'):\n"
         "            handler = None\n", ''),
        ("    async def _action_native_stage_local(self, command):\n"
         "        if self.measure_start_ns is not None:\n"
         "            raise NativeOverlapSessionError('native-stage-local-with-open-measurement')\n"
         "        return await self.control.native_stage_local(command.get('local'))\n\n", ''),
    ],
    'native_overlap_control.py': [
        ('def _stage_local():\n'
         '    """The C0 switch is exposed only where the engine installed it."""\n'
         "    return (os.environ.get('FLASH_S5_STAGE_LOCAL', '0') == '1'\n"
         "            and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')\n\n\n", ''),
        ("                    or action == 'native_progress' and _progress_burst()\n"
         "                    or action == 'native_stage_local' and _stage_local()):\n",
         "                    or action == 'native_progress' and _progress_burst()):\n"),
        ('    async def native_stage_local(self, local):\n'
         '        """C0 per-arm switch (installed only); the engine re-checks idle FULL."""\n'
         '        if type(local) is not bool:\n'
         "            raise NativeControlError('native-control-stage-local-invalid')\n"
         '        if self.engine_epoch is None:\n'
         '            await self.status()\n'
         "        reply = await self._call({'action': 'native_stage_local', 'local': local})\n"
         "        if reply.get('local') is not local:\n"
         "            raise NativeControlError('native-control-stage-local-not-set')\n"
         '        return reply\n\n', ''),
    ],
    'probe_control.py': [
        ("    # C0: the per-arm native_stage_local switch, only where it is installed.\n"
         "    if auto_actions and env.get('FLASH_S5_STAGE_LOCAL', '0') == '1':\n"
         "        auto_actions += ('native_stage_local',)\n", ''),
    ],
    'make_s5_overlap.py': [
        ("LAUNCHER_LIST_EXEMPT = ('native_auto_arm', 'stage_wire', 'stage_progress')\n", "LAUNCHER_LIST_EXEMPT = ('native_auto_arm',)\n"),
        ("'native_readiness',\n"
         "              # C0: stage-local prefill progress (FLASH_S5_STAGE_LOCAL).\n"
         "              'stage_wire', 'stage_progress')\n", "'native_readiness')\n"),
    ],
}
# File -> exact count of lines naming the new surface (outside the new modules).
NAMES = re.compile(r'StageScoped|STAGE_SKIP|stage_rpc|stage_progress|stage_wire|StagedSegmentDriver'
                   r'|stage_local_enabled|FLASH_S5_STAGE_LOCAL|native_stage_local|_stage_local'
                   r'|native-stage-local|stage-local-')
REFERENCES = {'decode_probe.py': 1, 'executor_hooks.py': 5, 'make_s5_overlap.py': 3,
              'native_overlap_control.py': 7, 'native_overlap_session.py': 5, 'probe_control.py': 3,
              's5_startup.py': 5}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, binary=False):
    out = subprocess.check_output(['git', *args], cwd=ROOT)
    return out if binary else out.decode()


def frozen_commit():
    return git('rev-parse', FROZEN).strip()


def prove(engine):
    """The receipt for the engine directory ``engine``; the negative controls pass mutated copies."""
    faults = []
    engine = Path(engine)
    names = sorted(p.name for p in engine.iterdir() if p.is_file())
    frozen_names = sorted(line[len(ENGINE):] for line in
                          git('ls-tree', '-r', '--name-only', FROZEN, ENGINE).splitlines()
                          if '/' not in line[len(ENGINE):])
    added = sorted(set(names) - set(frozen_names))
    removed = sorted(set(frozen_names) - set(names))
    if added != sorted(NEW_MODULES):
        faults.append('new engine files differ: %r' % (added,))
    if removed:
        faults.append('engine files removed: %r' % (removed,))
    identical, projected = [], {}
    for name in frozen_names:
        if name in removed:
            continue
        tip = (engine / name).read_bytes()
        parent = git('show', '%s:%s%s' % (FROZEN, ENGINE, name), binary=True)
        if tip == parent:
            identical.append(name)
            if name in PROJECTIONS:
                faults.append('declared projection on an unchanged file: ' + name)
            continue
        if name in DECLARED_META:
            projected[name] = dict(declared=DECLARED_META[name], tip=sha(tip), frozen=sha(parent))
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
        same = ast.dump(ast.parse(text)) == ast.dump(ast.parse(parent.decode()))
        if not same:
            faults.append(name + ': AST differs from the frozen file after projection')
        projected[name] = dict(tip=sha(tip), frozen=sha(parent), rows=len(rows), ast_identical=same)
    # Unreachable with the switch off: the new surface appears only on the declared lines.
    references = {}
    for path in sorted(engine.glob('*.py')):
        if path.name in NEW_MODULES:
            continue
        count = sum(bool(NAMES.search(line)) for line in path.read_text().splitlines())
        if count:
            references[path.name] = count
    if references != REFERENCES:
        faults.append('references to the new surface differ: %r' % (references,))
    # The only constructor of the wire record and the only importer of the driver.
    driver_text = (engine / 'stage_progress.py').read_text()
    constructors = [path.name for path in engine.glob('*.py')
                    if re.search(r'\bStageScoped\(', path.read_text())
                    and path.name not in ('stage_wire.py',)]
    if constructors != ['stage_progress.py'] or 'StageScoped(' not in driver_text:
        faults.append('the wire record has another constructor: %r' % (constructors,))
    return dict(schema='stage-default-off/1', verdict='FAIL' if faults else 'PASS',
                   frozen=frozen_commit(), tip=git('rev-parse', 'HEAD').strip(),
                   dirty=bool(git('status', '--porcelain', '--', ENGINE).strip()),
                   identical_files=len(identical), new_modules={
                       name: sha((engine / name).read_bytes()) for name in added},
                   projected=projected, references=references, faults=faults)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    options = parser.parse_args()
    receipt = prove(ROOT / ENGINE)
    faults = receipt['faults']
    text = json.dumps(receipt, indent=1, sort_keys=True)
    print(text)
    if options.out:
        Path(options.out).write_text(text + '\n')
    return 0 if not faults else 1


if __name__ == '__main__':
    sys.exit(main())
