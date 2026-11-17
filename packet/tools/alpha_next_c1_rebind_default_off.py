#!/usr/bin/env python3
"""alpha-next-c1 default-off proof for the C1 rebind and its per-arm selection (lead 2 Nov 21:0x CET, point 4).

Supersedes tools/c1_rebind_default_off.py (unchanged, the rebind builder's, against C1 cf120738) for this composition, against
the alpha-next-c1 tip before the rebind (``--base``, default 844dfe8d: alpha-next + C1 + the composition fixes):

1. ``stage_overlap.py``: the per-arm projections below (PER_ARM, exact text, exact count, in order) map this file back to the
   rebind's own text; then the rebind's PROJECTIONS (imported unchanged from tools/c1_rebind_default_off.py) apply; the result
   has the same AST as the base's file.
2. ``s5_startup.py``, ``native_overlap_session.py``, ``native_overlap_control.py``: their projections (OTHERS) give the same AST
   as the base's file.
3. Every other file under patches/s5_engine is byte-identical to the base, except ``native_candidate_review.json`` (the review
   record each chain's prepare-review step rewrites; it is not engine code and is pinned by the build itself).
4. Flag unset, field refused: REFUSAL_TESTS (harness/test_alpha_next_c1_defer_control.py) run in a child with both rebind switches
   removed from the environment and pass: the engine, the session router and the client each refuse ``defer_bind`` where the
   flag is not installed, and startup refuses either switch without C1.

Why each projection holds with FLASH_S5_STAGE_OVERLAP_DEFER_BIND and FLASH_S5_STAGE_OVERLAP_OBSERVE unset (``defer_bind`` and
``defer_bind_installed`` False, observer None): a removed statement is an assignment nothing on that path reads, a block that runs
only when ``defer_bind_installed`` is True, or a refusal (a raise) that fires only when a ``defer_bind`` field is sent where the
flag is not installed or a rebind switch is set without C1. The refusals are the one intended behaviour change with both unset:
the field is refused where before the rebind it did not exist. Prints a JSON receipt (``--out`` written once); exit 0 only when
all four hold.  usage: alpha_next_c1_rebind_default_off.py [--base REV] [--out FILE]
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import c1_rebind_default_off as rebind  # noqa: E402  (its PROJECTIONS and project(), unchanged)

ENGINE = 'patches/s5_engine'
TARGET = 'stage_overlap.py'
EXEMPT = ('native_candidate_review.json',)
SWITCHES = ('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE')
REFUSAL_TESTS = ('test_alpha_next_c1_defer_control.Install.test_refused_where_not_installed',
                 'test_alpha_next_c1_defer_control.RouterNotInstalled',
                 'test_alpha_next_c1_defer_control.Startup')

PER_ARM = [
    # __init__: the installed flag, read only by the per-arm field and the status row.
    ("        # alpha-next-c1: the startup flag installs the deferred bind (selected, as the rebind has it); the typed per-arm field\n"
     "        # ``defer_bind`` of native_stage_local selects it per arm while paused idle FULL, as fused and layer groups are.\n"
     "        self.defer_bind_installed = self.defer_bind\n", ''),
    # _stage_control: the field's checks only raise (a field sent where not installed is refused); without the field the
    # filter is the rebind's; the selection block runs only when installed.
    ("        # alpha-next-c1: ``defer_bind`` (installed only) is checked before anything is applied; it needs overlap in the\n"
     "        # same command, and C0's paused idle FULL rule (super()) refuses it mid-chunk like the other fields.\n"
     "        defer = command.get('defer_bind', None)\n"
     "        if 'defer_bind' in command:\n"
     "            if type(defer) is not bool:\n"
     "                raise DriverError('native-stage-overlap-defer-bind-invalid')\n"
     "            if not self.defer_bind_installed:\n"
     "                raise DriverError('native-stage-overlap-defer-bind-not-installed')\n"
     "            if defer and overlap is not True:\n"
     "                raise DriverError('native-stage-overlap-defer-bind-needs-overlap')\n"
     "        reply = super()._stage_control({k: v for k, v in command.items() if k not in ('overlap', 'defer_bind')})\n",
     "        reply = super()._stage_control({k: v for k, v in command.items() if k != 'overlap'})\n"),
    ("        if self.defer_bind_installed:\n"
     "            previous_defer = self.defer_bind\n"
     "            if not overlap:\n"
     "                defer = False\n"
     "            elif defer is None:\n"
     "                defer = previous_defer\n"
     "            self.defer_bind = defer\n"
     "            reply.update(defer_bind=defer, previous_defer_bind=previous_defer)\n", ''),
    # stage_status: installed == defer_bind at construction; both False when unset. Mapped to the rebind's row, which the
    # rebind's own projection then removes.
    ("        if self.defer_bind_installed:\n"
     "            result['overlap']['defer_bind'] = dict(installed=True, selected=self.defer_bind,\n"
     "                                                   counters=dict(self.defer_counters))\n",
     "        if self.defer_bind:\n"
     "            result['overlap']['defer_bind'] = dict(counters=dict(self.defer_counters))\n"),
]

OTHERS = {
    's5_startup.py': [
        # Refusals only: with both switches unset ('0') the loop raises nothing.
        ("    # alpha-next-c1: the C1 rebind switches are refused unless C1 is on (fail closed, never silently ignored).\n"
         "    for name in ('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE'):\n"
         "        value = env.get(name, '0')\n"
         "        if value not in ('0', '1'):\n"
         "            raise StartupError('invalid boolean setting: ' + name)\n"
         "        if value == '1' and env.get('FLASH_S5_STAGE_OVERLAP', '0') != '1':\n"
         "            raise StartupError(name + ' needs FLASH_S5_STAGE_OVERLAP=1')\n", ''),
    ],
    'native_overlap_session.py': [
        # A refusal where the flag is not installed; the pass-through runs only where it is installed.
        ("        if 'defer_bind' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', '0') != '1':\n"
         "            # alpha-next-c1: a requested deferred bind is never dropped silently where it is not installed.\n"
         "            raise NativeOverlapSessionError('native-stage-overlap-defer-bind-not-installed')\n", ''),
        ("            if 'defer_bind' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', '0') == '1':\n"
         "                # alpha-next-c1: the deferred bind rides on C1's command only where it is installed.\n"
         "                return await self.control.native_stage_overlap(command.get('local'), command['overlap'],\n"
         "                                                               command['defer_bind'])\n", ''),
    ],
    'native_overlap_control.py': [
        # A helper only the defer branch calls.
        ("def _stage_overlap_defer():\n"
         "    \"\"\"alpha-next-c1: the deferred-bind field is exposed only where the engine installed it "
         "(FLASH_S5_STAGE_OVERLAP_DEFER_BIND).\"\"\"\n"
         "    return _stage_overlap() and os.environ.get('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', '0') == '1'\n\n\n", ''),
        # defer_bind is None on every call where the flag is not installed (the session passes it only where installed): the
        # command is the same dict, the defer branches do not run.
        ("    async def native_stage_overlap(self, local, overlap, defer_bind=None):\n"
         "        \"\"\"C1 per-arm switch (installed only): C0's command with ``overlap``; the engine re-checks idle FULL.\n"
         "        alpha-next-c1: ``defer_bind`` (installed only) rides on the same command; the reply must acknowledge it.\"\"\"\n",
         "    async def native_stage_overlap(self, local, overlap):\n"
         "        \"\"\"C1 per-arm switch (installed only): C0's command with ``overlap``; the engine re-checks idle FULL.\"\"\"\n"),
        ("        command = {'action': 'native_stage_local', 'local': local, 'overlap': overlap}\n"
         "        if defer_bind is not None:\n"
         "            if not _stage_overlap_defer():\n"
         "                raise NativeControlError('native-control-stage-overlap-defer-bind-not-installed')\n"
         "            if type(defer_bind) is not bool:\n"
         "                raise NativeControlError('native-control-stage-overlap-defer-bind-invalid')\n"
         "            command['defer_bind'] = defer_bind\n"
         "        if self.engine_epoch is None:\n"
         "            await self.status()\n"
         "        reply = await self._call(command)\n"
         "        if reply.get('local') is not local or reply.get('overlap') is not overlap:\n"
         "            raise NativeControlError('native-control-stage-overlap-not-set')\n"
         "        if defer_bind is not None and reply.get('defer_bind') is not defer_bind:\n"
         "            raise NativeControlError('native-control-stage-overlap-defer-bind-not-set')\n",
         "        if self.engine_epoch is None:\n"
         "            await self.status()\n"
         "        reply = await self._call({'action': 'native_stage_local', 'local': local, 'overlap': overlap})\n"
         "        if reply.get('local') is not local or reply.get('overlap') is not overlap:\n"
         "            raise NativeControlError('native-control-stage-overlap-not-set')\n"),
    ],
}


def project(text, pairs):
    applied = []
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit('projection does not match exactly once: %r' % old[:100])
        text = text.replace(old, new)
        applied.append(dict(kind='text', removed_lines=old.count('\n') - new.count('\n')))
    return text, applied


def git_show(rev, path):
    return subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (rev, path)], check=True, capture_output=True).stdout


def same_ast(a, b):
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def refusals():
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    run = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', *REFUSAL_TESTS], cwd=str(ROOT / 'harness'), env=env,
                         capture_output=True, text=True, timeout=600)
    ran = [line for line in run.stderr.splitlines() if line.endswith(' ... ok')]
    return dict(tests=list(REFUSAL_TESTS), rc=run.returncode, ok_lines=len(ran), switches_unset=list(SWITCHES),
                tail=run.stderr.strip().splitlines()[-1:] if run.stderr.strip() else [])


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--base', default='844dfe8d')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    base = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', args.base], check=True, capture_output=True,
                          text=True).stdout.strip()
    listed = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', base, ENGINE + '/'],
                            check=True, capture_output=True, text=True).stdout.split()
    here = sorted(str(p.relative_to(ROOT)) for p in (ROOT / ENGINE).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    projected_files = {ENGINE + '/' + TARGET: None, **{ENGINE + '/' + name: pairs for name, pairs in OTHERS.items()}}
    differing, exempt, files = [], [], {}
    for path in sorted(set(listed) | set(here)):
        if path in projected_files:
            continue
        mine = (ROOT / path).read_bytes() if (ROOT / path).exists() else None
        theirs = git_show(base, path) if path in listed else None
        if mine != theirs:
            (exempt if Path(path).name in EXEMPT else differing).append(path)
    current = (ROOT / ENGINE / TARGET).read_text()
    step1, applied1 = project(current, PER_ARM)
    step2, applied2 = rebind.project(step1)
    files[TARGET] = dict(sha256=hashlib.sha256(current.encode()).hexdigest(), per_arm=applied1, rebind=applied2,
                         projected_ast_equals_base=same_ast(step2, git_show(base, ENGINE + '/' + TARGET).decode()))
    for name, pairs in OTHERS.items():
        text = (ROOT / ENGINE / name).read_text()
        projected, applied = project(text, pairs)
        files[name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(), projections=applied,
                           projected_ast_equals_base=same_ast(projected, git_show(base, ENGINE + '/' + name).decode()))
    refused = refusals()
    ok = (not differing and all(row['projected_ast_equals_base'] for row in files.values()) and refused['rc'] == 0
          and refused['ok_lines'] > 0)
    receipt = dict(schema='alpha-next-c1-rebind-default-off/1', base=base,
                   rebind_tool_sha256=hashlib.sha256((ROOT / 'tools/c1_rebind_default_off.py').read_bytes()).hexdigest(),
                   engine_files_compared=len(set(listed) | set(here)), other_engine_files_differing=differing,
                   exempt_differing=exempt, exempt_rule='review record rewritten by each chain (not engine code)',
                   files=files, flag_unset_field_refused=refused, verdict='PASS' if ok else 'FAIL', passed=ok)
    text = json.dumps(receipt, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as out:
            out.write(text + '\n')
    print(text)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
