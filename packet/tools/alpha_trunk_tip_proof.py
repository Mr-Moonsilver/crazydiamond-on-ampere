#!/usr/bin/env python3
"""Alpha trunk freeze proof: the composition and default-off identity, carried from the merge dd6d977 to the tip.

Three parts, one record (CPU only, no box, no GPU):
  merge  alpha_trunk_composition's source and overlay levels and alpha_trunk_default_off's two directions and negatives,
         re-run unchanged on the merge commit.  Every tree is a scratch clone at its commit, so moved worktrees cannot matter.
  tip    every path changed since the merge is classified.  In the engine only qsa_operation.py, qsa_driver.py,
         worker_bridge.py and the review changed.  The overlay regenerated from the tip differs from the merge's overlay
         only in those three files (plus generation metadata), each by exactly its source delta.
  off    with FLASH_S5_UNION_RETIRE unset the tip reduces to the merge plus three named unswitched deltas:
           m3-receipts       host-side counters and log fields, no device work (scanned);
           s1-accounting     the union set list and its live statistics;
           s1-release-clear  `run.sparse.union = None` at release, after the operation drained.
         Preconditions pin the latch, the operation's read, and the switch as the first statement of the retirement.
         Added definitions are pinned by body hash.  Negative controls must be refused.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_trunk_composition as C  # noqa: E402
import alpha_trunk_default_off as D  # noqa: E402
import native41_progress_freeze as P  # noqa: E402

ENGINE = D.ENGINE
OP = ENGINE + 'qsa_operation.py'
SWITCH = 'FLASH_S5_UNION_RETIRE'
LATCH = "self.native_union_retire = declared.get('FLASH_S5_UNION_RETIRE', '0') == '1'"
READ = "self.union_retire = getattr(device, 'native_union_retire', False) is True"
GUARD = 'if not self.union_retire:\n    return'
ENGINE_CHANGED = {'patches/s5_engine/qsa_operation.py', 'patches/s5_engine/qsa_driver.py',
                  'patches/s5_engine/worker_bridge.py', 'patches/s5_engine/native_candidate_review.json'}
OVERLAY_CHANGED = {'e3/qsa_operation.py', 'e3/qsa_driver.py', 'e3/worker_bridge.py'}

UNSWITCHED = {
    'm3-receipts': 'M3 a61289f: host-side receipts only.  Allocator counters on the existing memory line, new statistics '
                   'keys, the indexer size note and the native-alloc line, the last two behind the FLASH_S5_AUTO_ARM '
                   'receipt gate.  No device work (no_device_work below); a receipt failure is counted, never raised.',
    's1-accounting': 'S1 bef0900: the operation lists the union sets it allocated and reports the count and live bytes. '
                     'The list adds no lifetime: each set stays referenced by its layer until release either way.',
    's1-release-clear': 'S1 bef0900: release clears the sets, after the operation drained.  Reachable with the switch '
                        'unset (operator-accepted, lead 1 Nov 2026).',
    's1-retire-switched': 'Retirement behind FLASH_S5_UNION_RETIRE: with it unset the call returns at its first '
                          'statement (preconditions below), so it drops nothing and queries nothing.',
}
STATS = (", union_scratch_sets=0, union_scratch_retired=0, union_scratch_live_bytes=0, union_scratch_live_max_bytes=0, "
         "indexer_layers=0, indexer_rows=0, indexer_width=0, indexer_chunks=0, indexer_peak_bytes=0, "
         "indexer_total_bytes=0, indexer_errors=0, alloc_receipt_errors=0")
INVENTORY = ("union_scratch_sets=self.stats['union_scratch_sets'], union_scratch_retired=self.stats['union_scratch_retired'], "
             "union_scratch_live_max_bytes=self.stats['union_scratch_live_max_bytes'], ")
TIP_OFF = {
    'qsa_operation.py': {
        'NativeQsaDevice.__init__': dict(
            drop={LATCH: 1, "self.native_memory_receipts = declared.get('FLASH_S5_AUTO_ARM', '0') == '1'": 1}),
        'QsaPrefillOperation.__init__': dict(
            drop={READ: 1, "self.memory_receipts = getattr(device, 'native_memory_receipts', False) is True": 1,
                  'self._union_sets = []': 1},
            subs=((STATS, ''),)),
        'QsaPrefillOperation.sparse_union_scratch': dict(
            drop={'self._retire_union_scratch()': 1,
                  'self._union_sets.append((run.layer_name, self._union_scratch))': 1,
                  "self.stats['union_scratch_sets'] += 1": 1, 'self._union_live()': 1}),
        'QsaPrefillOperation.inventory': dict(subs=((INVENTORY, ''),)),
        'QsaPrefillOperation.release': dict(
            drop={'self._union_sets = []': 1, 'if run.sparse is not None:\n    run.sparse.union = None': 1,
                  'if self.memory_receipts:\n    self._alloc_receipt()': 1}),
    },
    'qsa_driver.py': {
        'QsaLayerDriver._pre': dict(
            drop={"if getattr(operation, 'memory_receipts', False):\n"
                  '    _note_indexer_scores(operation, self.forward_context, attn, num_tokens)': 1}),
    },
    'worker_bridge.py': {
        'NativeWorkerBridge._memory_receipt': dict(subs=(
            (' sync_all_streams=%d device_allocs=%d device_frees=%d', ''),
            (", row['sync_all_streams'], row['device_allocs'], row['device_frees']", ''),)),
    },
}
NAMES = {
    'qsa_operation.py:NativeQsaDevice.__init__': ['m3-receipts', 's1-retire-switched'],
    'qsa_operation.py:QsaPrefillOperation.__init__': ['m3-receipts', 's1-accounting', 's1-retire-switched'],
    'qsa_operation.py:QsaPrefillOperation.sparse_union_scratch': ['s1-accounting', 's1-retire-switched'],
    'qsa_operation.py:QsaPrefillOperation.inventory': ['s1-accounting'],
    'qsa_operation.py:QsaPrefillOperation.release': ['m3-receipts', 's1-accounting', 's1-release-clear'],
    'qsa_driver.py:QsaLayerDriver._pre': ['m3-receipts'],
    'worker_bridge.py:NativeWorkerBridge._memory_receipt': ['m3-receipts'],
}
# Definitions the tip adds, each with its name and a body hash pinned once the engine froze.
ADDED = {
    'qsa_driver.py:_note_indexer_scores': ('m3-receipts',
        '73f54e47040b14e76bc4ea3afc6f17b60eea6a1c0f3e7cf0c5ae4132039e6cb9'),
    'qsa_operation.py:indexer_score_shape': ('m3-receipts',
        '6bede35df67c244fea175dd5236ccd7fc7269cc0d68e203f26a7188b8266192f'),
    'qsa_operation.py:QsaPrefillOperation.note_indexer_scores': ('m3-receipts',
        'fbe3675d16ea9bb0241632cd5711d200ac08e4d0b4e61d221b77852bf61ac22b'),
    'qsa_operation.py:QsaPrefillOperation._alloc_receipt': ('m3-receipts',
        'd9bb8ac6ea317200c9b311bf1c2215b08d07989e14796d4fb4cd8d2bd23ff854'),
    'qsa_operation.py:QsaPrefillOperation._union_live': ('s1-accounting',
        '3cb482c21ae0c89af0081e66813bff52bb2b3787d876e9463193656645d24ab2'),
    'qsa_operation.py:QsaPrefillOperation._retire_union_scratch': ('s1-retire-switched',
        '3c28a5cdee9a42b903f4c6c02468ec00aca3c940a05998dfe7339295b1f17e51'),
}
RECEIPTS_ONLY = [k for k, v in ADDED.items() if v[0] != 's1-retire-switched']
DEVICE_WORDS = {'device', 'torch', 'cuda', 'query', 'wait', 'wait_event', 'record', 'sync', 'to', 'copy_', 'item', 'cpu'}
D.DECLARED['T'] = {'worker_bridge.py:module:_MEMORY_FIELDS':
                   'M3 (m3-receipts): three more torch.cuda.memory_stats keys read by the existing memory_receipt'}
D.UNNAMED['T'] = {}
D.PINS['T'] = dict(added=(6, 'adb5b599294c22d4501c922a0205fc1eca938256076f92293b88a9a71f268887'),
                   imports=(1, '4704bb01dcef639d3180d32e4ff53afec0fef9e46c8abd993a83483e6829f7a8'))


def no_device_work(texts):
    """The added receipt definitions name no device, torch, event or transfer call."""
    out = {}
    for key in RECEIPTS_ONLY:
        short, name = key.split(':', 1)
        node = P.function(texts[ENGINE + short], name)
        words = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        hit = sorted(words & DEVICE_WORDS)
        assert not hit, ('receipt definition names device work', key, hit)
        out[key] = 'no device word'
    return out


def preconditions(tip, merge):
    """The switch itself: latched once, read once, first statement of the retirement, one caller, one drop there."""
    text = tip[OP]
    device = P._strip_docstrings(P.function(text, 'NativeQsaDevice.__init__'))
    operation = P._strip_docstrings(P.function(text, 'QsaPrefillOperation.__init__'))
    retire = P._strip_docstrings(P.function(text, 'QsaPrefillOperation._retire_union_scratch'))
    assert LATCH in [ast.unparse(s) for s in device.body], 'the latch is not the declared statement'
    assert READ in [ast.unparse(s) for s in operation.body], "the operation's read is not the declared statement"
    assert ast.unparse(retire.body[0]) == GUARD, 'the switch is not the first statement of the retirement'
    nodes = [n for f, source in tip.items() if source is not None for n in ast.walk(ast.parse(source))]
    named = {SWITCH: 1, 'native_union_retire': 1}
    for value, count in named.items():
        seen = sum(isinstance(n, ast.Constant) and n.value == value for n in nodes)
        assert seen == count, ('string constant', value, seen)
    stores = {}
    for node in nodes:
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and node.attr in (
                'native_union_retire', 'union_retire'):
            stores[node.attr] = stores.get(node.attr, 0) + 1
    assert stores == dict(native_union_retire=1, union_retire=1), stores
    callers = sum(isinstance(n, ast.Attribute) and n.attr == '_retire_union_scratch' and isinstance(n.ctx, ast.Load)
                  for n in nodes)
    assert callers == 1, ('retirement callers', callers)
    code = sorted(ast.unparse(n) for n in nodes if isinstance(n, ast.Assign) and any(
        isinstance(target, ast.Attribute) and target.attr == 'union' for target in n.targets)
        and ast.unparse(n.value) == 'None')
    assert code == ['run.sparse.union = None', 'sparse.union = None'], ('union drops', code)
    assert not [n for f, source in merge.items() if source is not None for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Assign) and ast.unparse(n).endswith('.union = None')], 'the merge already dropped union sets'
    return dict(latch=LATCH, read=READ, first_statement=GUARD, retirement_callers=1,
                union_drops=sorted(code))


def added_bodies(tip):
    out = {}
    for key in ADDED:
        short, name = key.split(':', 1)
        node = P._strip_docstrings(P.function(tip[ENGINE + short], name))
        out[key] = hashlib.sha256(P._text(node).encode()).hexdigest()
    return out


def tip_off(tip, merge, files, *, pins=True):
    pre = preconditions(tip, merge)
    bodies = D.project_table(TIP_OFF, tip, merge)
    assert set(bodies) == set(NAMES), sorted(set(bodies) ^ set(NAMES))
    step = D.coverage('T', tip, merge, set(bodies), files)
    want_added = sorted(ADDED)
    assert step['added'] == want_added, ('added definitions differ', step['added'], want_added)
    scan = no_device_work(tip)
    hashes = added_bodies(tip)
    if pins:
        for key, (_, pin) in ADDED.items():
            assert pin in (None, hashes[key]), ('added body differs from the pin', key)
    return dict(switch_unset=SWITCH, target='the merge ' + D.REV['merge'], preconditions=pre, bodies=bodies,
                functions=len(bodies), delta_names=NAMES, unswitched=UNSWITCHED,
                added={k: dict(name=ADDED[k][0], body_sha256=hashes[k]) for k in ADDED},
                no_device_work=scan, coverage=step)


NEGATIVES = (
    ('retirement-guard-removed', 'QsaPrefillOperation._retire_union_scratch', 'if not self.union_retire:', 'if False:', OP),
    ('latch-default-on', 'NativeQsaDevice.__init__', "declared.get('FLASH_S5_UNION_RETIRE', '0')",
     "declared.get('FLASH_S5_UNION_RETIRE', '1')", OP),
    ('operation-read-forced-on', 'QsaPrefillOperation.__init__', "getattr(device, 'native_union_retire', False) is True",
     'True', OP),
    ('second-union-drop', 'QsaPrefillOperation._union_live', 'live = sum(',
     "self._layers['x'].sparse.union = None\n        live = sum(", OP),
    ('undeclared-change-in-tip-function', 'QsaPrefillOperation.release', None, None, OP),
    ('receipt-does-device-work', 'QsaPrefillOperation._alloc_receipt', 'stats, tx = self.stats, self.transaction',
     'self.device.query(None); stats, tx = self.stats, self.transaction', OP),
    ('added-receipt-body-changed', '_note_indexer_scores', "operation.stats['indexer_errors'] += 1",
     "operation.stats['indexer_errors'] += 2", ENGINE + 'qsa_driver.py'),
)


def negatives(tip, merge, files):
    out = {}
    for label, name, old, new, path in NEGATIVES:
        texts = dict(tip)
        texts[path] = D.mutate(tip[path], name, old, new)
        try:
            tip_off(texts, merge, files)
        except AssertionError as exc:
            out[label] = 'REFUSED: ' + str(exc)[:160]
            continue
        raise AssertionError('negative control accepted: ' + label)
    return out


def export(rev, into):
    """A scratch clone at REV: build_native43's stage needs a git tree, and no worktree of this repo is touched."""
    into.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '-q', '--shared', '--no-checkout', str(ROOT), str(into)], check=True)
    subprocess.run(['git', '-C', str(into), 'checkout', '-q', '--detach', rev], check=True)
    return into


def classify_paths(merge_rev, tip_rev):
    rows = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-status', merge_rev, tip_rev], text=True).splitlines()
    out = {}
    engine = set()
    for row in rows:
        status, path = row.split('\t', 1)
        if path.startswith('patches/'):
            engine.add(path)
            kind = 'engine'
        elif path.startswith('harness/'):
            kind = 'harness'
        elif path.startswith('tools/'):
            kind = 'tools'
        elif path.startswith(('results/hub/alpha-trunk/', 'docs/e3-briefs/')):
            kind = 'record'
        else:
            raise AssertionError('undeclared path since the merge: ' + path)
        out.setdefault(kind, []).append(status + ' ' + path)
    assert engine == ENGINE_CHANGED, ('engine paths since the merge', sorted(engine))
    return out


def delta(old, new):
    return [line for line in difflib.unified_diff(old.splitlines(True), new.splitlines(True), n=0)
            if line[0] in '+-' and line[:3] not in ('+++', '---')]


def tip_overlay(merged, tip, tip_tree, merged_tree):
    mr, tr = C.rows(merged), C.rows(tip)
    assert set(mr) == set(tr), sorted(set(mr) ^ set(tr))
    moved = sorted(n for n in mr if mr[n] != tr[n])
    metadata = sorted(n for n in moved if n in C.METADATA)
    code = set(moved) - C.METADATA
    assert code == OVERLAY_CHANGED, ('overlay code differs', sorted(code))
    for name in sorted(code):
        # The generator rewrites imports, so the claim is the delta: the lines added and removed in the overlay file are
        # exactly the lines added and removed in its source, in order.
        short = 'patches/s5_engine/' + Path(name).name
        source = delta((merged_tree / short).read_text(), (tip_tree / short).read_text())
        assert source and delta((merged / name).read_text(), (tip / name).read_text()) == source, (
            'overlay delta is not the source delta', name)
    return dict(code=sorted(code), metadata=metadata, merged_manifest=C.sha((merged / 'SHA256SUMS').read_bytes()),
                tip_manifest=C.sha((tip / 'SHA256SUMS').read_bytes()), files=len(tr))


def run(a):
    tip_rev = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', a.tip], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'patches', 'harness'], text=True)
    assert not dirty, 'engine or harness tree is not committed'
    D.REV['tip'] = tip_rev
    work = a.work
    work.mkdir(parents=True, exist_ok=False)
    files = D.engine_files()
    merged = {f: D.show('merge', f) for f in files}
    tip = {f: D.show('tip', f) for f in files}
    repo = C.Repo(ROOT)
    trees = {k: export(rev, work / 'tree' / k) for k, rev in (
        ('merged', D.REV['merge']), ('dense', D.REV['dense']), ('heldf', D.REV['heldf']), ('tip', tip_rev))}
    (work / 'overlay').mkdir()
    overlays = {k: C.regenerate(v, work / 'overlay' / k, a.tools) for k, v in trees.items()}
    base = Path(a.base_overlay)
    merge = dict(
        composition=dict(source=C.source_level(repo, D.REV['merge'], D.REV['dense'], D.REV['heldf']),
                         overlay=C.overlay_level(overlays['merged'], overlays['dense'], overlays['heldf'], base),
                         manifests={k: C.sha((overlays[k] / 'SHA256SUMS').read_bytes()) for k in ('merged', 'dense', 'heldf')},
                         base_manifest=C.sha((base / 'SHA256SUMS').read_bytes())),
        default_off=dict(progress_unset_is_dense_line=D.direction_h(merged, files, work),
                         policy_unset_is_heldf=D.direction_d(merged, files),
                         negative_controls=D.negatives(merged, files, work)))
    return dict(
        schema='alpha-trunk-tip-proof/1', verdict='PASS', tip=tip_rev, merge=D.REV['merge'], base=D.REV['base'],
        parents=dict(dense=D.REV['dense'], heldf=D.REV['heldf']), merge_proofs=merge,
        tip_paths=classify_paths(D.REV['merge'], tip_rev),
        tip_overlay=tip_overlay(overlays['merged'], overlays['tip'], trees['tip'], trees['merged']),
        union_retire_unset_is_merge=tip_off(tip, merged, files), tip_negative_controls=negatives(tip, merged, files),
        level='source and regenerated overlay; receipts and retirement have no box evidence',
        gpu_execution=False, box_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--tip', default='HEAD')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='new scratch dir')
    p.add_argument('--tools', default=str(ROOT / 'tools'))
    p.add_argument('--base-overlay', required=True)
    a = p.parse_args()
    record = run(a)
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=1)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'], tip=record['tip'], tip_overlay=record['tip_overlay']['code'],
                          off=record['union_retire_unset_is_merge']['functions'],
                          negatives=sorted(record['tip_negative_controls']),
                          manifests=record['merge_proofs']['composition']['manifests'],
                          tip_manifest=record['tip_overlay']['tip_manifest'])))
