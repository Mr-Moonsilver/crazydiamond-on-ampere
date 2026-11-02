#!/usr/bin/env python3
"""alpha-next-c1 composition proof: every merge from alpha-next's tip to HEAD is a clean merge or a declared resolution (CPU, git only).

tools/alpha_next_composition.py's merge method (its functions, imported unchanged: merge-tree re-run, clean paths byte-equal, each
conflict block's committed text declared; the review pin declared generated) over the first-parent chain from BASE (claude/alpha-next
eb3e0146, where C1 was merged) to HEAD. Unlike alpha-next's fixed step list, the chain here is walked: the coordinator merges
alpha-next's READY commit, this pack and the queue branch into claude/alpha-next-c1 after this tool is written, so every merge on the
chain is checked (each conflicted block must be in tools/alpha_next_c1_composition.json) and every non-merge commit is listed as
'own' with its paths (the engine part of those is proved by tools/alpha_next_c1_freeze.py, the harness part is reviewed and pinned by
the review). Also required: the merge of claude/c1 (C1_MERGE, second parent C1_TIP) is on the chain, and the pinned alpha-next build
source (tools/alpha_next_c1_parent.json) is an ancestor of HEAD.

    python3 tools/alpha_next_c1_composition.py --out FILE [--draft]
--draft writes the declaration the tree implies, for a reviewer to compare with (and commit as) the committed one.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_composition as comp  # noqa: E402

ROOT = comp.ROOT
DECLARATION = ROOT / 'tools/alpha_next_c1_composition.json'
BASE = 'eb3e0146'                        # claude/alpha-next tip the C1 merge was made on
C1_MERGE, C1_TIP = 'b58b6af7', 'cf120738'  # alpha-next-c1: merge claude/c1 (C1 two chunks in flight)
SCHEMA = 'alpha-next-c1-composition/1'


def chain():
    """(commit, parents) along the first-parent chain from BASE (exclusive) to HEAD, oldest first."""
    rows = comp.git('rev-list', '--first-parent', '--reverse', '--parents', comp.rev(BASE) + '..HEAD').splitlines()
    return [(r.split()[0], r.split()[1:]) for r in rows]


def draft():
    out = []
    for commit, parents in chain():
        if len(parents) != 2:
            continue
        tree, conflicted = comp.merge_rerun(parents[0], parents[1])
        for path in conflicted:
            if path == comp.REVIEW:
                continue
            marked = comp.blob(tree, path).decode()
            got = comp.resolutions_of(marked, comp.blob(commit, path).decode())
            blocks = [t for k, t in comp.blocks(marked) if k == 'block']
            for index, text in enumerate(got):
                out.append(dict(merge=commit, path=path, index=index, text=text,
                                marked_sha256=hashlib.sha256(blocks[index].encode()).hexdigest()))
    return dict(schema='alpha-next-c1-composition-declaration/1', base=comp.rev(BASE), blocks=out)


def prove(pinned_source=None):
    declaration = json.loads(DECLARATION.read_text())
    assert declaration['base'] == comp.rev(BASE), 'declaration base differs'
    steps, merges = [], []
    for commit, parents in chain():
        assert len(parents) in (1, 2), ('octopus merge', commit)
        if len(parents) == 2:
            steps.append(comp.check_merge(commit, parents[1], declaration['blocks']))
            merges.append(commit)
        else:
            steps.append(dict(kind='own', commit=commit, paths=comp.git('show', '--name-status', '--format=', commit).splitlines()))
    assert comp.rev(C1_MERGE) in merges and comp.rev(C1_MERGE + '^2') == comp.rev(C1_TIP), 'the C1 merge is not on the chain'
    used = {(b['merge'], b['path']) for b in declaration['blocks']}
    proved = {(s['commit'], p) for s in steps if s['kind'] == 'merge' for p, r in s['conflicted'].items() if r['kind'] == 'declared'}
    assert used == proved, ('declared blocks differ from the conflicts on the chain', sorted(used ^ proved))
    if pinned_source is not None:
        assert comp.git('merge-base', '--is-ancestor', pinned_source, 'HEAD') == '', 'pinned alpha-next source not merged'
    return dict(schema=SCHEMA, verdict='PASS', base=comp.rev(BASE), head=comp.rev('HEAD'), c1_merge=comp.rev(C1_MERGE),
                merges=merges, own=sum(1 for s in steps if s['kind'] == 'own'), steps=steps, parent_source=pinned_source,
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest(), box_action=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--draft', action='store_true')
    a = ap.parse_args(argv)
    if a.draft:
        record = draft()
    else:
        import alpha_next_c1_parent_pin as parent_pin
        record = prove(parent_pin.load()['source'])
    with Path(a.out).open('x') as handle:
        handle.write(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('steps', 'blocks')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
