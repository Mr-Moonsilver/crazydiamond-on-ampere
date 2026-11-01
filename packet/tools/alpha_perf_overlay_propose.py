#!/usr/bin/env python3
"""Alpha-perf spec aid: regenerate this tree's overlay under OUT and print the changed definitions per moved file.

Runs build_native43.stage + regenerate_overlay on a scratch bundle (never the held build) and proposes the
spec's engine_changed and overlay_definitions against the frozen held-f build-b parent overlay.  A reviewer commits the
proposal into the spec; nothing is applied by this tool.  No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_alpha_perf import b  # noqa: E402  (held-f build-b as the parent)


def propose(out):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    b.regenerate_overlay(out, bundle)
    overlay = bundle / 'native30_overlay'
    rows, parent = b.verify(overlay), b.verify(b.PARENT / 'native30_overlay')
    moved = sorted(n for n in parent if rows.get(n) != parent[n])
    engine = sorted(n for n in moved if n not in b.METADATA)
    definitions = {n: b.changed_definitions((b.PARENT / 'native30_overlay' / n).read_text(), (overlay / n).read_text())
                   for n in engine}
    return dict(moved=moved, engine_changed=engine, overlay_definitions=definitions,
                added=sorted(set(rows) - set(parent)), removed=sorted(set(parent) - set(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(propose(a.out), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
