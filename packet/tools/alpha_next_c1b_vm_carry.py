#!/usr/bin/env python3
"""alpha-next-c1b VM carry-over gate: every module VM <vm-id> ran from alpha-next-c1's overlay is byte-identical in the c1b bundle.

alpha-next-c1's delivered VM run (results/vm<vm-id>/alpha-next-c1-b-plain and -stage, probe.json, both pinned by sha in
tools/alpha_next_c1b_parent.json) records, in sections.local_overlap.mounted_origins, the 72 modules the engine imported from the
mounted overlay: (module, origin, sha256), origin the path inside the container. The plain and stage lists must be equal (72 rows).
c1b claims the engine bytes do not move, so for each row this gate finds the file of the c1b bundle that the engine mounts at that
origin and compares its sha256 to the recorded one.

Origin -> bundle file: read from the bundle's own launcher (native30_overlay/run-flash-s5-test.sh, the script that mounts the
overlay): every '-v "$OVERLAY/<src>:<target>:ro"' with the launcher's path variables (C=, V=, D=) expanded. A file mount maps one
origin; a directory mount (e2 -> flash_e2, e3 -> flash_e3) maps every origin below it; the core loop ('core/$f:$C/$f') maps
core/<f>. An exact file mount wins over a directory mount (docker's own precedence for a file bind below a directory bind). Each
file must also carry its own row in native30_overlay/SHA256SUMS with the same sha (the launcher refuses unchecksummed files).

PASS (rc 0) only when all 72 rows map to a bundle file whose sha256 equals the recorded one. Any differing, unmapped or missing
row: FAIL, rc 1, the rows listed (the chain stops; the lead must be told). Bad inputs (lists differ, not 72 rows, probe sha not
the pinned one): rc 2. --out is written once. CPU only, read-only on the bundle; no container, no box, no VM.

    python3 tools/alpha_next_c1b_vm_carry.py --bundle HELD --out FILE [--probe PLAIN --probe STAGE]
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SCHEMA = 'alpha-next-c1b-vm-carry/1'
ROWS = 72
LAUNCHER = 'run-flash-s5-test.sh'
MOUNT = re.compile(r'-v "\$OVERLAY/([^:"]+):([^:"]+):ro"')
VAR = re.compile(r'^\s*([A-Z])=(/[^\s;]+)\s*$', re.M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_probes():
    import alpha_next_c1b_parent_pin as parent_pin
    pins = parent_pin.load()
    return [(ROOT / rel, want) for rel, want in sorted(pins['vm_probes'].items())]


def origins(probes):
    """The recorded mounted_origins: one list, equal in every probe, ROWS rows, each (module, origin, sha256)."""
    lists = []
    for path, want in probes:
        if want is not None and sha(path) != want:
            raise ValueError('VM probe record is not the pinned one: %s' % path)
        lists.append(json.loads(Path(path).read_text())['sections']['local_overlap']['mounted_origins'])
    if any(rows != lists[0] for rows in lists[1:]):
        raise ValueError('mounted_origins differ between the VM probe records')
    rows = lists[0]
    if len(rows) != ROWS or any(set(r) != {'module', 'origin', 'sha256'} for r in rows):
        raise ValueError('mounted_origins: %d rows, want %d (module, origin, sha256)' % (len(rows), ROWS))
    if len({r['origin'] for r in rows}) != ROWS:
        raise ValueError('mounted_origins: duplicate origins')
    return rows


def mounts(launcher_text):
    """(file mounts {target: [src, ...]}, directory mounts [(target_dir, src_dir)]) of the launcher, variables expanded.

    A target can have more than one file source when a switch picks between them (the launcher's VISION=1 mounts
    vision/model.py at the model path that e3_sources/model.py takes otherwise): every candidate is kept, and the row names the
    one whose bytes are the recorded ones."""
    env = {}
    for name, value in VAR.findall(launcher_text):
        if name in env and env[name] != value:
            raise ValueError('launcher assigns %s twice with different values' % name)
        env[name] = value

    def expand(text):
        return re.sub(r'\$([A-Z])\b', lambda m: env[m.group(1)], text)

    files, dirs = {}, []
    for src, target in MOUNT.findall(launcher_text):
        if '$f' in src:          # the core loop: core/$f -> $C/$f
            assert src.endswith('/$f') and target.endswith('/$f'), (src, target)
            dirs.append((expand(target[:-len('$f')]).rstrip('/'), src[:-len('$f')].rstrip('/')))
            continue
        target = expand(target)
        if '$' in target or '$' in src:
            raise ValueError('unexpanded launcher mount: %s:%s' % (src, target))
        if target.endswith('.py') or '.' in Path(src).name:
            files.setdefault(target, [])
            if src not in files[target]:
                files[target].append(src)
        else:
            dirs.append((target, src))
    return files, dirs


def resolve(origin, files, dirs):
    """([candidate overlay paths], mount kind): the file mounts at ORIGIN, else the longest directory mount above it."""
    if origin in files:
        return list(files[origin]), 'file' if len(files[origin]) == 1 else 'file-switch-selected'
    best = None
    for target, src in dirs:
        if origin.startswith(target + '/') and (best is None or len(target) > len(best[0])):
            best = (target, src)
    if best is None:
        return [], None
    return [best[1] + '/' + origin[len(best[0]) + 1:]], 'directory'


def sums(overlay):
    out = {}
    for line in (overlay / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip('*')
        out[name[2:] if name.startswith('./') else name] = digest
    return out


def carry(bundle, probes):
    bundle = Path(bundle).resolve()
    overlay = bundle / 'native30_overlay'
    rows = origins(probes)
    files, dirs = mounts((overlay / LAUNCHER).read_text())
    listed = sums(overlay)
    table, problems = [], []
    for r in rows:
        candidates, how = resolve(r['origin'], files, dirs)
        present = [c for c in candidates if (overlay / c).is_file()]
        shas = {c: sha(overlay / c) for c in present}
        same = [c for c in present if shas[c] == r['sha256'] == listed.get(c)]
        src = same[0] if same else present[0] if present else candidates[0] if candidates else None
        row = dict(module=r['module'], origin=r['origin'], recorded_sha256=r['sha256'], overlay_path=src, mount=how,
                   candidates=candidates, bundle_sha256=shas.get(src), sums_sha256=listed.get(src))
        row['verdict'] = ('UNMAPPED' if not candidates else 'MISSING' if not present else 'SAME' if same else 'DIFFERS')
        table.append(row)
        if row['verdict'] != 'SAME':
            problems.append(row)
    return dict(schema=SCHEMA, verdict='FAIL' if problems else 'PASS', rows=len(rows),
                same=sum(1 for t in table if t['verdict'] == 'SAME'), problems=problems, table=table,
                bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay_manifest=sha(overlay / 'SHA256SUMS'),
                launcher_sha256=sha(overlay / LAUNCHER), probes={str(p): sha(p) for p, _ in probes},
                meaning='every module VM <vm-id> imported from alpha-next-c1\'s mounted overlay (plain and stage) is mounted at the '
                        'same origin from a byte-identical file of this bundle',
                gpu_execution=False, box_action=False, vm_action=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--probe', type=Path, action='append', help='a VM probe.json (tests; default: the pinned parent\'s two)')
    a = p.parse_args(argv)
    try:
        probes = [(path, None) for path in a.probe] if a.probe else pinned_probes()
        if len(probes) != 2:
            raise ValueError('two probe records (plain and stage) required')
        result = carry(a.bundle, probes)
    except (ValueError, KeyError, OSError) as exc:
        print('VM_CARRY BAD-INPUT %s: %s' % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    with a.out.open('x') as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write('\n')
    print('VM_CARRY %s %d/%d %s' % (result['verdict'], result['same'], result['rows'], sha(a.out)))
    for row in result['problems']:
        print('  %s %s %s' % (row['verdict'], row['origin'], row['overlay_path']))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
