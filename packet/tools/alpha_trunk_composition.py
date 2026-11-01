#!/usr/bin/env python3
"""Alpha trunk T1 composition proof: the merge of held-f (main c607333) into the dense line (le-fix f1e5820).

Source level (git, every path of the three trees):
  * a path both parents leave equal is the merged blob;
  * a path one parent changed is that parent's blob;
  * a path both changed is the clean `git merge-file` result, or one of the three declared resolutions:
      - patches/s5_engine/executor_hooks.py: the one conflict block is held-f's block with its broadcast write
        replaced by the dense line's bounded_enqueue call (wire_kwargs in place of kwargs); held-f's block and the
        dense line's block must be exactly the declared ones;
      - harness/selftest.py: each conflict block is the union of both blocks' suite rows (held-f's first, each suite
        once); the registered suite set is the union of the parents';
      - patches/s5_engine/native_candidate_review.json: top-level keys by the three-way rule; source_shas and
        harness_source_shas are the merged tree's own; cpu_regression/post_review_cpu are the dense parent's at the
        merge commit (re-prepared over the merged tree afterwards).
Overlay level (--overlay): the overlay is regenerated from each of the three trees with build_native43's own stage and
regenerate_overlay (scratch output); every non-metadata overlay file of the merged overlay is the dense one, the
held-f one, the clean three-way merge over the held-b overlay, or, for e3/executor_hooks.py, the declared resolution.
Default-off switches: the FLASH_S5_/FLASH_NATIVE_/VLLM_QSA_ names in each parent's added engine lines are listed per parent.
No box action, no GPU.  Writes one JSON record.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json', 'e3/native-overlap-readiness.json',
            'e3/decode-probe-evidence.json', 'core/SHA256SUMS', 'SHA256SUMS'}
HOOKS = 'patches/s5_engine/executor_hooks.py'
SELFTEST = 'harness/selftest.py'
REVIEW = 'patches/s5_engine/native_candidate_review.json'
DENSE_BLOCK = ('        bounded_enqueue(executor, executor.rpc_broadcast_mq,\n'
               '                        (send_method, args, kwargs, broadcast_rank))\n')
HELDF_WRITE = ('        executor.rpc_broadcast_mq.enqueue((send_method, args, wire_kwargs,\n'
               '                                           broadcast_rank))\n')
COMPOSED_WRITE = ('        bounded_enqueue(executor, executor.rpc_broadcast_mq,\n'
                  '                        (send_method, args, wire_kwargs, broadcast_rank))\n')
SUITE = re.compile(r"^\s*\('(test_[A-Za-z0-9_]+\.py|[A-Za-z0-9_/]+\.py)'")
ENV = re.compile(r"""['"]((?:FLASH_S5|FLASH_NATIVE|VLLM_QSA)_[A-Z0-9_]+)['"]""")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Repo:
    def __init__(self, root):
        self.root = Path(root)

    def run(self, *args, text=True):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True, text=text).stdout

    def blob(self, rev, path):
        r = subprocess.run(['git', '-C', str(self.root), 'show', '%s:%s' % (rev, path)], capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def paths(self, rev):
        return set(self.run('ls-tree', '-r', '--name-only', rev).split('\n')) - {''}


def merge_file(ours, base, theirs):
    """(text, clean) of git merge-file -p with labelled markers (ours=dense, theirs=held-f)."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (('dense', ours), ('base', base), ('heldf', theirs)):
            f = Path(d) / name
            f.write_bytes(data or b'')
            files.append(str(f))
        r = subprocess.run(['git', 'merge-file', '-p', '-L', 'dense', '-L', 'base', '-L', 'heldf', *files], capture_output=True)
        return r.stdout, r.returncode == 0


def blocks(text):
    """[(start, end, ours_text, theirs_text)] of the conflict blocks in a merge-file output."""
    out, i = [], 0
    while True:
        a = text.find('<<<<<<< dense\n', i)
        if a < 0:
            return out
        m = text.index('=======\n', a)
        z = text.index('>>>>>>> heldf\n', m)
        out.append((a, z + len('>>>>>>> heldf\n'), text[a + len('<<<<<<< dense\n'):m], text[m + len('=======\n'):z]))
        i = z


def resolve(text, rule):
    pieces, last = [], 0
    found = blocks(text)
    for a, z, ours, theirs in found:
        pieces.append(text[last:a])
        pieces.append(rule(ours, theirs))
        last = z
    pieces.append(text[last:])
    return ''.join(pieces), len(found)


def hooks_rule(ours, theirs, *, dense_block=DENSE_BLOCK, heldf_write=HELDF_WRITE, composed=COMPOSED_WRITE):
    assert ours == dense_block, ('dense conflict block is not the declared bounded_enqueue call', ours)
    assert theirs.count(heldf_write) == 1 and theirs.endswith(heldf_write), ('held-f block shape', theirs)
    return theirs.replace(heldf_write, composed)


def suite_union(ours, theirs):
    seen, out = set(), []
    for line in theirs.splitlines(True) + ours.splitlines(True):
        m = SUITE.match(line)
        key = m.group(1) if m else line
        if key not in seen:
            seen.add(key)
            out.append(line)
    return ''.join(out)


def registered(text):
    return sorted({m.group(1) for line in text.splitlines() for m in [SUITE.match(line)] if m and m.group(1).startswith('test_')})


def review_check(repo, merge, dense, heldf, base):
    m, o, t, b = (json.loads(repo.blob(r, REVIEW)) for r in (merge, dense, heldf, base))
    keys = {}
    for k in sorted(set(b) | set(o) | set(t) | set(m)):
        bo, oo, to = b.get(k), o.get(k), t.get(k)
        if k in ('source_shas', 'harness_source_shas'):
            continue
        if oo == to:
            want, rule = oo, 'equal'
        elif oo == bo:
            want, rule = to, 'held-f'
        elif to == bo:
            want, rule = oo, 'dense'
        elif k in ('cpu_regression', 'post_review_cpu'):
            want, rule = oo, 'dense-carried'
        else:
            raise AssertionError('review key changed on both sides without a rule: ' + k)
        assert m.get(k) == want, ('review key', k, rule)
        keys[k] = rule
    names = sorted(set(o['source_shas']) | set(t['source_shas']))
    assert sorted(m['source_shas']) == names
    for n in names:
        assert m['source_shas'][n] == sha(repo.blob(merge, 'patches/s5_engine/' + n)), n
    tracked = sorted(p for p in repo.paths(merge) if p.startswith('harness/'))
    want = {str(Path(p).relative_to('harness')): sha(repo.blob(merge, p)) for p in tracked}
    assert m['harness_source_shas'] == want, 'harness_source_shas are not the merged tree\'s'
    return dict(keys={k: v for k, v in keys.items() if v != 'equal'}, source_shas=len(names),
                harness_source_shas=len(want))


def source_level(repo, merge, dense, heldf):
    parents = repo.run('log', '-1', '--format=%P', merge).split()
    assert parents == [repo.run('rev-parse', dense).strip(), repo.run('rev-parse', heldf).strip()], parents
    base = repo.run('merge-base', dense, heldf).strip()
    counts, both, declared = {}, {}, {}
    for path in sorted(repo.paths(merge) | repo.paths(dense) | repo.paths(heldf)):
        mb, ob, tb, bb = (repo.blob(r, path) for r in (merge, dense, heldf, base))
        if ob == tb:
            kind = 'equal' if ob == bb else 'same-change'
            assert mb == ob, (path, kind)
        elif ob == bb:
            kind = 'held-f'
            assert mb == tb, (path, kind)
        elif tb == bb:
            kind = 'dense'
            assert mb == ob, (path, kind)
        else:
            text, clean = merge_file(ob, bb, tb)
            if clean:
                kind = 'clean-3way'
                assert mb == text, (path, kind)
                both[path] = kind
            elif path == HOOKS:
                kind = 'declared'
                want, n = resolve(text.decode(), hooks_rule)
                assert n == 1 and mb.decode() == want, path
                declared[path] = dict(rule='held-f progress meta, dense bounded_enqueue', blocks=n)
            elif path == SELFTEST:
                kind = 'declared'
                want, n = resolve(text.decode(), suite_union)
                assert mb.decode() == want, path
                union = sorted(set(registered(ob.decode())) | set(registered(tb.decode())))
                assert registered(mb.decode()) == union, 'registered suites are not the union'
                declared[path] = dict(rule='suite-row union', blocks=n, registered=len(union))
            elif path == REVIEW:
                kind = 'declared'
                declared[path] = dict(rule='review three-way, merged-tree shas', **review_check(repo, merge, dense, heldf, base))
            else:
                raise AssertionError('undeclared conflicted path: ' + path)
        counts[kind] = counts.get(kind, 0) + 1
    switches = {}
    for name, rev in (('dense', dense), ('held-f', heldf)):
        added = [line[1:] for line in repo.run('diff', '-U0', base, rev, '--', 'patches/s5_engine').splitlines()
                 if line.startswith('+') and not line.startswith('+++')]
        switches[name] = sorted({m for line in added for m in ENV.findall(line)})
    return dict(merge=repo.run('rev-parse', merge).strip(), parents=parents, base=base, counts=counts,
                both_changed_clean=sorted(both), declared=declared, switches_read_by_added_lines=switches)


def regenerate(tree, out, tools):
    """build_native43.stage + regenerate_overlay over TREE (its committed review), into OUT."""
    code = '''import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("b43", sys.argv[1] + "/build_native43.py")
b = importlib.util.module_from_spec(spec); sys.argv = sys.argv[:1] + sys.argv[2:]; spec.loader.exec_module(b)
b.ROOT = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]); out.mkdir()
bundle = out / "bundle"
b.stage(dict(tools=[]), bundle)
b.regenerate_overlay(out, bundle)
print(json.dumps(dict(overlay=str(bundle / "native30_overlay"))))
'''
    r = subprocess.run([sys.executable, '-B', '-c', code, str(tools), str(tree), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return Path(out) / 'bundle/native30_overlay'


def rows(overlay):
    return {line.split(None, 1)[1]: line.split(None, 1)[0] for line in (overlay / 'SHA256SUMS').read_text().splitlines()}


def overlay_level(merged, dense, heldf, base):
    mr, orr, tr, br = rows(merged), rows(dense), rows(heldf), rows(base)
    # A file one parent added is absent (None) on the other side and in the base.
    assert set(mr) == set(orr) | set(tr) and set(br) <= set(orr) & set(tr), 'overlay file sets are not the union'
    counts, detail = {}, {}
    for name in sorted(mr):
        if name in METADATA:
            continue
        m, o, t, b = mr[name], orr.get(name), tr.get(name), br.get(name)
        if o == t:
            kind = 'equal' if o == b else 'same-change'
            assert m == o, (name, kind)
        elif o == b:
            kind = 'held-f'
            assert m == t, (name, kind)
        elif t == b:
            kind = 'dense'
            assert m == o, (name, kind)
        else:
            text, clean = merge_file(*((root / name).read_bytes() if (root / name).is_file() else b''
                                       for root in (dense, base, heldf)))
            data = (merged / name).read_bytes()
            if clean:
                kind = 'clean-3way'
                assert data == text, (name, kind)
            elif name == 'e3/executor_hooks.py':
                kind = 'declared'
                want, n = resolve(text.decode(), hooks_rule)
                assert n == 1 and data.decode() == want, name
            else:
                raise AssertionError('undeclared conflicted overlay file: ' + name)
            detail[name] = kind
        counts[kind] = counts.get(kind, 0) + 1
    return dict(counts=counts, both_changed=detail, metadata_skipped=sorted(METADATA & set(mr)),
                added_by=dict(dense=sorted(set(orr) - set(br)), heldf=sorted(set(tr) - set(br))))


def main(a):
    repo = Repo(a.repo)
    record = dict(schema='alpha-trunk-composition/1', source=source_level(repo, a.merge, a.dense, a.heldf))
    if a.overlay:
        work = Path(a.work)
        trees = dict(merged=a.merged_tree, dense=a.dense_tree, heldf=a.heldf_tree)
        overlays = {k: (work / k / 'bundle/native30_overlay' if (work / k / 'bundle/native30_overlay/SHA256SUMS').is_file()
                        else regenerate(v, work / k, a.tools)) for k, v in trees.items()}
        base = Path(a.base_overlay)
        record['overlay'] = dict(overlay_level(overlays['merged'], overlays['dense'], overlays['heldf'], base),
                                 trees={k: str(v) for k, v in trees.items()},
                                 manifests={k: sha((v / 'SHA256SUMS').read_bytes()) for k, v in overlays.items()},
                                 base_overlay=str(base), base_manifest=sha((base / 'SHA256SUMS').read_bytes()))
    record.update(verdict='PASS', classification=None, box_action=False, gpu_execution=False)
    with Path(a.out).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(verdict='PASS', source=record['source']['counts'], declared=sorted(record['source']['declared']),
                          overlay=record.get('overlay', {}).get('counts'), switches=record['source']['switches_read_by_added_lines'])))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--repo', required=True)
    p.add_argument('--merge', required=True)
    p.add_argument('--dense', required=True)
    p.add_argument('--heldf', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--overlay', action='store_true')
    p.add_argument('--work')
    p.add_argument('--tools', help='tools dir holding build_native43.py')
    p.add_argument('--merged-tree')
    p.add_argument('--dense-tree')
    p.add_argument('--heldf-tree')
    p.add_argument('--base-overlay')
    raise SystemExit(main(p.parse_args()))
