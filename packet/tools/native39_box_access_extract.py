#!/usr/bin/env python3
"""Native39: real uvicorn access lines from retained box logs for the ingress check tests.

The trunk's ingress checks (e3_native39_trunk_checks ingress, ingress-w1) read
the engine's uvicorn access lines from the followed container log.  The queue
tests write those lines from a loopback leaf; this extract keeps real ones so
the parser and the span rules are also tested on box bytes.  Written once to
harness/fixtures/native39_box_access_extract.json (each source's sha256
recorded; refuses to overwrite):
  * q18b-w1ab2 run 1 arm B0 (31 Oct 2026, --host 127.0.0.1, plain w1_trace):
    every access line of engine.follow.1.log verbatim, the client's wall anchor
    and window bounds from w1.json;
  * q18a-native35-p-c run 1 (31 Oct 2026, --host 0.0.0.0): the count of
    access-shaped lines and of lines ACCESS_RE parses, counts per (loopback or
    not, method, path, code), and the first line of each such kind verbatim.
ACCESS_RE's pattern is recorded; the test requires the same pattern, so the
counts stay valid for the parser under test.  No secret is read: access lines
carry no headers.
"""
import hashlib
import json
import lzma
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
import e3_native39_trunk_checks as checks  # noqa: E402

FN = Path('<hub-dir>/agents/flash-next/fn-main/results/box')
PC = FN / 'q18a-native35-p-c-run-1/box/n35pc/q18a-native35-p-c'
B0 = FN / 'q18b-w1ab2-run-1/box/q18b-w1ab2/B0'
OUT = ROOT / 'harness/fixtures/native39_box_access_extract.json'
SOURCES = dict(p_c_engine_stream=PC / 'engine-stream.log.xz', w1ab2_b0_follow=B0 / 'engine.follow.1.log',
               w1ab2_b0_w1=B0 / 'w1.json')
SHAPE = re.compile(r'" \d{3}( |$)')  # any request line shape, parsed or not


def text(path):
    data = path.read_bytes()
    return (lzma.decompress(data) if path.suffix == '.xz' else data).decode(errors='replace')


def access(body):
    shaped = [line for line in body.splitlines() if 'HTTP/' in line and SHAPE.search(line)]
    return shaped, [line for line in shaped if checks.ACCESS_RE.search(line)]


def main():
    assert not OUT.exists(), 'extract exists'
    sources = {name: dict(path=str(path.relative_to(FN.parents[1])), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for name, path in SOURCES.items()}
    b0_shaped, b0_parsed = access(text(SOURCES['w1ab2_b0_follow']))
    w1 = json.loads(text(SOURCES['w1ab2_b0_w1']))
    pc_shaped, pc_parsed = access(text(SOURCES['p_c_engine_stream']))
    kinds, first = {}, {}
    for line in pc_parsed:
        m = checks.ACCESS_RE.search(line)
        key = '%s %s %s %s' % ('loopback' if m.group(1) == checks.LOOPBACK else 'lan', m.group(3),
                               m.group(4).split('?', 1)[0], m.group(5))
        kinds[key] = kinds.get(key, 0) + 1
        first.setdefault(key, line)
    extract = dict(
        schema='native39-box-access-extract/1', sources=sources, access_re=checks.ACCESS_RE.pattern,
        w1ab2_b0=dict(shaped_lines=len(b0_shaped), parsed_lines=len(b0_parsed), lines=b0_parsed,
                      wall_anchor_s=w1['wall_anchor'][0], streams=w1['streams'],
                      windows=[dict(repeat=x['repeat'], wall_t0=x['wall_t0'], wall_t1=x['wall_t1']) for x in w1['windows']]),
        p_c=dict(shaped_lines=len(pc_shaped), parsed_lines=len(pc_parsed), kinds=kinds,
                 first_of_kind=[first[k] for k in sorted(first)]))
    with OUT.open('x') as f:
        json.dump(extract, f, sort_keys=True, indent=1)
        f.write('\n')
    OUT.chmod(0o664)
    print('native39 box access extract written', OUT.stat().st_size, hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
