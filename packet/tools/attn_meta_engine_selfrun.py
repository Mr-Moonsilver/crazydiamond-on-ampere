#!/usr/bin/env python3
"""q18a-attn-meta engine self-run: c1's seven containers (tools/alpha_next_c1_engine_selfrun_b.py, the one c1's chain b ran),
unchanged, plus one attn-meta container. CPU only, no network.

Eight containers over one frozen bundle, all results kept under OUT:
  * progress/, dense/, fused/, groups/, stage/, composed/, overlap/: c1's parts, run by c1's run() text (the _b judge) with three
    exact-once swaps (SWAPS below: the eighth part appended to the run list, unpacked, and its suite names recorded); every
    attn-meta switch unset in all of them (FLASH_ATTN_META_GDN_BATCH is not in any of their command lines);
  * attn/: ATTN_SUITES (test_attn_meta_control: the typed control and the worker adapter's allowlist; test_attn_meta_c1_g1: the
    switch beside C1's per-arm selection at G1; test_attn_meta_box: the arm, decode arm, grade and read; test_attn_meta_queue: the
    derived queue texts, under real bash) in one unittest process with per-test outcomes, no connected cases (attn-meta adds no
    probe mode: its device path is the VM's, tools/attn_meta_vm_input_c.py). The suites that need the model config and the
    overlay qsa (test_attn_meta_select, test_attn_meta_probe_b, test_attn_meta_overlay) are the chain's own container steps
    (tools/attn_meta_selfrun5.py tests-c and probe-c, unchanged), not this record's.
OUT/result.json is the merged record (schema attn-meta-selfrun/1); its connected cases are exactly c1's expected_cases(). Verdict
PASS only when all eight are PASS and c1's pairs and early open hold (c1's rule, unchanged).

Comparison (--compare): against c1's frozen engine record (tools/attn_meta_parent.json engine_selfrun, sha-pinned), byte for byte
on the canonical trace: every connected case c1's record holds, in its own part. With the attn-meta switch unset nothing moves, so
attn-meta off == c1 exactly.

Negative (--negative): the candidate harness over the regenerated c1 overlay and sources (build_attn_meta's OUT/parent,
byte-identical to c1's): ATTN_SUITES run in full; exactly the pinned tests in tools/attn_meta_negative_pins.json do not pass, each
with its pinned outcome; c1's parent cases and C1's own (c1's PARENT_PASS + CASES) pass with c1's frozen traces. Two-phase, bounded
and fail-closed exactly as c1's negative (its judge, wall guard and INC rule; --observe once, OBSERVED rc 1; the pins are written
from that one observation after review).
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_engine_selfrun_b as c1b  # noqa: E402  (installs c1's and alpha-next's lines, writer resets, dense line)

base = sys.modules['alpha_next_c1_engine_selfrun']
SCHEMA = 'attn-meta-selfrun/1'
ATTN_SUITES = ('test_attn_meta_control', 'test_attn_meta_c1_g1', 'test_attn_meta_box', 'test_attn_meta_queue')
NEGATIVE_SUITES = ATTN_SUITES
PARTS = base.PARTS + ('attn',)
PINS = Path(__file__).resolve().with_name('attn_meta_negative_pins.json')
PARENT_PASS = list(base.PARENT_PASS) + list(base.CASES)
SWAPS = (
    ("           run_overlap(bundle, out / 'overlap', name + '-overlap')]\n",
     "           run_overlap(bundle, out / 'overlap', name + '-overlap'),\n"
     "           run_attn(bundle, out / 'attn', name + '-attn')]\n"),
    ("    p, d, f, g, s, n, o = parts\n", "    p, d, f, g, s, n, o, t = parts\n"),
    ("overlap_suites=list(OVERLAP_SUITES),", "overlap_suites=list(OVERLAP_SUITES), attn_suites=list(ATTN_SUITES),"),
)
NEGATIVE_SWAPS = (
    ("record = dict(schema='alpha-next-c1-negative/1', verdict=verdict,\n                  negative_control=True, "
     "negative_kind='alpha-next-parent',",
     "record = dict(schema='attn-meta-negative/1', verdict=verdict,\n                  negative_control=True, "
     "negative_kind='c1-parent',"),
)
COMPARE_SWAPS = (
    ("result = dict(schema='alpha-next-c1-default-off-comparison/1',", "result = dict(schema='attn-meta-default-off-comparison/1',"),
    ("meaning='with the C1 switch unset every alpha-next connected case has alpha-next\\'s exact canonical trace; with the '\n"
     "                          'switch installed and the arm on serial or local, each of those modes equals alpha-next\\'s alpha-c0 "
     "case of '\n                          'the same mode')",
     "meaning='with the attn-meta switch unset every c1 connected case has c1\\'s exact canonical trace')"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin():
    import attn_meta_parent_pin as parent_pin
    return parent_pin.load()


def frozen():
    """c1's frozen engine record, read from the parent file and pinned there."""
    pins = _pin()
    assert 'engine_selfrun' in pins, 'the parent pin has no engine self-run (a development parent cannot be compared)'
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'c1 engine record changed'
    return place


def part_of(mode):
    return base.part_of(mode)


def comparison_rules():
    """(part, mode, their part, against): every c1 connected case in its own part."""
    return [(part_of(m), m, part_of(m), m) for m in base.expected_cases()]


def expected_cases():
    """The connected cases the merged record holds: c1's (attn-meta adds none). The one source for the spec's runner swap."""
    return base.expected_cases()


def run_attn(bundle, out, name):
    """The eighth container: ATTN_SUITES, no connected case; its own record in OUT/attn/result.json."""
    step_a = base.step_a
    rc, result, cases, c, wall = step_a.container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, (),
                                                  ATTN_SUITES)
    reported = {test.split('.', 1)[0] for test in result.get('outcomes') or {}}
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and (result.get('tests') or 0) > 0 \
        and base.an.writer_resets(result) is not None and not cases and reported == set(ATTN_SUITES)
    record = dict(schema='attn-meta-attn-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'}, suite_names=list(ATTN_SUITES),
                  suites_reported=sorted(reported),
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  receipt_writer_resets=base.an.writer_resets(result), cases={},
                  manifest=sha(bundle / 'SHA256SUMS'), image=base.an.IMAGE, network='none', gpu_device_requests=0,
                  gpu_execution=False, box_action=False, container=name, wall_s=wall, command=c,
                  log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'))
    return 0 if ok else 1


def derive(text, swaps, what):
    for old, new in swaps:
        assert text.count(old) == 1, '%s changed: %s' % (what, old[:70])
        text = text.replace(old, new)
    return text


RUN_TEXT = derive(c1b.RUN_TEXT, SWAPS, 'c1 run()')
NEGATIVE_TEXT = derive(inspect.getsource(base.negative), NEGATIVE_SWAPS, 'c1 negative()')
COMPARE_TEXT = derive(inspect.getsource(base.compare), COMPARE_SWAPS, 'c1 compare()')
JUDGE_TEXT = inspect.getsource(base.judge)
NEGATIVE_JUDGEMENT_TEXT = inspect.getsource(base.negative_judgement)
# c1's globals, copied (nothing of c1's module is mutated), with attn-meta's names over them.
_ns = dict(base.__dict__, connected_records=c1b.connected_records, SCHEMA=SCHEMA, PARTS=PARTS, ATTN_SUITES=ATTN_SUITES,
           run_attn=run_attn, NEGATIVE_SUITES=NEGATIVE_SUITES, PINS=PINS, CASES=(), PARENT_PASS=PARENT_PASS, frozen=frozen,
           comparison_rules=comparison_rules, expected_cases=expected_cases)
# c1's run judges its connected cases against c1's own CASES (sorted(connected) == expected_cases(), C1's pairs): those keep
# c1's names in run(); CASES=() above is the negative's (attn-meta pins no probe case of its own).
_run_ns = dict(_ns, CASES=base.CASES)
exec(compile(RUN_TEXT, 'attn_meta_engine_selfrun:run', 'exec'), _run_ns)
for name, text in (('judge', JUDGE_TEXT), ('negative_judgement', NEGATIVE_JUDGEMENT_TEXT), ('negative', NEGATIVE_TEXT),
                   ('compare', COMPARE_TEXT)):
    exec(compile(text, 'attn_meta_engine_selfrun:' + name, 'exec'), _ns)
run = _run_ns['run']
judge = _ns['judge']
negative = _ns['negative']
compare = _ns['compare']

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path)
    p.add_argument('--tree', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name')
    p.add_argument('--overlay', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--negative', action='store_true')
    p.add_argument('--observe', action='store_true', help='negative only: record the outcomes, verdict OBSERVED, rc 1 (INC: rc 4)')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
