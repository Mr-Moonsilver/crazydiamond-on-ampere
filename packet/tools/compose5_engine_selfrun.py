#!/usr/bin/env python3
"""q18a-compose-throughput-b24 engine self-run, tools/inwait_engine_selfrun.py with this family's names and suites (discovery mode runs no --compare and no --negative): attn-meta's fixed form by name (claude/attn-meta-c1b: tools/attn_meta_engine_selfrun{,_b,_c}.py, flattened here, never edited
there): c1's seven containers unchanged plus one step container, compared against c1b's frozen record. CPU only, no network.

Eight containers over one frozen bundle, all results kept under OUT:
  * progress/, dense/, fused/, groups/, stage/, composed/, overlap/: c1's parts, run by c1's run() text (the _b judge) with three exact-once swaps (SWAPS below: the eighth part appended to
    the run list, unpacked, and its suite names recorded), OVERLAP_SUITES as c1b's; FLASH_S5_LAZY_RECV and FLASH_S5_DECODE_MERGE are in none of their command lines;
  * step/: STEP_SUITES (tools/inwait_engine_suites.json's engine_suites, the builder's; test_inwait_box: the arm, grade, order rules and read; test_inwait_queue: the derived texts,
    under real bash) in one unittest process with per-test outcomes, no connected cases (this family adds no probe mode).
OUT/result.json is the merged record (schema inwait-selfrun/1); its connected cases are exactly c1's expected_cases(). Verdict PASS only when all eight are PASS and c1's pairs and early
open hold (c1's rule, unchanged).

Comparison (--compare): c1b's compare (every case in its own part: both PASS and the canonical trace byte-equal; a case whose frozen trace has no canonical form, the four receipts-* cases, the
same verdict and check count) against c1b build-b2's engine record (tools/inwait_parent.json, sha-pinned): with the switch unset nothing moves.

Negative (--negative): the candidate harness over the regenerated c1b overlay and sources (build_inwait's OUT/parent, byte-identical to c1b's): STEP_SUITES run in full; exactly the
pinned tests in tools/inwait_negative_pins.json do not pass, each with its pinned outcome; c1's parent cases and C1's own pass with the frozen traces. Two-phase, bounded and fail-closed
exactly as c1's negative (its judge, wall guard and INC rule; --observe once, OBSERVED rc 1; the pins are written from that one observation after review). PARENT_PASS is a tuple, the type c1 hands to
the same text (attn-meta chain b's TypeError: harness/test_inwait_pack.py assembles it).
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_engine_selfrun_b as c1_b  # noqa: E402  (installs c1's and alpha-next's lines, writer resets, dense line)
import alpha_next_c1b_engine_selfrun as c1b_run  # noqa: E402

base = sys.modules['alpha_next_c1_engine_selfrun']
SCHEMA = 'compose5-selfrun/1'
HERE = Path(__file__).resolve()
ENGINE_SUITES = list(json.loads(HERE.with_name('compose5_engine_suites.json').read_text())['engine_suites'])
STEP_SUITES = tuple(ENGINE_SUITES) + ('test_compose5_box', 'test_compose5_queue')
NEGATIVE_SUITES = STEP_SUITES
PARTS = base.PARTS + ('step',)
PINS = HERE.with_name('compose5_negative_pins.json')
PARENT_PASS = tuple(base.PARENT_PASS) + tuple(base.CASES)
OVERLAP_SUITES = c1b_run.OVERLAP_SUITES
SWAPS = (
    ("           run_overlap(bundle, out / 'overlap', name + '-overlap')]\n",
     "           run_overlap(bundle, out / 'overlap', name + '-overlap'),\n"
     "           run_step(bundle, out / 'step', name + '-step')]\n"),
    ("    p, d, f, g, s, n, o = parts\n", "    p, d, f, g, s, n, o, t = parts\n"),
    ("overlap_suites=list(OVERLAP_SUITES),", "overlap_suites=list(OVERLAP_SUITES), step_suites=list(STEP_SUITES),"),
)
NEGATIVE_SWAPS = (
    ("record = dict(schema='alpha-next-c1-negative/1', verdict=verdict,\n                  negative_control=True, "
     "negative_kind='alpha-next-parent',",
     "record = dict(schema='inwait-negative/1', verdict=verdict,\n                  negative_control=True, "
     "negative_kind='c1b-parent',"),
)
COMPARE_SWAPS = (
    ("schema='alpha-next-c1b-engine-comparison/1'", "schema='inwait-default-off-comparison/1'"),
    ("meaning='c1b moves no engine byte: every connected case, overlap on included, has alpha-next-c1\\'s exact canonical '",
     "meaning='with the step switches unset every connected case, overlap on included, has alpha-next-c1b\\'s exact canonical '"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen():
    """c1b's frozen engine record, read from the parent file and pinned there."""
    import inwait_parent_pin as parent_pin
    pins = parent_pin.load()
    assert 'engine_selfrun' in pins, 'the parent pin has no engine self-run (a development parent cannot be compared)'
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'c1b engine record changed'
    return place


def part_of(mode):
    return base.part_of(mode)


def comparison_rules():
    """(part, mode, their part, against): every c1 connected case in its own part."""
    return [(part_of(m), m, part_of(m), m) for m in base.expected_cases()]


def expected_cases():
    """The connected cases the merged record holds: c1's (this family adds none). The one source for the spec's runner swap."""
    return base.expected_cases()


def run_step(bundle, out, name):
    """The eighth container: STEP_SUITES, no connected case; its own record in OUT/step/result.json."""
    step_a = base.step_a
    rc, result, cases, c, wall = step_a.container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, (),
                                                  STEP_SUITES)
    reported = {test.split('.', 1)[0] for test in result.get('outcomes') or {}}
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and (result.get('tests') or 0) > 0 \
        and base.an.writer_resets(result) is not None and not cases and reported == set(STEP_SUITES)
    record = dict(schema='compose5-part-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'}, suite_names=list(STEP_SUITES),
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


RUN_TEXT = derive(c1_b.RUN_TEXT, SWAPS, 'c1 run()')
NEGATIVE_TEXT = derive(inspect.getsource(base.negative), NEGATIVE_SWAPS, 'c1 negative()')
COMPARE_TEXT = derive(inspect.getsource(c1b_run.compare), COMPARE_SWAPS, 'c1b compare()')
JUDGE_TEXT = inspect.getsource(base.judge)
NEGATIVE_JUDGEMENT_TEXT = inspect.getsource(base.negative_judgement)
# c1's globals, copied (nothing of c1's module is mutated), with this family's names over them.
_ns = dict(base.__dict__, connected_records=c1_b.connected_records, SCHEMA=SCHEMA, PARTS=PARTS, STEP_SUITES=STEP_SUITES,
           run_step=run_step, NEGATIVE_SUITES=NEGATIVE_SUITES, PINS=PINS, CASES=(), PARENT_PASS=PARENT_PASS, frozen=frozen,
           comparison_rules=comparison_rules, expected_cases=expected_cases, OVERLAP_SUITES=OVERLAP_SUITES)
# c1's run judges its connected cases against c1's own CASES (sorted(connected) == expected_cases(), C1's pairs): those keep c1's names in
# run(); CASES=() above is the negative's (this family pins no probe case of its own).
_run_ns = dict(_ns, CASES=base.CASES)
exec(compile(RUN_TEXT, 'compose5_engine_selfrun:run', 'exec'), _run_ns)
exec(compile(inspect.getsource(base.run_overlap), 'compose5_engine_selfrun:run_overlap', 'exec'), _run_ns)
for _name, _text in (('judge', JUDGE_TEXT), ('negative_judgement', NEGATIVE_JUDGEMENT_TEXT), ('negative', NEGATIVE_TEXT)):
    exec(compile(_text, 'compose5_engine_selfrun:' + _name, 'exec'), _ns)
_compare_ns = dict(c1b_run.__dict__, frozen=frozen, expected_cases=expected_cases)
exec(compile(COMPARE_TEXT, 'compose5_engine_selfrun:compare', 'exec'), _compare_ns)
run = _run_ns['run']
judge = _ns['judge']
negative = _ns['negative']
compare = _compare_ns['compare']

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
