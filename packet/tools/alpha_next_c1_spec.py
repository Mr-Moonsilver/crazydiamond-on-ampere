#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-next-c1.json for HEAD: declarations computed from the tree, for a reviewer to commit.

tools/alpha_next_spec.py's generation over the alpha-next-c1 tool family. source_paths and definitions come from build_native43.propose
against the pinned alpha-next build source (through tools/build_alpha_next_c1.py); engine_changed and overlay_definitions from a
scratch overlay regeneration against alpha-next's overlay (tools/alpha_next_c1_overlay_propose.py), which must add exactly C1's
e3/stage_overlap.py and remove none. stage_overlap.py's bytes are never pinned here: they are read from HEAD each time. The runner
swaps are taken from alpha-next's own spec rows (tools/native43_spec.alpha-next.json, the text alpha-next's runner holds, asserted to
be the parent build's spec by sha) and this engine self-run's case list. The queue settings in the details are read from the queue's
own settings text. The source commit is HEAD (no pinned commit: the coordinator merges this branch), so run it with harness/ and
patches/ complete and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_alpha_next_c1 import NEW_OVERLAY, PINS, TOOLS_REPLACED, b  # noqa: E402
import alpha_next_c1_engine_selfrun as selfrun  # noqa: E402
import alpha_next_c1_overlay_propose as overlay  # noqa: E402
import alpha_next_engine_selfrun as parent_selfrun  # noqa: E402

NAME = 'native43_spec.alpha-next-c1.json'
PARENT_SPEC_PATH = TOOLS / 'native43_spec.alpha-next.json'
assert b.sha(PARENT_SPEC_PATH) == PINS['spec_sha256'] and Path(PINS['spec']).name == PARENT_SPEC_PATH.name, \
    'tools/native43_spec.alpha-next.json is not the pinned alpha-next build\'s spec'
STEP_B = PARENT = json.loads(PARENT_SPEC_PATH.read_text())   # STEP_B: the parent spec, the name the inherited code reads
QUEUE = ['q18a_alpha_next_c1.draft.sh', 'q18a_alpha_next_c1_declare.sh', 'q18a_alpha_next_c1_settings.sh']
SETTING_KEYS = ('FLASH_S5_AUTO_ARM', 'FLASH_S5_NATIVE_PROGRESS', 'FLASH_S5_NATIVE_QSA_POLICY', 'FLASH_S5_QSA_FUSED',
                'FLASH_S5_LAYER_GROUPS', 'FLASH_S5_STAGE_LOCAL', 'FLASH_S5_HANDLER_ANATOMY', 'FLASH_S5_UNION_RETIRE',
                'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER', 'FLASH_S5_STAGE_OVERLAP')


def _parent_swap(marker):
    """The text alpha-next's runner holds for one of its swaps: the new side of its spec's row, the one source."""
    rows = [row[1] for row in PARENT['runner_swaps'] if marker in row[1]]
    assert len(rows) == 1, marker
    return rows[0]


PARENT_CASES_TEXT = _parent_swap("sorted(engine.get('cases') or {})==")
assert PARENT_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(parent_selfrun.expected_cases()), \
    'alpha-next runner case list differs from its sources'
PARENT_QUEUES_TEXT = _parent_swap("result.get('alpha_next_queue')=='PASS'")
assert PARENT_QUEUES_TEXT.endswith(" and result.get('alpha_next_queue')=='PASS'"), PARENT_QUEUES_TEXT
RUNNER_SWAPS = [
    [_parent_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
    [PARENT_QUEUES_TEXT, PARENT_QUEUES_TEXT + " and result.get('alpha_next_c1_queue')=='PASS'"],
    [PARENT_CASES_TEXT, "sorted(engine.get('cases') or {})==" + repr(selfrun.expected_cases())],
]
HEADER = [
    'alpha-next-c1 packaged by tools/build_alpha_next_c1.py (spec alpha-next-c1); PREV is the alpha-next pre-VM build bundle and runner.',
    'Engine: claude/alpha-next-c1, alpha-next with C1 (FLASH_S5_STAGE_OVERLAP: two chunks in flight on the staged driver, one added '
    'module) composed, default off behind its switch; C1-off-is-alpha-next and both-off proofs in alpha-next-c1-freeze.json; the '
    'overlay files moved or added vs PREV are recomputed by the conf tool.',
    'Queue: alpha-next\'s startup and cleanup with FLASH_S5_STAGE_OVERLAP=1 added; C1 off (C0 local) and on (overlap) arms, matched by '
    'the typed native_stage_local selection with its overlap field; no production start.',
]
# Hub-side files the gate chain writes after the build: never bundled, so writing them cannot move the bundle it pins
# (tools/alpha_next_c1_queue_pins_write.py proves the same over each build OUT).
HUB_SIDE = ('tools/alpha_next_c1_queue_pins.json',)
FOCUSED_DELTA = ['test_e3_stage_overlap', 'test_alpha_next_c1_composed_fields', 'test_e3_stage_overlap_control',
                 'test_c1_rebind_witness', 'test_c1_rebind_on', 'test_alpha_next_c1_defer_control']


def queue_settings():
    """The declared switches as the queue's settings text sets them in EXTRA_ENV (None: not set there)."""
    text = (b.ROOT / 'harness/runs' / QUEUE[2]).read_text()
    lines = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', text, re.M)
    assert len(lines) == 1, 'one EXTRA_ENV line in ' + QUEUE[2]
    env = dict(tok.split('=', 1) for tok in lines[0].split() if '=' in tok)
    return {k: env.get(k) for k in SETTING_KEYS}


def expected_tests_from_sources():
    """The bundle self-run's test count: alpha-next's spec count plus the alpha-next-c1 queue suite's, equal to the committed derived
    CPU probe's EXPECTED_TESTS literal (the probe is regenerated and committed first; tools/alpha_next_c1_spec_regen.sh does both)."""
    import alpha_next_c1_tools_derive as derive
    assert derive.parent_tests() == PARENT['expected_tests'], (derive.parent_tests(), PARENT['expected_tests'])
    want = derive.expected()
    probe = (TOOLS / 'alpha_next_c1_cpu_probe.py').read_text()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', probe, re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


def queue_landed():
    return all((b.ROOT / 'harness/runs' / name).is_file() for name in QUEUE)


def compiled(paths, tools):
    """alpha-next's compiled list plus every added or modified python file under harness/, tools/ (one level) and patches/s5_engine/,
    each present where the dense container mounts it: harness/ is the bundle, patches/s5_engine/ its e3_builder, tools/ the bundle's
    native35_tools (the parent's tools, the spec's tools and the replaced ones)."""
    out = list(STEP_B['engine_selfrun']['compiled'])
    bundled = {p.name for p in (b.PARENT / 'native35_tools').glob('*.py')} | set(tools) | set(TOOLS_REPLACED)
    for row in paths:
        status, path = row.split('\t', 1)
        if status not in 'AM' or not path.endswith('.py') or path in out:
            continue
        if (path.startswith('harness/') and path.count('/') == 1) or (path.startswith('patches/s5_engine/') and path.count('/') == 2):
            out.append(path)
        elif path.startswith('tools/') and path.count('/') == 1:
            assert path[len('tools/'):] in bundled, ('compiled tool not bundled', path)
            out.append(path)
    for path in out:
        assert path.startswith(('harness/', 'tools/', 'patches/s5_engine/')), path
        if path.startswith('tools/'):
            assert path[len('tools/'):] in bundled, ('compiled tool not bundled', path)
    return out


def generate(scratch, expected_tests, pre_queue=False):
    if pre_queue:
        # Before the queue harness lands: everything the tree declares now, with the queue-derived fields left null and said so.
        # Such a spec cannot build (its checks name tools/alpha_next_c1_queue.py; build_native43 asserts the queue texts).
        assert not queue_landed(), 'the queue harness is here: generate the full spec'
        assert expected_tests is None, '--expected-tests comes from the queue suite; not with --pre-queue'
    else:
        want = expected_tests_from_sources()
        expected_tests = want if expected_tests is None else expected_tests
        assert expected_tests == want, (expected_tests, want)
    head = b.git('rev-parse', 'HEAD').strip()
    spec = dict(source_commit=head, tools=[])
    proposed = b.propose(spec)
    paths = proposed['source_paths']
    me = 'A\ttools/' + NAME
    if me not in paths:
        paths = sorted(paths + [me])
    # The gate chain's prepare-review commit always moves the review pin after this spec is written (Step B's rule, alpha-next's).
    review = 'M\t' + b.REVIEW_PATH
    if review not in paths:
        paths = sorted(paths + [review])
    for name in TOOLS_REPLACED:
        assert 'M\ttools/' + name in paths, ('replaced tool not an M row', name)
    tools = sorted(row.split('\t', 1)[1][len('tools/'):] for row in paths
                   if row.startswith('A\ttools/') and row.count('/') == 1 and row.endswith(('.py', '.json'))
                   and row.split('\t', 1)[1] not in HUB_SIDE)
    assert not set(HUB_SIDE) & {'tools/' + n for n in tools}, 'a hub-side file would be bundled'
    moved = overlay.propose(scratch)
    assert moved['added'] == sorted(NEW_OVERLAY) and not moved['removed'], moved
    dense = STEP_B['engine_selfrun']
    settings = {k: None for k in SETTING_KEYS} if pre_queue else queue_settings()
    assert pre_queue or (settings['FLASH_S5_STAGE_LOCAL'] == '1' and settings['FLASH_S5_STAGE_OVERLAP'] == '1'), settings
    pending = dict(pending='PRE-QUEUE: the queue harness has not landed; expected_tests and alpha_next_c1_settings are null and are '
                           'filled by regenerating this spec after it lands (tools/alpha_next_c1_spec_regen.sh); this spec cannot '
                           'build') if pre_queue else {}
    return dict(
        schema='native43-packaging-spec/1', label='alpha-next-c1', dry_run=False, source_commit=head,
        candidate='q18a-alpha-next-c1', arm='TRUNK', review='prepare', review_key='alpha_next_c1',
        details=dict(
            model_settings_changed=False, alpha_next_c1_settings=settings, **pending,
            queue='q18a-alpha-next-c1: alpha-next\'s queue with the overlap switch installed; C1 off (C0 local) against on (overlap) '
                  'arms over two bands, then a C1-on read-only tail, the arm table the queue\'s own; validity, engagement (early '
                  'open) and generated tokens gated on the core, speed, decode and memory reported; relative box budget only',
            scope='C1 composed onto alpha-next against alpha-next\'s pre-VM build; not a release assessment',
            engine_delta=dict(focused_suites=FOCUSED_DELTA, lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools, queue=QUEUE,
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-next-c1-selfrun-dense-part/1', compiled=compiled(paths, tools),
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_next_c1_cpu_probe.py'), clock_split=STEP_B['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_next_c1_queue.py', '--check'], ['tools/alpha_next_c1_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    p.add_argument('--expected-tests', type=int, help='asserted against the sources (default: taken from them)')
    p.add_argument('--pre-queue', action='store_true', help='the queue harness has not landed: write the spec with its fields null')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests, a.pre_queue), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
