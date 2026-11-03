#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-next-c1b.json for HEAD: declarations computed from the tree, for a reviewer to commit.

tools/alpha_next_c1_spec.py's generation over the alpha-next-c1b tool family. source_paths and definitions come from
build_native43.propose against the pinned alpha-next-c1 build source (through tools/build_alpha_next_c1b.py). engine_changed and
overlay_definitions come from a scratch overlay regeneration against c1's overlay (tools/alpha_next_c1b_overlay_propose.py,
--simulate-prepare), which must add and remove nothing, move no engine row, and move metadata only in review-bound keys: they are
asserted empty here (c1b is harness-only), never taken from the tree. The runner swaps come from c1's own spec rows
(tools/native43_spec.alpha-next-c1.json, the text c1's runner holds, asserted to be the parent build's spec by sha): the engine
record schema (alpha-next-c1b-selfrun/1) and the c1b queue receipt beside c1's; the case list is c1's, unchanged (asserted equal to
alpha_next_c1b_engine_selfrun.expected_cases()). The queue settings in the details are read from the queue's own settings text.
The source commit is HEAD, so run it with harness/ and patches/ complete and committed. Only tools/ may change after it.
--pre-queue: before the coordinator's queue harness lands; the queue-derived fields are null and the spec cannot build.
No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_alpha_next_c1b import NEW_OVERLAY, PINS, TOOLS_REPLACED, b  # noqa: E402
import alpha_next_c1b_engine_selfrun as selfrun  # noqa: E402
import alpha_next_c1b_overlay_propose as overlay  # noqa: E402

NAME = 'native43_spec.alpha-next-c1b.json'
PARENT_SPEC_PATH = TOOLS / 'native43_spec.alpha-next-c1.json'
assert b.sha(PARENT_SPEC_PATH) == PINS['spec_sha256'] and Path(PINS['spec']).name == PARENT_SPEC_PATH.name, \
    'tools/native43_spec.alpha-next-c1.json is not the pinned alpha-next-c1 build\'s spec'
STEP_B = PARENT = json.loads(PARENT_SPEC_PATH.read_text())   # STEP_B: the parent spec, the name the inherited code reads
QUEUE = ['q18a_alpha_next_c1b.draft.sh', 'q18a_alpha_next_c1b_declare.sh', 'q18a_alpha_next_c1b_settings.sh']
SETTING_KEYS = ('FLASH_S5_AUTO_ARM', 'FLASH_S5_NATIVE_PROGRESS', 'FLASH_S5_NATIVE_QSA_POLICY', 'FLASH_S5_QSA_FUSED',
                'FLASH_S5_LAYER_GROUPS', 'FLASH_S5_STAGE_LOCAL', 'FLASH_S5_HANDLER_ANATOMY', 'FLASH_S5_UNION_RETIRE',
                'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER', 'FLASH_S5_STAGE_OVERLAP',
                'FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE')


def _parent_swap(marker):
    """The text c1's runner holds for one of its swaps: the new side of its spec's row, the one source."""
    rows = [row[1] for row in PARENT['runner_swaps'] if marker in row[1]]
    assert len(rows) == 1, marker
    return rows[0]


PARENT_CASES_TEXT = _parent_swap("sorted(engine.get('cases') or {})==")
assert PARENT_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(selfrun.expected_cases()), \
    'c1 runner case list differs from the c1b engine self-run\'s (they must be the same 65 cases)'
PARENT_SCHEMA_TEXT = _parent_swap("engine.get('schema')==")
assert PARENT_SCHEMA_TEXT == "engine.get('schema')=='alpha-next-c1-selfrun/1'", PARENT_SCHEMA_TEXT
PARENT_QUEUES_TEXT = _parent_swap("result.get('alpha_next_c1_queue')=='PASS'")
assert PARENT_QUEUES_TEXT.endswith(" and result.get('alpha_next_c1_queue')=='PASS'"), PARENT_QUEUES_TEXT
RUNNER_SWAPS = [
    [PARENT_SCHEMA_TEXT, "engine.get('schema')==" + repr(selfrun.SCHEMA)],
    [PARENT_QUEUES_TEXT, PARENT_QUEUES_TEXT + " and result.get('alpha_next_c1b_queue')=='PASS'"],
]
HEADER = [
    'alpha-next-c1b packaged by tools/build_alpha_next_c1b.py (spec alpha-next-c1b); PREV is the alpha-next-c1 delivered build bundle '
    'and runner.',
    'Engine: alpha-next-c1\'s, unchanged (harness-only package): no engine overlay row moved; the overlay metadata moved only through '
    'the re-prepared source review (alpha-next-c1b-freeze.json); the overlay files moved vs PREV are recomputed by the conf tool.',
    'Queue: q18a-alpha-next-c1b, the coordinator\'s c1b queue over c1\'s engine; no production start.',
]
# Hub-side files the gate chain writes after the build: never bundled, so writing them cannot move the bundle it pins.
HUB_SIDE = ('tools/alpha_next_c1b_queue_pins.json',)
FOCUSED_DELTA = ['test_e3_c1_open_rows', 'test_alpha_next_c1b_queue', 'test_alpha_next_c1_send_gate',
                 'test_alpha_next_c1_defer_control_b']


def queue_settings():
    """The declared switches as the queue's settings text sets them in EXTRA_ENV (None: not set there)."""
    text = (b.ROOT / 'harness/runs' / QUEUE[2]).read_text()
    lines = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', text, re.M)
    assert len(lines) == 1, 'one EXTRA_ENV line in ' + QUEUE[2]
    env = dict(tok.split('=', 1) for tok in lines[0].split() if '=' in tok)
    return {k: env.get(k) for k in SETTING_KEYS}


def expected_tests_from_sources():
    """The bundle self-run's test count: c1's spec count plus the c1b queue suite's, equal to the committed derived CPU probe's
    EXPECTED_TESTS literal (the probe is regenerated and committed first; tools/alpha_next_c1b_spec_regen.sh does both)."""
    import alpha_next_c1b_tools_derive as derive
    assert derive.parent_tests() == PARENT['expected_tests'], (derive.parent_tests(), PARENT['expected_tests'])
    want = derive.expected()
    probe = (TOOLS / 'alpha_next_c1b_cpu_probe.py').read_text()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', probe, re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


def queue_landed():
    return all((b.ROOT / 'harness/runs' / name).is_file() for name in QUEUE)


def compiled(paths, tools):
    """c1's compiled list plus every added or modified python file under harness/, tools/ (one level) and patches/s5_engine/, each
    present where the dense container mounts it (harness/ the bundle, patches/s5_engine/ its e3_builder, tools/ native35_tools)."""
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
        assert not queue_landed(), 'the queue harness is here: generate the full spec'
        assert expected_tests is None, '--expected-tests comes from the queue suite; not with --pre-queue'
    else:
        want = expected_tests_from_sources()
        expected_tests = want if expected_tests is None else expected_tests
        assert expected_tests == want, (expected_tests, want)
    head = b.git('rev-parse', 'HEAD').strip()
    proposed = b.propose(dict(source_commit=head, tools=[]))
    paths = proposed['source_paths']
    me = 'A\ttools/' + NAME
    if me not in paths:
        paths = sorted(paths + [me])
    review = 'M\t' + b.REVIEW_PATH          # the chain's prepare-review commit always moves the review pin after this spec
    if review not in paths:
        paths = sorted(paths + [review])
    for name in TOOLS_REPLACED:
        assert 'M\ttools/' + name in paths, ('replaced tool not an M row', name)
    engine_rows = [r for r in paths if r.startswith(('A\tpatches/s5_engine/', 'M\tpatches/s5_engine/', 'D\tpatches/s5_engine/'))
                   and r != review]
    assert not engine_rows, ('engine source rows in a harness-only package', engine_rows)
    tools = sorted(row.split('\t', 1)[1][len('tools/'):] for row in paths
                   if row.startswith('A\ttools/') and row.count('/') == 1 and row.endswith(('.py', '.json'))
                   and row.split('\t', 1)[1] not in HUB_SIDE)
    assert not set(HUB_SIDE) & {'tools/' + n for n in tools}, 'a hub-side file would be bundled'
    moved = overlay.propose(scratch, simulate_prepare=True)
    assert moved['added'] == sorted(NEW_OVERLAY) == [] and not moved['removed'], moved
    assert moved['engine_changed'] == [] and moved['overlay_definitions'] == {}, ('engine overlay rows moved', moved)
    assert not moved['metadata_review_only_problems'], moved['metadata_review_only_problems']
    dense = STEP_B['engine_selfrun']
    settings = {k: None for k in SETTING_KEYS} if pre_queue else queue_settings()
    pending = dict(pending='PRE-QUEUE: the queue harness has not landed; expected_tests and alpha_next_c1b_settings are null and are '
                           'filled by regenerating this spec after it lands (tools/alpha_next_c1b_spec_regen.sh); this spec cannot '
                           'build') if pre_queue else {}
    return dict(
        schema='native43-packaging-spec/1', label='alpha-next-c1b', dry_run=False, source_commit=head,
        candidate='q18a-alpha-next-c1b', arm='TRUNK', review='prepare', review_key='alpha_next_c1b',
        details=dict(
            model_settings_changed=False, alpha_next_c1b_settings=settings, **pending,
            queue='q18a-alpha-next-c1b: the c1b queue (harness-only changes over alpha-next-c1) on c1\'s unchanged engine; the arm '
                  'table and gates are the queue\'s own; relative box budget only',
            scope='harness-only package over alpha-next-c1\'s delivered build; engine bytes unchanged; not a release assessment',
            engine_delta=dict(focused_suites=FOCUSED_DELTA, lineage_rows=[], engine_changed=[],
                              overlay_metadata_moved=moved['moved'])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=[], overlay_definitions={}, tools=tools,
        queue=QUEUE, expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-next-c1b-selfrun-dense-part/1', compiled=compiled(paths, tools),
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_next_c1b_cpu_probe.py'), clock_split=STEP_B['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_next_c1b_queue.py', '--check'], ['tools/alpha_next_c1b_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    p.add_argument('--expected-tests', type=int, help='asserted against the sources (default: taken from them)')
    p.add_argument('--pre-queue', action='store_true', help='the queue harness has not landed: write the spec with its fields null')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests, a.pre_queue), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
