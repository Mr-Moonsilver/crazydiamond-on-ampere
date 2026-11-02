#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-next.json for HEAD: declarations computed from the tree, for a reviewer to commit.

tools/step_b_spec.py's generation, with C0's additions (tools/alpha_c0_spec.py: the overlay modules the regeneration adds). source_paths
and definitions come from build_native43.propose against Step B's build source 1a29d84f (through tools/build_alpha_next.py);
engine_changed and overlay_definitions from a scratch overlay regeneration against Step B's frozen overlay
(tools/alpha_next_overlay_propose.py), which must add exactly the three declared modules and remove none. The runner swaps are taken
from Step B's own spec rows (the text Step B's runner holds) and the alpha-next engine self-run's case list. The queue settings in
the details are read from the queue's own settings text. The source commit is HEAD, so run it with harness/ and patches/ complete
and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_alpha_next import NEW_OVERLAY, TOOLS_REPLACED, b  # noqa: E402
import alpha_next_engine_selfrun as selfrun  # noqa: E402
import alpha_next_overlay_propose as overlay  # noqa: E402
import step_b_engine_selfrun as step_b_selfrun  # noqa: E402

NAME = 'native43_spec.alpha-next.json'
STEP_B = json.loads((TOOLS / 'native43_spec.step-b.json').read_text())
QUEUE = ['q18a_alpha_next.draft.sh', 'q18a_alpha_next_declare.sh', 'q18a_alpha_next_settings.sh']
SETTING_KEYS = ('FLASH_S5_AUTO_ARM', 'FLASH_S5_NATIVE_PROGRESS', 'FLASH_S5_NATIVE_QSA_POLICY', 'FLASH_S5_QSA_FUSED',
                'FLASH_S5_LAYER_GROUPS', 'FLASH_S5_STAGE_LOCAL', 'FLASH_S5_HANDLER_ANATOMY', 'FLASH_S5_UNION_RETIRE',
                'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER')


def _step_b_swap(marker):
    """The text Step B's runner holds for one of its swaps: the new side of its spec's row, the one source."""
    rows = [row[1] for row in STEP_B['runner_swaps'] if marker in row[1]]
    assert len(rows) == 1, marker
    return rows[0]


STEP_B_CASES_TEXT = _step_b_swap("sorted(engine.get('cases') or {})==")
assert STEP_B_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(step_b_selfrun.expected_cases()), \
    'Step B runner case list differs from its sources'
STEP_B_QUEUES_TEXT = "result.get('step_a_queue')=='PASS' and result.get('step_b_queue')=='PASS'"
assert STEP_B_QUEUES_TEXT in _step_b_swap("result.get('step_b_queue')=='PASS'")
RUNNER_SWAPS = [
    [_step_b_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
    [STEP_B_QUEUES_TEXT, STEP_B_QUEUES_TEXT + " and result.get('alpha_next_queue')=='PASS'"],
    [STEP_B_CASES_TEXT, "sorted(engine.get('cases') or {})==" + repr(selfrun.expected_cases())],
]
HEADER = [
    'alpha-next packaged by tools/build_alpha_next.py (spec alpha-next); PREV is the Step B build e bundle and runner.',
    'Engine: claude/alpha-next, the stop-fix trunk with held-g (progress anatomy, lean replies), Step B (with Step A), C0 (staged '
    'driver, two added modules) and the stage-1 handler anatomy (one added module) composed, each default off behind its switch; '
    'composition and default-off proofs in alpha-next-freeze.json; the overlay files moved or added vs PREV are recomputed by the '
    'conf tool.',
    'Queue: Step B\'s startup and cleanup with FLASH_S5_STAGE_LOCAL=1 added; core ABBA C0 on/off arms at size 2 over both bands, then '
    'the size 4 C0-on tail (reads only), matched by the typed native_stage_local and native_layer_group selections; no production start.',
]
# Hub-side files the gate chain writes after the build: never bundled, so writing them cannot move the bundle it pins
# (tools/alpha_next_queue_pins_write.py proves the same over each build OUT).
HUB_SIDE = ('tools/alpha_next_queue_pins.json',)
FOCUSED_DELTA = ['test_e3_stage_local', 'test_e3_stage_local_control', 'test_e3_stage_local_qsa', 'test_stage1_handler_anatomy',
                 'test_e3_native_anatomy', 'test_e3_native_lean_replies', 'test_alpha_next_stopfix_matrix', 'test_alpha_next_c0_ring',
                 'test_alpha_next_peak_receipt', 'test_alpha_next_receipts', 'test_alpha_next_receipts_mutants',
                 'test_alpha_next_stage1_slow_rank', 'test_alpha_next_path_qualification', 'test_alpha_next_probe_launch',
                 'test_alpha_next_wall_guard']


def queue_settings():
    """The declared switches as the queue's settings text sets them in EXTRA_ENV (None: not set there)."""
    text = (b.ROOT / 'harness/runs' / QUEUE[2]).read_text()
    lines = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', text, re.M)
    assert len(lines) == 1, 'one EXTRA_ENV line in ' + QUEUE[2]
    env = dict(tok.split('=', 1) for tok in lines[0].split() if '=' in tok)
    return {k: env.get(k) for k in SETTING_KEYS}


def expected_tests_from_sources():
    """The bundle self-run's test count: Step B's spec count plus the alpha-next queue suite's, equal to the committed derived CPU
    probe's EXPECTED_TESTS literal (the probe is regenerated and committed first; tools/alpha_next_spec_regen.sh does both)."""
    import alpha_next_tools_derive as derive
    assert derive.PARENT_TESTS == STEP_B['expected_tests'], (derive.PARENT_TESTS, STEP_B['expected_tests'])
    want = derive.expected()
    probe = (TOOLS / 'alpha_next_cpu_probe.py').read_text()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', probe, re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


def queue_landed():
    return all((b.ROOT / 'harness/runs' / name).is_file() for name in QUEUE)


def compiled(paths, tools):
    """Step B's compiled list plus every added or modified python file under harness/, tools/ (one level) and patches/s5_engine/,
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
        # Such a spec cannot build (its checks name tools/alpha_next_queue.py; build_native43 asserts the queue texts).
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
    # The gate chain's prepare-review commit always moves the review pin after this spec is written (Step B's rule).
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
    assert pre_queue or (settings['FLASH_S5_STAGE_LOCAL'] == '1' and settings['FLASH_S5_LAYER_GROUPS'] == '1'), settings
    pending = dict(pending='PRE-QUEUE: the queue harness has not landed; expected_tests and alpha_next_settings are null and are '
                           'filled by regenerating this spec after it lands (tools/alpha_next_spec_regen.sh); this spec cannot '
                           'build') if pre_queue else {}
    return dict(
        schema='native43-packaging-spec/1', label='alpha-next', dry_run=False, source_commit=head,
        candidate='q18a-alpha-next', arm='TRUNK', review='prepare', review_key='alpha_next',
        details=dict(
            model_settings_changed=False, alpha_next_settings=settings, **pending,
            queue='q18a-alpha-next: Step B\'s queue with the stage-local switch; core ABBA C0 on/off at size 2 over two bands, then '
                  'the size 4 C0-on tail as reads; validity, engagement and generated tokens gated on the core, speed, decode and '
                  'memory reported; 9,000 s box budget relative to the run\'s start',
            scope='the composed candidate (held-g, Step A/B, C0, stage-1 anatomy on the stop-fix trunk) against Step B; not a '
                  'release assessment',
            engine_delta=dict(focused_suites=FOCUSED_DELTA, lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools, queue=QUEUE,
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-next-selfrun-dense-part/1', compiled=compiled(paths, tools),
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_next_cpu_probe.py'), clock_split=STEP_B['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_next_queue.py', '--check'], ['tools/alpha_next_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    p.add_argument('--expected-tests', type=int, help='asserted against the sources (default: taken from them)')
    p.add_argument('--pre-queue', action='store_true', help='the queue harness has not landed: write the spec with its fields null')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests, a.pre_queue), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
