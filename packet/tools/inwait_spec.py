#!/usr/bin/env python3
"""Write tools/native43_spec.inwait.json for HEAD: declarations computed from the tree, for a reviewer to commit.

tools/attn_meta_spec_c.py's generation by name over the q18a-inwait tool family. source_paths and definitions come from build_native43.propose against the pinned c1b build source (through
tools/build_inwait.py); engine_changed and overlay_definitions from a scratch overlay regeneration against c1b's overlay (tools/inwait_overlay_propose.py), which must add exactly e3/native_fifo.py and remove nothing. The
runner swaps are taken from c1b's own spec rows (tools/native43_spec.alpha-next-c1b.json, asserted to be the pinned c1b build's spec by sha): the engine record schema (inwait-selfrun/1) and the queue
receipt (inwait_queue PASS beside c1b's); the connected case list is c1's unchanged (inwait adds no probe mode), so it needs no swap and is asserted equal. The queue settings are read from the
queue's own settings text. The source commit is HEAD, so run it with harness/ and patches/ complete and committed. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_inwait import NEW_OVERLAY, PINS, TOOLS_REPLACED, b  # noqa: E402
import inwait_engine_selfrun as selfrun  # noqa: E402
import inwait_overlay_propose as overlay  # noqa: E402
import alpha_next_c1b_engine_selfrun as parent_selfrun  # noqa: E402

NAME = 'native43_spec.inwait.json'
PARENT_SPEC_PATH = TOOLS / 'native43_spec.alpha-next-c1b.json'
assert b.sha(PARENT_SPEC_PATH) == PINS['spec_sha256'] and Path(PINS['spec']).name == PARENT_SPEC_PATH.name, \
    'tools/native43_spec.alpha-next-c1b.json is not the pinned c1b build\'s spec'
STEP_B = PARENT = json.loads(PARENT_SPEC_PATH.read_text())   # STEP_B: the parent spec, the name the inherited code reads
QUEUE = ['q18a_inwait.draft.sh', 'q18a_inwait_declare.sh', 'q18a_inwait_settings.sh']
SETTING_KEYS = ('FLASH_S5_AUTO_ARM', 'FLASH_S5_NATIVE_PROGRESS', 'FLASH_S5_NATIVE_QSA_POLICY', 'FLASH_S5_QSA_FUSED',
                'FLASH_S5_LAYER_GROUPS', 'FLASH_S5_STAGE_LOCAL', 'FLASH_S5_HANDLER_ANATOMY', 'FLASH_S5_UNION_RETIRE',
                'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER', 'FLASH_S5_STAGE_OVERLAP',
                'FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE', 'FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE', 'FLASH_ATTN_META_GDN_BATCH', 'FLASH_S5_INWAIT')


def _parent_swap(marker):
    """The text c1's runner holds for one of its swaps: the new side of its spec's row, the one source."""
    rows = [row[1] for row in PARENT['runner_swaps'] if marker in row[1]]
    assert len(rows) == 1, marker
    return rows[0]


# c1b's spec has no case-list row (its runner keeps c1's list): the text is c1's spec row.
PARENT_CASES_TEXT = [row[1] for row in json.loads((TOOLS / 'native43_spec.alpha-next-c1.json').read_text())['runner_swaps']
                     if "sorted(engine.get('cases') or {})==" in row[1]][0]
assert PARENT_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(parent_selfrun.expected_cases()), \
    'c1 runner case list differs from its sources'
assert PARENT_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(selfrun.expected_cases()), \
    'inwait adds a connected case: the case list would need a swap'
PARENT_QUEUES_TEXT = _parent_swap("result.get('alpha_next_c1b_queue')=='PASS'")
assert PARENT_QUEUES_TEXT.endswith(" and result.get('alpha_next_c1b_queue')=='PASS'"), PARENT_QUEUES_TEXT
RUNNER_SWAPS = [
    [_parent_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
    [PARENT_QUEUES_TEXT, PARENT_QUEUES_TEXT + " and result.get('inwait_queue')=='PASS'"],
    # Decode-phase build-3: one launch one value, tools/sm_split_queue.py's rows on claude/sm-split. Read once at the runner's start, refused before any action.
    ["QNAME='q18a-inwait'", "SMSPLIT_DECODE_SMS=os.environ.get('SMSPLIT_DECODE_SMS','44')\n"
                            "if not re.fullmatch('[2-6][02468]',SMSPLIT_DECODE_SMS) or not 24<=int(SMSPLIT_DECODE_SMS)<=60:\n"
                            "    raise SystemExit('SMSPLIT_DECODE_SMS must be even, 24..60')\n"
                            "CHUNK_TOKENS=os.environ.get('CHUNK_TOKENS','1024')\n"
                            "if CHUNK_TOKENS not in ('1024','1536','2048'):\n"
                            "    raise SystemExit('CHUNK_TOKENS must be 1024, 1536 or 2048')\n"
                            "QNAME='q18a-ple-private-'+CHUNK_TOKENS"],
    ["OVERLAY=Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-inwait')", "OVERLAY=Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-'+QNAME)"],
    ["S5_REQUESTED_DECODE_SMS='44'", "S5_REQUESTED_DECODE_SMS=SMSPLIT_DECODE_SMS,S5_REQUESTED_CHUNK_TOKENS=CHUNK_TOKENS"],
]
HEADER = [
    'q18a-inwait packaged by tools/build_inwait.py (spec inwait); PREV is the alpha-next-c1b READY build bundle and runner. Discovery mode: strict and focused logs are the parent tree\'s.',
    'Engine: claude/inwait, compose-2 build-2 with native handoffs in the stock wait (FLASH_S5_INWAIT, default off, selected per arm by typed control).',
    'Queue: compose-2\'s startup and cleanup with FLASH_S5_INWAIT=1 beside its three flags for the whole run; one decode-only baseline, six core arms, C1 defer, attn-meta and L1 + L2 in every arm: in-wait off and on; band 1 live x2, solo x1 each; no fixed-schedule check.',
]
# Hub-side files the gate chain writes after the build: never bundled, so writing them cannot move the bundle it pins
# (tools/alpha_next_c1_queue_pins_write.py proves the same over each build OUT).
HUB_SIDE = ('tools/inwait_queue_pins.json', 'tools/inwait_negative_pins.json')
FOCUSED_DELTA = selfrun.ENGINE_SUITES + ['test_inwait_box', 'test_inwait_queue', 'test_inwait_robust_swap', 'test_native43_engine_death_h', 'test_alpha_next_stop_race_h',
                                         'test_e3_c1_open_rows']


def queue_settings():
    """The declared switches as the queue's settings text sets them in EXTRA_ENV (None: not set there)."""
    text = (b.ROOT / 'harness/runs' / QUEUE[2]).read_text()
    lines = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', text, re.M)
    assert len(lines) == 1, 'one EXTRA_ENV line in ' + QUEUE[2]
    env = dict(tok.split('=', 1) for tok in lines[0].split() if '=' in tok)
    return {k: env.get(k) for k in SETTING_KEYS}


def expected_tests_from_sources():
    """The bundle self-run's test count: c1b's spec count plus the inwait queue suite's, equal to the committed derived CPU
    probe's EXPECTED_TESTS literal (the probe is regenerated and committed first)."""
    import inwait_tools_derive as derive
    assert derive.parent_tests() == PARENT['expected_tests'], (derive.parent_tests(), PARENT['expected_tests'])
    want = derive.expected()
    probe = (TOOLS / 'inwait_cpu_probe.py').read_text()
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
        # Such a spec cannot build (build_native43 asserts the queue texts).
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
    assert pre_queue or all(settings[k] == '1' for k in ('FLASH_S5_STAGE_LOCAL', 'FLASH_S5_STAGE_OVERLAP',
                                                         'FLASH_S5_STAGE_OVERLAP_DEFER_BIND', 'FLASH_S5_STAGE_OVERLAP_OBSERVE',
                                                         'FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE', 'FLASH_ATTN_META_GDN_BATCH', 'FLASH_S5_INWAIT')), settings
    pending = dict(pending='PRE-QUEUE: the queue harness has not landed; expected_tests and inwait_settings are null; this '
                           'spec cannot build') if pre_queue else {}
    return dict(
        schema='native43-packaging-spec/1', label='inwait', dry_run=False, source_commit=head,
        candidate='q18a-inwait', arm='TRUNK', review='prepare', review_key='inwait',
        details=dict(
            model_settings_changed=False, inwait_settings=settings, **pending,
            queue='q18a-inwait: compose-2\'s queue with FLASH_S5_INWAIT=1 for the whole run; one decode-only baseline, six core arms, C1 defer, in-wait selected per arm while paused idle FULL; compose-2\'s grade plus the in-wait selection and engagement check',
            scope='in-wait handoffs on compose-2\'s engine against c1b\'s READY build, discovery mode; not a release assessment and not a numerics result',
            engine_delta=dict(focused_suites=FOCUSED_DELTA, lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools, queue=QUEUE,
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='inwait-selfrun-dense-part/1', compiled=compiled(paths, tools),
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='inwait_cpu_probe.py'), clock_split=STEP_B['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/inwait_queue.py', '--check'], ['tools/inwait_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    p.add_argument('--expected-tests', type=int, help='asserted against the sources (default: taken from them)')
    p.add_argument('--pre-queue', action='store_true', help='the queue harness has not landed: write the spec with its fields null')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests, a.pre_queue), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
