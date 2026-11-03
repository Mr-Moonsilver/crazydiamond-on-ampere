#!/usr/bin/env python3
"""Derive the q18a-attn-meta c1b-base tool family: each tool below is the chain-a tool (never edited; superseded by name) with the declared
exact-count swaps of TABLE. Lead 3 Nov 16:49: the parent is alpha-next-c1b b2's READY build (tools/attn_meta_parent_b.json), the box and
queue are the _b ones, the death rule the _h one, the negative pins hub-side so that pinning them does not move the bundle. Written as real
files so a reviewer can diff them; --check verifies them; harness/test_attn_meta_pack.py runs it. No GPU, no box action.
Not derived, unchanged on purpose: attn_meta_default_off (rows onto fbb9123e; c1b moved no engine byte), attn_meta_base_proof and
harness/attn_meta_robust_swap (re-derived on this tree 3 Nov 17:00: the same four ids fail by construction, nothing else), attn_meta_review_pins,
attn_meta_staging_check, attn_meta_queue_pins_write, attn_meta_selfrun5.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PIN = ('import attn_meta_parent_pin as parent_pin', 'import attn_meta_parent_pin_b as parent_pin', 1)
HEADER = ('derived by tools/attn_meta_b_derive.py from tools/%s (never edited; superseded by name for the c1b base)\n')
TABLE = {
    'build_attn_meta_b.py': ('build_attn_meta.py', (
        PIN,
        ("PARENT_BUILD['name'] == 'q18a-alpha-next-c1'", "PARENT_BUILD['name'] == 'q18a-alpha-next-c1b'", 1),
        ("b.PARENT_CANDIDATE = 'q18a-alpha-next-c1'", "b.PARENT_CANDIDATE = 'q18a-alpha-next-c1b'", 1),
        ("b.PARENT_STEM = 'q18a_alpha_next_c1'", "b.PARENT_STEM = 'q18a_alpha_next_c1b'", 1),
        ('import attn_meta_freeze as freeze', 'import attn_meta_freeze_b as freeze', 1),
    )),
    'attn_meta_freeze_b.py': ('attn_meta_freeze.py', (
        PIN,
        ("if (record.get('source') or {}).get('tip') != PARENT_SOURCE:", "if (record.get('engine_sources') or {}).get('source') != PARENT_SOURCE:", 1),
        ("proved_commit=(record.get('source') or {}).get('tip')", "proved_commit=(record.get('engine_sources') or {}).get('source')", 1),
    )),
    'attn_meta_overlay_propose_b.py': ('attn_meta_overlay_propose.py', (
        ('from build_attn_meta import b', 'from build_attn_meta_b import b', 1),
    )),
    'attn_meta_overlay_tamper_b.py': ('attn_meta_overlay_tamper.py', (
        ('import attn_meta_freeze as freeze', 'import attn_meta_freeze_b as freeze', 1),
        ("over tools/attn_meta_freeze.py'", "over tools/attn_meta_freeze_b.py'", 1),
    )),
    'attn_meta_negative_pins_write_b.py': ('attn_meta_negative_pins_write.py', (
        ('import attn_meta_engine_selfrun as selfrun', 'import attn_meta_engine_selfrun_b as selfrun', 1),
    )),
    'attn_meta_precheck_conf_b.py': ('attn_meta_precheck_conf.py', (
        PIN,
        ("prev['name'] == 'q18a-alpha-next-c1'", "prev['name'] == 'q18a-alpha-next-c1b'", 1),
        ('import alpha_next_c1_precheck_conf as c1', 'import alpha_next_c1b_precheck_conf as c1', 1),
        ("SEAMS = ('test_attn_meta_queue.py e3_attn_meta_box.py ",
         "SEAMS = ('native43_engine_death_h.py alpha_next_stop_race_h.py test_native43_engine_death_h.py test_alpha_next_stop_race_h.py '\n"
         "         'e3_attn_meta_box_b.py test_attn_meta_box_b.py test_attn_meta_queue.py e3_attn_meta_box.py ", 1),
        ('# attn-meta on the composed engine. PREV = exact alpha-next-c1 (%s)', '# attn-meta on the composed engine. PREV = exact alpha-next-c1b (%s)', 1),
    )),
    'attn_meta_tools_derive_b.py': ('attn_meta_tools_derive.py', (
        ("PARENT_CANDIDATE = 'q18a-alpha-next-c1'", "PARENT_CANDIDATE = 'q18a-alpha-next-c1b'", 1),
        ("PARENT_PROBE = 'alpha_next_c1_cpu_probe.py'", "PARENT_PROBE = 'alpha_next_c1b_cpu_probe.py'", 1),
        ('import attn_meta_parent_pin as parent_pin', 'import attn_meta_parent_pin_b as parent_pin', 1),
        ("pattern='test_alpha_next_c1_queue.py'))\\n\"\n    return (", "pattern='test_alpha_next_c1b_queue.py'))\\n\"\n    return (", 1),
        ('the %d alpha-next-c1 probe cases', 'the %d alpha-next-c1b probe cases', 1),
        ("('import alpha_next_c1_gate\\nactive_and_cleanup=alpha_next_c1_gate.ACTIVE_S+alpha_next_c1_gate.CLEANUP_S\\n',\n         'import alpha_next_c1_gate\\nimport attn_meta_gate\\n'",
         "('import alpha_next_c1b_gate\\nactive_and_cleanup=alpha_next_c1b_gate.ACTIVE_S+alpha_next_c1b_gate.CLEANUP_S\\n',\n         'import alpha_next_c1b_gate\\nimport attn_meta_gate\\n'", 1),
        ("(\"+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS\\n\",\n         \"+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS+",
         "(\"+sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS\\n\",\n         \"+sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS+", 1),
        ("/source/harness/runs/q18a_alpha_next_c1_settings.sh", "/source/harness/runs/q18a_alpha_next_c1b_settings.sh", 1),
        ("(\"record=dict(alpha_next_c1_queue='PASS',\",", "(\"record=dict(alpha_next_c1b_queue='PASS',\",", 1),
        ("\"alpha_next_c1_queue='PASS',\"),", "\"alpha_next_c1b_queue='PASS',\"),", 1),
        ("OUTPUTS = {'attn_meta_cpu_probe.py':", "OUTPUTS = {'attn_meta_cpu_probe_b.py':", 1),
    )),
    'attn_meta_spec_b.py': ('attn_meta_spec.py', (
        ('from build_attn_meta import', 'from build_attn_meta_b import', 1),
        ('import attn_meta_engine_selfrun as selfrun', 'import attn_meta_engine_selfrun_b as selfrun', 1),
        ('import attn_meta_overlay_propose as overlay', 'import attn_meta_overlay_propose_b as overlay', 1),
        ('import alpha_next_c1_engine_selfrun as parent_selfrun', 'import alpha_next_c1b_engine_selfrun as parent_selfrun', 1),
        ("NAME = 'native43_spec.attn-meta.json'", "NAME = 'native43_spec.attn-meta-b.json'", 1),
        ("PARENT_SPEC_PATH = TOOLS / 'native43_spec.alpha-next-c1.json'", "PARENT_SPEC_PATH = TOOLS / 'native43_spec.alpha-next-c1b.json'", 1),
        ("'tools/native43_spec.alpha-next-c1.json is not the pinned c1 build\\'s spec'", "'tools/native43_spec.alpha-next-c1b.json is not the pinned c1b build\\'s spec'", 1),
        ("PARENT_CASES_TEXT = _parent_swap(\"sorted(engine.get('cases') or {})==\")",
         "# c1b's spec has no case-list row (its runner keeps c1's list): the text is c1's spec row.\n"
         "PARENT_CASES_TEXT = [row[1] for row in json.loads((TOOLS / 'native43_spec.alpha-next-c1.json').read_text())['runner_swaps']\n"
         "                     if \"sorted(engine.get('cases') or {})==\" in row[1]][0]", 1),
        ("PARENT_QUEUES_TEXT = _parent_swap(\"result.get('alpha_next_c1_queue')=='PASS'\")", "PARENT_QUEUES_TEXT = _parent_swap(\"result.get('alpha_next_c1b_queue')=='PASS'\")", 1),
        ("PARENT_QUEUES_TEXT.endswith(\" and result.get('alpha_next_c1_queue')=='PASS'\")", "PARENT_QUEUES_TEXT.endswith(\" and result.get('alpha_next_c1b_queue')=='PASS'\")", 1),
        ("PREV is the alpha-next-c1 delivered build bundle and runner.", "PREV is the alpha-next-c1b READY build bundle and runner.", 1),
        ("HUB_SIDE = ('tools/attn_meta_queue_pins.json',)", "HUB_SIDE = ('tools/attn_meta_queue_pins.json', 'tools/attn_meta_negative_pins.json')", 1),
        ("'test_attn_meta_default_off', 'test_attn_meta_robust_swap']", "'test_attn_meta_default_off', 'test_attn_meta_robust_swap', 'test_attn_meta_box_b',\n                 'test_native43_engine_death_h', 'test_alpha_next_stop_race_h', 'test_e3_c1_open_rows']", 1),
        ("import attn_meta_tools_derive as derive", "import attn_meta_tools_derive_b as derive", 1),
        ("probe = (TOOLS / 'attn_meta_cpu_probe.py').read_text()", "probe = (TOOLS / 'attn_meta_cpu_probe_b.py').read_text()", 1),
        ("bundle_selfrun=dict(arm='TRUNK', probe='attn_meta_cpu_probe.py')", "bundle_selfrun=dict(arm='TRUNK', probe='attn_meta_cpu_probe_b.py')", 1),
        ("checks=[['tools/attn_meta_queue.py', '--check'], ['tools/attn_meta_tools_derive.py', '--check']]",
         "checks=[['tools/attn_meta_queue_b.py', '--check'], ['tools/attn_meta_tools_derive_b.py', '--check']]", 1),
    )),
}
FOCUSED_OLD = '    test_attn_meta_queue test_attn_meta_box test_attn_meta_control'
FOCUSED_NEW = ('    test_attn_meta_box_b test_native43_engine_death_h test_alpha_next_stop_race_h test_e3_c1_open_rows test_alpha_next_c1b_queue '
               'test_alpha_next_c1_send_gate test_alpha_next_c1_defer_control_b \\\n' + FOCUSED_OLD)
GATE_SWAPS = (
    ('tools/attn_meta_parent_pin.py --pinned', 'tools/attn_meta_parent_pin_b.py --pinned', 1),
    ('unset ATTN_META_PARENT ALPHA_NEXT_C1_PARENT', 'unset ATTN_META_PARENT ATTN_META_PARENT_B ALPHA_NEXT_C1_PARENT', 1),
    (FOCUSED_OLD, FOCUSED_NEW, 1),
    ('attn_meta_robust_swap.py --base-receipt "$BASE_RECEIPT" test_alpha_next_composition test_alpha_next_c1_pack 2>&1',
     'attn_meta_robust_swap.py --base-receipt "$BASE_RECEIPT" test_alpha_next_composition test_alpha_next_c1_pack test_alpha_next_c1b_pack 2>&1', 1),
    ('tools/build_attn_meta.py', 'tools/build_attn_meta_b.py', 2),
    ('tools/attn_meta_engine_selfrun.py', 'tools/attn_meta_engine_selfrun_b.py', 5),
    ('tools/attn_meta_overlay_tamper.py', 'tools/attn_meta_overlay_tamper_b.py', 1),
    ('tools/attn_meta_clock_scan.py', 'tools/attn_meta_clock_scan_b.py', 1),
    ('tools/attn_meta_precheck_conf.py', 'tools/attn_meta_precheck_conf_b.py', 1),
)
GATE_NOTE = ('# attn_meta_gates_b.sh: derived by tools/attn_meta_b_derive.py from tools/attn_meta_gates.sh (chain a ran it; never edited): the\n'
             '# c1b-base family, the parent c1b b2\'s READY build, focused plus the c1b and _h suites. Everything below is that chain.\n')
CONT_HEAD = '''#!/usr/bin/env bash
# attn_meta_gates_b_cont.sh: the continuation of attn_meta_gates_b.sh after its negative observation stop (rc 5), derived by
# tools/attn_meta_b_derive.py: the same preamble (wait_clear, step), then the chain's text from the engine negative to its stop line,
# unchanged. Refuses unless the steps file ends at 'engine-negative-awaiting-pins rc=5', the pins are filled and committed (hub-side,
# so the bundle did not move: the build's manifest is re-checked) and the tree is clean. Same usage, same steps file; appended to it.
'''
CONT_REFUSE = '''cd "$WT" || exit 3
[[ -f $STEPS && -f $OUT/build.json ]] || exit 3
[[ $(grep -E ' rc=[0-9]+$' "$STEPS" | tail -n 1 | cut -d' ' -f2-) == 'engine-negative-awaiting-pins rc=5' ]] || { echo 'continuation refused: not at the observation stop' >>"$STEPS"; exit 3; }
[[ -z $(git status --porcelain -- harness tools patches) ]] || { echo 'continuation refused: dirty' >>"$STEPS"; exit 3; }
python3 -I -c "import json,sys;sys.exit(1 if str(json.load(open(sys.argv[1]))['status']).startswith('AWAITING-OBSERVATION') else 0)" tools/attn_meta_negative_pins.json \\
    || { echo 'continuation refused: pins not filled' >>"$STEPS"; exit 3; }
HEAD=$(git rev-parse HEAD)
printf 'continuation head %s\\n' "$HEAD" >>"$STEPS"
PIN=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'])" "$OUT/build.json")
P12=${PIN:0:12}
HELD=$OUT/q18a-attn-meta-bundle-$P12
RUNNER=$OUT/q18a-attn-meta-owner-run-$P12.py
[[ $(sha256sum "$HELD/SHA256SUMS" | cut -d' ' -f1) == "$PIN" ]] || { echo 'continuation refused: the bundle moved' >>"$STEPS"; exit 3; }
ENGINE=$OUT/engine-selfrun
grep -qE ' engine-selfrun-2 rc=0$' "$STEPS" && ENGINE=$OUT/engine-selfrun-2
'''


def swap(text, swaps, what):
    for old, new, count in swaps:
        assert text.count(old) == count, (what, old[:90], text.count(old), count)
        text = text.replace(old, new)
    return text


def gates():
    text = swap((TOOLS / 'attn_meta_gates.sh').read_text(), GATE_SWAPS, 'gates')
    first, rest = text.split('\n', 1)
    return first + '\n' + GATE_NOTE + rest


def continuation():
    text = gates()
    start, stop = text.index('set -u\n'), text.index('cd "$WT" || exit 3\n')
    preamble = text[start:stop].replace('[[ ! -e $STEPS && ! -e $OUT && -f $SPEC && -n $CLEAR ]] || exit 3\n', '')
    assert preamble.count('printf') >= 2 and '$STEPS' in preamble
    tail = text[text.index('wait_clear engine-negative\n'):]
    return CONT_HEAD + preamble + CONT_REFUSE + tail


def generate():
    out = {}
    for name, (source, swaps) in TABLE.items():
        text = swap((TOOLS / source).read_text(), swaps, source)
        first, rest = text.split('\n', 1)
        out[name] = first + '\n# ' + HEADER % source + rest
    out['attn_meta_gates_b.sh'] = gates()
    out['attn_meta_gates_b_cont.sh'] = continuation()
    return out


if __name__ == '__main__':
    check = '--check' in sys.argv
    for name, text in generate().items():
        path = TOOLS / name
        if check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o775 if name.endswith('.sh') else 0o664)
    print('attn-meta c1b-base tools ' + ('verified' if check else 'written'))
