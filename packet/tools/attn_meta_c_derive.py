#!/usr/bin/env python3
"""Derive the q18a-attn-meta chain c family (lead 3 Nov ~20:00): each tool below is the _b tool (chain b ran it; never edited) with the
declared exact-count swaps. Written as real files so a reviewer can diff them; --check verifies them;
harness/test_attn_meta_engine_selfrun_c.py runs it. No GPU, no box action.

What chain c changes, all of it:
  * the engine self-run is tools/attn_meta_engine_selfrun_c.py (the negative's list/tuple defect fixed), the pin writer reads it;
  * the spec is tools/native43_spec.attn-meta-c.json (the b spec stays as chain b built from it);
  * wait_clear: the docker-name wait only, every wait logged; the clearance file is gone (lead: chain b's release file no longer applies);
  * the carry rule: the build's manifest is compared with chain b's e447b8103f2a...; only if sha256-equal do chain b's bundle self-run,
    engine self-run and comparison carry; any other manifest runs them in full. engine-negative-observe always runs.
Every other step, its order and its bounds are chain b's text.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from attn_meta_b_derive import CONT_REFUSE, swap  # noqa: E402

CARRY_MANIFEST = 'e447b8103f2a6eefed073087d26a0f668412645a2374444c147241f862bf22cd'
CARRY_OUT = '<hub-dir>/agents/flash-next/tmp/attn-meta/build-b'
HEADER = '# derived by tools/attn_meta_c_derive.py from tools/%s (chain b ran it; never edited; superseded by name for chain c)\n'
TABLE = {
    'attn_meta_negative_pins_write_c.py': ('attn_meta_negative_pins_write_b.py', (
        ('import attn_meta_engine_selfrun_b as selfrun', 'import attn_meta_engine_selfrun_c as selfrun', 1),
    )),
    'attn_meta_spec_c.py': ('attn_meta_spec_b.py', (
        ('import attn_meta_engine_selfrun_b as selfrun', 'import attn_meta_engine_selfrun_c as selfrun', 1),
        ("NAME = 'native43_spec.attn-meta-b.json'", "NAME = 'native43_spec.attn-meta-c.json'", 1),
    )),
}
CARRY_BLOCK = '''# Lead 3 Nov ~20:00, item 3: chain b's three records carry by manifest identity, only if byte-identical; otherwise they run in full.
CARRY_MANIFEST=%s
CARRY_OUT=%s
# >>> carry_check (harness/test_attn_meta_engine_selfrun_c.py runs this block)
CARRY=
SELFRUN=$OUT/selfrun
ENGINE=$OUT/engine-selfrun
ACTUAL=$(sha256sum "$HELD/SHA256SUMS" | cut -d' ' -f1)
if [[ $PIN == "$CARRY_MANIFEST" && $ACTUAL == "$CARRY_MANIFEST" ]]; then
  CARRY=1
  SELFRUN=$CARRY_OUT/selfrun
  ENGINE=$CARRY_OUT/engine-selfrun
  printf 'carry-by-manifest-identity manifest %%s selfrun %%s engine %%s comparison %%s\\n' "$ACTUAL" \\
      "$(sha256sum "$SELFRUN/result.json" | cut -d' ' -f1)" "$(sha256sum "$ENGINE/result.json" | cut -d' ' -f1)" \\
      "$(sha256sum "$CARRY_OUT/engine-comparison.json" | cut -d' ' -f1)" >>"$STEPS"
else
  printf 'carry-check differs: build %%s, on disk %%s, chain b %%s; bundle-selfrun, engine-selfrun and engine-comparison run in full\\n' \\
      "$PIN" "$ACTUAL" "$CARRY_MANIFEST" >>"$STEPS"
fi
# <<< carry_check

''' % (CARRY_MANIFEST, CARRY_OUT)
BUNDLE_RUN = '''wait_clear bundle-selfrun
python3 -B tools/native43_bundle_selfrun.py --spec "$SPEC" --bundle "$HELD" --out "$OUT/selfrun" \\
    --name "attn-meta-hub-cpu-$P12" --runner "$RUNNER" >"$OUT/selfrun-stdout.json" 2>&1
step bundle-selfrun $?
'''
ENGINE_DEF = '''engine() { python3 -B tools/attn_meta_engine_selfrun_c.py --spec "$SPEC" --tree "$WT" --bundle "$HELD" --out "$OUT/$1" \\
    --name "attn-meta-$2-cpu-$P12" >"$OUT/$1-stdout.log" 2>&1; }
'''
ENGINE_RUN = '''ENGINE=$OUT/engine-selfrun
wait_clear engine-selfrun
engine engine-selfrun engine; RC=$?
if [[ $RC != 0 ]] && flaky engine "$ENGINE"; then
  note_rc engine-selfrun-1 "$RC"; ENGINE=$OUT/engine-selfrun-2; wait_clear engine-selfrun-2; engine engine-selfrun-2 engine-2; step engine-selfrun-2 $?
else step engine-selfrun "$RC"; fi

python3 -B tools/attn_meta_engine_selfrun_c.py --compare "$ENGINE" --out "$OUT/engine-comparison.json" \\
    >"$OUT/engine-comparison-stdout.log" 2>&1
step engine-comparison $?
'''
GATE_SWAPS = (
    ('tools/attn_meta_engine_selfrun_b.py', 'tools/attn_meta_engine_selfrun_c.py', 5),
    ('WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6 CLEAR=$7\n', 'WT=$1 SPEC=$2 OUT=$3 EVID=$4 TAG=$5 NOTE=$6\n', 1),
    ('[[ ! -e $STEPS && ! -e $OUT && -f $SPEC && -n $CLEAR ]] || exit 3\n', '[[ ! -e $STEPS && ! -e $OUT && -f $SPEC ]] || exit 3\n', 1),
    ('    [[ -e $CLEAR ]] || names="${names}hold:no-clearance-file"\n', '', 1),
    ("whose name contains alpha-next-c1, append-yield or returning-append (a failing docker ps counts as not clear) AND the\n"
     "#     clearance file CLEAR (argument 7) exists: the lead's hold on all hub container steps (3 Nov 13:0x) is lifted only on the\n"
     "#     lead's word, and the file records it. waiting-before / clear-before lines (no rc field) go to the steps file.\n",
     "whose name contains alpha-next-c1, append-yield or returning-append (a failing docker ps counts as not clear). No\n"
     "#     clearance file in chain c (lead 3 Nov ~20:00). waiting-before / clear-before lines (no rc field) go to the steps file.\n", 1),
    ('# usage: attn_meta_gates.sh WT SPEC OUT EVID TAG NOTE CLEAR\n', '# usage: attn_meta_gates_c.sh WT SPEC OUT EVID TAG NOTE\n', 1),
    ('RUNNER=$OUT/q18a-attn-meta-owner-run-$P12.py\n\n' + BUNDLE_RUN,
     'RUNNER=$OUT/q18a-attn-meta-owner-run-$P12.py\n\n' + CARRY_BLOCK + 'if [[ -z $CARRY ]]; then\n' + BUNDLE_RUN + 'fi\n', 1),
    (ENGINE_RUN, 'if [[ -z $CARRY ]]; then\n' + ENGINE_RUN + 'fi\n', 1),
    ('$OUT/selfrun/result.json', '$SELFRUN/result.json', 2),
)
GATE_NOTE = ('# attn_meta_gates_c.sh: derived by tools/attn_meta_c_derive.py from tools/attn_meta_gates_b.sh (chain b ran it; never edited): the\n'
             '# fixed engine self-run, the docker-name wait with no clearance file, the carry check. Everything below is that chain.\n')
CONT_HEAD = '''#!/usr/bin/env bash
# attn_meta_gates_c_cont.sh: the continuation of attn_meta_gates_c.sh after its negative observation stop (rc 5), derived by
# tools/attn_meta_c_derive.py: the same preamble (wait_clear, step), then the chain's text from the engine negative to its stop line,
# unchanged. Refuses unless the steps file ends at 'engine-negative-awaiting-pins rc=5', the pins are filled and committed (hub-side,
# so the bundle did not move: the build's manifest is re-checked) and the tree is clean. Same usage, same steps file; appended to it.
'''
CONT_ENGINE = ('ENGINE=$OUT/engine-selfrun\n'
               'grep -qE \' engine-selfrun-2 rc=0$\' "$STEPS" && ENGINE=$OUT/engine-selfrun-2\n')
CONT_CARRY = ('SELFRUN=$OUT/selfrun\n' + CONT_ENGINE +
              'if grep -q \'^carry-by-manifest-identity \' "$STEPS"; then SELFRUN=%s/selfrun; ENGINE=%s/engine-selfrun; fi\n' % (CARRY_OUT, CARRY_OUT))


def gates():
    text = swap((TOOLS / 'attn_meta_gates_b.sh').read_text(), GATE_SWAPS, 'gates')
    first, rest = text.split('\n', 1)
    # The _b note is the second and third line; chain c's note goes above it, the rest stays the record of what it derives from.
    return first + '\n' + GATE_NOTE + rest


def continuation():
    text = gates()
    start, stop = text.index('set -u\n'), text.index('cd "$WT" || exit 3\n')
    preamble = text[start:stop].replace('[[ ! -e $STEPS && ! -e $OUT && -f $SPEC ]] || exit 3\n', '')
    assert preamble.count('printf') >= 2 and '$STEPS' in preamble and '$CLEAR' not in preamble
    tail = text[text.index('wait_clear engine-negative\n'):]
    assert CONT_REFUSE.count(CONT_ENGINE) == 1
    return CONT_HEAD + preamble + CONT_REFUSE.replace(CONT_ENGINE, CONT_CARRY) + tail


def generate():
    out = {}
    for name, (source, swaps) in TABLE.items():
        text = swap((TOOLS / source).read_text(), swaps, source)
        first, rest = text.split('\n', 1)
        out[name] = first + '\n' + HEADER % source + rest
    out['attn_meta_gates_c.sh'] = gates()
    out['attn_meta_gates_c_cont.sh'] = continuation()
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
    print('attn-meta chain c tools ' + ('verified' if check else 'written'))
