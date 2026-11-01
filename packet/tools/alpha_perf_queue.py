#!/usr/bin/env python3
"""Derive the q18a-alpha-perf box scripts and band 2 manifest from the committed held-f (6e36495) texts.

Kept from held-f, byte for byte: pins, preflight, startup, bind, host headroom, the unarmed check, the end identity
and the whole owned cleanup/restore tail. Declared deltas, each applied exactly once:
  names   q18a_native41_progress -> q18a_alpha_perf, q18a-native41-progress -> q18a-alpha-perf;
  budget  draft and window read alpha_perf_gate (ACTIVE_S is the 2.5 h box budget, LAST_ARM_ENTRY_S the arm entry rule);
  dense   FLASH_S5_NATIVE_QSA_POLICY=dense in EXTRA_ENV, as l-f3 configures it, asserted in declaration and preflight,
          with no trace and no stop frontier;
  death   the le-fix engine death scan in host_crash_detected and engine_watched (native43_le_fix_queue_deltas D3+D4,
          unchanged), the watch around every arm and every typed FULL, the tool in the preflight list;
  guard   the /tokenize length guard runs on both manifests (w2-length, w2-length-band2);
  m3      alpha_m3_queue_deltas M3 unchanged: after the engine-log step the per-phase allocator receipts over the
          engine-watch records (per arm: allocator retries, device-wide syncs). Receipt only, never a verdict input;
  body    held-f's eight arms unchanged in order and label, then the band 2 arms; the grade is e3_alpha_perf_box.
No model setting, gate threshold, launcher or restore change. --check verifies the committed files.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import native41_progress_queue as held_f  # noqa: E402
import native43_le_fix_queue_deltas as le_fix  # noqa: E402
import alpha_m3_queue_deltas as m3  # noqa: E402

OLD, NEW = 'q18a_native41_progress', 'q18a_alpha_perf'
OLD_Q, NEW_Q = 'q18a-native41-progress', 'q18a-alpha-perf'
DENSE = 'FLASH_S5_NATIVE_QSA_POLICY=dense'
BAND1_MANIFEST = 'workloads/q18a-native41-progress.json'
BAND2_MANIFEST = 'workloads/q18a-alpha-perf-band2.json'
BAND2_PROMPT_TOKENS = 132916            # 131,094 tokens by the length model, residue 150; append 135,085, residue 93
BAND2_STANDING_BUDGET = 32768
ARMS_BAND1 = held_f.ARMS
ARMS_BAND2_REQUIRED = ('burst-solo-2', 'burst-live-2', 'burst-live-3', 'burst-solo-3')
ARMS_BAND2_OPTIONAL = ('burst-solo-4', 'burst-live-4', 'burst-live-5', 'burst-solo-5')
ARMS = ARMS_BAND1 + ARMS_BAND2_REQUIRED + ARMS_BAND2_OPTIONAL
HEADER_NOTE = ('# Alpha-perf: this queue measures prefill speed and decode-during-prefill on the composed alpha engine, '
               'dense QSA policy on\n# (FLASH_S5_NATIVE_QSA_POLICY=dense, as l-f3): held-f\'s eight arms unchanged (band 1), '
               'then burst live and solo with a\n# 131,094-token fresh prompt past the 97,152-row arena (band 2), solo first in '
               'each round so a live failure keeps its solo\n# number. Speed is descriptive. Every cap is alpha_perf_gate.\n#\n')
BODY = r'''# Alpha-perf: band 1 (held-f's eight arms) then band 2 (burst, 131,094 tokens, past the arena). Receipts only; speed is
# descriptive, never a gate. Every timeout below is one alpha_perf_gate cap (instruments, arm_entries, progress_arms,
# ingress, end_full_idle, grade); each arm enters only inside LAST_ARM_ENTRY_S. _cid is the engine container id read
# before the unarmed check. Every engine-facing client runs under the le-fix engine watch.
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
  --out "$Q/instruments-start.json" > "$Q/instruments-start.log" 2>&1
INSTRUMENTS_START=$?
record instruments-start "$INSTRUMENTS_START" 'replay observer library mapped in no rank process (observer and offers off)'
GRADE_ARGS=()
LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)
if [[ $INSTRUMENTS_START == 0 ]]; then
  for _label in current-solo-0 burst-solo-0 burst-live-0 current-live-0 current-live-1 burst-live-1 burst-solo-1 current-solo-1 burst-solo-2 burst-live-2 burst-live-3 burst-solo-3 burst-solo-4 burst-live-4 burst-live-5 burst-solo-5; do
    IFS=- read -r _mode _cohort _pair <<< "$_label"
    if [[ $_pair == [01] ]]; then _workload=$H/workloads/q18a-native41-progress.json
    else _workload=$H/workloads/q18a-alpha-perf-band2.json; fi
    if (( SECONDS > LAST_ARM_ENTRY_S )); then
      if [[ $_pair == [45] ]]; then
        # An optional band 2 round past the budget: declared to the grade, no verdict changes.
        PHASE_VERDICT[arm-$_label]=NOT_RUN; echo "STEP arm-$_label NOT_RUN box budget: elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"
        GRADE_ARGS+=(--not-run "$_label")
      else
        record "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"
      fi
      continue
    fi
    n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
    if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
      record "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
      continue
    fi
    # The typed all-rank FULL takes the manual lease the previous arm released.
    engine_watched "entry-$_label" timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
    _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
    if [[ $_entry_rc != 0 ]]; then
      record "arm-$_label" 2 'NOT_RUN: arm entry not a typed idle FULL'
      continue
    fi
    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \
    PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    engine_watched "arm-$_label" timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_perf_box.py" arm \
      --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --mode "$_mode" --cohort "$_cohort" --index "$_pair" \
      --workload "$_workload" --control-timeout-s 60 \
      --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
    record "arm-$_label" "$?" 'auto arm: 8 standing decoders, native_progress mode, fresh, append; live keeps them, solo cancels'
    GRADE_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
    _span=$(python3 - "$Q/arm-$_label.json" <<'PY_SPAN'
import json,sys
try:d=json.load(open(sys.argv[1]))
except (OSError,ValueError):raise SystemExit(0)
s,e=d.get('wall_start_s'),d.get('wall_end_s')
if type(s) in (int,float) and type(e) in (int,float):print(repr(float(s)),repr(float(e)))
PY_SPAN
)
    if [[ -n $_span ]]; then
      read -r _s _e <<< "$_span"
      timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label "$_label" --start "$_s" --end "$_e" \
        --generation 10 --out "$Q/ingress-$_label.json" > "$Q/ingress-$_label.log" 2>&1
      record "ingress-$_label" "$?" 'access log: every request in the arm span is the arm (8 standing, fresh, append), else INVALID'
    else
      record "ingress-$_label" 2 'arm span missing; ingress not checked'
    fi
    GRADE_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  done
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || ! health_ok; then
    ENGINE_CRASHED=1; record arms-engine 1 'engine unhealthy or auto-restarted; no restart attempted'
  else
    record arms-engine 0 'engine healthy with unchanged NRestarts'
  fi
else
  record arms 2 'NOT_RUN: observer and offers off not proven on every rank process (instruments-start)'
fi
if [[ $INSTRUMENTS_START == 0 ]] && health_ok; then
  timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-start.json" \
    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1
  record instruments-end "$?" 'after the arms: the same rank processes, replay observer mapped in none'
  engine_watched end-full-idle timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/end-full-idle.json" > "$Q/end-full-idle.log" 2>&1
  record end-full-idle "$?" 'typed all-rank FULL at the end: resumed, idle, workers empty'
fi
[[ ${#GRADE_ARGS[@]} -gt 0 ]] || GRADE_ARGS=(--arm "$Q/arm-none.json")
timeout 600s python3 "$H/e3_alpha_perf_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --out "$Q/alpha-perf-grade.json" > "$Q/alpha-perf-grade.log" 2>&1
record alpha-perf "$?" 'per-arm validity (crash-free, 0 disagree), complete progress lines; speed per band descriptive'
'''
GUARD_NEW = '''# Alpha-perf: the fresh/append counts both manifests declare, from this engine's /tokenize, must plan.
timeout 60s python3 "$H/e3_native39_w2_length.py" --url "$PROBE_URL" --model "$MODEL" \\
  --workload "$H/workloads/q18a-native41-progress.json" --out "$Q/w2-length.json" > "$Q/w2-length.log" 2>&1
rc=$?; record w2-length "$rc" 'band 1 fresh/append counts equal /tokenize and both have a native plan'
timeout 60s python3 "$H/e3_native39_w2_length.py" --url "$PROBE_URL" --model "$MODEL" \\
  --workload "$H/workloads/q18a-alpha-perf-band2.json" --out "$Q/w2-length-band2.json" > "$Q/w2-length-band2.log" 2>&1
rc=$?; record w2-length-band2 "$rc" 'band 2 fresh/append counts equal /tokenize and both have a native plan'
'''
BUDGET_OLD = ("from native41_progress_gate import ACTIVE_S,CLEANUP_S,PHASES\n"
              "print(ACTIVE_S if sys.argv[1]=='active' else CLEANUP_S if sys.argv[1]=='cleanup' else PHASES[sys.argv[1]])\n")
BUDGET_NEW = ("from alpha_perf_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n"
              "print(ACTIVE_S if sys.argv[1]=='active' else CLEANUP_S if sys.argv[1]=='cleanup' else "
              "LAST_ARM_ENTRY_S if sys.argv[1]=='last_arm_entry' else PHASES[sys.argv[1]])\n")
NOTE_OLD = '# Held-f: FLASH_S5_NATIVE_PROGRESS=burst below, the only setting '
SETTINGS_NOTE = ('# Alpha-perf: FLASH_S5_NATIVE_QSA_POLICY=dense below, the dense policy as q18a-native43-lifecycle-f configures it '
                 '(no trace,\n# no stop frontier), the only setting added to held-f.\n')
DRAFT_DENSE = ("assert env.get('FLASH_S5_NATIVE_QSA_POLICY') == 'dense', 'alpha-perf dense policy'\n"
               "assert 'FLASH_S5_UNION_RETIRE' not in env, 'alpha-perf: S1 retirement stays off'\n"
               "assert 'FLASH_S5_NATIVE_QSA_TRACE' not in env and 'FLASH_S5_NATIVE_QSA_STOP_AFTER' not in env, "
               "'alpha-perf: no trace and no stop frontier, the prefill must complete'\n")
DECLARE_DENSE = ("assert env.get('FLASH_S5_NATIVE_QSA_POLICY')=='dense', 'alpha-perf runs the dense policy'\n"
                 "assert 'FLASH_S5_UNION_RETIRE' not in env, 'alpha-perf: S1 retirement stays off'\n"
                 "assert 'FLASH_S5_NATIVE_QSA_TRACE' not in env and 'FLASH_S5_NATIVE_QSA_STOP_AFTER' not in env, "
                 "'alpha-perf: no trace and no stop frontier, the prefill must complete'\n")
BAND2_NOTES = ('Alpha-perf band 2: held-f\'s manifest (corpus, seeds, decode streams and prompt, timeouts, reserve, route plan, '
               'claims) with the fresh prompt_tokens set to {p} ({nf} tokens, residue {rf}; about 33 native chunks of 1024 '
               'past the 97,152-row arena on the dense route) and the same 4096-token append suffix ({na} tokens, residue '
               '{ra}), both with a native plan, and the declared standing budget below. length_declaration is derived by '
               'e3_native39_w2_length.declaration and re-derived by the queue guard.')
BAND2_BASIS = ('decode.generation_budget stays at the validated 8192; the arm client sends the eight standing decoders with '
               'this declared budget instead, so all eight stay open through a live band 2 fresh prefill up to about 5 s per '
               'chunk (held-f run 2 decoded 47.5 tok/s per stream). The streams stop at the first token of the append; budget '
               'is not reserved KV. prompt 153537 + 32768 = 186305 tokens per stream, below the 262144 model length.')


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, held_f.BODY, BODY)
            text = replace_once(text, held_f.GUARD_NEW, GUARD_NEW)
            text = replace_once(text, BUDGET_OLD, BUDGET_NEW)
            text = replace_once(text, 'from native41_progress_gate import window_ok', 'from alpha_perf_gate import window_ok')
            text = replace_once(text, '  e3_native41_progress_box.py native41_progress_gate.py; do',
                                '  e3_native41_progress_box.py native41_progress_gate.py \\\n'
                                '  e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py; do')
            text = replace_once(text, '   -r $H/workloads/q18a-native41-progress.json &&',
                                '   -r $H/workloads/q18a-native41-progress.json && -r $H/workloads/q18a-alpha-perf-band2.json &&')
            old = "assert env.get('FLASH_S5_NATIVE_PROGRESS') == 'burst', 'held-f native progress switch'\n"
            text = replace_once(text, old, old + DRAFT_DENSE)
            for a, b in (("perf=p.parent/'native41-progress-grade.json'", "perf=p.parent/'alpha-perf-grade.json'"),
                         ("'missing':['native41-progress-grade-absent']", "'missing':['alpha-perf-grade-absent']")):
                text = replace_once(text, a, b)
            text = replace_once(text, le_fix.D3_OLD, le_fix.D3_NEW)
            text = replace_once(text, m3.M3_OLD, m3.M3_NEW)
            text = replace_once(text, 'Every cap is native41_progress_gate.PHASES.', 'Every cap is alpha_perf_gate.PHASES.')
            first = text.index('# q18a-native41-progress:')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, ' FLASH_S5_AUTO_ARM=1 ' + held_f.SWITCH + ' ', ' FLASH_S5_AUTO_ARM=1 ' + held_f.SWITCH + ' ' + DENSE + ' ')
            text = replace_once(text, NOTE_OLD, SETTINGS_NOTE + NOTE_OLD)
        else:
            old = "assert env.get('FLASH_S5_NATIVE_PROGRESS')=='burst', 'held-f native progress switch must be burst'\n"
            text = replace_once(text, old, old + DECLARE_DENSE)
            old = '      e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py \\\n'
            text = replace_once(text, old, old + '      e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py '
                                                 'e3_native41_progress_box.py native41_progress_gate.py \\\n'
                                                 '      workloads/q18a-native41-progress.json workloads/q18a-alpha-perf-band2.json \\\n')
        text = text.replace(OLD, NEW).replace(OLD_Q, NEW_Q)
        # Keep the unchanged band 1 manifest and arm module names (replace() above renamed nothing else).
        text = text.replace('workloads/q18a-alpha-perf.json', 'workloads/q18a-native41-progress.json')
        out['harness/runs/' + NEW + suffix] = text
    return out


def band2_manifest():
    import e3_native39_w2_length as length
    data = json.loads((ROOT / 'harness' / BAND1_MANIFEST).read_text())
    out = copy.deepcopy(data)
    out['workload'] = 'q18a-alpha-perf-band2'
    out['prefill']['fresh']['prompt_tokens'] = BAND2_PROMPT_TOKENS
    out['progress'] = dict(schema=data['progress']['schema'], standing_generation_budget=BAND2_STANDING_BUDGET,
                           basis=BAND2_BASIS)
    out.pop('length_declaration')
    out['length_declaration'] = length.declaration(out)
    decl = out['length_declaration']
    out['notes'] = BAND2_NOTES.format(p=BAND2_PROMPT_TOKENS, nf=decl['fresh']['tokens'], rf=decl['fresh']['residue'],
                                      na=decl['append']['tokens'], ra=decl['append']['residue'])
    return json.dumps(out, indent=1, sort_keys=False) + '\n'


def derive_all():
    return derive() | {'harness/' + BAND2_MANIFEST: band2_manifest()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    for name, text in derive_all().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-perf queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
