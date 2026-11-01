#!/usr/bin/env python3
"""Generate the q18a-native39-trunk queue from the q18a-native35-p-c texts.

Trunk = P-c under fresh names on the native39 engine (native37 robust + native38
plain bind + native39 P1/P2), for the W1/W2 speed run (lead, 31 Oct 2026).
Every P-c phase stays: native28 startup (the P-c startup step reuses the
native38 startup fence, test_native39_trunk.StartupStep), row1, rule4, row4 with
the P-c capacity continuation, native W1, two W2 rounds.  Declared deltas only:
  1. W2 manifest: workloads/q18a-w2-overlap-native39.json (fresh prompt_tokens
     199934 -> 197270 tokens, both fresh and append plan) in the request plan,
     preflight, rounds and declaration pins; a guard before row1
     (e3_native39_w2_length) counts both prompts with the engine's /tokenize and
     stops the queue if either differs from the declaration or would be refused;
  2. loopback API server: EXTRA_ARGS = the found production value + --host
     127.0.0.1 (last-wins, after the launcher's --host 0.0.0.0, as q18b-w1ab);
     the found value is checked silently at preflight, the declared command's
     last --host at declaration and preflight, and after identity-start the
     captured command's last --host plus the API server's own 'Starting vLLM
     server on http://127.0.0.1:8000' line in the followed log (effective-host);
  3. bind receipt: the restart health loop samples /health every second until
     P-c's 540 s deadline and keeps each request's sent/returned time in
     Q/health.tsv; the first 'flash-s5 startup owner bound' line of the followed
     container log (host observer, run-scoped) must precede the first 200
     request (bind-receipt), and its engine_epoch must be the startup grade's
     (bind-epoch);
  4. plain W1 control: the 24 Oct client (w1_trace.py, byte-identical to
     q18b-w1ab2's) right after the native W1, 8 x 153537, 3 x 20 s, its
     q18b-w1ab2 box corpus (sha256 prefix checked at preflight); then the engine
     must be idle (running + waiting 0) before W2 runs; a trunk summary reads
     W2 beside both W1 baselines (descriptive);
  5. window: native39_trunk_gate (P's derivation plus the added phases' caps),
     outer timeout 16200 s;
  6. Codex's queue review of 1c334bc (four blockers, 1 Nov 2026):
     a. observer and offers off: settings declare FLASH_S5_REPLAY_OBSERVER=0
        and FLASH_S5_ENGINE_OFFERS=0, preflight and declaration assert both;
        row4 (it needs both) is declared NOT_RUN and the gates carry that
        declaration (e3_native39_round reads exactly it for the unchanged
        gate); the actual per-rank state (instruments: the replay observer
        library is mapped in no rank process) gates W1, each W2 round
        (same rank processes as W1) and is re-proved at the end;
     b. sealed cohort: the native W1 is e3_native39_w1 (exact cohort ids and
        no prefill at every entry and barrier; FULL restored on any error),
        and the access log must show only harness requests in every W1
        repeat, the plain W1 and each W2 round (ingress, else INVALID); W2
        needs both W1 ingress receipts;
     c. the plain W1 runs only after a typed all-rank FULL, resumed and idle
        (full-idle), and its route generation is unchanged after it
        (route-check);
     d. each W2 round starts idle; the return-timing reader and the summary
        (plain ratios only from a valid plain W1) close the run;
  plus names and the header comment.  Every replacement must match exactly once.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
P_C, TRUNK = 'q18a_native35_p_c', 'q18a_native39_trunk'
MANIFEST_P_C, MANIFEST = 'workloads/q18a-w2-overlap.json', 'workloads/q18a-w2-overlap-native39.json'

PC_HEADER = ('# Native35-p-c: P-b with FLASH_NATIVE30_LOCALIZE=0 (plugins present, inert): native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt; decode cohort stopped at the window boundary) -> W1 -> two retained W2 rounds -> owned restore.\n')
TRUNK_HEADER = ('# Native39 trunk: P-c on the native39 engine with a plannable W2 length, a loopback API server, the native38 bind receipt and a plain W1 control: native28 startup -> bind receipt -> W2 length guard -> row1 -> rule4_controls -> rule4_candidate -> row4 declared NOT_RUN (observer and offers off for the speed engine) -> instruments-off proof -> sealed native W1 -> typed FULL + idle -> plain W1 control -> idle -> two retained W2 rounds (each idle, instruments-off) -> access-log ingress per span -> return timing -> trunk summary -> owned restore.\n')
PC_TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 14400s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
TRUNK_TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 16200s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
PC_SLOT_NOTE = '# entry unless its 14400 s timeout plus 900 s kill margin fits that window (native35p_gate derivation).\n'
TRUNK_SLOT_NOTE = '# entry unless its 16200 s timeout plus 900 s kill margin fits that window (native39_trunk_gate derivation).\n'
PC_SLOT = 'from native35p_gate import speed_window_ok as window_ok\n'
TRUNK_SLOT = 'from native39_trunk_gate import trunk_window_ok as window_ok\n'
PC_OFFLINE = '  HEALTH_TRIES=${Q17_TEST_HEALTH_TRIES:-2}; HEALTH_SLEEP=${Q17_TEST_HEALTH_SLEEP:-0.2}\n'
TRUNK_OFFLINE = ('  HEALTH_DEADLINE_S=${Q17_TEST_HEALTH_DEADLINE_S:-2}; HEALTH_SLEEP=${Q17_TEST_HEALTH_SLEEP:-0.2}\n'
                 '  CORPUS=${Q17_TEST_CORPUS:?offline test corpus}; CORPUS_SHA12=${Q17_TEST_CORPUS_SHA12:?offline corpus pin}\n'
                 '  W1_PLAIN_WINDOW_S=${Q17_TEST_W1_PLAIN_WINDOW_S:-1}; W1_PLAIN_SETTLE_S=0.2; W1_PLAIN_START_DEADLINE_S=20\n'
                 '  W1_PLAIN_TIMEOUT_S=60; BIND_WAIT_S=2; IDLE_TIMEOUT_S=5\n')
PC_BOX = '  HEALTH_TRIES=36; HEALTH_SLEEP=10\n'
TRUNK_BOX = ('  # Native39: one /health sample per second until the 540 s P-c allowed (36 x (5 s curl + 10 s sleep)).\n'
             '  HEALTH_DEADLINE_S=540; HEALTH_SLEEP=1\n'
             '  # The plain W1 control as q18b-w1ab2 ran it on this box (corpus sha256 prefix from its preflight).\n'
             '  CORPUS=/home/<service-user>/p2p-tests/corpus.txt; CORPUS_SHA12=2d5bb2ad5f42\n'
             '  W1_PLAIN_WINDOW_S=20; W1_PLAIN_SETTLE_S=3; W1_PLAIN_START_DEADLINE_S=900\n'
             '  W1_PLAIN_TIMEOUT_S=1500; BIND_WAIT_S=30; IDLE_TIMEOUT_S=120\n')
PC_INTERFACES = '  e3_native35_l2_placement.py e3_native35_p_c_placement.py e3_observer_census.py e3_native35_p_c_continuation.py; do\n'
TRUNK_INTERFACES = ('  e3_native35_l2_placement.py e3_native35_p_c_placement.py e3_observer_census.py e3_native35_p_c_continuation.py \\\n'
                    '  e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py \\\n'
                    '  e3_native39_w1.py e3_native39_round.py; do\n')
PC_REQUIRED = '   -r $H/agent_replay.py && -r $H/workloads/q18a-w2-overlap.json &&\n'
TRUNK_REQUIRED = '   -r $H/agent_replay.py && -r $H/workloads/q18a-w2-overlap-native39.json &&\n'
PC_AUTH = '[[ $AUTH_LEN -ge 8 ]] || fail_early auth-missing\n'
TRUNK_AUTH = (PC_AUTH
              + '# Native39: bash clock for health samples, the plain W1 corpus, and the found EXTRA_ARGS that the\n'
                '# loopback setting carries forward (compared silently; no defaults value is printed).\n'
                '[[ -n ${EPOCHREALTIME:-} ]] || fail_early bash-without-EPOCHREALTIME\n'
                '[[ -f $CORPUS && ! -L $CORPUS && -r $CORPUS ]] || fail_early plain-w1-corpus-missing\n'
                '[[ $(sha256sum "$CORPUS" | cut -c1-12) == "$CORPUS_SHA12" ]] || fail_early plain-w1-corpus-differs\n'
                'python3 "$H/e3_native39_trunk_checks.py" found-extra-args --defaults "$DEFAULTS" --expected "$TRUNK_PROD_EXTRA_ARGS" \\\n'
                '  --out "$Q/found-extra-args.json" > "$Q/found-extra-args.log" 2>&1 || fail_early found-extra-args-differ\n')
PC_PREFLIGHT_AUTH = "assert env.get('VLLM_API_KEY') == {'redacted_length': auth_len}\n"
TRUNK_PREFLIGHT_AUTH = (PC_PREFLIGHT_AUTH
                        + "args = ident['command']['args']\n"
                          "hosts = [args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host']\n"
                          "assert hosts and hosts[-1] == '127.0.0.1', 'declared API server must bind 127.0.0.1 (last --host)'\n"
                          "assert defaults.get('EXTRA_ARGS') == os.environ['TRUNK_PROD_EXTRA_ARGS'].split() + ['--host', '127.0.0.1']\n")
PC_PLAN = "assert plan.get('overlap') == 'workloads/q18a-w2-overlap.json'\n"
TRUNK_PLAN = "assert plan.get('overlap') == 'workloads/q18a-w2-overlap-native39.json'\n"
PC_RESTART = ('timeout --signal=TERM --kill-after=20s 240s systemctl restart "$SERVICE" > "$Q/restart.log" 2>&1\n'
              'rc=$?\n'
              'if [[ $rc == 0 ]]; then\n'
              '  rc=1\n'
              '  for _ in $(seq 1 "$HEALTH_TRIES"); do\n'
              '    health_ok && { rc=0; break; }\n'
              '    systemctl is-failed -q "$SERVICE" && break\n')
TRUNK_RESTART = ('printf \'%s restart-begin\\n\' "$EPOCHREALTIME" > "$Q/health.tsv"\n'
                 'timeout --signal=TERM --kill-after=20s 240s systemctl restart "$SERVICE" > "$Q/restart.log" 2>&1\n'
                 'rc=$?\n'
                 'printf \'%s restart-returned rc=%s\\n\' "$EPOCHREALTIME" "$rc" >> "$Q/health.tsv"\n'
                 'if [[ $rc == 0 ]]; then\n'
                 '  rc=1\n'
                 '  # Native39: one row per /health request, "<sent> <returned> <code>", for the bind receipt.\n'
                 '  _health_end=$((SECONDS + HEALTH_DEADLINE_S))\n'
                 '  while (( SECONDS < _health_end )); do\n'
                 '    _sent=$EPOCHREALTIME\n'
                 '    _code=$(curl -s -o /dev/null -w \'%{http_code}\' --max-time 5 "$PROBE_URL/health" 2>/dev/null) || true\n'
                 '    printf \'%s %s %s\\n\' "$_sent" "$EPOCHREALTIME" "${_code:-000}" >> "$Q/health.tsv"\n'
                 '    [[ $_code == 200 ]] && { rc=0; break; }\n'
                 '    systemctl is-failed -q "$SERVICE" && break\n')
PC_RESTART_STEP = ('step restart "$rc" "NRestarts $N0 -> $n"\n'
                   '[[ $rc == 0 ]] || { DONE=1; exit 1; }\n')
TRUNK_RESTART_STEP = (PC_RESTART_STEP
                      + '# Native39: the native38 bind receipt, from the followed (persisted, run-scoped) container log.\n'
                        '_cid=$(docker inspect --format \'{{.Id}}\' "$CONTAINER" 2>/dev/null) || _cid=\'\'\n'
                        'python3 "$H/e3_native39_trunk_checks.py" bind --q "$Q" --container-id "${_cid:-none}" --wait-s "$BIND_WAIT_S" \\\n'
                        '  --out "$Q/bind-receipt.json" > "$Q/bind-receipt.log" 2>&1\n'
                        "record bind-receipt \"$?\" \"first 'flash-s5 startup owner bound' line before the first /health 200 request\"\n")
PC_IDENTITY = ('rc=$?; step engine.identity-start "$rc" "$(identity_cost "$Q/engine.identity-start.json")"\n'
               '[[ $rc == 0 ]] || { DONE=1; exit 1; }\n')
TRUNK_IDENTITY = (PC_IDENTITY
                  + 'python3 "$H/e3_native39_trunk_checks.py" host --identity "$R/startup.json" --identity "$Q/engine.identity-start.json" \\\n'
                    '  --expected-extra-args "$TRUNK_PROD_EXTRA_ARGS --host 127.0.0.1" --q "$Q" \\\n'
                    '  --out "$Q/effective-host.json" > "$Q/effective-host.log" 2>&1\n'
                    "rc=$?; record effective-host \"$rc\" 'declared, captured and served API server host 127.0.0.1 (last --host; engine log)'\n"
                    '[[ $rc == 0 ]] || { DONE=1; exit 1; }\n')
PC_EPOCH = 'export EPOCH\n'
TRUNK_EPOCH = (PC_EPOCH
               + 'python3 "$H/e3_native39_trunk_checks.py" bind-epoch --bind "$Q/bind-receipt.json" --startup "$Q/startup-grade.json" \\\n'
                 '  --out "$Q/bind-epoch.json" > "$Q/bind-epoch.log" 2>&1\n'
                 "record bind-epoch \"$?\" 'bind line engine_epoch equals the startup grade epoch'\n")
PC_MODEL = ('PY_MODEL\n'
            ') || { step workload-plan 1; DONE=1; exit 1; }\n')
TRUNK_MODEL = (PC_MODEL
               + '# Native39: the W2 counts the trunk manifest declares, from this engine\'s /tokenize, must plan.\n'
                 'python3 "$H/e3_native39_w2_length.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
                 '  --workload "$H/workloads/q18a-w2-overlap-native39.json" --out "$Q/w2-length.json" > "$Q/w2-length.log" 2>&1\n'
                 "rc=$?; record w2-length \"$rc\" 'declared W2 fresh/append counts equal /tokenize and both have a native plan'\n"
                 '[[ $rc == 0 ]] || { DONE=1; exit "$rc"; }\n')
PC_W1 = ('if [[ $qualification_ready == 1 ]]; then\n'
         '  WORKLOAD=$H/workloads/q17b-w1.json READ_TIMEOUT_S=$Q18A_FIRST_ADVANCE_READ_TIMEOUT_S \\\n'
         '    python3 "$H/e3_native_w1.py" "$PROBE_URL" "$MODEL" "$Q/w1.json" > "$Q/w1.log" 2>&1\n'
         "  record w1_controls \"$?\" 'unchanged fixed W1 requests, six native full/split windows'\n"
         'else\n'
         "  record w1_controls 2 'W1 withheld until row1/placement PASS and rule4 PASS or reproduced bounded numerical outcome'\n"
         'fi\n')
TRUNK_W1 = r"""W1_PLAIN_IDLE=0; INSTRUMENTS_W1=2; W1_INGRESS=2; PLAIN_INGRESS=2
if [[ $qualification_ready == 1 ]]; then
  # Native39: observer and offers off, proven on every rank process before W1 (the flags are not the proof).
  _cid=$(docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null) || _cid=''
  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
    --out "$Q/instruments-w1.json" > "$Q/instruments-w1.log" 2>&1
  INSTRUMENTS_W1=$?; record instruments-w1 "$INSTRUMENTS_W1" 'replay observer library mapped in no rank process (actual state; observer and offers off)'
fi
if [[ $qualification_ready == 1 && $INSTRUMENTS_W1 == 0 ]]; then
  WORKLOAD=$H/workloads/q17b-w1.json READ_TIMEOUT_S=$Q18A_FIRST_ADVANCE_READ_TIMEOUT_S \
    python3 "$H/e3_native39_w1.py" "$PROBE_URL" "$MODEL" "$Q/w1.json" > "$Q/w1.log" 2>&1
  record w1_controls "$?" 'unchanged fixed W1 requests, six native full/split windows; cohort sealed at every entry and barrier'
  python3 "$H/e3_native39_trunk_checks.py" ingress-w1 --q "$Q" --w1 "$Q/w1.json" \
    --out "$Q/ingress-w1.json" > "$Q/ingress-w1.log" 2>&1
  W1_INGRESS=$?; record ingress-w1 "$W1_INGRESS" 'access log: every request in each native W1 repeat is its sealed cohort, else INVALID'
  # Native39: the plain 24 Oct client (a control, not a gate) only on a typed all-rank FULL, resumed and idle.
  python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/w1-plain-full-idle.json" > "$Q/w1-plain-full-idle.log" 2>&1
  _full_rc=$?; record w1-plain-full-idle "$_full_rc" 'typed all-rank FULL, resumed, running 0 waiting 0, workers empty'
  if [[ $_full_rc == 0 ]]; then
    _plain_start=$EPOCHREALTIME
    WINDOWS=3 WINDOW_S=$W1_PLAIN_WINDOW_S SETTLE_S=$W1_PLAIN_SETTLE_S START_DEADLINE_S=$W1_PLAIN_START_DEADLINE_S \
      CORPUS=$CORPUS timeout --signal=TERM --kill-after=30s "${W1_PLAIN_TIMEOUT_S}s" \
      python3 "$H/w1_trace.py" "$PROBE_URL" "$MODEL" 8 153537 plain-w1 "$Q/w1-plain.json" > "$Q/w1-plain.log" 2>&1
    record w1_plain "$?" 'plain 24 Oct client (q18b-w1ab2 w1_trace.py): 8 x 153537, 3 x 20 s, on the proven FULL route'
    _plain_end=$EPOCHREALTIME
    python3 "$H/e3_native39_trunk_checks.py" route-check --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --control-timeout-s 60 --full-idle "$Q/w1-plain-full-idle.json" --out "$Q/w1-plain-route.json" > "$Q/w1-plain-route.log" 2>&1
    record w1-plain-route "$?" 'route full at the full-idle generation after the plain W1: no transition inside it'
    python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label plain-w1 --start "$_plain_start" --end "$_plain_end" \
      --generation 8 --out "$Q/ingress-plain.json" > "$Q/ingress-plain.log" 2>&1
    PLAIN_INGRESS=$?; record ingress-plain "$PLAIN_INGRESS" 'access log: every request in the plain W1 span is the plain client, else INVALID'
    # Its requests end on disconnect; W2 waits for an idle engine.
    python3 "$H/e3_native39_trunk_checks.py" idle --url "$PROBE_URL" --timeout-s "$IDLE_TIMEOUT_S" \
      --out "$Q/w1-plain-idle.json" > "$Q/w1-plain-idle.log" 2>&1
    _idle_rc=$?; record w1-plain-idle "$_idle_rc" 'engine idle (running 0, waiting 0) before W2'
    [[ $_idle_rc == 0 ]] && W1_PLAIN_IDLE=1
  else
    record w1_plain 2 'plain W1 withheld: typed all-rank FULL and idle not proven after the native W1'
  fi
elif [[ $qualification_ready == 1 ]]; then
  record w1_controls 2 'W1 withheld: observer and offers off not proven on every rank process (instruments-w1)'
  record w1_plain 2 'plain W1 control withheld with the native W1'
else
  record w1_controls 2 'W1 withheld until row1/placement PASS and rule4 PASS or reproduced bounded numerical outcome'
  record w1_plain 2 'plain W1 control withheld with the native W1'
fi
"""
PC_W2_IF = ('gate_rc=$?;step overlap-gates "$gate_rc"\n')
TRUNK_W2_IF = (PC_W2_IF
               + '# Native39: W2 runs only on an idle engine after the plain W1 control.\n'
                 'W2_READY=$gate_rc\n'
                 'if [[ $W2_READY == 0 && $W1_PLAIN_IDLE != 1 ]]; then\n'
                 '  W2_READY=2\n'
                 "  step w2-idle-precondition 2 'engine not idle after the plain W1 control; W2 not run'\n"
                 'fi\n'
                 '# Native39: and only after every W1 span was proven to hold only harness requests (access log).\n'
                 'if [[ $W2_READY == 0 && ( $W1_INGRESS != 0 || $PLAIN_INGRESS != 0 ) ]]; then\n'
                 '  W2_READY=2\n'
                 "  step w2-ingress-precondition 2 'a W1 span is not proven to hold only harness requests; W2 not run'\n"
                 'fi\n')
PC_W2_GATE = 'if [[ $gate_rc == 0 ]]; then\n  for _round in 0 1; do\n'
TRUNK_W2_GATE = 'if [[ $W2_READY == 0 ]]; then\n  for _round in 0 1; do\n'
PC_W2_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap.json"'
TRUNK_W2_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap-native39.json"'
PC_W2_ELSE = "  record overlap 2 'native performance NOT_RUN: same-queue gates refused continuation'\n"
TRUNK_W2_ELSE = "  record overlap 2 'native performance NOT_RUN: same-queue gates refused continuation, the engine was not idle after the plain W1 control, or a W1 span was not proven harness-only'\n"
PC_GRADE = "record native35-performance \"$?\" 'both retained rounds beside the W1 baseline; raw rates, counts and durations; 80/80 is a marker, never a gate'\n"
TRUNK_GRADE = (PC_GRADE
               + '# Native39: the end instruments receipt brackets every return; the return reader is descriptive.\n'
                 'if [[ $INSTRUMENTS_W1 == 0 ]] && health_ok; then\n'
                 '  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\\n'
                 '    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-w1.json" \\\n'
                 '    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1\n'
                 "  record instruments-end \"$?\" 'after W1 and W2: the same rank processes, replay observer mapped in none'\n"
                 'fi\n'
                 'if [[ -f $Q/w1.json ]]; then\n'
                 '  python3 "$H/e3_native39_trunk_checks.py" returns --q "$Q" --out "$Q/returns.json" > "$Q/returns.log" 2>&1\n'
                 "  step returns \"$?\" 'descriptive split->FULL return timing (row1 1.5 ms, row4 42.76 s); no observation window at any return'\n"
                 'fi\n'
                 'python3 "$H/e3_native39_trunk_checks.py" summary --q "$Q" --out "$Q/trunk-summary.json" > "$Q/trunk-summary.log" 2>&1\n'
                 "step trunk-summary \"$?\" 'descriptive: W2 beside the native W1 FULL and plain W1 baselines; 80/80 is a marker'\n")

PC_SETTINGS_ARRAY = 'S5_SETTINGS=(\n'
TRUNK_SETTINGS_ARRAY = ('# Native39 trunk: the API server binds 127.0.0.1.  The found production EXTRA_ARGS (checked at\n'
                        '# preflight) is carried forward with --host 127.0.0.1 appended; the launcher appends EXTRA_ARGS after\n'
                        '# its own --host 0.0.0.0 and the last --host wins (q18b-w1ab and q18b-w1ab2 on this box).\n'
                        'TRUNK_PROD_EXTRA_ARGS="--prefix-match-unit 176 --enable-prompt-tokens-details"\n'
                        '# Native39 speed engine: FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0 below (q18b-w1ab2\n'
                        '# measured ~5 % decode for the two together); row4 placement needs both and is declared NOT_RUN.\n'
                        'export TRUNK_PROD_EXTRA_ARGS\n'
                        'S5_SETTINGS=(\n')
PC_SETTINGS_AUTH = ' "FLASH_S5_AUTH_ENV_FILE=$S5_AUTH_ENV_FILE"\n'
TRUNK_SETTINGS_AUTH = (PC_SETTINGS_AUTH + ' "EXTRA_ARGS=$TRUNK_PROD_EXTRA_ARGS --host 127.0.0.1"\n')

PC_DECL_PLAN = "    overlap='workloads/q18a-w2-overlap.json',\n"
TRUNK_DECL_PLAN = "    overlap='workloads/q18a-w2-overlap-native39.json',\n"
PC_DECL_REQUIRED = '   -r $H/workloads/q18a-w2-overlap.json && -r $H/workloads/q17b-w1.json &&\n'
TRUNK_DECL_REQUIRED = '   -r $H/workloads/q18a-w2-overlap-native39.json && -r $H/workloads/q17b-w1.json &&\n'
# Native39 (Codex 1c334bc review): row4 declared NOT_RUN, the gates carry the declaration.
PC_ROW4 = 'qualification_phase rule4_candidate\nqualification_phase row4\n'
TRUNK_ROW4 = ('qualification_phase rule4_candidate\n'
              '# Native39: the speed engine runs observer and offers off; row4 placement needs both, so it is\n'
              '# declared NOT_RUN for this run (lead option: dropped).  No row4 client, no observer action.\n'
              'ROW4_DECLARED_NOT_RUN=1\n'
              'PHASE_VERDICT[row4]=NOT_RUN\n'
              "echo 'STEP row4 NOT_RUN declared: observer and offers off for the speed engine (native39_trunk_gate.ROW4_NOT_RUN_REASON)'\n"
              'export ROW4_DECLARED_NOT_RUN\n')
PC_READY_ROW4 = '  if [[ $ROW4_CAPACITY_CONTINUATION == 1 && $_gate == row4 ]]; then continue; fi\n'
TRUNK_READY_ROW4 = (PC_READY_ROW4
                    + '  if [[ ${ROW4_DECLARED_NOT_RUN:-0} == 1 && $_gate == row4 && ${PHASE_VERDICT[row4]:-} == NOT_RUN ]]; then continue; fi\n')
PC_GATES_CALL = 'python3 - "$Q/gates.json" <<\'PY_GATES\'\n'
TRUNK_GATES_CALL = 'python3 - "$Q/gates.json" "$H" <<\'PY_GATES\'\n'
PC_GATES_CAP = "capacity=os.environ.get('ROW4_CAPACITY_CONTINUATION')=='1'\n"
TRUNK_GATES_CAP = (PC_GATES_CAP
                   + "declared=os.environ.get('ROW4_DECLARED_NOT_RUN')=='1'\n"
                     "sys.path.insert(0,sys.argv[2])\n"
                     "from native39_trunk_gate import ROW4_NOT_RUN_REASON\n")
PC_GATES_VALID = "          or (capacity and k=='row4' and v=='INCONCLUSIVE') for k,v in d.items())\n"
TRUNK_GATES_VALID = ("          or (capacity and k=='row4' and v=='INCONCLUSIVE')\n"
                     "          or (declared and k=='row4' and v=='NOT_RUN') for k,v in d.items())\n")
PC_GATES_OUT = "    'row4_capacity_matched':capacity_matched},f)\n"
TRUNK_GATES_OUT = ("    'row4_capacity_matched':capacity_matched,'row4_declared_not_run':declared,\n"
                   "    'row4_not_run_reason':ROW4_NOT_RUN_REASON if declared else None},f)\n")
PC_ENV_OBSERVER = "assert env.get('FLASH_S5_REPLAY_OBSERVER') == '1'\n"
TRUNK_ENV_OBSERVER = ("# Native39: the speed engine runs observer and offers off (lead, 1 Nov 2026).\n"
                      "assert env.get('FLASH_S5_REPLAY_OBSERVER') == '0'\n"
                      "assert env.get('FLASH_S5_ENGINE_OFFERS') == '0'\n")
PC_ROUND_PRE = ('        continue\n      fi\n    fi\n'
                '    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\\n')
TRUNK_ROUND_PRE = ('        continue\n      fi\n    fi\n'
                   '    # Native39: each round starts idle, on the W1 rank processes, with the observer still not mapped.\n'
                   '    python3 "$H/e3_native39_trunk_checks.py" idle --url "$PROBE_URL" --timeout-s "$IDLE_TIMEOUT_S" \\\n'
                   '      --out "$Q/w2-idle-round-$_round.json" > "$Q/w2-idle-round-$_round.log" 2>&1\n'
                   "    _pre_rc=$?; record \"w2-idle-round-$_round\" \"$_pre_rc\" 'engine idle (running 0, waiting 0) before the round'\n"
                   '    if [[ $_pre_rc == 0 ]]; then\n'
                   '      python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\\n'
                   '        --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-w1.json" \\\n'
                   '        --out "$Q/instruments-round-$_round.json" > "$Q/instruments-round-$_round.log" 2>&1\n'
                   "      _pre_rc=$?; record \"instruments-round-$_round\" \"$_pre_rc\" 'the W1 rank processes; replay observer mapped in none'\n"
                   '    fi\n'
                   '    if [[ $_pre_rc != 0 ]]; then\n'
                   "      record \"overlap-round-$_round\" 2 'NOT_RUN: engine not idle, or observer and offers off not proven on the W1 rank processes'\n"
                   '      continue\n'
                   '    fi\n'
                   '    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\\n')
PC_ROUND_CLI = '    python3 "$H/e3_native35_performance.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
TRUNK_ROUND_CLI = '    python3 "$H/e3_native39_round.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
PC_ROUND_RECORD = ("    record \"overlap-round-$_round\" \"$?\" 'actual native fresh/append in full-solo, matched split-solo and "
                   "decode-during-prefill; round salt namespace; 80/80 reported target'\n")
TRUNK_ROUND_RECORD = (PC_ROUND_RECORD
                      + '    python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label "round-$_round" \\\n'
                        '      --round "$Q/overlap-round-$_round.json" --out "$Q/ingress-round-$_round.json" > "$Q/ingress-round-$_round.log" 2>&1\n'
                        "    record \"ingress-round-$_round\" \"$?\" 'access log: every request in the round span is the round, else INVALID'\n")
PC_SETTINGS_FLAGS = 'FLASH_S5_REPLAY_OBSERVER=1 FLASH_S5_ENGINE_OFFERS=1'
TRUNK_SETTINGS_FLAGS = 'FLASH_S5_REPLAY_OBSERVER=0 FLASH_S5_ENGINE_OFFERS=0'

PC_DECL_OBSERVER = "assert env.get('FLASH_S5_REPLAY_OBSERVER')=='1', 'observer not declared'\n"
TRUNK_DECL_OBSERVER = ("assert env.get('FLASH_S5_REPLAY_OBSERVER')=='0', 'observer must be off for the speed engine'\n"
                       "assert env.get('FLASH_S5_ENGINE_OFFERS')=='0', 'engine offers must be off for the speed engine'\n"
                       "args=d['identity']['command']['args']\n"
                         "hosts=[args[j+1] for j,x in enumerate(args[:-1]) if x=='--host']\n"
                         "assert hosts and hosts[-1]=='127.0.0.1', 'declared API server must bind 127.0.0.1 (last --host)'\n")
PC_DECL_PINS = '           "$H/workloads/q17b-w1.json" "$H/workloads/q18a-w2-overlap.json" \\\n'
TRUNK_DECL_PINS = '           "$H/workloads/q17b-w1.json" "$H/workloads/q18a-w2-overlap-native39.json" \\\n'
PC_DECL_NATIVE = '      canary_hub_candidate_runner.py flash_canary_candidate.sh \\\n'
TRUNK_DECL_NATIVE = (PC_DECL_NATIVE
                     + '      e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py \\\n'
                       '      e3_native39_w1.py e3_native39_round.py \\\n')

DRAFT_SWAPS = ((PC_HEADER, TRUNK_HEADER), (PC_TIMEOUT, TRUNK_TIMEOUT), (PC_SLOT_NOTE, TRUNK_SLOT_NOTE),
               (PC_SLOT, TRUNK_SLOT), (PC_OFFLINE, TRUNK_OFFLINE), (PC_BOX, TRUNK_BOX),
               (PC_INTERFACES, TRUNK_INTERFACES), (PC_REQUIRED, TRUNK_REQUIRED), (PC_AUTH, TRUNK_AUTH),
               (PC_PREFLIGHT_AUTH, TRUNK_PREFLIGHT_AUTH), (PC_PLAN, TRUNK_PLAN), (PC_RESTART, TRUNK_RESTART),
               (PC_RESTART_STEP, TRUNK_RESTART_STEP), (PC_IDENTITY, TRUNK_IDENTITY), (PC_EPOCH, TRUNK_EPOCH),
               (PC_MODEL, TRUNK_MODEL), (PC_W1, TRUNK_W1), (PC_W2_IF, TRUNK_W2_IF), (PC_W2_GATE, TRUNK_W2_GATE),
               (PC_W2_WORKLOAD, TRUNK_W2_WORKLOAD), (PC_W2_ELSE, TRUNK_W2_ELSE), (PC_GRADE, TRUNK_GRADE),
               (PC_ROW4, TRUNK_ROW4), (PC_READY_ROW4, TRUNK_READY_ROW4), (PC_GATES_CALL, TRUNK_GATES_CALL),
               (PC_GATES_CAP, TRUNK_GATES_CAP), (PC_GATES_VALID, TRUNK_GATES_VALID), (PC_GATES_OUT, TRUNK_GATES_OUT),
               (PC_ENV_OBSERVER, TRUNK_ENV_OBSERVER), (PC_ROUND_PRE, TRUNK_ROUND_PRE), (PC_ROUND_CLI, TRUNK_ROUND_CLI),
               (PC_ROUND_RECORD, TRUNK_ROUND_RECORD))
SETTINGS_SWAPS = ((PC_SETTINGS_ARRAY, TRUNK_SETTINGS_ARRAY), (PC_SETTINGS_AUTH, TRUNK_SETTINGS_AUTH),
                  (PC_SETTINGS_FLAGS, TRUNK_SETTINGS_FLAGS))
DECLARE_SWAPS = ((PC_DECL_PLAN, TRUNK_DECL_PLAN), (PC_DECL_REQUIRED, TRUNK_DECL_REQUIRED),
                 (PC_DECL_OBSERVER, TRUNK_DECL_OBSERVER), (PC_DECL_PINS, TRUNK_DECL_PINS),
                 (PC_DECL_NATIVE, TRUNK_DECL_NATIVE))


def swap(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


def names(text):
    # Only the q18a-prefixed queue names; e3_native35_p_c_* modules and native35 graders stay.
    assert 'q18a-native39-trunk' not in text and 'q18a_native39_trunk' not in text
    return text.replace('q18a-native35-p-c', 'q18a-native39-trunk').replace(P_C, TRUNK)


def derive(path, swaps):
    text = path.read_text()
    for old, new in swaps:
        text = swap(text, old, new)
    text = names(text)
    assert MANIFEST_P_C not in text, path.name + ': P-c W2 manifest still referenced'
    return text


def generate():
    return {RUNS / (TRUNK + '.draft.sh'): derive(RUNS / (P_C + '.draft.sh'), DRAFT_SWAPS),
            RUNS / (TRUNK + '_settings.sh'): derive(RUNS / (P_C + '_settings.sh'), SETTINGS_SWAPS),
            RUNS / (TRUNK + '_declare.sh'): derive(RUNS / (P_C + '_declare.sh'), DECLARE_SWAPS)}


if __name__ == '__main__':
    check = '--check' in sys.argv
    # --rewrite: replace the drafts in place.  Only for drafts no queue has run (these never ran on any
    # box or VM; 1 Nov 2026 rewrite for Codex's 1c334bc review); a run draft gets a new name instead.
    rewrite = '--rewrite' in sys.argv
    for path, text in generate().items():
        if check:
            assert path.read_text() == text, path.name
        else:
            assert rewrite or not path.exists(), path.name + ' exists; regenerate only under --check or --rewrite'
            path.write_text(text)
            path.chmod(0o664)
    print('native39 trunk queue drafts ' + ('verified' if check else 'rewritten' if rewrite else 'written'))
