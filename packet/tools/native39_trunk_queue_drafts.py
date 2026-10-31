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
  plus names and the header comment.  Every replacement must match exactly once.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
P_C, TRUNK = 'q18a_native35_p_c', 'q18a_native39_trunk'
MANIFEST_P_C, MANIFEST = 'workloads/q18a-w2-overlap.json', 'workloads/q18a-w2-overlap-native39.json'

PC_HEADER = ('# Native35-p-c: P-b with FLASH_NATIVE30_LOCALIZE=0 (plugins present, inert): native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt; decode cohort stopped at the window boundary) -> W1 -> two retained W2 rounds -> owned restore.\n')
TRUNK_HEADER = ('# Native39 trunk: P-c on the native39 engine with a plannable W2 length, a loopback API server, the native38 bind receipt and a plain W1 control: native28 startup -> bind receipt -> W2 length guard -> row1 -> rule4_controls -> rule4_candidate -> row4 (P-c collector and capacity continuation) -> W1 -> plain W1 control -> idle -> two retained W2 rounds -> trunk summary -> owned restore.\n')
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
                    '  e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py; do\n')
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
TRUNK_W1 = ('W1_PLAIN_IDLE=0\n'
            'if [[ $qualification_ready == 1 ]]; then\n'
            '  WORKLOAD=$H/workloads/q17b-w1.json READ_TIMEOUT_S=$Q18A_FIRST_ADVANCE_READ_TIMEOUT_S \\\n'
            '    python3 "$H/e3_native_w1.py" "$PROBE_URL" "$MODEL" "$Q/w1.json" > "$Q/w1.log" 2>&1\n'
            "  record w1_controls \"$?\" 'unchanged fixed W1 requests, six native full/split windows'\n"
            '  # Native39: the plain 24 Oct client beside the native W1 (a control, not a gate). Its requests end\n'
            '  # on disconnect; W2 waits for an idle engine.\n'
            '  WINDOWS=3 WINDOW_S=$W1_PLAIN_WINDOW_S SETTLE_S=$W1_PLAIN_SETTLE_S START_DEADLINE_S=$W1_PLAIN_START_DEADLINE_S \\\n'
            '    CORPUS=$CORPUS timeout --signal=TERM --kill-after=30s "${W1_PLAIN_TIMEOUT_S}s" \\\n'
            '    python3 "$H/w1_trace.py" "$PROBE_URL" "$MODEL" 8 153537 plain-w1 "$Q/w1-plain.json" > "$Q/w1-plain.log" 2>&1\n'
            "  record w1_plain \"$?\" 'plain 24 Oct client (q18b-w1ab2 w1_trace.py): 8 x 153537, 3 x 20 s, beside the native W1'\n"
            '  python3 "$H/e3_native39_trunk_checks.py" idle --url "$PROBE_URL" --timeout-s "$IDLE_TIMEOUT_S" \\\n'
            '    --out "$Q/w1-plain-idle.json" > "$Q/w1-plain-idle.log" 2>&1\n'
            "  _idle_rc=$?; record w1-plain-idle \"$_idle_rc\" 'engine idle (running 0, waiting 0) before W2'\n"
            '  [[ $_idle_rc == 0 ]] && W1_PLAIN_IDLE=1\n'
            'else\n'
            "  record w1_controls 2 'W1 withheld until row1/placement PASS and rule4 PASS or reproduced bounded numerical outcome'\n"
            "  record w1_plain 2 'plain W1 control withheld with the native W1'\n"
            'fi\n')
PC_W2_IF = ('gate_rc=$?;step overlap-gates "$gate_rc"\n')
TRUNK_W2_IF = (PC_W2_IF
               + '# Native39: W2 runs only on an idle engine after the plain W1 control.\n'
                 'W2_READY=$gate_rc\n'
                 'if [[ $W2_READY == 0 && $W1_PLAIN_IDLE != 1 ]]; then\n'
                 '  W2_READY=2\n'
                 "  step w2-idle-precondition 2 'engine not idle after the plain W1 control; W2 not run'\n"
                 'fi\n')
PC_W2_GATE = 'if [[ $gate_rc == 0 ]]; then\n  for _round in 0 1; do\n'
TRUNK_W2_GATE = 'if [[ $W2_READY == 0 ]]; then\n  for _round in 0 1; do\n'
PC_W2_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap.json"'
TRUNK_W2_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap-native39.json"'
PC_W2_ELSE = "  record overlap 2 'native performance NOT_RUN: same-queue gates refused continuation'\n"
TRUNK_W2_ELSE = "  record overlap 2 'native performance NOT_RUN: same-queue gates refused continuation, or the engine was not idle after the plain W1 control'\n"
PC_GRADE = "record native35-performance \"$?\" 'both retained rounds beside the W1 baseline; raw rates, counts and durations; 80/80 is a marker, never a gate'\n"
TRUNK_GRADE = (PC_GRADE
               + 'python3 "$H/e3_native39_trunk_checks.py" summary --q "$Q" --out "$Q/trunk-summary.json" > "$Q/trunk-summary.log" 2>&1\n'
                 "step trunk-summary \"$?\" 'descriptive: W2 beside the native W1 FULL and plain W1 baselines; 80/80 is a marker'\n")

PC_SETTINGS_ARRAY = 'S5_SETTINGS=(\n'
TRUNK_SETTINGS_ARRAY = ('# Native39 trunk: the API server binds 127.0.0.1.  The found production EXTRA_ARGS (checked at\n'
                        '# preflight) is carried forward with --host 127.0.0.1 appended; the launcher appends EXTRA_ARGS after\n'
                        '# its own --host 0.0.0.0 and the last --host wins (q18b-w1ab and q18b-w1ab2 on this box).\n'
                        'TRUNK_PROD_EXTRA_ARGS="--prefix-match-unit 176 --enable-prompt-tokens-details"\n'
                        'export TRUNK_PROD_EXTRA_ARGS\n'
                        'S5_SETTINGS=(\n')
PC_SETTINGS_AUTH = ' "FLASH_S5_AUTH_ENV_FILE=$S5_AUTH_ENV_FILE"\n'
TRUNK_SETTINGS_AUTH = (PC_SETTINGS_AUTH + ' "EXTRA_ARGS=$TRUNK_PROD_EXTRA_ARGS --host 127.0.0.1"\n')

PC_DECL_PLAN = "    overlap='workloads/q18a-w2-overlap.json',\n"
TRUNK_DECL_PLAN = "    overlap='workloads/q18a-w2-overlap-native39.json',\n"
PC_DECL_REQUIRED = '   -r $H/workloads/q18a-w2-overlap.json && -r $H/workloads/q17b-w1.json &&\n'
TRUNK_DECL_REQUIRED = '   -r $H/workloads/q18a-w2-overlap-native39.json && -r $H/workloads/q17b-w1.json &&\n'
PC_DECL_OBSERVER = "assert env.get('FLASH_S5_REPLAY_OBSERVER')=='1', 'observer not declared'\n"
TRUNK_DECL_OBSERVER = (PC_DECL_OBSERVER
                       + "args=d['identity']['command']['args']\n"
                         "hosts=[args[j+1] for j,x in enumerate(args[:-1]) if x=='--host']\n"
                         "assert hosts and hosts[-1]=='127.0.0.1', 'declared API server must bind 127.0.0.1 (last --host)'\n")
PC_DECL_PINS = '           "$H/workloads/q17b-w1.json" "$H/workloads/q18a-w2-overlap.json" \\\n'
TRUNK_DECL_PINS = '           "$H/workloads/q17b-w1.json" "$H/workloads/q18a-w2-overlap-native39.json" \\\n'
PC_DECL_NATIVE = '      canary_hub_candidate_runner.py flash_canary_candidate.sh \\\n'
TRUNK_DECL_NATIVE = (PC_DECL_NATIVE
                     + '      e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py \\\n')

DRAFT_SWAPS = ((PC_HEADER, TRUNK_HEADER), (PC_TIMEOUT, TRUNK_TIMEOUT), (PC_SLOT_NOTE, TRUNK_SLOT_NOTE),
               (PC_SLOT, TRUNK_SLOT), (PC_OFFLINE, TRUNK_OFFLINE), (PC_BOX, TRUNK_BOX),
               (PC_INTERFACES, TRUNK_INTERFACES), (PC_REQUIRED, TRUNK_REQUIRED), (PC_AUTH, TRUNK_AUTH),
               (PC_PREFLIGHT_AUTH, TRUNK_PREFLIGHT_AUTH), (PC_PLAN, TRUNK_PLAN), (PC_RESTART, TRUNK_RESTART),
               (PC_RESTART_STEP, TRUNK_RESTART_STEP), (PC_IDENTITY, TRUNK_IDENTITY), (PC_EPOCH, TRUNK_EPOCH),
               (PC_MODEL, TRUNK_MODEL), (PC_W1, TRUNK_W1), (PC_W2_IF, TRUNK_W2_IF), (PC_W2_GATE, TRUNK_W2_GATE),
               (PC_W2_WORKLOAD, TRUNK_W2_WORKLOAD), (PC_W2_ELSE, TRUNK_W2_ELSE), (PC_GRADE, TRUNK_GRADE))
SETTINGS_SWAPS = ((PC_SETTINGS_ARRAY, TRUNK_SETTINGS_ARRAY), (PC_SETTINGS_AUTH, TRUNK_SETTINGS_AUTH))
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
    for path, text in generate().items():
        if check:
            assert path.read_text() == text, path.name
        else:
            assert not path.exists(), path.name + ' exists; regenerate only under --check'
            path.write_text(text)
            path.chmod(0o664)
    print('native39 trunk queue drafts ' + ('verified' if check else 'written'))
