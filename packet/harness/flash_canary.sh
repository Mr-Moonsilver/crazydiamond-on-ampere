#!/bin/bash
# Real-route leak canary: N concurrent Claude Code sessions on the local model (<llm-gateway>, ~/.claude-local, ds-work's
# local flags), each with a unique session code in its brief and a unique note in a file. Graded by canary_grade.py
# (next to this script): whole output + transcript scanned for foreign codes/notes, rc=0 and exact counts required,
# tool chain checked. usage: flash_canary.sh N ROUNDS OUTDIR   exit = canary_grade.py's (0 PASS, 1 FAIL, 2 INCONCLUSIVE)
set -u
N=$1; R=$2; OUT=$3; mkdir -p "$OUT"
names=(AMBER COBALT JUNIPER OSPREY TUNDRA VIOLET MAGPIE QUARTZ SIENNA FALCON HAZEL RIVET)
for r in $(seq 1 "$R"); do
  for i in $(seq 1 "$N"); do
    d=$(mktemp -d /tmp/flashcanary.XXXXXX)
    while :; do
      code="${names[$((RANDOM % 12))]}-$((RANDOM % 9000 + 1000))"; note="note-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
      grep -qF -e " $code " -e " $note " "$OUT/expected.txt" 2>/dev/null || break
    done
    echo "$note" > "$d/notes.txt"; printf 'filler line %s\n' $(seq 1 40) >> "$d/notes.txt"
    echo "$r $i $code $note $d" >> "$OUT/expected.txt"
    ( cd "$d" && CLAUDE_CONFIG_DIR=/home/<service-user>/.claude-local timeout 900 "$HOME/.local/bin/claude" -p --model Qwen3.8-Flash-Next \
        --tools "Bash,Read,Edit,Write,Grep,Glob,LSP" --strict-mcp-config --dangerously-skip-permissions --output-format text \
        <<<"Canary check of the local model, no real work. Your session code is $code. Do these steps in order: 1) Read the file notes.txt in the current directory with the Read tool. 2) Run ls -la with Bash. 3) Run wc -l notes.txt with Bash. 4) Then reply with exactly one line: CODE=<your session code> NOTE=<first line of notes.txt>. Nothing else." \
        > "$OUT/r$r-s$i.out" 2>&1; echo "rc=$?" >> "$OUT/r$r-s$i.out" ) &
    sleep 1
  done
  wait
done
python3 "$(dirname "$0")/canary_grade.py" "$OUT" /home/<service-user>/.claude-local/projects "$OUT/grade.json"
