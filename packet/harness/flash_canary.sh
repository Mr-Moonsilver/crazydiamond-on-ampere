#!/bin/bash
# Real-route leak canary: N concurrent Claude Code sessions on the local model (<llm-gateway>, ~/.claude-local, ds-work's
# local flags), each with a unique session code in its brief and a unique note in a file. Pass = every final line
# names its own code and note; any other session's code/note = LEAK. usage: flash_canary.sh N ROUNDS OUTDIR
set -u
N=$1; R=$2; OUT=$3; mkdir -p "$OUT"
names=(AMBER COBALT JUNIPER OSPREY TUNDRA VIOLET MAGPIE QUARTZ SIENNA FALCON HAZEL RIVET)
for r in $(seq 1 "$R"); do
  for i in $(seq 1 "$N"); do
    d=$(mktemp -d /tmp/flashcanary.XXXXXX)
    code="${names[$((RANDOM % 12))]}-$((RANDOM % 9000 + 1000))"; note="note-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
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
ok=0; bad=0
while read -r r i code note d; do
  f="$OUT/r$r-s$i.out"; line=$(grep -E "CODE=" "$f" | tail -1)
  others=$(awk -v me="$code" '$3!=me{print $3; print $4}' "$OUT/expected.txt" | grep -F -x -v "$note" | while read -r x; do grep -qF "$x" <<<"$line" && echo "$x"; done)
  if [ -n "$others" ]; then echo "LEAK r$r s$i want $code $note got: $line (foreign: $others)"; bad=$((bad+1))
  elif grep -qF "$code" <<<"$line" && grep -qF "$note" <<<"$line"; then ok=$((ok+1))
  else echo "WRONG r$r s$i want $code $note got: ${line:-$(tail -3 "$f" | tr '\n' ' ')}"; bad=$((bad+1)); fi
done < "$OUT/expected.txt"
echo "canary sessions $((N*R)): ok $ok bad $bad"
[ "$bad" = 0 ]
