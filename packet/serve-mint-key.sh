#!/usr/bin/env bash
# Final-engine serve: mint ONE fresh 64-lowercase-hex engine key (run on the hub; prints lengths only).
#   box copy : CT <ct-id> /root/.flashnext-serve-<tag>/vllm-api.env  (new root 0700 dir, 0600 file) -> the engine arm's AUTH_ENV
#   hub copy : /home/<service-user>/.flashnext/serve-<tag>/vllm-api.env  (new 0700 dir, 0600 file)    -> the A/B driver's --auth-env-file
# Both refuse if the dir exists (fresh key per tag).  The key travels hub -> box on ssh stdin only (never argv, never a log).
# The stock arm needs no key (the found launcher has no private-auth path; vLLM without a key accepts and ignores the bearer).
set -euo pipefail
PK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$PK/serve-params.env"
PVE=${PVE:-ssh -F /dev/null -i /home/<service-user>/.ssh/<ssh-key> -o IdentitiesOnly=yes -o BatchMode=yes root@<pve-host>}
HUBD=${HUBD:-/home/<service-user>/.flashnext/serve-$SERVE_TAG}; HUBF=$HUBD/vllm-api.env
BOXD=${R:-}/root/.flashnext-serve-$SERVE_TAG
umask 077
test ! -e "$HUBD" || { echo "hub key dir exists: $HUBD (fresh tag needed)"; exit 2; }
k=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'); test ${#k} = 64
mkdir -p "$(dirname "$HUBD")"; mkdir -m 0700 "$HUBD"
( set -C; printf 'VLLM_API_KEY=%s\n' "$k" > "$HUBF" ); chmod 600 "$HUBF"
# bash -s reads its script from the pipe one line at a time, so the `read` consumes the very next line: the key.
{ printf 'set -euo pipefail; umask 077\nread -r k\n%s\n' "$k"; cat <<BOX
BOXD='$BOXD'; BOXF=\$BOXD/vllm-api.env
test \${#k} = 64; test ! -e \$BOXD; mkdir -m 0700 \$BOXD
( set -C; printf 'VLLM_API_KEY=%s\n' "\$k" > \$BOXF ); chmod 600 \$BOXF
stat -c 'box %n %U %a %s bytes' \$BOXD \$BOXF | sed 's#.*/root/#/root/#' | tr '\n' ';'; echo
echo BOX_KEY_MINTED len=\${#k}
BOX
} | $PVE 'pct exec <ct-id> -- bash -s'
stat -c 'hub %n %U %a %s bytes' "$HUBD" "$HUBF" | tr '\n' ';'; echo
echo "SERVE_KEY_MINTED len=${#k} box=$BOXD/vllm-api.env hub=$HUBF"
