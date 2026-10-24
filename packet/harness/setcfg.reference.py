#!/usr/bin/env python3
# Write /etc/default/vllm-flash.next from the current settings with KEY=VALUE overrides (VALUE "-" deletes the key).
# Values are written with systemd EnvironmentFile quoting (double quotes, inner " and \ escaped).
import sys
cur = open('/etc/default/vllm-flash').read().splitlines()
over = dict(a.split('=', 1) for a in sys.argv[1:])
out, seen = [], set()
for line in cur:
    k = line.split('=', 1)[0] if '=' in line and not line.lstrip().startswith('#') else None
    if k in over:
        seen.add(k)
        if over[k] != '-':
            out.append(f'{k}="' + over[k].replace('\\', '\\\\').replace('"', '\\"') + '"')
        continue
    out.append(line)
for k, v in over.items():
    if k not in seen and v != '-':
        out.append(f'{k}="' + v.replace('\\', '\\\\').replace('"', '\\"') + '"')
open('/etc/default/vllm-flash.next', 'w').write('\n'.join(out) + '\n')
print('next:', {k: v for k, v in over.items()})
