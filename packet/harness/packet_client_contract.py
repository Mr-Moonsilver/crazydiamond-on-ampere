"""Generated packet client argv, stream capture and bounded failure diagnostics."""
import json
import os
from pathlib import Path
import re
import shlex
import sys
import traceback


def report_exception(exc, *, private=False):
    message='*** (private declaration)' if private else str(exc).replace('\n',' ').replace('\r',' ')
    for key,value in os.environ.items():
        if value and any(word in key for word in ('KEY','TOKEN','SECRET','PASSWORD')):
            message=message.replace(value,'***')
    frames=traceback.extract_tb(exc.__traceback__)
    site=frames[-1] if frames else traceback.extract_stack()[-2]
    text=f'{type(exc).__name__}: {message[:512]} at {Path(site.filename).name}:{site.lineno}'
    print('PACKET_CLIENT_ERROR '+text,file=sys.stderr,flush=True)
    return text


def capture_streams(script):
    anchor='    --container "$CONTAINER" -- "$@"'
    if script.count(anchor)!=1:
        raise ValueError('engine-watch-capture-anchor')
    begin='  local name=$1 rc\n  shift\n'
    if script.count(begin)!=1:raise ValueError('engine-watch-client-wrapper-anchor')
    wrapped=begin+'''  local -a client=("$@")
  local i j found=0
  for ((i=0; i<${#client[@]}; i++)); do
    if [[ ${client[i]} == python3 ]]; then
      j=$((i+1))
      [[ ${client[j]} != -B ]] || j=$((j+1))
      client=("${client[@]:0:j}" "$H/packet_client_exec.py" "${client[@]:j}")
      found=1; break
    fi
  done
  [[ $found == 1 ]] || { echo "PACKET_CLIENT_ERROR non-Python client: $name" >&2; return 2; }
'''
    return script.replace(begin,wrapped,1).replace(anchor,
        '    --container "$CONTAINER" -- "${client[@]}" >> "$Q/client-$name.stdout.log" 2>> "$Q/client-$name.stderr.log"',1)


def client_commands(script,name):
    """Extract the real generated invocations; unknown substitutions refuse freeze."""
    commands=[]
    labels=sorted(set(re.findall(r'core:([\w.-]+)',script))) or ['n3.g2-live-0']
    for number,line in enumerate(script.replace('\\\n',' ').splitlines(),1):
        match=re.search(r'(?:^|\s)engine_watched\s+(?!\()(.+)',line)
        if match is None:continue
        words=shlex.split('engine_watched '+match[1])
        words=words[:next((i for i,w in enumerate(words) if w in ('>','>>','2>','2>>')),len(words))]
        for label in labels if '$_label' in line else [labels[0]]:
            selector,cohort,index=label.split('-',2);kind,group=selector.split('.',1)
            variables=dict(H='/src/harness',Q='/scratch/clients/'+name,PROBE_URL='http://127.0.0.1:1',MODEL='qwen38-w8a8-tp4',
                           RUN_ID='packet-client-selftest',EPOCH='1',IDLE_TIMEOUT_S='90',_label=label,_c1=kind,_group=group,
                           _cohort=cohort,_pair=index,_cid='fixture-container',_workload='/src/harness/workloads/'+
                           ('q18a-native41-progress.json' if index.isdigit() else 'q18a-alpha-perf-band2.json'))
            def expand(word):
                word=word.replace('${_c1#c1}',kind.removeprefix('c1')).replace('${_group#g}',group.removeprefix('g')).replace('${_cid:-none}',variables['_cid'])
                word=re.sub(r'\$([A-Za-z_]\w*)',lambda m:variables[m[1]] if m[1] in variables else m[0],word)
                if '$' in word:raise ValueError('unpriced-client-substitution:'+word)
                return word
            argv=[expand(w) for w in words[2:]]
            if 'python3' not in argv:raise ValueError('python-client-required')
            start=argv.index('python3')+1
            if argv[start]=='-B':start+=1
            commands.append(dict(label=expand(words[1]),line=number,argv=argv,python_argv=argv[start:]))
    if not commands:raise ValueError('no-engine-watch-client-commands')
    return commands


def freeze_clients(script,name,bundle):
    path=Path(bundle)/'packet-client-commands.json'
    with path.open('x') as f:json.dump(dict(schema='packet-clients/1',name=name,commands=client_commands(script,name)),f,sort_keys=True,indent=2);f.write('\n')


def full_protocol_result(command, *, stdout, stderr, returncode, verdict_prefix,
                         expected_verdict, expected_steps, observed_steps):
    """Standing check: first-I/O probes alone never qualify a packet client."""
    if not command or not expected_steps or not verdict_prefix or not expected_verdict:
        raise ValueError('whole-protocol-contract-missing')
    last = next((line for line in reversed(stdout.splitlines()) if line.startswith(verdict_prefix)), None)
    missing = sorted(set(expected_steps) - set(observed_steps))
    expected_rc = {'FIT':0, 'PASS':0, 'GO':0, 'HOLD':2, 'FAIL':1, 'NO-GO':1}[expected_verdict]
    if returncode != expected_rc or last is None or last.split()[1] != expected_verdict or missing:
        raise ValueError('whole-protocol-client-incomplete:'+','.join(missing)+':'+stderr[-512:])
    return dict(argv=list(command), final_verdict_line=last, returncode=returncode,
                observed_steps=sorted(set(observed_steps)), scope='schema-valid stub only; no device qualification')


def startup_vram_gate(script):
    """A startup-only peer census must pass before any diagnostic/native arm."""
    anchor='  engine_watched numeric-smoke '
    if script.count(anchor)!=1 or 'engine_watched peer-vram-budget ' in script:
        raise ValueError('peer-vram-budget-template-anchor')
    command='''  engine_watched peer-vram-budget timeout 30s python3 "$H/peer_vram_budget.py" --engine-log "$Q/startup-speculation.log" --startup "$Q/startup-grade.json" --out "$Q/peer-vram-budget.json" > "$Q/peer-vram-budget.log" 2>&1
  _vram_rc=$?; record peer-vram-budget "$_vram_rc" 'all four synchronized capture/driver budgets required before native admission'
  [[ $_vram_rc == 0 ]] || fail_early peer-vram-budget
'''
    return script.replace(anchor,command+anchor,1)
