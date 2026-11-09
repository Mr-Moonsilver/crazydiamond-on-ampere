"""Declare engine identity from sources. Never inspects/starts a container.

Only actual Docker call: image inspect. Launcher executes with a private PATH
containing non-executing docker/sudo/rm stubs and read-only sha256sum/grep.
Only reviewed launcher/setcfg source hashes are accepted.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import engine_identity as identity

HERE=Path(__file__).resolve().parent
TASK='/data/vllm/flash-next-tp4'
LAUNCHER_SHA='98d6d2a21927b8943f1ae9b0b43616757e80adac0c628e65f482e4332e3cf126'
E2_LAUNCHER_SHA='2630f86c961c3c5ecff9b1be87b1602fca182f48484991ea1744071dea5f7d8b'
# Frozen from the verified overlay launcher; build_inwait checks the two pins agree.
S5_LAUNCHER_SHA='0bf5b9a9c0bde25f0b7c4650507d0aa1a90fa393414b47f5ba98cf08d224a224'
FOUND_SHA='45cc5d813dca6610dd871b99b095956f307f29ab5f8de4993400018c1ab80c88'
SETCFG_SHA='f3cae1379fa9079785c03a5555dbf739c687c5dcde9b1c6e058b38e8849754fe'
PP_SETTINGS=dict(OVERLAY=TASK+'/prod/overlay-v16g',PP_LAYER_PARTITION='24,24',MODEL_DIR='-',VISION='-',VISION_ARGS='-',
    BREAKABLE_CG='-',COMPILATION_CONFIG='-',FLASH_SCHED_STATS='200',PREFILL_CHUNK='1024',FLASH_PD_ALTERNATE='1',
    QSA_COW_VERIFY='-',EXTRA_ENV='QSA_STAGE_TIMING=1 FLASH_PP_GROUPING=1 FLASH_POOL_STATS=200')
PP_CORE=('block_pool.py','kv_cache_manager.py','kv_cache_utils.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(words):
    out={}
    for word in words:
        k,sep,v=word.partition('=')
        identity.need(sep and identity.NAME.fullmatch(k) and k not in out and '\x00' not in v)
        out[k]=v
    return out


def apply_setcfg(raw, overrides, source):
    # Exact f3cae137 text algorithm. Unlike sourcing EnvironmentFile, no expansion
    # or command substitution occurs, and neither /etc/default path is written.
    identity.need(sha(source)==SETCFG_SHA)
    out,seen=[],set()
    def assignment(k,v): return k+'="'+v.replace('\\','\\\\').replace('"','\\"')+'"'
    for line in raw.splitlines():
        k=line.split('=',1)[0] if '=' in line and not line.lstrip().startswith('#') else None
        if k in overrides:
            seen.add(k)
            if overrides[k]!='-': out.append(assignment(k,overrides[k]))
            continue
        out.append(line)
    for k,v in overrides.items():
        if k not in seen and v!='-': out.append(assignment(k,v))
    return '\n'.join(out)+'\n'


def raw_defaults(raw):
    return identity.defaults_snapshot(raw.encode(),redact=False)['values']


def host_path(path, mirror):
    if mirror is None: return Path(path)
    p=Path(path)
    identity.need(p.is_relative_to(TASK))
    return Path(mirror)/p.relative_to(TASK)


def logical_path(path, mirror):
    if mirror is None: return path
    p=Path(path)
    if p.is_relative_to(mirror): return str(Path(TASK)/p.relative_to(mirror))
    return path


def render(launcher, values, image_id, scratch, mirror=None):
    identity.need(sha(launcher) in (LAUNCHER_SHA,E2_LAUNCHER_SHA,S5_LAUNCHER_SHA))
    scratch=Path(scratch);scratch.mkdir(mode=0o700)
    bindir=scratch/'bin';bindir.mkdir(mode=0o700)
    for name in ('docker','sudo','rm'):
        p=bindir/name
        p.write_text('#!'+sys.executable+'\n'+(HERE/'identity_declare_stub.py').read_text())
        p.chmod(0o700)
    # No inherited PATH or shell startup file. Hash/grep are the only real child
    # programs the reviewed script can reach; the rest are Bash builtins.
    for name in ('sha256sum','grep'):
        executable=shutil.which(name);identity.need(executable is not None)
        (bindir/name).symlink_to(Path(executable).resolve())
    dangerous={'BASH_ENV','ENV','SHELLOPTS','BASHOPTS','CDPATH','GLOBIGNORE','LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'}
    identity.need(not dangerous.intersection(values))
    env=dict(values)
    task_dir=str(host_path(TASK,mirror))
    for key in ('OVERLAY','MODEL_DIR','MPS_PIPE'):
        if env.get(key): env[key]=str(host_path(env[key],mirror))
    identity.need(not env.get('MPS_PIPE')) # sockets / MPS are outside this gate
    # Raw values exist only in this private directory and child environment.
    env.update(PATH=str(bindir),DECLARE_TASK_DIR=task_dir,DECLARE_IMAGE=image_id,LC_ALL='C')
    script=Path(launcher).read_text()
    if mirror is not None:
        old='task_dir='+TASK+'\n';identity.need(script.count(old)==1)
        script=script.replace(old,'task_dir='+shlex.quote(task_dir)+'\n')
    copy_path=scratch/'launcher.sh';copy_path.write_text(script);copy_path.chmod(0o600)
    p=subprocess.run(['/bin/bash','--noprofile','--norc',str(copy_path)],env=env,cwd=scratch,
                     capture_output=True,timeout=120)
    # Never echo stdout/stderr: launcher errors may contain expanded env values.
    identity.need(p.returncode==0)
    return json.loads(p.stdout)


def docker_spec(argv,image,mirror=None,private_envfile=None):
    identity.need(argv and argv[0]=='run')
    env={};mounts=[];i=1;options={};envfile_seen=False;log_opts=[];core_limit=False
    while i<len(argv) and argv[i].startswith('-'):
        opt,eq,attached=argv[i].partition('=');i+=1
        if opt in ('--rm','--init'):
            identity.need(opt not in options);options[opt]=True;continue
        identity.need(opt in ('--name','--label','--gpus','--ipc','--network','-e','-v','--env-file','--log-opt','--ulimit'))
        if eq: val=attached
        else:
            identity.need(i<len(argv));val=argv[i];i+=1
        if opt=='--ulimit':
            identity.need(val=='core=0' and not core_limit)
            core_limit=True
        elif opt=='--log-opt':
            identity.need(val in ('max-size=512m','max-file=4') and val not in log_opts)
            log_opts.append(val)
        elif opt=='--env-file':
            identity.need(not envfile_seen and private_envfile is not None and val==private_envfile)
            identity.need('VLLM_API_KEY' not in env)
            from e3_private_auth import read_private
            env['VLLM_API_KEY']=read_private(val)
            envfile_seen=True
        elif opt=='-e':
            k,sep,v=val.partition('=');identity.need(sep and identity.NAME.fullmatch(k) and k not in env);env[k]=v
        elif opt=='-v':
            fields=val.split(':');identity.need(len(fields) in (2,3))
            source,destination=fields[:2];mode=fields[2] if len(fields)==3 else 'rw'
            identity.need(source.startswith('/') and destination.startswith('/') and mode in ('ro','rw'))
            mounts.append(dict(type='bind',source=logical_path(source,mirror),destination=destination,
                               rw=mode!='ro',propagation='rprivate'))
        else:
            identity.need(opt not in options);options[opt]=val
    identity.need(envfile_seen == (private_envfile is not None))
    identity.need(log_opts in ([], ['max-size=512m','max-file=4']))
    identity.need(core_limit == bool(log_opts))
    identity.need(options=={'--rm':True,'--init':True,'--name':'qwen-flash-prod',
        '--label':'qwen.flash-next-production=true','--gpus':'all','--ipc':'host','--network':'host'})
    identity.need(argv[i]==image['Id']);i+=1
    config=image['Config'];identity.need(not config.get('Volumes'))
    entry=config.get('Entrypoint') or []
    command=entry+(argv[i:] or config.get('Cmd') or [])
    identity.need(len(command)>1 and all(isinstance(v,str) for v in command))
    base={}
    for item in config.get('Env') or []:
        k,sep,v=item.partition('=');identity.need(sep and k not in base);base[k]=v
    base.update(env)
    return dict(image={'id':image['Id'],'repo_digests':sorted(image.get('RepoDigests') or [])},
                command={'path':command[0],'args':identity.clean_args(command[1:])},env=identity.clean_env(base),
                mounts=sorted(mounts,key=lambda m:m['destination']))


def hash_sources(mounts, policy, mirror, max_bytes, max_files, timeout):
    identity.need(set(policy)<=set(m['destination'] for m in mounts))
    queries=[]
    for m in mounts:
        p=policy.get(m['destination'],{})
        identity.need(set(p)<= {'exclusions','watch'})
        exclusions=p.get('exclusions',[]);watch=p.get('watch',[])
        identity.validate_content(dict(kind='directory',entries=[],exclusions=exclusions))
        identity.need(len(set(watch))==len(watch) and all(identity.relative(w) for w in watch))
        # Use the existing probe's exact hashing and race-detection semantics on
        # host sources. Never substitutes these for capture's in-container reads.
        queries.append(dict(destination=str(host_path(m['source'],mirror)),exclusions=exclusions,watch=watch))
    proc=subprocess.run([sys.executable,str(HERE/'identity_probe.py')],
                        input=json.dumps(dict(mounts=queries,max_bytes=max_bytes,max_files=max_files)),
                        capture_output=True,text=True,timeout=timeout)
    identity.need(proc.returncode==0)
    result=json.loads(proc.stdout);identity.need(len(result['mounts'])==len(mounts))
    for m,q,r in zip(mounts,queries,result['mounts']):
        identity.need(set(r)=={'destination','content','watch'} and r['destination']==q['destination'])
        m.update(content=r['content'],watch=r['watch'])
    return dict(bytes_read=result['bytes_read'],files_read=result['files_read'])


def no_unknown(value):
    if isinstance(value,dict): return value.get('unclassified') is not True and all(no_unknown(v) for v in value.values())
    if isinstance(value,list): return all(no_unknown(v) for v in value)
    return True


def project_launcher(spec, launcher):
    """Declare the one planned host-file replacement visible through /work."""
    work=next(m for m in spec['mounts'] if m['destination']=='/work')
    entry=next(e for e in work['content']['entries'] if e['path']=='prod/run-flash-prod.sh')
    identity.need(entry['kind']=='file' and entry['sha256'] in (FOUND_SHA,LAUNCHER_SHA,E2_LAUNCHER_SHA))
    future_sha=sha(launcher)
    identity.need(future_sha in (LAUNCHER_SHA,E2_LAUNCHER_SHA,S5_LAUNCHER_SHA))
    entry.update(bytes=Path(launcher).stat().st_size,sha256=future_sha)


def declare(a, overrides, image, policy, scratch, install_launcher=False, hash_reference=None):
    raw=Path(a.defaults).read_text()
    future=apply_setcfg(raw,overrides,a.setcfg)
    values=raw_defaults(future)
    args=render(a.launcher,values,image['Id'],scratch,a.source_mirror)
    private_envfile=values.get('FLASH_S5_AUTH_ENV_FILE') if values.get('S5')=='1' else None
    if private_envfile is not None:
        identity.need(sha(a.launcher)==S5_LAUNCHER_SHA)
    spec=docker_spec(args,image,a.source_mirror,private_envfile=private_envfile)
    spec['defaults']=identity.defaults_snapshot(future.encode())['values']
    if hash_reference is None:
        io=hash_sources(spec['mounts'],policy,a.source_mirror,a.max_bytes,a.max_files,a.timeout)
        if install_launcher: project_launcher(spec,a.launcher)
    else:
        old=hash_reference['identity']['mounts']
        identity.need(spec['mounts']==[{k:v for k,v in m.items() if k not in ('content','watch')} for m in old])
        spec['mounts']=copy.deepcopy(old)
        io=dict(bytes_read=0,files_read=0,reused_declared_source_snapshot=True)
    identity.validate_identity(spec)
    identity.need(no_unknown(spec))
    return {'schema':identity.REQUEST,'identity':spec},io


def assert_pair(on,verify):
    x=copy.deepcopy(verify)
    for section in ('env','defaults'):
        identity.need('QSA_COW_VERIFY' not in on['identity'][section])
        identity.need(x['identity'][section].pop('QSA_COW_VERIFY',None)=='1')
    identity.need(x==on)
    spec=on['identity'];identity.need(spec['defaults'].get('PP_LAYER_PARTITION')=='24,24')
    work=next(m for m in spec['mounts'] if m['destination']=='/work')
    identity.need({w['path']:w['kind'] for w in work['watch']}==
                  {'rowcache.off':'absent','rowcache.report':'absent','retain.off':'absent'})
    root=spec['defaults']['OVERLAY'];base='/usr/local/lib/python3.12/dist-packages/vllm/v1/core/'
    for f in PP_CORE:
        identity.need(any(m['source']==root+'/core/'+f and m['destination']==base+f for m in spec['mounts']))


def load_image(a):
    image_id=Path(a.image_file).read_text().strip()
    identity.need(re.fullmatch(r'sha256:[a-f0-9]{64}',image_id))
    raw=json.loads(Path(a.image_inspect).read_text()) if a.image_inspect else identity.run(['docker','image','inspect',image_id],a.timeout)
    identity.need(isinstance(raw,list) and len(raw)==1 and raw[0]['Id']==image_id)
    return raw[0]


def run(a):
    out=Path(a.out).absolute();identity.need(not out.exists())
    image=load_image(a)
    policy_path=a.mount_policy or (HERE/'pp_release_mount_policy.json' if a.action=='declare-pair' else None)
    policy=json.loads(Path(policy_path).read_text()) if policy_path else {}
    if a.action=='declare-pair':
        identity.need(not a.settings and not a.source_mirror and not getattr(a,'project_launcher',False))
        identity.need(sha(a.found_launcher)==FOUND_SHA)
        out.mkdir(mode=0o700)
        scratch=out/'private-render';scratch.mkdir(mode=0o700)
        on,io0=declare(a,PP_SETTINGS,image,policy,scratch/'on',True)
        verify,io1=declare(a,dict(PP_SETTINGS,QSA_COW_VERIFY='1'),image,policy,scratch/'verify',True,on)
        assert_pair(on,verify)
        # Check exact payload, not merely that three filenames exist.
        payload=Path(a.harness)/'ppg'
        for name in PP_CORE:
            identity.need(sha(payload/name)==sha(Path(PP_SETTINGS['OVERLAY'])/'core'/name))
        identity.write(out/'on.json',on);identity.write(out/'verify.json',verify)
        shutil.copyfile(a.found_launcher,out/'launcher.found');(out/'launcher.found').chmod(0o600)
        pins=[out/'on.json',out/'verify.json',out/'launcher.found',Path(a.launcher),Path(a.setcfg),
              Path(a.reconfig),Path(a.queue),Path(PP_SETTINGS['OVERLAY'])/'SHA256SUMS',
              Path(PP_SETTINGS['OVERLAY'])/'core/SHA256SUMS']
        if policy_path: pins.append(Path(policy_path))
        identity.need(Path(a.launcher).resolve()==(Path(a.harness)/'run-flash-prod.sh.q9').resolve())
        with (out/'SHA256SUMS').open('x') as f:
            for p in pins: f.write(sha(p)+'  '+str(p.absolute())+'\n')
        identity.write(out/'declaration.json',dict(image=image['Id'],launcher_sha256=sha(a.launcher),
            setcfg_sha256=sha(a.setcfg),found_launcher_sha256=FOUND_SHA,pair_delta='QSA_COW_VERIFY=1 only',
            content_io=[io0,io1],policy=policy,claim='declared sources, not a running engine; no container commands'))
    else:
        scratch=Path(tempfile.mkdtemp(prefix='identity-render-',dir=out.parent));scratch.chmod(0o700)
        declaration,_=declare(a,settings(a.settings),image,policy,scratch/'render',
                              install_launcher=getattr(a,'project_launcher',False))
        identity.write(out,declaration)
    return 0
