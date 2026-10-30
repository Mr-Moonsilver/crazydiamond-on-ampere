"""Standalone function replacements inserted into the frozen owner runner.

Never executed by Codex against the box. Staging does not query or control the
running engine; arming is an explicit, separate owner action.
"""


def live_guards(cap):
    check_slot()
    require(os.geteuid()==0,'owner-root-required')
    stopped_unit(public_output(['systemctl','show','--property=ActiveState,SubState,MainPID,ControlPID','vllm-flash']))
    raw=public_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'])
    devices=[tuple(int(part.strip()) for part in line.split(',')) for line in raw.splitlines()]
    require(len(devices)==4 and {i for i,_ in devices}==set(range(4)) and all(v==0 for _,v in devices),'four-idle-gpus-required')
    print('ARM_GUARD PASS stopped=true idle_gpus=4',flush=True)


def preflight(cap):
    require(os.geteuid()==0,'owner-root-required')
    require(type(cap) is int and cap==512*1024**2,'snapshot-cap-invalid')
    require(sha(H/'SHA256SUMS')==BUNDLE,'harness-manifest-pin')
    rows=manifest(H);require(inventory(H)==set(rows)|{'SHA256SUMS'},'harness-inventory')
    require(sha(BASE/'SHA256SUMS')==BASE_MANIFEST,'base-manifest-pin');manifest(BASE)
    for p in (RAW,OVERLAY,BUILD_LOGS):require(not p.exists() and not p.is_symlink(),'fresh-staging-path-required')
    for suffix in ('','-requests','.log','-declare.log','-declare-chain.log','-staging.json'):
        p=S0/(QNAME+suffix);require(not p.exists() and not p.is_symlink(),'fresh-output-path-required')
    require(sha(Path('/data/vllm/flash-next-tp4/prod/run-flash-prod.sh'))=='45cc5d813dca6610dd871b99b095956f307f29ab5f8de4993400018c1ab80c88','found-launcher-pin')
    require(sha(Path('/etc/default/vllm-flash'))=='f9fb82486ef3132ac3e6fa9c9ed12fb33225b08e1727727376d3e5ac1a678311','found-defaults-pin')
    sys.path[:0]=[str(H/'e3_builder'),str(H)]
    print('STAGING_INPUTS PASS engine_untouched=true',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description='Native30: stage while production serves, arm only after owner stop')
    action=p.add_mutually_exclusive_group(required=True);action.add_argument('--staging',action='store_true');action.add_argument('--arm',action='store_true')
    p.add_argument('--auth-env-file',type=Path);p.add_argument('--selfrun-record',type=Path)
    p.add_argument('--model',default='qwen38-w8a8-tp4');p.add_argument('--snapshot-max-bytes',type=int,default=512*1024**2)
    p.add_argument('--sync',choices=('none','native','decode','both'),default='none')
    a=p.parse_args(argv)
    global GREEN,MANIFEST
    receipt=S0/(QNAME+'-staging.json')
    sys.path[:0]=[str(H/'e3_builder'),str(H)]
    from e3_private_auth import read_private
    if a.staging:
        require(a.auth_env_file is not None and a.selfrun_record is not None,'staging-auth-and-selfrun-record-required')
        preflight(a.snapshot_max_bytes)
        result=json.loads(a.selfrun_record.read_text())
        require(result.get('bundle_manifest')==BUNDLE and result.get('verdict')=='PASS' and result.get('rc')==0
                and result.get('gpu_device_requests')==0 and result.get('image')==IMAGE
                and result.get('skips')==0 and result.get('tests_run',0)==35
                and result.get('plugin_loader')=='PASS'
                and result.get('identity_capture_check')=='PASS'
                and result.get('identity_timeout_negative_rejected') is True,'exact-frozen-selfrun-required')
        GREEN,MANIFEST,proof=cpu_build()
        with (BUILD_LOGS/'owner-build-receipt.json').open('x') as f:json.dump(proof,f,sort_keys=True,indent=2);f.write('\n')
        env=environment(read_private(a.auth_env_file),a.model,a.snapshot_max_bytes);env['FLASH_NATIVE30_SYNC']=a.sync
        rc=phase('q18a_native30_localize_declare.sh',S0/(QNAME+'-declare-chain.log'),env)
        require(rc==0,'staging-declaration-refused')
        stage=dict(schema='native30-staging/1',verdict='PASS',bundle_sha256=BUNDLE,overlay_manifest=MANIFEST,green_sha256=GREEN,
            review_sha256=REVIEW,model=a.model,snapshot_max_bytes=a.snapshot_max_bytes,sync=a.sync,
            selfrun_sha256=sha(a.selfrun_record),declaration_sha256=sha(S0/(QNAME+'-requests')/'startup.json'),
            engine_untouched=True,engine_stop_or_idle_required=False,box_compilation=False)
        with receipt.open('x') as f:json.dump(stage,f,sort_keys=True,indent=2);f.write('\n')
        receipt.chmod(0o444)
        print('STAGING PASS engine_untouched=true next=owner-stop-then-arm receipt='+str(receipt),flush=True)
        return 0
    stage=json.loads(receipt.read_text())
    require(stage.get('verdict')=='PASS' and stage.get('bundle_sha256')==BUNDLE and stage.get('review_sha256')==REVIEW,'staging-receipt-required')
    live_guards(stage['snapshot_max_bytes'])
    GREEN=stage['green_sha256'];MANIFEST=stage['overlay_manifest']
    env=environment(read_private(S0/(QNAME+'-requests')/'private/vllm-api.env'),stage['model'],stage['snapshot_max_bytes'])
    env.update(FLASH_NATIVE30_SYNC=stage['sync'],REQUESTS=str(S0/(QNAME+'-requests')),Q=str(S0/QNAME))
    rc=phase('q18a_native30_localize.draft.sh',S0/(QNAME+'.log'),env)
    print('OWNER_CHAIN DONE rc='+str(rc)+' queue_started=true engine-stopped-before-restore production-never-started',flush=True)
    return rc


def check_slot():
    from native30_gate import window_ok
    require(window_ok(time.time()),'operator-window-not-before-or-cleanup-margin')
