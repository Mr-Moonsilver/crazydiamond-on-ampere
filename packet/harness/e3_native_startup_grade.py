#!/usr/bin/env python3
"""Query and grade actual native construction; no decode-probe open record."""
import argparse
import json
from pathlib import Path

from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details
from e3_finalize_package import manifest


def grade(prepare, identity, *, epoch, package=None):
    failures,missing,workers=[],[],{}
    def need(label,ok):
        if not ok:missing.append(label)
    def require(label,ok):
        if not ok:failures.append(label)
    client=NativeRuntimeClient.__new__(NativeRuntimeClient);client.epoch=epoch
    try:
        client.require_acks(prepare,'full')
        require('prepare-route',prepare.get('route')=='full')
    except Exception:
        missing.append('actual-typed-full-prepare')
    try:
        rows=client.workers(identity)
    except Exception:
        rows={};missing.append('queried-construction-ranks')
    incarnations=set()
    storage=set()
    for rank,row in rows.items():
        prefix='rank%d:'%rank
        inv=row.get('native_inventory')
        if type(inv) is not dict:
            missing.append(prefix+'native-inventory');continue
        require(prefix+'schema',inv.get('schema')=='s5-native-construction/1')
        require(prefix+'rank-epoch',inv.get('worker_rank')==rank and inv.get('engine_epoch')==epoch
                and inv.get('engine_bound') is True and row.get('route')=='full')
        require(prefix+'generation',row.get('route_generation')==inv.get('route_generation'))
        require(prefix+'identity',type(inv.get('worker_incarnation')) is str and bool(inv['worker_incarnation'])
                and inv['worker_incarnation']==row.get('worker_incarnation')
                and type(inv.get('decode_owner_incarnation')) is str
                and inv['decode_owner_incarnation']==inv['worker_incarnation'])
        incarnations.add(row.get('worker_incarnation'));storage.add(row.get('storage_epoch'))
        split=inv.get('split') or {};membership=split.get('membership') or {}
        require(prefix+'split-counts',split.get('requested_decode_sms')==44 and split.get('decode_sms')==44
                and split.get('decode_sms',0)+split.get('prefill_sms',0)==split.get('total_sms'))
        need(prefix+'measured-membership',split.get('basis')=='measured'
             and membership.get('ok') is True and membership.get('disjoint') is True
             and membership.get('decode_seen')==44
             and membership.get('prefill_seen')==split.get('prefill_sms')
             and membership.get('total')==split.get('total_sms'))
        need(prefix+'captured-bank',inv.get('split_bank_basis')=='measured'
             and type(inv.get('split_bank_bytes')) is int and inv['split_bank_bytes']>=0)
        comm=inv.get('communicators') or {}
        need(prefix+'lane-communicators',type(comm) is dict and comm.get('tp_warmed') is True
             and comm.get('tp_world_size')==2 and type(inv.get('pp_channel_bytes')) is int
             and inv['pp_channel_bytes']>0)
        owners=inv.get('prefill_streams') or {}
        need(prefix+'prefill-owners',owners.get('state')=='ready' and owners.get('basis')=='prefill-green'
             and owners.get('original_owners_restored') is True)
        private=inv.get('private_bindings') or {}
        need(prefix+'private-owners',private.get('workspace_prepared') is True and private.get('failed') is False
             and private.get('workspace_basis')=='extra_last_lane' and type(private.get('workspace_bytes')) is int
             and private['workspace_bytes']>=0 and type(private.get('uid_rows')) is int and private['uid_rows']>=1
             and type(private.get('qsa_layers')) is list and bool(private['qsa_layers']))
        # The partial audit remains partial: admission/request-row, source lease,
        # row-cache bypass and per-call selection are graded in their own rows.
        router=inv.get('router') or {}
        need(prefix+'router',router.get('engine_epoch')==epoch and router.get('pending')==0
             and router.get('storage_epoch')==row.get('storage_epoch'))
        alloc,memory=row.get('allocator') or {},row.get('memory') or {}
        need(prefix+'queried-memory',alloc.get('basis')=='torch-cuda-allocator-queried-v1'
             and memory.get('basis')=='torch-cuda-mem-get-info-v1'
             and all(type(memory.get(k)) is int and memory[k]>0 for k in ('free_bytes','total_bytes')))
        workers[rank]={'split':split,'native_inventory':inv,'allocator':alloc,'memory':memory}
    require('four-distinct-incarnations',len(incarnations)==4)
    need('one-retained-storage-epoch',len(storage)==1 and all(type(v) is int and v>0 for v in storage))
    if package is not None:
        try:
            root=Path(package);declared=manifest(root)
            actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            require('package-file-set',actual==set(declared)|{'SHA256SUMS'})
            generation=json.loads((root/'S5-OVERLAP-GENERATION.json').read_text())
            need('native-build',type(generation.get('native_build')) is dict)
            require('default-off',generation.get('default_feature')=='FLASH_S5_OVERLAP=0')
        except Exception as exc:
            missing.append('package-verification:'+type(exc).__name__)
    return {'schema':'s5-native-startup-grade/1','verdict':'FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
            'failures':failures,'missing':missing,'engine_epoch':epoch,'workers':workers,
            'scope':'actual queried construction, SM membership, bank, owners, memory and typed FULL fence; no performance qualification'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--package',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if Path(a.out).exists():return 2
    record={}
    try:
        client=NativeRuntimeClient(a.url,a.run_id)
        prepare=client.prepare('full','full-prefill-solo')
        identity=client.call('identity')
        record=grade(prepare,identity,epoch=client.epoch,package=a.package)
        record.update(prepare=prepare,identity=identity,resume=client.resume())
    except Exception as exc:
        record.update(verdict='INCONCLUSIVE',failures=[],missing=['native-startup-query'],error=exception_details(exc))
    with open(a.out,'x') as handle:json.dump(record,handle,sort_keys=True)
    print('NATIVE_STARTUP '+record['verdict'])
    return {'PASS':0,'FAIL':1}.get(record['verdict'],2)


if __name__=='__main__':raise SystemExit(main())
