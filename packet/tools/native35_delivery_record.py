#!/usr/bin/env python3
"""Assemble the machine delivery records for a native35 arm (results/hub/native35-<arm>-a/)."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def precheck_summary(log):
    text=Path(log).read_text();summary=re.search(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$',text,re.M)
    assert summary and summary.group(1)==summary.group(2) and summary.group(3)=='0',log
    return summary.group(2)+'/'+summary.group(1)


def record(build,freeze,selfrun,compile_proof,precheck_conf,precheck_log,strict_log,review,out):
    b=json.loads(Path(build).read_text());s=json.loads(Path(selfrun).read_text());c=json.loads(Path(compile_proof).read_text())
    assert s['verdict']=='PASS' and s['rc']==0 and s['bundle_manifest']==b['manifest'] and s['frozen_runner_sha256']==b['runner_sha256'] and s['tests_run']==b['expected_selfrun_tests'] and s['skips']==0
    assert c['verdict']=='PASS' and c['manifest']==b['manifest'] and c['runner_sha256']==b['runner_sha256']
    strict=dict(b['regression'],log=str(strict_log),verdict='PASS',gpu_execution=False,scope='full strict registered CPU regression on the native35 final tree')
    assert strict['log_sha256']==sha(strict_log)
    builder_view=s.get('native35_builder_view') or {}
    row=dict(name=b['name'],arm=b['arm'],source=b['source'],bundle=b['bundle'],manifest=b['manifest'],overlay_manifest=b['overlay_manifest'],coverage=b['checksummed_files'],
        runner=b['runner'],runner_sha256=b['runner_sha256'],runner_size=b['runner_size'],review_sha256=b['review_sha256'],
        selfrun_result=str(selfrun),selfrun_result_sha256=sha(selfrun),selfrun_tests=s['tests_run'],skips=s['skips'],
        readback_check=str(Path(selfrun).with_name('readback-check.json')),readback_check_sha256=sha(Path(selfrun).with_name('readback-check.json')),
        builder_view=str(Path(selfrun).with_name('native35-builder-view.json')),builder_view_sha256=sha(Path(selfrun).with_name('native35-builder-view.json')),
        builder_view_verdict=builder_view.get('verdict'),builder_repro=dict(old_shared_refusal=s['readback_check']['negative_refusal'],native_private=s['readback_check']['native_private_positive']['verdict'],ordinary_after_native=s['readback_check']['ordinary_after_native_state']),
        draft_slots=str(len(s['draft_window_receipts']))+' real-clock entries incl. the three native35 drafts plus the exact old inline slot refusal',
        freeze_proof=str(freeze),freeze_proof_sha256=sha(freeze),compile_files=c['files'],compile_failures=len(c['failures']),
        precheck_conf=str(precheck_conf),precheck=precheck_summary(precheck_log),precheck_log=str(precheck_log),precheck_log_sha256=sha(precheck_log),
        review=str(review),review_file_sha256=sha(review),strict=strict,verdict='PASS',classification=None,gpu_execution=False,box_action=False,
        scope='frozen input gates and the fresh strict regression; no device run; the deciding O-style box result owns the fix conclusion')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in (('delivery.json',row),('frozen-gates.json',{k:v for k,v in row.items() if k!='strict'}),('strict.json',strict)):
        with (out/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(out=str(out),verdict='PASS',manifest=b['manifest'][:12]),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('build','freeze','selfrun','compile-proof','precheck-conf','precheck-log','strict-log','review','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();record(a.build,a.freeze,a.selfrun,a.compile_proof,a.precheck_conf,a.precheck_log,a.strict_log,a.review,a.out)
