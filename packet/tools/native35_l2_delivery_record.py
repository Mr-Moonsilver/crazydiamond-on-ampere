#!/usr/bin/env python3
"""Assemble the machine delivery records for native35 L2 (results/hub/native35-l2-a/)."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def precheck_summary(log):
    text=Path(log).read_text();summary=re.search(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$',text,re.M)
    assert summary and summary.group(1)==summary.group(2) and summary.group(3)=='0',log
    return summary.group(2)+'/'+summary.group(1)


def owner_commands(b):
    pin=b['manifest'][:12];runner=Path(b['runner']).name
    return dict(note='box-side owner only; the L procedure with L2 names: held bundle at H=/home/<service-user>/harness/'+pin+', runner '+runner+' and the hub self-run result copied byte-exact to /home/<service-user>/s0/q18a-native35-l2-selfrun-record.json; never edit them',
        staging='python3 -B <box path>/'+runner+' --staging --auth-env-file <owner-minted private env file> --selfrun-record /home/<service-user>/s0/q18a-native35-l2-selfrun-record.json',
        arm='python3 -B <box path>/'+runner+' --arm',
        order='staging (engine untouched; receipt /home/<service-user>/s0/q18a-native35-l2-staging.json) -> owner stops the engine -> arm (ARM guard: stopped unit, idle GPUs, native35_gate.lifetime_window_ok: 7200 s active + 420 s cleanup before HARD_END 2026-11-02T06:00Z)')


def record(build,freeze,selfrun,compile_proof,precheck_conf,precheck_log,strict_log,focused_log,review,out):
    b=json.loads(Path(build).read_text());s=json.loads(Path(selfrun).read_text());c=json.loads(Path(compile_proof).read_text());f=json.loads(Path(freeze).read_text())
    assert b['arm']=='L2' and b['name']=='q18a-native35-l2'
    assert s['verdict']=='PASS' and s['rc']==0 and s['bundle_manifest']==b['manifest'] and s['frozen_runner_sha256']==b['runner_sha256'] and s['tests_run']==b['expected_selfrun_tests'] and s['skips']==0
    assert s['native35_arm']=='L2' and s['native35_l2']=='PASS' and s['gpu_device_requests']==0 and s['overlay_manifest']==b['overlay_manifest']
    assert c['verdict']=='PASS' and c['manifest']==b['manifest'] and c['runner_sha256']==b['runner_sha256']
    assert f['verdict']=='PASS' and f['manifest']==b['manifest'] and f['overlay_byte_identical'] is True and f['engine_changed']==[]
    strict=dict(b['regression'],log=str(strict_log),verdict='PASS',gpu_execution=False,scope='full strict registered CPU regression on the native35 L2 final tree (test_native35_l2.py is not registered in selftest.py; it runs in the focused log and the pinned-image probe)')
    assert strict['log_sha256']==sha(strict_log)
    focused=dict(b['focused'],log=str(focused_log));assert focused['log_sha256']==sha(focused_log)
    l2check=Path(selfrun).with_name('native35-l2-check.json')
    row=dict(name=b['name'],arm=b['arm'],source=b['source'],bundle=b['bundle'],manifest=b['manifest'],overlay_manifest=b['overlay_manifest'],coverage=b['checksummed_files'],
        parent_manifest=b['parent_manifest'],runner=b['runner'],runner_sha256=b['runner_sha256'],runner_size=b['runner_size'],review_sha256=b['review_sha256'],
        selfrun_result=str(selfrun),selfrun_result_sha256=sha(selfrun),selfrun_tests=s['tests_run'],skips=s['skips'],image=s['image'],network='none',
        l2_check=str(l2check),l2_check_sha256=sha(l2check),l2_timing=s['native35_l2_timing_receipts'],l2_cli=s['native35_l2_cli_receipts'],
        draft_slots=str(len(s['draft_window_receipts']))+' real-clock entries incl. the R/O/L drafts, the L2 draft and the exact old inline slot refusal',
        window=dict(source='native35_gate.lifetime_window_ok(time.time())',not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=7200,cleanup_margin_s=420),
        freeze_proof=str(freeze),freeze_proof_sha256=sha(freeze),freeze_changed=f['changed'],freeze_added=f['added'],freeze_removed=f['removed'],
        compile_files=c['files'],compile_failures=len(c['failures']),
        precheck_conf=str(precheck_conf),precheck=precheck_summary(precheck_log),precheck_log=str(precheck_log),precheck_log_sha256=sha(precheck_log),
        review=str(review),review_file_sha256=sha(review),focused=focused,strict=strict,owner_commands=owner_commands(b),
        unproved=['box behaviour of the resume-receipt window, observer_end and the row4 regrade over the ~83 s observer interval under the 512 MiB snapshot cap (never reached on any box run of this collector)',
                  'the 45.5 s engine idle before the native handback step seen in L row4 (engine observation, not a gate input)',
                  'lifecycle/hygiene/leak phases (never executed on the box: gated on row4 PASS in native28 and L)'],
        verdict='PASS',classification=None,gpu_execution=False,box_action=False,
        scope='frozen input gates and the fresh strict regression; no device run; the box L2 run owns the row4/lifetime/leak conclusions')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in (('delivery.json',row),('frozen-gates.json',{k:v for k,v in row.items() if k!='strict'}),('strict.json',strict)):
        with (out/name).open('x') as handle:json.dump(value,handle,sort_keys=True,indent=2);handle.write('\n')
    print(json.dumps(dict(out=str(out),verdict='PASS',manifest=b['manifest'][:12]),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('build','freeze','selfrun','compile-proof','precheck-conf','precheck-log','strict-log','focused-log','review','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();record(a.build,a.freeze,a.selfrun,a.compile_proof,a.precheck_conf,a.precheck_log,a.strict_log,a.focused_log,a.review,a.out)
