#!/usr/bin/env python3
"""Assemble the machine delivery records for native35 P-b (results/hub/native35-p-b/)."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


W2_EVIDENCE=('No box W2 run of q18a-w2-overlap.json exists in results/box: all 19 queued attempts that reached it recorded '
    '"overlap NOT_RUN" (same-queue gates refused), and no W1 ran either, so there is no measured 200k-prefill duration or standing-decoder '
    'rate on this engine. Nearest box datum: L row4 (31 Oct) native prefill of the 4096-token placement plan (4167 prompt tokens) took '
    '83.25 s to its first token, including a 45.5 s span with no worker execute_model (reviews/claude-20261031-native35-l-row4-window.md).')


def precheck_summary(log):
    text=Path(log).read_text();summary=re.search(r'^SUMMARY checks=(\d+) pass=(\d+) fail=(\d+)$',text,re.M)
    assert summary and summary.group(1)==summary.group(2)=='15' and summary.group(3)=='0',log
    return summary.group(2)+'/'+summary.group(1)


def record(build,freeze,freeze_check,selfrun,compile_proof,precheck_conf,precheck_log,strict_log,focused_log,note,out):
    b=json.loads(Path(build).read_text());s=json.loads(Path(selfrun).read_text());c=json.loads(Path(compile_proof).read_text());f=json.loads(Path(freeze_check).read_text())
    assert b['arm']=='P-B' and s['verdict']=='PASS' and s['rc']==0 and s['bundle_manifest']==b['manifest'] and s['frozen_runner_sha256']==b['runner_sha256']
    assert s['tests_run']==b['expected_selfrun_tests'] and s['skips']==0 and s['native35_arm']=='P-B' and s['native35_performance']=='PASS' and s['native35_row4_l2']=='PASS' and s['gpu_device_requests']==0
    assert s['overlay_manifest']==b['overlay_manifest'] and s['owner_prelaunch']['verdict']=='PASS' and s['image']==b['image']
    assert c['verdict']=='PASS' and c['manifest']==b['manifest'] and c['runner_sha256']==b['runner_sha256']
    assert f['verdict']=='PASS' and f['manifest']==b['manifest'] and f['freeze_proof_sha256']==sha(freeze)
    strict=dict(b['regression'],log=str(strict_log),verdict='PASS',gpu_execution=False,scope='full strict registered CPU regression on the native35 P-b build source tree')
    assert strict['log_sha256']==sha(strict_log)
    focused=dict(b['focused'],log=str(focused_log));assert focused['log_sha256']==sha(focused_log)
    perf=Path(selfrun).with_name('native35-p-performance-check.json');slots=Path(selfrun).with_name('draft-window-check.json')
    details=json.loads((Path(b['bundle'])/'native30-provenance.json').read_text())['native35_p_b']
    row4=Path(selfrun).with_name('native35-p-b-row4-check.json')
    row=dict(name=b['name'],arm='P-B',source=b['source'],predecessor=details['predecessor'],row4=details['row4'],l2_runner_sha256=b['l2_runner_sha256'],
        row4_check=str(row4),row4_check_sha256=sha(row4),row4_cases=[r['case'] for r in s['native35_p_b_row4_receipts']]+sorted(r['case'] for r in s['native35_l2_timing_receipts']),bundle=b['bundle'],manifest=b['manifest'],overlay_manifest=b['overlay_manifest'],overlay_unchanged_from_o=True,
        coverage=b['checksummed_files'],runner=b['runner'],runner_sha256=b['runner_sha256'],runner_size=b['runner_size'],review_sha256=b['review_sha256'],image=b['image'],
        parent=dict(name='q18a-native35-o',manifest=b['parent_manifest'],runner_sha256=b['parent_runner_sha256']),
        selfrun_result=str(selfrun),selfrun_result_sha256=sha(selfrun),selfrun_tests=s['tests_run'],skips=s['skips'],selfrun_container=s['container'],
        performance_check=str(perf),performance_check_sha256=sha(perf),performance_cases=sorted(r['case'] for r in s['native35_performance_receipts']),
        draft_slots=str(slots),draft_slots_sha256=sha(slots),draft_slot_count=len(s['draft_window_receipts']),
        owner_prelaunch=s['owner_prelaunch'],freeze_proof=str(freeze),freeze_proof_sha256=sha(freeze),freeze_check=str(freeze_check),freeze_check_sha256=sha(freeze_check),
        freeze_differences=dict(bundle_changed=f['bundle_changed'],bundle_added=f['bundle_added'],bundle_removed=f['bundle_removed'],runner=f['runner_declared_differences'],
                                queue='freeze-proof.json queue_differences (unified diff of each P queue script vs O with names normalized)'),
        compile_proof=str(compile_proof),compile_proof_sha256=sha(compile_proof),compile_files=c['files'],compile_failures=len(c['failures']),
        precheck_conf=str(precheck_conf),precheck=precheck_summary(precheck_log),precheck_log=str(precheck_log),precheck_log_sha256=sha(precheck_log),
        window=dict(not_before=details['not_before'],hard_end=details['hard_end'],active_s=details['active_s'],cleanup_margin_s=details['cleanup_margin_s'],
                    source=details['window_source'],active_derivation=details['active_derivation'],cleanup_derivation=details['cleanup_derivation'],
                    latest_arm_utc='2026-11-02T01:45:00Z'),
        expected_runtime_s=details['expected_runtime_s'],
        w2_prefill_evidence=W2_EVIDENCE,
        owner_commands=['sudo python3 %s --staging --auth-env-file PRIVATE_AUTH_FILE --selfrun-record EXACT_RESULT_JSON --snapshot-max-bytes 536870912 --sync none'%Path(b['runner']).name,
                        'owner stops production (vllm-flash inactive/failed, four idle GPUs), then: sudo python3 %s --arm'%Path(b['runner']).name],
        note=str(note),note_sha256=sha(note),strict=strict,focused=focused,verdict='PASS',classification=None,gpu_execution=False,box_action=False,
        scope='frozen input gates and the fresh strict regression; no device run; the speed numbers come only from the box run, 80/80 is a reported marker, never a gate')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in (('delivery.json',row),('frozen-gates.json',{k:v for k,v in row.items() if k not in ('strict','focused')}),('strict.json',strict)):
        with (out/name).open('x') as fh:json.dump(value,fh,sort_keys=True,indent=2);fh.write('\n')
    print(json.dumps(dict(out=str(out),verdict='PASS',manifest=b['manifest'][:12]),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('build','freeze','freeze-check','selfrun','compile-proof','precheck-conf','precheck-log','strict-log','focused-log','note','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();record(a.build,a.freeze,a.freeze_check,a.selfrun,a.compile_proof,a.precheck_conf,a.precheck_log,a.strict_log,a.focused_log,a.note,a.out)
