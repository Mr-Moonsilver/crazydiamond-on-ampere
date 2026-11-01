#!/usr/bin/env python3
"""Held-f exact pinned-container CPU engine gates; no GPU, network or container removal.

Cases, all against one bundle's overlay and sources:

* held-e's 18 cases unchanged (the connected auto-arm probe modes of
  e3_native41_auto_probe and the receipt probe modes), with
  FLASH_S5_NATIVE_PROGRESS unset;
* the eight connected progress probe modes (e3_native41_progress_probe, cases
  progress-<mode>): unset, trace, trace-skew, burst, burst-skew, burst-current,
  burst-solo, current-solo;
* the suites: held-e's engine suites plus the held-f queue and reply suites,
  one unittest process, per-test outcomes recorded.

Default-off comparison (``--compare``): every held-e case with a canonical
trace (trace, request_tokens, rpc_methods, cohort_tokens, ticks, geometry)
equals held-e2's frozen engine record byte for byte, and so do the progress
modes that must not change behaviour (unset, trace, burst-current against
held-e2's auto; trace-skew against its tp-skew).

Negative control (``--negative``): the candidate harness over held-e2's
(regenerated, byte-identical) overlay and sources. Both held-f suites run in
full: exactly the NEGATIVE_TESTS ids do not pass, each with its pinned outcome
and reason, and every other test passes, among them the two current-mode
head-of-line repros (the parent has the reply delay they show). The progress
probe needs held-f engine names in every mode, so all four progress cases fail
with their pinned errors; auto and off pass with held-e2's exact canonical traces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from native37_robust_selfrun import IMAGE, overlay_mounts
import native41_autoarm_selfrun as held_e

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'native41-progress-selfrun/1'
SUITES = held_e.SUITES + ('test_native41_progress_queue', 'test_e3_native_reply_hol')
PROGRESS_MODES = ('unset', 'trace', 'trace-skew', 'burst', 'burst-skew', 'burst-current', 'burst-solo', 'current-solo')
CASES = held_e.CASES + tuple('progress-' + m for m in PROGRESS_MODES)
CANONICAL = ('trace', 'request_tokens', 'rpc_methods', 'cohort_tokens', 'ticks', 'geometry')
# Progress modes whose behaviour must equal held-e2's: receipts only, or burst
# installed with the arm on current. Burst changes the tick schedule by design.
SAME_AS = {'progress-unset': 'auto', 'progress-trace': 'auto', 'progress-burst-current': 'auto',
           'progress-trace-skew': 'tp-skew'}
HELD_E2_ENGINE = Path('<hub-dir>/agents/flash-next/tmp/native41-autoarm-evidence-e2/frozen-engine')
HELD_E2_ENGINE_SHA = '00ceccaf2c7f23234af6a5d190ce6a3119524d7a79988d3a432515db29cb3e03'
HELD_E2_BUNDLE = Path('<hub-dir>/agents/flash-next/tmp/native41-autoarm-held-e2/'
                      'q18a-native41-autoarm-e-bundle-fc9fee4268dd')
# Negative (held-e2 engine under the candidate harness): both held-f suites run in full; exactly these
# tests do not pass, each with its pinned outcome and reason, and every other test passes.
NEGATIVE_SUITES = ('test_native41_progress_queue', 'test_e3_native_reply_hol')
# Pinned from the first negative run over held-e2's engine (dev bundle e3f22438e21e, run neg-discovery-1):
# each test needs a held-f engine name or argument the parent lacks. The InWaitPairBoundary pins (build-b)
# come from the same suite run over build-a's regenerated held-e2 parent.
NO_PAIR = "AttributeError: type object 'NativeSegmentDriver' has no attribute 'progress_in_wait'"
NEGATIVE_TESTS = {
    'test_e3_native_reply_hol.DirectReplies.test_direct_command_failure_poisons_at_once_ahead_of_decode':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.DirectReplies.test_direct_reply_is_accepted_only_as_the_exact_direct_head':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.DirectReplies.test_direct_reply_passes_the_pending_decode_output':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.DirectReplies.test_direct_reply_pickles_with_its_sequence':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.DirectReplies.test_each_class_stays_fifo_under_a_decode_stream':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.DirectReplies.test_off_has_no_direct_fifo_and_refuses_a_direct_submission':
        ('error', "AttributeError: 'ReplyTransport' object has no attribute '_direct'"),
    'test_e3_native_reply_hol.DirectReplies.test_wire_with_no_expectation_in_its_class_poisons':
        ('error', "AttributeError: module 'executor_hooks' has no attribute 'DirectReply'"),
    'test_e3_native_reply_hol.DirectReplies.test_worker_refuses_a_direct_bit_on_an_ineligible_method':
        ('error', "TypeError: install() got an unexpected keyword argument 'direct_replies'"),
    'test_e3_native_reply_hol.InWaitLoop.test_cap_bounds_sends_and_is_counted':
        ('error', "AttributeError: module 'engine_driver' has no attribute 'NativeProgress'"),
    'test_e3_native_reply_hol.InWaitLoop.test_deadline_ends_the_loop_without_a_send':
        ('error', "AttributeError: module 'engine_driver' has no attribute 'NativeProgress'"),
    'test_e3_native_reply_hol.InWaitLoop.test_refused_permit_and_ready_end_the_loop':
        ('error', "AttributeError: module 'engine_driver' has no attribute 'NativeProgress'"),
    'test_e3_native_reply_hol.InWaitLoop.test_wait_needs_burst_owner_thread_and_a_sample_last':
        ('error', "AttributeError: module 'engine_driver' has no attribute 'NativeProgress'"),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_a_permit_that_holds_after_the_drain_sends_once':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_a_refused_permit_never_drains':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_a_reply_not_yet_arrived_stops_the_send':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_a_reply_that_ends_the_layers_stops_the_send':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_a_stock_execute_after_the_drain_stops_the_send':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_the_leaf_holds_every_field_the_real_functions_read':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.InWaitPairBoundary.test_the_mutant_without_the_post_drain_recheck_sends':
        ('error', NO_PAIR),
    'test_e3_native_reply_hol.ProgressReceipts.test_line_parts_stay_below_the_docker_split_and_reassemble':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_e3_native_reply_hol.ProgressReceipts.test_setting_and_meta_contract':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'progress_mode'"),
    'test_e3_native_reply_hol.ProgressReceipts.test_trace_separates_rank2_reply_wait_from_worker_host_time':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_e3_native_reply_hol.ProgressReceipts.test_unrecorded_rpcs_carry_no_meta_and_leave_no_rows':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Arm.test_a_preemption_over_the_fresh_prompt_is_a_reason':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
    'test_native41_progress_queue.Arm.test_held_e_router_stops_the_arm_at_the_mode_switch':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
    'test_native41_progress_queue.Arm.test_mode_and_cohort_arms_through_the_exact_cli':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
    'test_native41_progress_queue.Receipts.test_dropped_rows_stray_lines_and_ingress':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Receipts.test_each_completeness_defect_is_inconclusive':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Receipts.test_each_wrong_arm_shape_is_inconclusive':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Receipts.test_eight_complete_arms_pass_with_the_exact_decomposition':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Receipts.test_grade_cli_reads_both_streams_and_never_overwrites':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'ProgressTrace'"),
    'test_native41_progress_queue.Receipts.test_schema_and_line_path_are_the_emitters':
        ('error', "AttributeError: module 'native_measurement' has no attribute 'PROGRESS_ROWS'"),
    'test_native41_progress_queue.Router.test_both_switches_forward_native_progress_to_the_real_session':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
    'test_native41_progress_queue.Router.test_every_other_setting_refuses_native_progress_exactly_as_held_e':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
    'test_native41_progress_queue.RouterTraceInstalled.test_an_engine_without_burst_refuses_the_switch':
        ('error', "AttributeError: 'NativeSegmentDriver' object has no attribute 'install_progress'"),
}
# Connected cases on the parent: (verdict, error_type, error). The progress probe needs held-f engine names in
# every mode, so even progress-unset fails there; the parent's own unset behaviour is held-e's auto and off,
# which pass with held-e2's exact canonical traces.
NO_WAIT = ('AttributeError', "'DriverWithCPUWorkers' object has no attribute 'progress_in_wait'")
NEGATIVE_CASES = {'auto': ('PASS', None, None), 'off': ('PASS', None, None),
                  'progress-unset': ('FAIL', 'AttributeError',
                                     "module 'vllm.v1.flash_e3.engine_driver' has no attribute '_progress_logger'"),
                  'progress-trace': ('FAIL',) + NO_WAIT, 'progress-burst': ('FAIL',) + NO_WAIT,
                  'progress-burst-solo': ('FAIL',) + NO_WAIT}

SUITES_PY = r'''import json,sys,traceback,unittest
from pathlib import Path
class Record(unittest.TextTestResult):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.rows={}
    def addSuccess(self,test):
        super().addSuccess(test);self.rows.setdefault(test.id(),('ok',''))
    def _bad(self,kind,test,err):
        last=''.join(traceback.format_exception_only(err[0],err[1])).strip().splitlines()
        self.rows[test.id()]=(kind,(last[-1] if last else '')[:400])
    def addFailure(self,test,err):
        super().addFailure(test,err);self._bad('fail',test,err)
    def addError(self,test,err):
        super().addError(test,err);self._bad('error',test,err)
    def addSubTest(self,test,sub,err):
        super().addSubTest(test,sub,err)
        if err is not None:self._bad('fail' if issubclass(err[0],test.failureException) else 'error',test,err)
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.rows[test.id()]=('skip',reason[:400])
if __name__ == '__main__':
    import torch
    files=list(Path('/source/harness').rglob('*.py'))
    for p in files: compile(p.read_bytes(),str(p),'exec')
    names=sys.argv[1:]
    r=unittest.TextTestRunner(verbosity=2,resultclass=Record).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    assert not torch.cuda.is_initialized()
    Path('/out/suites.json').write_text(json.dumps(dict(tests=r.testsRun,skips=len(r.skipped),failures=len(r.failures),
        errors=len(r.errors),ok=r.wasSuccessful(),compiled=len(files),cuda_initialized=False,
        outcomes={k:list(v) for k,v in sorted(r.rows.items())}),sort_keys=True))
    raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)
'''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(record):
    if not all(k in record for k in CANONICAL):
        return None
    return json.dumps({k: record[k] for k in CANONICAL}, sort_keys=True, separators=(',', ':')).encode()


def command_lines(modes, suites):
    lines = ['cd /source/harness']
    if suites:
        lines += ['python3 -B /out/suites.py ' + ' '.join(suites) + ' > /out/suites.log 2>&1', 'echo $? > /out/suites.rc']
    for mode in modes:
        if mode.startswith('progress-'):
            probe, arg, auto = 'e3_native41_progress_probe.py', mode[len('progress-'):], '1'
        elif mode.startswith('receipts-'):
            probe, arg = 'e3_native41_receipts_probe.py', mode[len('receipts-'):]
            auto = '0' if arg == 'off' else '1'
        else:
            probe, arg, auto = 'e3_native41_auto_probe.py', mode, '0' if mode == 'off' else '1'
        # FLASH_S5_NATIVE_PROGRESS is never set here: the progress probe passes its
        # setting to install_engine through its env argument.
        lines += ['env -u FLASH_S5_NATIVE_PROGRESS FLASH_S5_AUTO_ARM=%s python3 -B %s --mode %s --out /out/%s.json '
                  '> /out/%s.log 2>&1' % (auto, probe, arg, mode, mode), 'echo $? > /out/%s.rc' % mode]
    return lines


def run(bundle, out, name, overlay=None, sources=None, negative=False):
    from build_native30_held import verify
    bundle = bundle.resolve()
    verify(bundle)
    overlay = overlay.resolve() if overlay else bundle / 'native30_overlay'
    verify(overlay)
    sources = sources.resolve() if sources else bundle / 'e3_builder'
    out = out.resolve()
    out.mkdir()
    out.chmod(0o777)
    (out / 'suites.py').write_text(SUITES_PY)
    modes = tuple(NEGATIVE_CASES) if negative else CASES
    suites = NEGATIVE_SUITES if negative else SUITES
    (out / 'inner.sh').write_text('\n'.join(command_lines(modes, suites)) + '\n')
    c = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', name, '--runtime', 'runc', '--network', 'none',
         '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
         '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0',
         '-e', 'VLLM_CACHE_ROOT=/tmp/vllm-cache', '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=',
         '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts = dict(overlay_mounts(overlay))
    mounts.update({bundle: '/source/harness', sources: '/source/patches/s5_engine', bundle / 'native35_tools': '/source/tools',
                   (ROOT / 'reference/vllm-src').resolve(): '/source/reference/vllm-src'})
    for s, t in mounts.items():
        c += ['-v', '%s:%s:ro' % (s, t)]
    c += ['-v', '%s:/out' % out, '--entrypoint', 'bash', IMAGE, '/out/inner.sh']
    start = time.monotonic()
    with (out / 'container.log').open('x') as f:
        rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
    inspected = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]['HostConfig']
    assert not inspected.get('DeviceRequests') and not inspected.get('Devices') and inspected['NetworkMode'] == 'none'

    def read(n):
        return json.loads((out / n).read_text()) if (out / n).exists() else {}
    result = read('suites.json')
    cases = {mode: read(mode + '.json') for mode in modes}
    reasons = []
    if negative:
        outcomes = result.get('outcomes') or {}
        for test, (kind, text) in NEGATIVE_TESTS.items():
            got = outcomes.get(test)
            if not got or got[0] != kind or text not in got[1]:
                reasons.append('negative-test-differs:%s:%s' % (test, got))
        for test, (kind, text) in outcomes.items():
            if test not in NEGATIVE_TESTS and kind != 'ok':
                reasons.append('undeclared-negative-outcome:%s:%s:%s' % (test, kind, text))
        if not outcomes or not NEGATIVE_TESTS:
            reasons.append('negative-suites-not-run')
        for mode, expected in NEGATIVE_CASES.items():
            got = (cases[mode].get('verdict'), cases[mode].get('error_type'), cases[mode].get('error'))
            if got != expected:
                reasons.append('negative-case-differs:%s:%s' % (mode, got))
        assert sha(HELD_E2_ENGINE / 'result.json') == HELD_E2_ENGINE_SHA, 'held-e2 engine record changed'
        for mode, against in (('auto', 'auto'), ('off', 'off')):
            theirs = canonical(json.loads((HELD_E2_ENGINE / (against + '.json')).read_text()))
            if not cases[mode] or canonical(cases[mode]) != theirs:
                reasons.append('negative-trace-differs-from-held-e2:%s' % mode)
        ok = not reasons and result.get('skips') == 0
    else:
        ok = result.get('ok') is True and result.get('skips') == 0 and all(
            v.get('verdict') == 'PASS' for v in cases.values())
    record = dict(schema=SCHEMA, verdict='PASS' if (rc == 0 or negative) and ok else 'FAIL',
                  negative_control=negative, negative_kind='held-e2-parent' if negative else None,
                  classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', container_rc=rc, reasons=reasons,
                  suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  cases={m: dict(verdict=r.get('verdict'), checks=len(r.get('checks', [])), error_type=r.get('error_type'),
                                 error=(r.get('error') or '')[:400] or None,
                                 canonical_sha256=hashlib.sha256(canonical(r)).hexdigest() if r and canonical(r) else None,
                                 sha256=sha(out / (m + '.json')) if r else None) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources), image=IMAGE, network='none',
                  gpu_device_requests=0, gpu_execution=False, box_action=False, container=name,
                  wall_s=round(time.monotonic() - start, 3), command=c,
                  log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'))
    return 0 if record['verdict'] == 'PASS' else 1


def compare(candidate, out, parent=HELD_E2_ENGINE):
    """Default-off: canonical traces equal held-e2's frozen engine record."""
    assert sha(parent / 'result.json') == HELD_E2_ENGINE_SHA, 'held-e2 engine record changed'
    rows, problems = {}, []
    for mode in held_e.CASES + tuple(SAME_AS):
        ours = json.loads((candidate / (mode + '.json')).read_text())
        theirs = json.loads((parent / (SAME_AS.get(mode, mode) + '.json')).read_text())
        a, b = canonical(ours), canonical(theirs)
        if b is None:
            rows[mode] = dict(compared=False, why='no canonical trace in the held-e2 record')
            continue
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and a == b
        rows[mode] = dict(compared=True, same=same, against=SAME_AS.get(mode, mode),
                          canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    result = dict(schema='native41-progress-default-off-comparison/1', verdict='FAIL' if problems else 'PASS',
                  differs=problems, fields=CANONICAL, modes=rows, parent_record=str(parent / 'result.json'),
                  parent_record_sha256=HELD_E2_ENGINE_SHA, candidate=str(candidate),
                  candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with FLASH_S5_NATIVE_PROGRESS unset every held-e connected case has held-e2\'s exact '
                          'canonical trace; receipts only (trace) and burst installed with the arm on current '
                          'change nothing either')
    with out.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print(result['verdict'], problems)
    return 0 if not problems else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name')
    p.add_argument('--overlay', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--negative', action='store_true')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(a.compare.resolve(), a.out))
    raise SystemExit(run(a.bundle, a.out, a.name, a.overlay, a.sources, a.negative))
