#!/usr/bin/env python3
"""One-card receipt proof for the real NativeSegmentDriver refusal path.

The two local profiles are deliberate CPU fixture leaves. CUDA establishes the
one-card venue, but this proof does not diagnose the native26 TP mismatch.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_ast(source, method):
    tree = ast.parse(Path(source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'NativeSegmentDriver')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == method)
    return ast.dump(node, include_attributes=False)


def run_receipt_case(out, graph):
    from vllm.v1.flash_e3 import engine_driver as mounted
    source = Path('/proof/patches/s5_engine/engine_driver.py')
    generated = Path(mounted.__file__)
    assert sha(generated) == graph['source_hashes']['e3/engine_driver.py']
    for name in ('_refuse_profile', '_profile_refusal_receipt',
                 '_status_record', '_record'):
        assert method_ast(source, name) == method_ast(generated, name), name
    # The source module is the real driver class, with the same relevant method
    # bodies as the mounted package. Its existing CPU fixture supplies typed
    # observations and a deterministic reply transport; no driver method is patched.
    sys.path[:0] = ['/proof/harness', '/proof/patches/s5_engine']
    import test_e3_native_profile_observability as test
    case = test.ProfileObservability('runTest')
    case.setUpClass()
    case.setUp()
    try:
        case.case.selection_counts = (4097, 97153)
        world, driver, command = case.case.boundary((True, True))
        observations = case.case.query(world, driver)
        driver._apply_profile(observations)
        event = next(row for row in driver._journal
                     if row['kind'] == 'schedule_profile_refused')
        assert event['reason'] == 'unsupported-native-schedule-profile'
        assert [p['selected_count'] for p in event['peer_profiles']] == [4097, 97153]
        for index in range(driver.journal_limit + 1):
            driver._record('vm-receipt-overflow', index=index)
        assert not any(row['kind'] == 'schedule_profile_refused'
                       for row in driver._journal)
        retained = driver._status_record()['schedule_profile_refusals']
        assert retained['count'] == 1
        assert retained['first'] == retained['latest'] == event
        cleanup = {'steps': {'leave_full': {
            'census_before_return': {'snapshot': driver._status_record()}}}}
        import e3_native27_profile_receipt as receipt
        phase = receipt.from_cleanup(cleanup)
        assert phase == retained
        (out / 'qual-row1.json').write_text(json.dumps({
            'verdict': 'INCONCLUSIVE', 'schedule_profile_refusals': phase},
            sort_keys=True) + '\n')
        grade = receipt.queue_grade_receipts(out)
        assert grade['phases']['row1'] == retained and grade['unavailable'] == {}
        (out / 'queue-grade.json').write_text(json.dumps(grade, sort_keys=True) + '\n')
        return {'verdict': 'PASS', 'real_driver_class':
                driver.__class__.__module__ + '.' + driver.__class__.__name__,
                'mounted_driver_sha256': sha(generated),
                'source_driver_sha256': sha(source),
                'method_bodies_equal': True,
                'journal_limit': driver.journal_limit,
                'journal_dropped': driver._journal_dropped,
                'journal_refusal_evicted': True,
                'status_count': retained['count'],
                'status_peer_selected_counts': [p['selected_count']
                                                for p in retained['first']['peer_profiles']],
                'status_peer_bucket_waves': [p['bucket_waves']
                                             for p in retained['first']['peer_profiles']],
                'phase_receipt_exact': phase == retained,
                'queue_grade_exact': grade['phases']['row1'] == retained,
                'qualification_sha256': sha(out / 'qual-row1.json'),
                'grade_sha256': sha(out / 'queue-grade.json')}
    finally:
        case.doCleanups()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('gpu', 'cpu-shape'), required=True)
    args = parser.parse_args()
    out = args.out.parent
    graph = json.loads(args.graph.read_text())
    record = {'schema': 'native27-profile-vm-probe/1', 'verdict': 'FAIL',
              'sections': {}, 'failures': [], 'gpu_execution': False,
              'engine_constructed': False, 'box_action': False}
    try:
        record['sections']['receipt_path'] = run_receipt_case(out, graph)
        if args.mode == 'gpu':
            import torch
            tensor = torch.arange(64, device='cuda:0', dtype=torch.int32)
            assert int(tensor.sum().item()) == 2016
            torch.cuda.synchronize()
            record['gpu_execution'] = True
        record['verdict'] = 'PASS'
    except Exception as exc:
        record['failures'] = [type(exc).__name__ + ': ' + str(exc)]
        record['traceback'] = traceback.format_exc()
    args.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print('NATIVE27_PROFILE ' + record['verdict'], flush=True)
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
