#!/usr/bin/env python3
"""Connected profile rejection through the actual pinned host scheduler.

Worker model/tensor/event/PP/graph/MQ leaves are synthetic and more convenient
than the engine. This proves host control/lifetime ordering, not box readiness.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def run_host(cfg, scheduler, leases, owner, cache_record):
    from e3_native_cow_lease_probe import engine_request, host_preprocess
    from vllm.v1.flash_e3 import scheduler_hooks as sh
    from vllm.v1.flash_e3 import admission_metadata as am
    import async_handoff as ah
    import executor_hooks as eh
    from vllm.v1.request import RequestStatus
    from vllm.v1.engine import FinishReason
    from vllm.v1.outputs import ModelRunnerOutput
    import test_e3_native_schedule_profile as proof

    class MountedProfile(proof.ScheduleProfile):
        @classmethod
        def setUpClass(cls):
            # Execute the actual mounted helper source with declared tensor and
            # kernel leaves. No builder generation or reference-path substitute.
            cls.generated_ops = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py').read_text()
            cls.generated_qsa = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py').read_text()

    MountedProfile.setUpClass()
    case = MountedProfile('runTest')
    case.setUp()
    try:
        case.selection_counts = (4097, 97153)
        request = host_preprocess(owner, engine_request('profile-refusal', [1] * 65, 'refusal', 4))
        scheduler.add_request(request)
        controller = sh.PrefillAdmissionController(
            scheduler, leases=leases, frontier=ah.CommittedFrontier(),
            engine_epoch=7, chunk_tokens=64, ranks=(0, 1, 2, 3), minimum_stage_rows=64).attach()
        am.install_admission_metadata(scheduler)
        network = proof.ProfileNetwork(case, scheduler, controller)
        ticks = network.drain_profile_refusal()
        assert network.driver._profile_failure[1] == 'unsupported-native-schedule-profile'
        assert network.driver._profile_timeouts == 0
        counter = network.driver.qsa_control.snapshot()['layers'][0]
        assert counter['boundary_commands'] == counter['profile_polls'] == counter['profile_mismatches'] == 1
        assert counter['bucket_commands'] == counter['profile_timeouts'] == 0
        assert network.driver._last_native_closure['complete']
        assert leases.shutdown()['clean'] and not network.transport._calls
        assert not network.driver._split_ready and not controller.active
        assert not request.is_finished(), 'output was consumed outside the engine step'
        # Actual pinned EngineCore idle seam: the parked request leaves no
        # ordinary waiting work, yet its ERROR must be delivered exactly once.
        from types import SimpleNamespace
        engine = SimpleNamespace(_flash_s5_driver=network.driver, scheduler=scheduler)
        output = eh.engine_idle_tick(engine)
        assert output is not None
        rows, did_work = output
        assert did_work is False and list(rows) == [request.client_index]
        message = rows[request.client_index].outputs[0]
        assert message.request_id == request.request_id
        assert message.finish_reason == FinishReason.ERROR
        assert message.stop_reason == 'unsupported-native-schedule-profile'
        assert message.new_token_ids == []
        assert request.status == RequestStatus.FINISHED_ERROR
        assert network.driver.error is None and not network.driver.active
        assert eh.engine_idle_tick(engine) is None
        assert request.request_id not in scheduler.requests
        # Subsequent real scheduler/allocator request must still complete.
        # ModelRunnerOutput below is a declared synthetic CPU sampling leaf.
        next_request = host_preprocess(owner, engine_request('after-refusal', [2, 3, 4], 'next', 4))
        scheduler.add_request(next_request)
        for step in range(40):
            out = scheduler.schedule(False)
            ids = list(out.num_scheduled_tokens)
            tokens = [[17] if scheduler.requests[k].num_computed_tokens >= scheduler.requests[k].num_prompt_tokens else [] for k in ids]
            scheduler.update_from_output(out, ModelRunnerOutput(req_ids=ids,
                req_id_to_index={k: i for i, k in enumerate(ids)}, sampled_token_ids=tokens))
            if next_request.is_finished():
                break
        assert next_request.is_finished() and next_request.num_output_tokens == 4
        assert leases.shutdown()['clean'] and not network.transport._calls
        return dict(verdict='PASS', actual_scheduler=type(scheduler).__module__+'.'+type(scheduler).__name__,
            actual_request=type(request).__module__+'.'+type(request).__name__,
            drain_ticks=ticks, all_rank_close=network.driver._last_native_closure,
            mismatch_count=network.driver._profile_mismatches, timeout_count=network.driver._profile_timeouts,
            qsa_control=network.driver.qsa_control.snapshot(), error_delivered_once=True,
            request_error_reason=message.stop_reason, driver_error=None,
            subsequent_request_completed=True, subsequent_tokens=next_request.num_output_tokens,
            leases_clean=True, credits_returned=True, cache_configuration=cache_record,
            runtime_methods_patched=False, worker_model_tensor_event_leaves_synthetic=True,
            sampling_output_synthetic=True, box_evidence=False)
    finally:
        case.doCleanups()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('cpu-full',), default='cpu-full')
    args = p.parse_args()
    start = time.monotonic()
    record = dict(verdict='INCONCLUSIVE', rc=2, failure_class='NONE', sections={}, box_evidence=False)
    try:
        # Same mounted peer namespace as production, with proof-only fixtures
        # on a separate path. No runtime copy exists in the fixture tree.
        sys.path[:0] = ['/proof/harness', '/proof/builders', '/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3']
        graph = json.loads(args.graph.read_text())
        for row in graph['modules']:
            name = row['module']
            if name.startswith(('vllm.v1.flash_e3.', 'vllm.v1.flash_e2.')) and row['active_mount']:
                module = importlib.import_module(name)
                leaf = name.rsplit('.', 1)[-1]
                prior = sys.modules.setdefault(leaf, module)
                assert prior is module, leaf
        pins = []
        for name in ('native_measurement', 'engine_driver', 'executor_hooks', 'wire_v1', 'qsa_operation', 'qsa_driver', 'lane_dispatch', 'worker_bridge'):
            module = importlib.import_module(name)
            path = Path(inspect.getfile(module))
            expected = next(x for x in graph['modules'] if x['destination'] == str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected['sha256'], name
            pins.append(dict(module=name, origin=str(path), sha256=digest))
        record['sections']['mounted_sources'] = pins
        args.out.write_text(json.dumps(record, indent=2)+'\n')
        from native18_scheduler_fixture import with_actual_scheduler
        record['sections']['profile_refusal'] = with_actual_scheduler(run_host)
        import torch
        assert not torch.cuda.is_initialized()
        record.update(verdict='PASS', rc=0, cuda_initialized=False)
    except Exception as exc:
        record.update(verdict='FAIL', rc=1, failure_class='OPEN', error_type=type(exc).__name__,
                      error_message=str(exc), exception_chain=traceback.format_exc())
    record['wall_s'] = time.monotonic() - start
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record.get(k) for k in ('verdict', 'rc', 'failure_class', 'error_type', 'error_message', 'wall_s')}), flush=True)
    return record['rc']


if __name__ == '__main__':
    raise SystemExit(main())
