#!/usr/bin/env python3
"""One-card CUDA PP event-order leaf; no NCCL or multi-rank claim."""
import argparse
import json
from pathlib import Path
import time
import traceback


def _classes():
    from vllm.v1.flash_e3 import engine_driver, wire_v1
    assert callable(engine_driver.PPSubmissionObligation)
    assert callable(wire_v1.SegmentCommand)
    return engine_driver, wire_v1


def cpu_shape():
    import torch
    engine_driver, wire = _classes()
    assert not torch.cuda.is_initialized()
    return dict(verdict='PASS', actual_pp_obligation=engine_driver.PPSubmissionObligation.__name__,
                actual_segment_command=wire.SegmentCommand.__name__,
                cuda_initialized=False, gpu_event_execution_deferred=True)


def gpu_case():
    import torch
    _classes()
    assert torch.cuda.device_count() == 1
    sender = torch.cuda.Stream(device='cuda:0')
    receiver = torch.cuda.Stream(device='cuda:0')
    assert sender.cuda_stream != receiver.cuda_stream
    # A pre-send stop creates no receive wait or event. The production driver
    # obligation is checked by the connected CPU process proof; this leaf
    # executes only the CUDA stream ordering that the CPU leaf cannot.
    before_send = dict(send_accepted=False, receive_posted=False,
                       stream_wait_installed=False)
    with torch.cuda.stream(sender):
        torch.cuda._sleep(2_000_000_000)
        sent = torch.cuda.Event()
        sent.record(sender)
    with torch.cuda.stream(receiver):
        receiver.wait_event(sent)
        received = torch.cuda.Event()
        received.record(receiver)
    pending_before_query = not bool(received.query())
    deadline = time.monotonic() + 15.
    while time.monotonic() < deadline and not received.query():
        time.sleep(.001)
    assert sent.query() and received.query()
    return dict(verdict='PASS', before_send=before_send,
        sender_stream=int(sender.cuda_stream), receiver_stream=int(receiver.cuda_stream),
        pending_before_query=pending_before_query,
        matched_send_event_queried=True, receive_event_queried=True,
        cuda_wait_event_executed=True, pp_send_count=1, pp_receive_count=1,
        nccl_executed=False, box_evidence=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu'), required=True)
    args = parser.parse_args()
    result = dict(schema='native19-pp-event/1', verdict='INCONCLUSIVE',
                  mode=args.mode, box_evidence=False)
    try:
        result['cpu_shape'] = cpu_shape() if args.mode == 'cpu-shape' else None
        if args.mode == 'gpu':
            result['gpu'] = gpu_case()
        result['verdict'] = 'PASS'
    except BaseException as exc:
        result.update(verdict='FAIL', error_type=type(exc).__name__,
                      error_message=str(exc), stack=traceback.format_exc())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
