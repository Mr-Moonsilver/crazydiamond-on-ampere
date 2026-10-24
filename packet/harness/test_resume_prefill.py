#!/usr/bin/env python3
"""CPU-only S4 tests: real generated control flow with explicitly FAKE tensor/layer ops.

No torch import, sockets, inference or GPU. These prove protocol/control-flow
properties, not CUDA numerical equivalence, memory isolation or execution times.
"""
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import hashlib
import json
from itertools import islice
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patches/resume"))
from make_resume import generate, method, RUNNER_GUARD, transform_model, transform_runner
from protocol import ChunkKey, ChunkLedger, Completion, Phase, ProtocolError, StageSlot, Ticket

REF = ROOT / "reference/vllm-src"
GENERATED = ROOT / "patches/resume/generated"
KEY = ChunkKey("private-A", 1, 0, 7392, 1024)


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


def synthetic_segments():
    yield (0, 2, None)
    yield (2, 4, "stage-output")


class ProtocolTests(unittest.TestCase):
    def ledger(self, layers=2):
        return ChunkLedger(KEY, ((0, 4), (4, 7)), layers=layers)

    def complete(self, ledger, ticket, success=True):
        for rank in range(ledger.tp_size):
            ledger.acknowledge(Completion(ticket, success), rank)
        if ledger.handoff_stage is not None:
            source = ledger.handoff_stage
            for rank in range(ledger.tp_size):
                ledger.acknowledge_handoff(ledger.key, source, rank)

    def drain(self, ledger):
        for stage in range(len(ledger.ranges)):
            for rank in range(ledger.tp_size):
                ledger.acknowledge_quiescent(ledger.key, stage, rank)
        return ledger.release_cancelled()

    def test_commit_only_after_last_stage_all_ranks(self):
        ledger = self.ledger()
        ranges = []
        while ledger.phase is Phase.ACTIVE:
            ticket = ledger.issue()
            ranges.append((ticket.stage, ticket.lo, ticket.hi))
            with self.assertRaises(ProtocolError):
                ledger.commit()
            ledger.acknowledge(Completion(ticket, True), 1)
            self.assertEqual(ledger.computed_tokens, KEY.base_tokens)
            self.assertEqual(ledger.inflight, ticket)
            with self.assertRaises(ProtocolError):
                ledger.issue()
            ledger.acknowledge(Completion(ticket, True), 0)
            if ledger.handoff_stage is not None:
                source = ledger.handoff_stage
                with self.assertRaises(ProtocolError):
                    ledger.issue()
                for rank in range(ledger.tp_size):
                    ledger.acknowledge_handoff(KEY, source, rank)
        self.assertEqual(ranges, [(0, 0, 2), (0, 2, 4), (1, 4, 6), (1, 6, 7)])
        self.assertEqual(ledger.commit(), KEY.base_tokens + 1024)
        with self.assertRaises(ProtocolError):
            ledger.commit()

    def test_stage_partition_26_22_every_segment_size(self):
        for layers in (1, 2, 4, 8, 100):
            with self.subTest(layers=layers):
                ledger = ChunkLedger(KEY, layers=layers)
                seen = []
                while ledger.phase is Phase.ACTIVE:
                    ticket = ledger.issue()
                    self.assertLessEqual(ticket.hi-ticket.lo, layers)
                    seen.extend(range(ticket.lo, ticket.hi))
                    self.complete(ledger, ticket)
                self.assertEqual(seen, list(range(48)))
                self.assertEqual(ledger.computed_tokens, KEY.base_tokens)
                ledger.commit()

    def test_duplicate_foreign_and_stale_completion_rejected(self):
        ledger = self.ledger()
        ticket = ledger.issue()
        for wrong in (replace(ticket, key=replace(KEY, generation=2)),
                      replace(ticket, key=replace(KEY, chunk_id=1)),
                      replace(ticket, stage=1), replace(ticket, ordinal=9), replace(ticket, hi=3)):
            with self.assertRaises(ProtocolError):
                ledger.acknowledge(Completion(wrong, True), 0)
        with self.assertRaises(ProtocolError):
            ledger.acknowledge(ticket, 0)
        ledger.acknowledge(Completion(ticket, True), 0)
        with self.assertRaises(ProtocolError):
            ledger.acknowledge(Completion(ticket, True), 0)
        ledger.acknowledge(Completion(ticket, True), 1)
        ledger.issue()
        with self.assertRaises(ProtocolError):
            ledger.acknowledge(Completion(ticket, True), 1)

    def test_cancel_before_dispatch_and_at_every_boundary(self):
        for count in range(5):
            ledger = self.ledger()
            for _ in range(count):
                self.complete(ledger, ledger.issue())
            ledger.cancel()
            with self.assertRaises(ProtocolError):
                ledger.commit()
            with self.assertRaises(ProtocolError):
                ledger.release_cancelled()
            self.assertEqual(self.drain(ledger), KEY.base_tokens)
            self.assertEqual(ledger.phase, Phase.CANCELLED)

    def test_cancel_inflight_waits_for_all_writers(self):
        ledger = self.ledger()
        ticket = ledger.issue()
        ledger.acknowledge(Completion(ticket, True), 0)
        ledger.cancel()
        with self.assertRaises(ProtocolError):
            ledger.acknowledge_quiescent(KEY, 0, 1)
        ledger.acknowledge(Completion(ticket, True), 1)  # late success cannot commit
        self.assertEqual(self.drain(ledger), KEY.base_tokens)

    def test_failed_worker_completion_cancels_transaction(self):
        ledger = self.ledger()
        ticket = ledger.issue()
        ledger.acknowledge(Completion(ticket, False), 0)
        ledger.acknowledge(Completion(ticket, True), 1)
        self.assertEqual(ledger.phase, Phase.CANCELLING)
        self.assertEqual(self.drain(ledger), KEY.base_tokens)

    def test_old_generation_drain_cannot_release_new_owner(self):
        ledger = self.ledger()
        ledger.cancel()
        with self.assertRaises(ProtocolError):
            ledger.acknowledge_quiescent(replace(KEY, generation=2), 0, 0)
        with self.assertRaises(ProtocolError):
            ledger.release_cancelled()

    def test_pp_transfer_cancel_and_stale_receive(self):
        ledger = self.ledger(layers=4)
        ticket = ledger.issue()
        for rank in range(2):
            ledger.acknowledge(Completion(ticket, True), rank)
        self.assertEqual(ledger.handoff_stage, 0)
        with self.assertRaises(ProtocolError):
            ledger.issue()
        ledger.cancel()
        with self.assertRaises(ProtocolError):
            ledger.acknowledge_quiescent(KEY, 0, 0)
        with self.assertRaises(ProtocolError):
            ledger.acknowledge_handoff(replace(KEY, generation=2), 0, 0)
        ledger.acknowledge_handoff(KEY, 0, 0)
        with self.assertRaises(ProtocolError):
            ledger.release_cancelled()
        ledger.acknowledge_handoff(KEY, 0, 1)
        self.assertEqual(self.drain(ledger), KEY.base_tokens)

    def test_invalid_dimensions_and_key(self):
        for layers in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.ledger(layers)
        for ranges in ((), ((1, 4),), ((0, 4), (5, 8)), ((0, 0),)):
            with self.assertRaises(ValueError):
                ChunkLedger(KEY, ranges)
        for kwargs in ({"generation": 0}, {"chunk_tokens": 0}, {"request_id": ""}, {"base_tokens": True}):
            with self.assertRaises(ValueError):
                replace(KEY, **kwargs)

    def test_workspace_fence_and_context_rebinding(self):
        ledger = self.ledger()
        fence = Fence()
        context_log = []
        @contextmanager
        def bind():
            context_log.append("enter")
            yield
            context_log.append("exit")
        slot = StageSlot(KEY, 0, (0, 4), synthetic_segments(), {"metadata": "private"})
        ticket = ledger.issue()
        slot.advance(ticket, bind, lambda: fence)
        self.assertIsNone(slot.poll())
        self.assertEqual(slot.cursor, 0)
        with self.assertRaises(ProtocolError):
            slot.advance(ticket, bind, lambda: fence)
        with self.assertRaises(ProtocolError):
            slot.output(KEY)
        fence.ready = True
        self.assertEqual(slot.poll(), Completion(ticket, True))
        self.assertIsNone(slot.poll())
        self.complete(ledger, ticket)
        ticket2 = ledger.issue()
        slot.advance(ticket2, bind, lambda: Fence(True))
        self.assertEqual(slot.poll(), Completion(ticket2, True))
        self.assertEqual(slot.output(KEY), "stage-output")
        with self.assertRaises(ProtocolError):
            slot.output(KEY)  # each consumer must register its own fence before another borrow
        with self.assertRaises(ProtocolError):
            slot.close(KEY)
        consumer = Fence()
        slot.hold_output(consumer)
        with self.assertRaises(ProtocolError):
            slot.close(KEY)
        consumer.ready = True
        slot.close(KEY)
        self.assertIsNone(slot.resources)
        self.assertEqual(context_log, ["enter", "exit", "enter", "exit"])

    def test_workspace_cancel_keeps_resources_until_fenced(self):
        ledger = self.ledger()
        fence = Fence()
        resources = {"pages": [23, 42]}
        slot = StageSlot(KEY, 0, (0, 4), synthetic_segments(), resources)
        ticket = ledger.issue()
        slot.advance(ticket, nullcontext, lambda: fence)
        slot.cancel(KEY)
        with self.assertRaises(ProtocolError):
            slot.close(KEY)
        self.assertIs(slot.resources, resources)
        fence.ready = True
        self.assertEqual(slot.poll(), Completion(ticket, False))
        slot.close(KEY)
        self.assertIsNone(slot.resources)

    def test_workspace_cancel_without_submission(self):
        slot = StageSlot(KEY, 0, (0, 4), synthetic_segments(), object())
        slot.cancel(KEY)
        slot.close(KEY)
        with self.assertRaises(ProtocolError):
            slot.close(KEY)

    def test_workspace_wrong_generation_and_range(self):
        slot = StageSlot(KEY, 0, (0, 4), synthetic_segments(), object())
        ticket = self.ledger().issue()
        with self.assertRaises(ProtocolError):
            slot.advance(replace(ticket, key=replace(KEY, generation=2)), nullcontext, lambda: Fence(True))
        with self.assertRaises(ProtocolError):
            slot.cancel(replace(KEY, generation=2))
        with self.assertRaises(ProtocolError):
            slot.advance(replace(ticket, lo=1), nullcontext, lambda: Fence(True))
        slot.cancel(KEY)
        slot.close(KEY)

    def test_bad_segment_output_drains_as_failure(self):
        def wrong():
            yield (0, 3, None)
        slot = StageSlot(KEY, 0, (0, 4), wrong(), object())
        ticket = self.ledger().issue()
        with self.assertRaises(ProtocolError):
            slot.advance(ticket, nullcontext, lambda: Fence(True))
        completion = slot.poll()
        self.assertEqual(completion, Completion(ticket, False))
        slot.close(KEY)

    def test_exception_after_submission_requires_fence(self):
        def failing():
            raise RuntimeError("layer failure")
            yield
        slot = StageSlot(KEY, 0, (0, 4), failing(), object())
        fence = Fence()
        with self.assertRaisesRegex(RuntimeError, "layer failure"):
            slot.advance(self.ledger().issue(), nullcontext, lambda: fence)
        with self.assertRaises(ProtocolError):
            slot.close(KEY)
        fence.ready = True
        self.assertFalse(slot.poll().succeeded)
        slot.close(KEY)

    def test_missing_fence_quarantines_workspace(self):
        slot = StageSlot(KEY, 0, (0, 4), synthetic_segments(), object())
        with self.assertRaises(ProtocolError):
            slot.advance(self.ledger().issue(), nullcontext, lambda: None)
        slot.cancel(KEY)
        self.assertIsNone(slot.poll())
        with self.assertRaises(ProtocolError):
            slot.close(KEY)


class Tensor:
    """Tiny mutable CPU stand-in. Arithmetic deliberately has no CUDA/BF16 semantics."""
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)

    def repeat(self, _, count):
        return Tensor(self.value * count)

    def __add__(self, other):
        return Tensor(self.value + other.value)


class HC:
    def __init__(self, model):
        self.model = model

    def combine(self, hidden, block, injection):
        self.model.combines += 1
        return Tensor(hidden.value + (0 if block is None else block.value * injection.value))

    def combine_and_mix(self, hidden, block, injection):
        hidden = self.combine(hidden, block, injection)
        return hidden, Tensor(hidden.value + 7), None


class Layer:
    def __init__(self, model, index, ple=False):
        self.model, self.index = model, index
        self.ple = object() if ple else None
        self.mlp_hyper_connection = HC(model)

    def __call__(self, hidden_states, prev_block_output, prev_injection, positions,
                 input_ids, query_start_loc, ngram_context):
        m = self.model
        hidden = self.mlp_hyper_connection.combine(hidden_states, prev_block_output, prev_injection)
        if self.ple is not None:
            if m.prefetch is None or m.prefetch[0] != m.active:
                raise AssertionError("PLE prefetch was lost or belongs to another request")
            hidden = Tensor(hidden.value + m.prefetch[1])
            m.prefetch = None
        # Mutable scratch can be overwritten by a different decode between yields.
        block = Tensor((hidden.value + self.index + positions.value) % 103)
        inj = Tensor(self.index % 3 + 1)
        m.scratch = [hidden, block, inj]
        m.updates.setdefault(m.active, []).append((self.index, hidden.value, block.value))
        return hidden, block, inj


def load_model_methods(source):
    methods = [method(source, "Qwen4ExpModel", "forward")]
    if "    def flash_prefill_segments(" in source:
        methods.append(method(source, "Qwen4ExpModel", "flash_prefill_segments"))
    # Only the actual generated methods are compiled. All GPU/layer functions are stubs.
    ns = {"get_pp_group": lambda: GROUP, "IntermediateTensors": dict, "islice": islice}
    code = "from __future__ import annotations\n" + "\n".join(textwrap.dedent(m) for m in methods)
    exec(compile(code, "<actual-model-methods-with-fake-layers>", "exec"), ns)
    return ns


GROUP = SimpleNamespace(is_first_rank=True, is_last_rank=True)
MODEL_SOURCE = (GENERATED / "model.py.v1").read_text()
METHODS = load_model_methods(MODEL_SOURCE)


class Model:
    forward = METHODS["forward"]
    flash_prefill_segments = METHODS["flash_prefill_segments"]

    def __init__(self, lo=0, hi=5):
        self.start_layer, self.end_layer = lo, hi
        self.config = SimpleNamespace(hc_count=4)
        self.layers = [Layer(self, i, ple=(i == 1)) for i in range(hi)]
        self.hyper_connection_mixer = HC(self)
        self._mtp_hidden_buffer = None
        self.prefetch, self.scratch = None, []
        self.updates, self.combines, self.active = {}, 0, "A"
        self.prefetch_count = {}

    def embed_input_ids(self, ids):
        return Tensor(ids.value + 3)

    def _start_layer_ple_prefetch(self, layer, hidden, ids, starts, context):
        if layer.ple is not None:
            if self.prefetch is not None:
                raise AssertionError("unconsumed shared prefetch")
            self.prefetch = (self.active, ids.value + starts.value + context.value)
            self.prefetch_count[self.active] = self.prefetch_count.get(self.active, 0) + 1


def inputs(first=True):
    return dict(input_ids=Tensor(11) if first else None, positions=Tensor(17),
                query_start_loc=Tensor(2) if first else None,
                ngram_context=Tensor(5) if first else None,
                intermediate_tensors=None if first else {"hidden_states": Tensor(97)})


def value(output):
    return output["hidden_states"].value if isinstance(output, dict) else output.value


class GeneratedModelTests(unittest.TestCase):
    def setUp(self):
        GROUP.is_first_rank = GROUP.is_last_rank = True

    def run_segmented(self, model, args, n, interleave=False):
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS=str(n)):
            iterator = model.flash_prefill_segments(**args, flash_chunk_key=KEY.model_owner, flash_num_requests=1)
            outputs = []
            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                outputs.append(result)
                if result[2] is None and interleave:
                    self.assertIsNone(model.prefetch, "no shared PLE result may cross a yield")
                    # Scratch mutation emulates the next graph replay's buffer reuse.
                    for tensor in model.scratch:
                        tensor.value = -9999
                    model.active = "decode-B"
                    model.forward(**inputs(GROUP.is_first_rank))
                    model.active = "A"
                    # Runner input arrays are reusable too, not request-owned storage.
                    for name in ("input_ids", "positions", "query_start_loc", "ngram_context"):
                        if args[name] is not None:
                            args[name].value = -777
                    if args["intermediate_tensors"] is not None:
                        args["intermediate_tensors"]["hidden_states"].value = -777
            self.assertIsNone(model._flash_prefill_live)
            return outputs

    def test_exact_fake_output_and_updates_across_segments_and_decodes(self):
        for first, last, lo, hi in ((True, True, 0, 48), (True, False, 0, 26), (False, True, 26, 48)):
            GROUP.is_first_rank, GROUP.is_last_rank = first, last
            for n in (1, 2, 3, 4, 8, 64):
                with self.subTest(first=first, last=last, n=n):
                    baseline, candidate = Model(lo, hi), Model(lo, hi)
                    expected = baseline.forward(**inputs(first))
                    actual = self.run_segmented(candidate, inputs(first), n, interleave=True)
                    self.assertEqual(value(actual[-1][2]), value(expected))
                    self.assertEqual(candidate.updates["A"], baseline.updates["A"])
                    self.assertEqual(candidate.prefetch_count.get("A", 0), int(first))
                    self.assertEqual([i for a,b,_ in actual for i in range(a,b)], list(range(lo,hi)))
                    self.assertTrue(all(out is None for _,_,out in actual[:-1]))

    def test_no_extra_hc_combine_at_yield(self):
        baseline, candidate = Model(), Model()
        expected = baseline.forward(**inputs())
        actual = self.run_segmented(candidate, inputs(), 1)
        self.assertEqual(value(actual[-1][2]), value(expected))
        self.assertEqual(candidate.combines, baseline.combines)

    def test_cancel_each_boundary_then_reuse(self):
        for stop in range(1, 6):
            with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="1"):
                model = Model()
                iterator = model.flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1)
                for _ in range(stop):
                    next(iterator)
                self.assertIsNone(model.prefetch)
                iterator.close()  # synchronous FAKE kernels; real adapter must fence first
                self.assertIsNone(model._flash_prefill_live)
                self.assertEqual(len(model.updates["A"]), stop)
                model.active = "B"
                fresh = model.flash_prefill_segments(**inputs(), flash_chunk_key=("B", 2, 0), flash_num_requests=1)
                self.assertIsNotNone(list(fresh)[-1][2])

    def test_one_active_prefill_and_close_ownership(self):
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="1"):
            model = Model()
            a = model.flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1)
            next(a)
            b = model.flash_prefill_segments(**inputs(), flash_chunk_key=("B", 2, 0), flash_num_requests=1)
            with self.assertRaisesRegex(RuntimeError, "one parked"):
                next(b)
            self.assertEqual(model._flash_prefill_live, KEY.model_owner)
            a.close()

    def test_unsupported_inputs_fail(self):
        for n in ("0", "-1", "garbage"):
            with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS=n):
                with self.assertRaises(ValueError):
                    next(Model().flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1))
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="2"):
            for count in (0, 2, True):
                with self.assertRaises(ValueError):
                    next(Model().flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=count))
            for key in (("", 1, 0), ("A", 0, 0), ("A", True, 0)):
                with self.assertRaises(ValueError):
                    next(Model().flash_prefill_segments(**inputs(), flash_chunk_key=key, flash_num_requests=1))
            m = Model(); m._mtp_hidden_buffer = object()
            with self.assertRaises(NotImplementedError):
                next(m.flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1))
            with self.assertRaises(NotImplementedError):
                next(Model().flash_prefill_segments(**inputs(), deepstack_input_embeds={},
                     flash_chunk_key=KEY.model_owner, flash_num_requests=1))

    def test_layer_failure_quarantines_model_owner(self):
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="1"):
            model = Model()
            def broken(**kwargs):
                raise RuntimeError("simulated partial layer submission")
            broken.ple = None
            model.layers[0] = broken
            gen = model.flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1)
            with self.assertRaisesRegex(RuntimeError, "partial layer"):
                next(gen)
            gen.close()
            self.assertEqual(model._flash_prefill_live, KEY.model_owner)
            with self.assertRaisesRegex(RuntimeError, "one parked"):
                next(model.flash_prefill_segments(**inputs(), flash_chunk_key=("B", 2, 0), flash_num_requests=1))

    def test_generated_continuation_owner_is_held_until_cancel_fence(self):
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="1"):
            model = Model()
            iterator = model.flash_prefill_segments(**inputs(), flash_chunk_key=KEY.model_owner, flash_num_requests=1)
            slot = StageSlot(KEY, 0, (0, 5), iterator, {"private_metadata": object()})
            fence = Fence()
            ticket = Ticket(KEY, 0, 0, 0, 1)
            slot.advance(ticket, nullcontext, lambda: fence)
            slot.cancel(KEY)
            with self.assertRaises(ProtocolError):
                slot.close(KEY)
            self.assertEqual(model._flash_prefill_live, KEY.model_owner)
            fence.ready = True
            self.assertEqual(slot.poll(), Completion(ticket, False))
            slot.close(KEY)
            self.assertIsNone(model._flash_prefill_live)


class GeneratorTests(unittest.TestCase):
    def test_anchored_reproduction_and_hashes(self):
        self.assertTrue(REF.is_dir(), "read-only reference source is required; do not skip coverage")
        manifest = json.loads((GENERATED / "manifest.json").read_text())
        for name, transform in (("model.py.v1", transform_model), ("model_runner.py.v1", transform_runner)):
            entry = manifest["files"][name]
            source = (REF / entry["source"]).read_bytes()
            output = transform(source.decode()).encode()
            self.assertEqual(hashlib.sha256(source).hexdigest(), entry["source_sha256"])
            self.assertEqual(hashlib.sha256(output).hexdigest(), entry["output_sha256"])
            self.assertEqual(output, (GENERATED / name).read_bytes())
            compile(output, name, "exec")

    def test_original_forward_and_graph_execute_methods_unchanged(self):
        model = (REF / "models/qwen4_exp/nvidia/model.py").read_text()
        self.assertEqual(method(model, "Qwen4ExpModel", "forward"),
                         method(MODEL_SOURCE, "Qwen4ExpModel", "forward"))
        runner = (REF / "v1/worker/gpu/model_runner.py").read_text()
        patched = (GENERATED / "model_runner.py.v1").read_text()
        self.assertEqual(patched.replace(RUNNER_GUARD, "", 1), runner)
        for name in ("execute_model", "sample_tokens", "prepare_inputs"):
            self.assertEqual(method(runner, "GPUModelRunner", name), method(patched, "GPUModelRunner", name))

    def test_missing_duplicate_and_already_patched_anchors_fail(self):
        model = (REF / "models/qwen4_exp/nvidia/model.py").read_text()
        for changed in (model.replace("        if self.start_layer < self.end_layer:\n", "        if False:\n"),
                        model + model):
            with self.assertRaises(ValueError):
                transform_model(changed)
        runner = (REF / "v1/worker/gpu/model_runner.py").read_text()
        with self.assertRaises(ValueError):
            transform_runner(runner + runner)
        with self.assertRaises(ValueError):
            transform_model(MODEL_SOURCE)
        with self.assertRaises(ValueError):
            transform_runner((GENERATED / "model_runner.py.v1").read_text())

    def test_runner_guard_off_and_explicit_unsupported_on(self):
        ns = {}
        exec("def guard():\n" + textwrap.indent(textwrap.dedent(RUNNER_GUARD), "    "), ns)
        with patch.dict(os.environ, clear=True):
            self.assertIsNone(ns["guard"]())
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="0"):
            self.assertIsNone(ns["guard"]())
        with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS="2"):
            with self.assertRaisesRegex(NotImplementedError, "scheduler, private metadata"):
                ns["guard"]()
        for n in ("-1", "oops"):
            with patch.dict(os.environ, FLASH_PREFILL_SEGMENT_LAYERS=n):
                with self.assertRaises(ValueError):
                    ns["guard"]()

    def test_generator_cli_output_directory_is_never_overwritten(self):
        # Keep artifacts in the agent home. No cleanup/deletion of fixtures.
        parent = Path(tempfile.mkdtemp(prefix="resume-test-", dir=ROOT.parent / "tmp"))
        output = parent / "generated"
        generated = generate(REF, output)
        self.assertEqual(generated, json.loads((GENERATED / "manifest.json").read_text()))
        with self.assertRaises(FileExistsError):
            generate(REF, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
