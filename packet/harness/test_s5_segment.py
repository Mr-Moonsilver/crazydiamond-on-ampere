#!/usr/bin/env python3
"""Offline shape/verdict/queue checks and reproductions of decode_ab v3 findings.

No torch, CUDA, sockets, or engine. Review reproductions assert the *current*
undesirable verdict, not an approval of it; see reviews/codex-20261024f.md.
"""
import ast
from contextlib import redirect_stdout
import copy
import importlib.util
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(HERE))
from s5_segment import MODES, plan, grade_pair, combine


def fixture(pair="0,2"):
    ranks = []
    for rank in (0,1):
        modes = {}
        for mode in MODES:
            modes[mode] = dict(correct=True,decode_steps_s=100,prefill_steps_s=50 if "green" in mode else 100,
                               gap_p50_ms=10,gap_p99_ms=10,gap_max_ms=11,
                               prefill_covers_decode_tail=True,device_pcie={"available":True,"samples":20,"mean_rx_GB_s":5})
            if mode.startswith("green"):
                n = int(mode[5:])
                modes[mode].update(sm_graph_restriction=True,sm_split=[n,82-n])
            if mode=="serial":modes[mode].update(decode_steps_s=30,prefill_steps_s=70)
        ranks.append(dict(rank=rank,device_sms=82,peak_vram_gib=4,calibration=dict(decode_stage_ms=9,prefill_stage_ms=131),
                          cuda_visible_devices=pair,pci_bus_id=f'0000:0{int(pair.split(",")[rank])+1}:00.0',
                          cuda_graph_equal=True,finite=True,reduction_equal=True,gdn_dispatch_evidence=True,
                          clock_units_validated=True,modes=modes))
    return dict(pair=pair,run_spec={"fixture":True},correctness=True,ranks=ranks)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT/"reference/model-config.json").read_text())

    def test_real_shapes(self):
        p = plan(self.c)
        self.assertEqual(p["marlin_w13"],[256,160,2560])
        self.assertEqual(p["marlin_w2"],[256,40,5120])
        self.assertEqual(p["gdn_state"],[4,24,128,128])
        self.assertEqual(p["selection_width"],2051)
        self.assertEqual(p["expert_weight_bytes"],629145600)
        self.assertEqual(p["parked_hc_bytes"],26222592)
        self.assertEqual(p["measured_repeats"],1)
        self.assertEqual(tuple(p["modes"]),MODES)

    def test_mixes(self):
        for n in (4,8,12):
            p = plan(self.c,n)
            self.assertEqual(p["mix"].count("linear_attention"),3*n//4)
            self.assertEqual(p["mix"].count("full_attention"),n//4)

    def test_reject_wrong_config_or_scope(self):
        for key,val in (("hidden_size",4096),("mamba_ssm_dtype","bfloat16"),("ple_layer_ids",[])):
            c = copy.deepcopy(self.c);c["text_config"][key] = val
            with self.assertRaises(ValueError):plan(c)
        for n in (0,1,3,5,True):
            with self.assertRaises(ValueError):plan(self.c,n)
        with self.assertRaises(ValueError):plan(self.c,4,6)
        self.c["quantization_config"]["sym"] = False
        with self.assertRaises(ValueError):plan(self.c)

    def test_plan_cli_imports_no_gpu(self):
        p = subprocess.run([sys.executable,str(HERE/"s5_segment.py"),"--plan"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)["hidden"],2560)

    def test_gpu_code_compiles_as_python_without_importing(self):
        for name in ("s5_segment.py","s5_segment_gpu.py"):
            ast.parse((HERE/name).read_text())

    def test_source_manifest_matches_read_only_snapshot(self):
        d = json.loads((HERE/"s5_source_manifest.json").read_text())
        self.assertEqual(d["model_config_sha256"],hashlib.sha256((ROOT/"reference/model-config.json").read_bytes()).hexdigest())
        for module,sha in d["modules"].items():
            path = ROOT/"reference/vllm-src"/(module.removeprefix("vllm.").replace(".","/")+".py")
            self.assertEqual(sha,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_graph_replay_restores_captured_output_after_eager_check(self):
        tree = ast.parse((HERE/"s5_segment_gpu.py").read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DecodeGraph")
        ns = {}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),"<real DecodeGraph wrapper>","exec"),ns)
        out, eager = object(), object()
        lane = SimpleNamespace(last_output=eager)
        calls = []
        graph = SimpleNamespace(replay=lambda:calls.append("replay"),reset=lambda:calls.append("reset"))
        wrapped = ns["DecodeGraph"](graph,lane,out)
        wrapped.replay();self.assertIs(lane.last_output,out)
        wrapped.reset();self.assertEqual(calls,["replay","reset"])

    def test_queue_syntax_and_no_automatic_engine_restore(self):
        p = subprocess.run(["bash","-n",str(HERE/"runs/q7-draft.sh")],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        src = (HERE/"runs/q7-draft.sh").read_text()
        self.assertNotIn("systemctl start",src)
        self.assertNotIn("flash-reconfig.sh",src)
        self.assertIn("gpu_idle",src)


class VerdictTests(unittest.TestCase):
    def test_go_needs_both_pairs(self):
        self.assertEqual(combine([fixture(),fixture("1,3")])["verdict"],"GO")
        self.assertEqual(combine([fixture()])["verdict"],"INCONCLUSIVE")

    def test_no_go_on_one_pair(self):
        b = fixture("1,3")
        for m in b["ranks"][1]["modes"].values():
            m["gap_p99_ms"] = 20
        b["ranks"][1]["modes"]["decode"]["gap_p99_ms"] = 10
        self.assertEqual(combine([fixture(),b])["verdict"],"NO-GO")

    def test_best_split_prefers_decode_gate_then_prefill(self):
        d = fixture()
        for r in d["ranks"]:
            r["modes"]["green24"].update(decode_steps_s=94,prefill_steps_s=99)
            r["modes"]["green34"].update(prefill_steps_s=55)
        self.assertEqual(grade_pair(d)["best"]["mode"],"green34")

    def test_prefill_floor(self):
        d = fixture()
        for r in d["ranks"]:
            for name in ("green24","green34","green44"):
                r["modes"][name]["prefill_steps_s"] = 39
        self.assertEqual(grade_pair(d)["verdict"],"NO-GO")

    def test_normalized_total_work_must_not_regress(self):
        d = fixture()
        for r in d["ranks"]:r["modes"]["serial"].update(decode_steps_s=100,prefill_steps_s=100)
        self.assertEqual(grade_pair(d)["verdict"],"NO-GO")

    def test_gpu_clock_evidence_required(self):
        d = fixture();d["ranks"][0]["clock_units_validated"] = False
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_missing_pcie_evidence_is_inconclusive(self):
        d = fixture();d["ranks"][0]["modes"]["green24"]["device_pcie"] = {}
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_wrong_or_repeated_physical_pair_cannot_pass(self):
        a,b = fixture(),fixture("1,3")
        b["ranks"][0]["pci_bus_id"] = a["ranks"][0]["pci_bus_id"]
        self.assertEqual(combine([a,b])["verdict"],"INCONCLUSIVE")
        a["ranks"][0]["cuda_visible_devices"] = "1,3"
        self.assertEqual(grade_pair(a)["verdict"],"INCONCLUSIVE")

    def test_failure_file_wins_even_if_other_pair_missing(self):
        directory = Path(tempfile.mkdtemp(prefix="s5-failure-",dir=ROOT.parent/"tmp"))
        x,y = directory/"x.json",directory/"y.json"
        (directory/"x.json.rank0.error.json").write_text(json.dumps({"fatal_correctness":True}))
        p = subprocess.run([sys.executable,str(HERE/"s5_segment.py"),"--combine",str(x),str(y)],capture_output=True,text=True)
        self.assertEqual(p.returncode,1,p.stderr)
        self.assertEqual(json.loads(p.stdout)["verdict"],"FAIL")

    def test_calibration_miss_is_inconclusive(self):
        d = fixture();d["ranks"][0]["calibration"]["decode_stage_ms"] = 4
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_correctness_wins_over_calibration(self):
        d = fixture();d["ranks"][1]["finite"] = False
        d["ranks"][0]["calibration"] = {}
        self.assertEqual(grade_pair(d)["verdict"],"FAIL")
        self.assertEqual(combine([d])["verdict"],"FAIL")

    def test_missing_rank_mode_or_duplicate_pair(self):
        d = fixture();d["ranks"].pop()
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")
        d = fixture();d["ranks"][0]["modes"].pop("green44")
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")
        self.assertEqual(combine([fixture(),fixture()])["verdict"],"INCONCLUSIVE")

    def test_invalid_numbers(self):
        for x in (None,float("nan"),float("inf"),0,-1,True):
            d = fixture();d["ranks"][1]["modes"]["green24"]["decode_steps_s"] = x
            self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_graph_restriction_and_split_must_be_proven(self):
        for key,val in (("sm_graph_restriction",False),("sm_graph_restriction",None),("sm_split",[24,82])):
            d = fixture();d["ranks"][0]["modes"]["green24"][key] = val
            self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_no_dispatch_evidence_no_go_claim(self):
        d = fixture();d["ranks"][0]["gdn_dispatch_evidence"] = False
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_tail_must_remain_loaded_except_serial(self):
        d = fixture();d["ranks"][0]["modes"]["serial"]["prefill_covers_decode_tail"] = False
        self.assertEqual(grade_pair(d)["verdict"],"GO")
        d["ranks"][0]["modes"]["green24"]["prefill_covers_decode_tail"] = False
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")

    def test_mismatched_workloads_and_memory(self):
        d = fixture("1,3");d["run_spec"] = {"different":True}
        self.assertEqual(combine([fixture(),d])["verdict"],"INCONCLUSIVE")
        d = fixture();d["ranks"][0]["peak_vram_gib"] = 19
        self.assertEqual(grade_pair(d)["verdict"],"INCONCLUSIVE")


class DecodeReviewReproductions(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("review_decode_ab",HERE/"decode_ab.py")
        self.m = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.m.N,self.m.LEN,self.m.PLEN,self.m.LONG,self.m.MIN_MIXED = 2,70,1,1,1
        self.pre = dict(t0=0,t1=10,usage=dict(prompt_tokens=1,prompt_tokens_details=dict(cached_tokens=0)))

    def record(self):
        return dict(tokens=["token_id:7"]*70,lp=[-.01]*70,top=[{"token_id:7":-.01,"token_id:8":-2.} for _ in range(70)],
                    ts=[2.]*70,usage=dict(prompt_tokens=1,prompt_tokens_details=dict(cached_tokens=0)))

    def finish(self,runs):
        directory = Path(tempfile.mkdtemp(prefix="s5-review-",dir=ROOT.parent/"tmp"))
        out = directory/"record.json"
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ex:
            self.m.finish(runs,self.pre,True,str(out))
        return ex.exception.code,json.loads(out.read_text())

    def test_review_empty_saved_runs_are_inconclusive(self):  # decode_ab v4: was a false PASS
        rc,d = self.finish({k:[] for k in ("alone-a","alone-b","under-prefill")})
        self.assertEqual(rc,2)
        self.assertEqual(d["summary"]["final"],"INCONCLUSIVE")

    def test_review_noise_elsewhere_masks_clear_candidate_failure(self):
        runs = {k:[self.record(),self.record()] for k in ("alone-a","alone-b","under-prefill")}
        r = runs["alone-b"][0];r["lp"][0] = -.5;r["top"][0]["token_id:7"] = -.5
        r = runs["under-prefill"][1];r["tokens"][5] = "token_id:8"
        r["top"][5] = {"token_id:8":-.01,"token_id:7":-2.}
        rc,d = self.finish(runs)
        self.assertEqual(d["summary"]["noise"][1]["status"],"ok")
        self.assertEqual(d["summary"]["under_prefill"][1]["status"],"fail")
        self.assertEqual(rc,1,"decode_ab v4: a failure on a clean control stays FAIL")

    def test_failure_on_its_own_noisy_control_is_inconclusive(self):
        runs = {k:[self.record(),self.record()] for k in ("alone-a","alone-b","under-prefill")}
        r = runs["alone-b"][1];r["lp"][0] = -.5;r["top"][0]["token_id:7"] = -.5
        r = runs["under-prefill"][1];r["tokens"][5] = "token_id:8"
        r["top"][5] = {"token_id:8":-.01,"token_id:7":-2.}
        rc,d = self.finish(runs)
        self.assertEqual((rc,d["summary"]["per_prompt"]),(2,["ok","inconclusive"]))

    def test_stop_exclusion_does_not_hide_nonstop_wrong_winner(self):
        r = self.record()
        r["top"][0].update({"token_id:248046":0.,"token_id:9":-.001})
        self.assertIn("not the scored top-1",self.m.problems(r,1)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
