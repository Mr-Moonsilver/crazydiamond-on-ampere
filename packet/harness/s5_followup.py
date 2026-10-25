#!/usr/bin/env python3
"""Offline q11 liveness grader and full-slice measurement reader. No GPU imports.

PASS for q11 means both diagnostic modes finished correctly on all four ranks,
not that the S5 service/latency targets passed. Missing or malformed evidence is
INCONCLUSIVE. Explicit correctness failure wins over missing/partial evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from s5_segment import PAIRS, combine, EXIT

IMAGE = "sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b"
LIVENESS_MODES = ("prio", "green24")


def decode_json(text):
    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON number')
        return number
    return json.loads(text, parse_float=real,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def load(path):
    return decode_json(Path(path).read_text())


def numeric(x, positive=False):
    try:
        return type(x) in (int, float) and math.isfinite(x) and (x > 0 if positive else True)
    except OverflowError:
        return False


def mapping(x):
    return x if isinstance(x, dict) else {}


def records(x):
    return x if isinstance(x, list) else []


def explicit_failure(obj):
    if not isinstance(obj, dict):
        return False
    if obj.get("fatal_correctness") is True or mapping(obj.get("grade")).get("verdict") == "FAIL":
        return True
    if any(obj.get(k) is False for k in ("correctness", "correct", "cuda_graph_equal", "finite", "reduction_equal")):
        return True
    return (any(explicit_failure(r) for r in records(obj.get("ranks")))
            or any(explicit_failure(m) for m in mapping(obj.get("modes")).values())
            or explicit_failure(obj.get("partial_rank")))


def rank_progress(pairfile, rank):
    base = str(pairfile) + f".rank{rank}"
    row = {"rank": rank, "phase": "unavailable", "last_lane_events": {}, "errors": []}
    try:
        p = load(base + ".progress.json")
        if not isinstance(p, dict) or p.get("rank") != rank:
            raise ValueError("progress identity")
        row.update(phase=p.get("phase", "unavailable"), timeout_s=p.get("timeout_s"))
    except (ValueError, OSError, TypeError):
        row["errors"].append("missing/malformed progress snapshot")
    try:
        # Keep only the final issue events per lane. These are host issue order,
        # never claims of GPU completion of individual collectives.
        with Path(base + ".progress.jsonl").open() as history:
            for line in history:
                e = decode_json(line)
                if e.get("rank") != rank:
                    raise ValueError("event rank")
                if e.get("kind") == "lane" and e.get("lane") in ("decode", "prefill"):
                    tail = row["last_lane_events"].setdefault(e["lane"], [])
                    tail.append({k: e[k] for k in ("seq", "phase", "stage", "step", "collective", "backend", "monotonic_ns") if k in e})
                    tail[:] = tail[-4:]
    except (ValueError, OSError, TypeError, AttributeError):
        row["errors"].append("missing/malformed progress history")
    err = Path(base + ".error.json")
    if err.exists():
        try:
            e = load(err)
            row["failure_artifact"] = str(err)
            row["watchdog"] = e.get("watchdog", False) is True or "watchdog" in str(e.get("error", "")).lower()
            row["fatal_correctness"] = explicit_failure(e)
            # Do not echo free-form errors/tracebacks; leave those in the source artifact.
        except (ValueError, OSError, AttributeError, TypeError):
            row["errors"].append("malformed rank error")
    return row


def pair_evidence(path, mode=None):
    result = {"artifact": str(path), "rank_progress": [rank_progress(path, i) for i in (0, 1)],
              "reasons": [], "fatal_correctness": False}
    reasons = result["reasons"]
    for error in path.parent.glob(path.name + ".rank*.error.json"):
        reasons.append("rank error artifact: " + error.name)
        try:
            result["fatal_correctness"] |= explicit_failure(load(error))
        except (ValueError, OSError, AttributeError, TypeError):
            reasons.append("malformed rank error: " + error.name)
    try:
        d = load(path)
        result["fatal_correctness"] |= explicit_failure(d)
        if not isinstance(d, dict):
            raise ValueError("pair document")
        result["data"] = d
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if mode is None:
            return result
        spec = d.get("run_spec", {})
        if not isinstance(spec, dict):
            raise ValueError("run specification")
        expected = {"modes": [mode], "steps": 60, "layers": 4, "decode_rows": 4,
                    "chunk": 1024, "history_decode": 153537, "history_prefill": 200000,
                    "gdn_decode": "packed", "pp_partition": [24, 24],
                    "collectives": {"decode": "CustomAllreduce", "prefill": "PyNccl"}}
        for k, v in expected.items():
            if spec.get(k) != v:
                reasons.append("q11 specification mismatch: " + k)
        manifest = load(Path(__file__).with_name("s5_source_manifest.json"))
        if spec.get("model_config_sha256") != manifest["model_config_sha256"]:
            reasons.append("model config identity not pinned")
        if d.get("correctness") is not True:
            reasons.append("pair correctness not established")
        ranks = d.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 2 or any(not isinstance(r, dict) for r in ranks):
            raise ValueError("exactly two rank objects required")
        if any(type(r.get("rank")) is not int for r in ranks) or sorted(r.get("rank", -1) for r in ranks) != [0, 1]:
            reasons.append("distinct rank IDs 0 and 1 required")
        for r in ranks:
            if r.get("source_sha256") != manifest["modules"]:
                reasons.append("rank sources differ from pinned manifest")
            for key in ("diagnostic_completed", "cuda_graph_equal", "finite", "reduction_equal",
                        "clock_units_validated", "gdn_dispatch_evidence"):
                if r.get(key) is not True:
                    reasons.append(f"rank {r.get('rank')} missing {key}")
            if r.get("cuda_visible_devices") != d.get("pair") or not r.get("pci_bus_id"):
                reasons.append("rank device identity mismatch")
            if r.get("reduction_paths") != {"decode": ["CustomAllreduce"], "prefill": ["PyNccl"]}:
                reasons.append("wrong reduction paths")
            if r.get("clock_check", {}).get("valid") is not True:
                reasons.append("clock not validated")
            if not numeric(r.get("peak_vram_gib"), True) or r["peak_vram_gib"] > 18:
                reasons.append("VRAM missing or exceeds slice bound")
            modes = r.get("modes", {})
            if set(modes) != {mode} or not isinstance(modes.get(mode), dict):
                raise ValueError("wrong mode evidence")
            m = modes[mode]
            if m.get("correct") is not True:
                reasons.append("mode correctness not established")
            if len(m.get("raw_decode", [])) != 60 or not m.get("raw_prefill"):
                reasons.append("diagnostic needs 60 decode steps and actual prefill work")
            for lane in ("decode", "prefill"):
                rows = m.get("raw_" + lane, [])
                if any(not isinstance(x, dict) or not numeric(x.get("observed_s"), True) for x in rows):
                    reasons.append("invalid completed step records")
                elif any(b["observed_s"] <= a["observed_s"] for a, b in zip(rows, rows[1:])):
                    reasons.append("non-monotonic completion records")
            if mode.startswith("green") and m.get("sm_graph_restriction") is not True:
                reasons.append("green SM restriction unproven")
        for p in result["rank_progress"]:
            if p["phase"] != "completed" or p["errors"]:
                reasons.append(f"rank {p['rank']} did not leave complete progress evidence")
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        reasons.append("missing/malformed pair evidence")
    return result


def q11(root):
    stages, reasons, fatal = {}, [], False
    all_specs, all_devices = [], []
    for mode in LIVENESS_MODES:
        dirs = sorted(p for p in root.glob("s5-*-" + mode) if p.is_dir())
        stage = stages[mode] = {"pairs": {}}
        if len(dirs) != 1:
            stage["verdict"] = "INCONCLUSIVE"
            reasons.append(f"{mode}: expected exactly one stage directory, found {len(dirs)}")
            # A correctness failure in an ambiguous run must still be reported.
        for directory in dirs:
            pair_specs, devices = [], []
            for pair in PAIRS:
                p = directory / ("pair-" + pair.replace(",", "-") + ".json")
                info = pair_evidence(p, mode)
                fatal |= info["fatal_correctness"]
                d = info.pop("data", {})
                if d.get("pair") != pair:
                    info["reasons"].append("wrong/missing pair identity")
                pair_specs.append(d.get("run_spec"))
                devices += [r.get("pci_bus_id") for r in records(d.get("ranks")) if isinstance(r, dict)]
                stage["pairs"][pair] = info
                reasons.extend(f"{mode}/{pair}: {x}" for x in info["reasons"])
            if not pair_specs[0] or pair_specs[0] != pair_specs[1]:
                reasons.append(f"{mode}: different or missing pair specifications")
            if len(devices) != 4 or any(not isinstance(x, str) for x in devices) or len(set(devices)) != 4:
                reasons.append(f"{mode}: four distinct GPUs not established")
            all_devices.append(devices)
            if isinstance(pair_specs[0], dict):
                all_specs.append({k: v for k, v in pair_specs[0].items() if k != "modes"})
            try:
                if (directory / "image.txt").read_text().split()[0] != IMAGE:
                    raise ValueError("image mismatch")
            except (OSError, IndexError, ValueError):
                reasons.append(f"{mode}: pinned image identity missing or mismatched")
        stage["verdict"] = ("FAIL" if any(p["fatal_correctness"] for p in stage["pairs"].values()) else
                            "INCONCLUSIVE" if any(r.startswith(mode + ":") or r.startswith(mode + "/") for r in reasons) else "PASS")
    if len(all_specs) != 2 or all_specs[0] != all_specs[1] or len(all_devices) != 2 or all_devices[0] != all_devices[1]:
        reasons.append("workload/device assignments differ between prio and green24, or are incomplete")
    verdict = "FAIL" if fatal else "INCONCLUSIVE" if reasons else "PASS"
    return {"schema": 1, "scope": "q11 diagnostic liveness ONLY; not S5 GO or release", "root": str(root),
            "verdict": verdict, "measurement_allowed": verdict == "PASS", "stages": stages,
            "reasons": sorted(set(reasons))}


def measurement(root):
    pairs, reasons, fatal, tables = [], [], False, {}
    for pair in PAIRS:
        e = pair_evidence(root / ("pair-" + pair.replace(",", "-") + ".json"))
        fatal |= e["fatal_correctness"]
        reasons += e["reasons"]
        if any(r['phase'] != 'completed' or r['errors'] for r in e['rank_progress']):
            reasons.append(f'{pair}: rank cleanup/progress evidence incomplete')
        d = e.get("data", {})
        pairs.append(d)
        tables[pair] = {"rank_progress": e["rank_progress"], "ranks": []}
        for r in records(d.get("ranks")):
            if not isinstance(r, dict):
                continue
            modes, rows = mapping(r.get("modes")), {}
            for name, m in modes.items():
                m = mapping(m)
                rows[name] = {k: m.get(k) for k in ("decode_steps_s", "prefill_steps_s", "prefill_steps_s_upper",
                    "gap_p50_ms", "gap_p99_ms", "gap_max_ms", "prefill_covers_decode_tail", "sm_split")}
                for metric in ("decode_steps_s", "prefill_steps_s", "gap_p99_ms"):
                    base, val = mapping(modes.get("serial")).get(metric), m.get(metric)
                    rows[name][metric + "_vs_serial"] = val/base if numeric(base, True) and numeric(val) else None
            tables[pair]["ranks"].append({"rank": r.get("rank"), "modes": rows})
    try:
        grade = combine(pairs)
    except (TypeError, ValueError, AttributeError, KeyError, ZeroDivisionError):
        grade = {"verdict": "INCONCLUSIVE", "reasons": ["malformed measurement artifacts"]}
    if fatal:
        grade = {"verdict": "FAIL", "reasons": ["explicit correctness failure artifact"]}
    elif reasons and grade["verdict"] != "FAIL":
        grade = {"verdict": "INCONCLUSIVE", "reasons": reasons}
    return {"schema": 1, "scope": "4-layer slice rates, NOT full-model tokens/s or release", **grade, "measurements": tables}


def brief(result):
    lines = [result['verdict'] + ': ' + result['scope']]
    for mode, stage in result.get('stages', {}).items():
        lines.append(f"{mode}: {stage['verdict']}")
        for pair, data in stage['pairs'].items():
            for rank in data['rank_progress']:
                desc = f"  pair={pair} rank={rank['rank']} phase={rank['phase']}"
                if rank.get('watchdog'):
                    desc += ' WATCHDOG'
                if rank.get('fatal_correctness'):
                    desc += ' CORRECTNESS-FAIL'
                for lane, events in rank['last_lane_events'].items():
                    if events:
                        e = events[-1]
                        desc += f" {lane}={e.get('stage')} step={e.get('step')} collective={e.get('collective', '-')}"
                lines.append(desc)
    for pair, data in result.get('measurements', {}).items():
        for rank in data['ranks']:
            for mode, m in rank['modes'].items():
                def fmt(k):
                    v = m.get(k)
                    return f'{v:.3f}' if numeric(v) else 'n/a'
                lines.append(f"pair={pair} rank={rank['rank']} {mode}: decode={fmt('decode_steps_s')} steps/s "
                             f"prefill={fmt('prefill_steps_s')} steps/s p99={fmt('gap_p99_ms')} ms "
                             f"decode/serial={fmt('decode_steps_s_vs_serial')}x")
        for r in data['rank_progress']:
            if r['phase'] != 'completed':
                lines.append(f"pair={pair} rank={r['rank']} last_phase={r['phase']}")
    lines += ['reason: ' + x for x in result.get('reasons', [])]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("q11", "measurement"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, help="fresh file, never overwritten")
    ap.add_argument("--brief", action="store_true", help="human summary; --out still writes complete JSON")
    a = ap.parse_args()
    result = (q11 if a.kind == "q11" else measurement)(a.root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.out:
        with a.out.open("x") as f:
            f.write(text)
    print(brief(result) if a.brief else text, end="\n" if a.brief else "")
    return 0 if result["verdict"] == "PASS" else EXIT[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
