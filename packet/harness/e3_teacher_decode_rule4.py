#!/usr/bin/env python3
"""Teacher-forced decode rule v4: control-envelope comparison of schema/2 evidence.

Rule v3 (``teacher_forced.RULE`` and ``e3_teacher_decode_grade``) stays frozen and
unchanged.  Rule v4 does not reinterpret the collector artifact: it treats
``s5-teacher-decode/2`` as collection evidence, runs the unchanged v3 grader as a
mandatory structural gate, and then applies the operator-authorized comparison
policy separately.  The retained artifact still declares ``rule.version == 3``;
the v4 policy is reported under its own keys.

Policy (operator authorization 27 Oct 2026):

* controls are A1/A2 and A2/C (NOT the v3 historical A1/C pair);
* per run E = max of the two control-pair metrics;
* allowance(continuous) = min(hard_cap, max(floor, 2*E + 0.125));
* allowance(flips) = min(1577, 2*E_flips + 41);
* split-B is compared against EVERY control run (A1, A2, C) with symmetric
  winners/top5/shared-information metrics and both flip directions;
* a control pair that is non-finite or above a hard cap makes the run
  INCONCLUSIVE; a split outside any allowance or a flip over the declared
  flip-margin allowance makes it FAIL; structural row-mapping/identity/coverage
  defects are INCONCLUSIVE and a forced-token/data failure stays FAIL;
* no retries, no outlier exclusion, no favourable-control selection, no
  tolerance fitting.

The calibration input was recomputed directly from the pinned full12 receipt
(``teacher-forced.json``, SHA256 ``a6d830af...901fc``); the client summary was
not used.  ``e3_teacher_decode_rule4_calibration.json`` records the exact
control scalars and the frozen envelope/floors/caps.  Its canonical JSON digest
is pinned below, so an edited calibration is rejected rather than silently
changing the policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import e3_teacher_decode_grade as v3

VERSION = 4
POLICY = "control-envelope-v4"
SCHEMA = 1
N = 2048
STREAMS = 8
STEPS = 256
TOP_ORDER_BASIS = "response-insertion-order/1"
SOURCE_SHA256 = "a6d830af15d58598f8dd95e053214ca3511266048dec13ae8f5dd60f860901fc"
CALIBRATION_FILENAME = "e3_teacher_decode_rule4_calibration.json"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().with_name(CALIBRATION_FILENAME)
# sha256 of json.dumps(calibration, sort_keys=True, separators=(',',':'), allow_nan=False)
PINNED_CALIBRATION_SHA256 = "bf5737ac001d8c64a006e3ca8d30ec7d8236586885d95de8dd2e29a068744689"

CONTROL_PAIRS = (("full-A1", "full-A2"), ("full-A2", "full-C"))
SPLIT = "split-B"
CONTINUOUS = ("winners", "top5", "shared_info", "flip_margin")
COUNTING = ("flips",)
MAX_REASONS = 32

V3_CONTROL_INSTABILITY = "same-instrument control instability under frozen rule v3"
V3_SPLIT_FAILURE = "split outside frozen rule v3"
NOTE = ("schema/2 is collection evidence; the v4 comparison policy is labelled "
        "separately and the retained artifact still declares rule v3")


class EvidenceError(ValueError):
    """Malformed or inadmissible input; callers map it to INCONCLUSIVE."""


def _canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _ceil1e6(value):
    return math.ceil(value * 1e6) / 1e6


def load_calibration(path=None):
    """Load and validate the pinned calibration; return (calibration, canonical_sha, file_sha).

    Raises EvidenceError for a missing, unreadable or altered calibration.  The
    canonical digest pin is the policy freeze: a forged floor/cap/factor cannot
    be selected by pointing at another file.
    """
    target = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise EvidenceError("calibration unreadable") from exc
    try:
        calibration = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise EvidenceError("calibration unreadable") from exc
    if type(calibration) is not dict:
        raise EvidenceError("calibration shape invalid")
    try:
        canonical_sha = _canonical_sha(calibration)
    except (ValueError, TypeError) as exc:
        raise EvidenceError("calibration identity differs") from exc
    if canonical_sha != PINNED_CALIBRATION_SHA256:
        raise EvidenceError("calibration identity differs")
    if calibration.get("version") != VERSION or calibration.get("policy") != POLICY:
        raise EvidenceError("calibration policy differs")
    if calibration.get("source_sha256") != SOURCE_SHA256:
        raise EvidenceError("calibration source differs")
    geometry = calibration.get("geometry")
    if (type(geometry) is not dict or geometry.get("N") != N
            or geometry.get("streams") != STREAMS or geometry.get("positions") != STEPS):
        raise EvidenceError("calibration geometry differs")
    controls = calibration.get("controls")
    expected_pairs = {f"{left}|{right}" for left, right in CONTROL_PAIRS}
    if type(controls) is not dict or set(controls) != expected_pairs:
        raise EvidenceError("calibration controls differ")
    for metrics in controls.values():
        if type(metrics) is not dict or any(type(metrics.get(k)) not in (int, float)
                                           for k in CONTINUOUS + COUNTING):
            raise EvidenceError("calibration control metrics differ")
    envelope = calibration.get("envelope")
    if type(envelope) is not dict:
        raise EvidenceError("calibration envelope differs")
    for metric in CONTINUOUS:
        if envelope.get(metric) != _ceil1e6(max(controls[label][metric] for label in controls)):
            raise EvidenceError("calibration envelope differs")
    if envelope.get("flips") != max(controls[label]["flips"] for label in controls):
        raise EvidenceError("calibration envelope differs")
    floors, caps = calibration.get("floors"), calibration.get("caps")
    if type(floors) is not dict or type(caps) is not dict:
        raise EvidenceError("calibration allowances differ")
    if (calibration.get("factor") != 2.0 or calibration.get("offset") != 0.125
            or calibration.get("flip_offset") != 41):
        raise EvidenceError("calibration policy differs")
    for metric in CONTINUOUS:
        if floors.get(metric) is None or caps.get(metric) is None:
            raise EvidenceError("calibration allowances differ")
        if caps[metric] != round(calibration["factor"] * envelope[metric] + calibration["offset"], 6):
            raise EvidenceError("calibration caps differ")
    expected_flip_cap = min(N, int(calibration["factor"] * envelope["flips"]
                                   + math.ceil(0.02 * N)))
    if caps.get("flips") != expected_flip_cap or calibration.get("flip_offset") != 41:
        raise EvidenceError("calibration caps differ")
    return calibration, canonical_sha, hashlib.sha256(raw).hexdigest()


def _ordered(row):
    """Return (token_id, logprob) pairs in response insertion order.

    ``v3._rows(..., ordered=True)`` already restored the response order into the
    row's ``top`` dict; iterating it preserves the original tie order.  This
    helper is defensive: structural validation is the caller's gate.
    """
    top = row.get("top") if type(row) is dict else None
    if type(top) is not dict or not 20 <= len(top) <= 21:
        raise EvidenceError("original top20 logprobs incomplete")
    seq = []
    for token, value in top.items():
        if (type(token) is not str or not token.startswith("token_id:")
                or not token[9:].isdigit() or type(value) not in (int, float)
                or not math.isfinite(value)):
            raise EvidenceError("original top20 logprobs incomplete")
        seq.append((token, float(value)))
    return seq


def _winner(seq):
    """Greedy winner; ties keep the original response insertion order."""
    best_token, best_value = seq[0]
    for token, value in seq[1:]:
        if value > best_value:
            best_token, best_value = token, value
    return best_token


def _top5(seq):
    """v3-compatible top-5 set: stable ascending sorted(seq)[-5:].

    This is the selection used by the frozen v3 rule and by the independent
    full12 recomputation; equal values keep their response insertion order.
    """
    return {token for token, _ in sorted(seq, key=lambda item: item[1])[-5:]}


def pair_metrics(left, right, *, flip_margin_limit=None):
    """Symmetric metrics for two mapped runs ({position: {'top': ...}}).

    Missing winner or top-5 support is infinite, never dropped.  ``shared_info``
    is the intersection of the two top-20 sets and is infinite when empty.
    ``flips_over`` counts flips whose maximum two-direction margin exceeds the
    supplied limit; it stays zero when no limit is supplied.
    """
    if type(left) is not dict or type(right) is not dict:
        raise EvidenceError("row coverage differs")
    if set(left) != set(right) or len(left) != N or len(right) != N:
        raise EvidenceError("row coverage differs")
    out = {"n": 0, "winners": 0.0, "top5": 0.0, "shared_info": 0.0,
           "flip_margin": 0.0, "flips": 0, "flips_over": 0}
    for key in left:
        out["n"] += 1
        a, b = _ordered(left[key]), _ordered(right[key])
        da, db = dict(a), dict(b)
        wa, wb = _winner(a), _winner(b)

        def delta(token):
            if token not in da or token not in db:
                return math.inf
            return abs(da[token] - db[token])

        out["winners"] = max(out["winners"], delta(wa), delta(wb))
        out["top5"] = max(out["top5"], max(delta(token) for token in _top5(a) | _top5(b)))
        shared = set(da) & set(db)
        out["shared_info"] = max(
            out["shared_info"],
            max(abs(da[token] - db[token]) for token in shared) if shared else math.inf)
        if wa != wb:
            out["flips"] += 1
            if wa not in da or wb not in da or wb not in db or wa not in db:
                margin = math.inf
            else:
                margin = max(da[wa] - da[wb], db[wb] - db[wa])
            out["flip_margin"] = max(out["flip_margin"], margin)
            if flip_margin_limit is not None and margin > flip_margin_limit:
                out["flips_over"] += 1
    return out


def _allowances(calibration, control_max):
    """Per-run allowances from E and the frozen floors/caps."""
    allowances = {}
    for metric in CONTINUOUS:
        allowances[metric] = min(
            calibration["caps"][metric],
            max(calibration["floors"][metric],
                calibration["factor"] * control_max[metric] + calibration["offset"]))
    allowances["flips"] = int(min(
        calibration["caps"]["flips"],
        calibration["factor"] * control_max["flips"] + calibration["flip_offset"]))
    return allowances


def _control_violations(calibration, controls):
    violations = []
    for label, metrics in controls.items():
        for metric in CONTINUOUS:
            value = metrics[metric]
            if type(value) not in (int, float) or not math.isfinite(value):
                violations.append(f"control {label}: {metric} evidence incomplete/nonfinite")
            elif value > calibration["caps"][metric]:
                violations.append(
                    f"control {label}: {metric} {value} exceeds hard cap "
                    f"{calibration['caps'][metric]}")
        if metrics["flips"] > calibration["caps"]["flips"]:
            violations.append(
                f"control {label}: flips {metrics['flips']} exceeds hard cap "
                f"{calibration['caps']['flips']}")
    return violations


def _public_metrics(metrics):
    """Keep the verdict dict JSON-safe without changing the decision values."""
    result = dict(metrics)
    for metric in CONTINUOUS:
        if not math.isfinite(result[metric]):
            result[metric] = "inf"
    return result


def _comparison_violations(label, metrics, allowances):
    violations = []
    for metric in ("winners", "top5", "shared_info"):
        value = metrics[metric]
        if type(value) not in (int, float) or not math.isfinite(value):
            violations.append(f"{label}: {metric} evidence incomplete/nonfinite")
        elif value > allowances[metric]:
            violations.append(f"{label}: {metric} {value} > allowance {allowances[metric]}")
    flips = metrics["flips"]
    if type(flips) is not int or flips < 0:
        violations.append(f"{label}: flip count invalid")
    elif flips > allowances["flips"]:
        violations.append(f"{label}: flips {flips} > allowance {allowances['flips']}")
    if flips:
        margin = metrics["flip_margin"]
        if type(margin) not in (int, float) or not math.isfinite(margin):
            violations.append(f"{label}: flip_margin evidence incomplete/nonfinite")
        elif metrics.get("flips_over", 0):
            violations.append(
                f"{label}: {metrics['flips_over']} flip(s) exceed flip_margin "
                f"allowance {allowances['flip_margin']}")
    return violations


def _structural_problem(v3_result):
    """Translate unchanged-v3 output into the v4 structural gate.

    The two v3 numerical outcomes are replaced by the v4 policy.  Every other
    v3 outcome is a geometry/schema/row-mapping/data finding and is propagated.
    """
    verdict = v3_result.get("verdict")
    if verdict == "PASS":
        return None
    if verdict == "INCONCLUSIVE" and v3_result.get("missing") == [V3_CONTROL_INSTABILITY]:
        return None
    if verdict == "FAIL" and v3_result.get("failures") == [V3_SPLIT_FAILURE]:
        return None
    if verdict == "FAIL":
        return ("FAIL", list(v3_result.get("failures") or ["forced-token or data failure"]))
    return ("INCONCLUSIVE", list(v3_result.get("missing") or ["structural evidence invalid"]))


def _mapped_rows(evidence):
    """Run the unchanged v3 gate, then re-extract rows with proven tie order."""
    if type(evidence) is not dict:
        raise EvidenceError("evidence shape invalid")
    structural = _structural_problem(v3.grade(evidence))
    if structural is not None:
        return None, structural
    if evidence.get("top_order_basis") != TOP_ORDER_BASIS:
        return None, ("INCONCLUSIVE", [
            "original top order unproved; rule v4 ties require response insertion order"])
    mapped = {}
    for name in v3.RUNS:
        mapped[name] = v3._rows(
            evidence["runs"][name],
            "split" if name == SPLIT else "full",
            evidence["fixed_positions"],
            evidence["prefix_digests"],
            True)
    return mapped, None


def _finish(out, verdict, reasons):
    bounded = [str(reason) for reason in reasons][:MAX_REASONS]
    out["verdict"] = verdict
    out["reasons"] = bounded
    out["missing"] = list(bounded) if verdict == "INCONCLUSIVE" else []
    out["failures"] = list(bounded) if verdict == "FAIL" else []
    return out


def grade(evidence, *, calibration_path=None):
    """Apply rule v4 to schema/2 teacher evidence; return a bounded verdict dict.

    The result keeps ``verdict``/``missing``/``failures`` compatible with the
    existing client consumer.  It additionally labels the collector artifact and
    the separate v4 policy, and never includes token ids or per-position rows.
    """
    out = {
        "schema": SCHEMA,
        "version": VERSION,
        "policy": POLICY,
        "comparison_policy": POLICY,
        "collector_schema": evidence.get("schema") if type(evidence) is dict else None,
        "collector_rule_version": (
            evidence.get("rule", {}).get("version")
            if type(evidence) is dict and type(evidence.get("rule")) is dict else None),
        "top_order_basis_required": TOP_ORDER_BASIS,
        "calibration_sha256": None,
        "calibration_file_sha256": None,
        "calibration_source_sha256": SOURCE_SHA256,
        "controls": {},
        "control_max": {},
        "calibration_envelope": {},
        "allowances": {},
        "comparisons": {},
        "note": NOTE,
    }
    try:
        calibration, canonical_sha, file_sha = load_calibration(calibration_path)
    except (EvidenceError, OSError, ValueError, TypeError, KeyError) as exc:
        reason = str(exc) if isinstance(exc, EvidenceError) else "calibration unreadable"
        return _finish(out, "INCONCLUSIVE", [reason])
    out["calibration_sha256"] = canonical_sha
    out["calibration_file_sha256"] = file_sha
    out["calibration_source_sha256"] = calibration["source_sha256"]
    out["geometry"] = dict(calibration["geometry"])
    out["calibration_envelope"] = dict(calibration["envelope"])

    try:
        mapped, structural = _mapped_rows(evidence)
        if structural is not None:
            return _finish(out, structural[0], structural[1])
        if mapped is None:
            raise EvidenceError("structured rows unavailable")

        controls = {}
        for left, right in CONTROL_PAIRS:
            controls[f"{left}|{right}"] = pair_metrics(mapped[left], mapped[right])
        out["controls"] = {label: _public_metrics(metrics)
                           for label, metrics in controls.items()}
        control_violations = _control_violations(calibration, controls)
        if control_violations:
            return _finish(out, "INCONCLUSIVE", control_violations)

        control_max = {
            metric: max(metrics[metric] for metrics in controls.values())
            for metric in CONTINUOUS + COUNTING}
        out["control_max"] = control_max
        allowances = _allowances(calibration, control_max)
        out["allowances"] = allowances

        violations = []
        for right in ("full-A1", "full-A2", "full-C"):
            label = f"{SPLIT}|{right}"
            metrics = pair_metrics(
                mapped[SPLIT], mapped[right],
                flip_margin_limit=allowances["flip_margin"])
            out["comparisons"][label] = _public_metrics(metrics)
            violations.extend(_comparison_violations(label, metrics, allowances))
        if violations:
            return _finish(out, "FAIL", violations)
        return _finish(out, "PASS", [])
    except Exception as exc:
        reason = str(exc) if isinstance(exc, EvidenceError) else "invalid evidence shape"
        return _finish(out, "INCONCLUSIVE", [reason])
