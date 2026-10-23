#!/usr/bin/env python3
"""Production summary for a time window from Prometheus (vLLM at 250 ms, NVML at 500 ms).
usage: window_report.py START_UTC END_UTC   (e.g. 2026-10-23T01:50 2026-10-23T02:12; END may be 'now')"""
import calendar, json, sys, time, urllib.parse, urllib.request

P = "http://localhost:9090/api/v1/"
to_ts = lambda s: time.time() if s == "now" else calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M"))
start, end = to_ts(sys.argv[1]), to_ts(sys.argv[2])
W = f"{int(end - start)}s"
V, G = 'job="vllm"', 'job="nvml_gpu"'


def q(expr):
    args = {"query": expr, "time": end}
    return json.load(urllib.request.urlopen(P + "query?" + urllib.parse.urlencode(args)))["data"]["result"]


def one(expr):
    r = q(expr)
    return float(r[0]["value"][1]) if r else float("nan")


def series(expr, step="2s"):
    args = {"query": expr, "start": start, "end": end, "step": step}
    r = json.load(urllib.request.urlopen(P + "query_range?" + urllib.parse.urlencode(args)))["data"]["result"]
    return [float(v[1]) for v in r[0]["values"]] if r else []


pct = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))] if a else float("nan")
hq = lambda m, x: one(f'histogram_quantile({x}, sum by (le) (increase(vllm:{m}_bucket{{{V}}}[{W}])))')
run = series(f'sum(vllm:num_requests_running{{{V}}})')
gen = series(f'sum(rate(vllm:generation_tokens_total{{{V}}}[10s]))')
per = series(f'sum(rate(vllm:generation_tokens_total{{{V}}}[30s])) / clamp_min(sum(vllm:num_requests_running{{{V}}}), 1)', "10s")
comp = one(f'sum(increase(vllm:prompt_tokens_by_source_total{{{V},source="local_compute"}}[{W}]))')
hit = one(f'sum(increase(vllm:prompt_tokens_by_source_total{{{V},source="local_cache_hit"}}[{W}]))')
gtot = one(f'sum(increase(vllm:generation_tokens_total{{{V}}}[{W}]))')
reqs = one(f'sum(increase(vllm:request_success_total{{{V}}}[{W}]))')
print(f"window {time.strftime('%H:%M', time.gmtime(start))}-{time.strftime('%H:%M', time.gmtime(end))}Z ({(end - start) / 60:.0f} min)")
print(f"  running mean {sum(run) / len(run):.1f} (p50 {pct(run, .5):.0f})  requests {reqs:.0f} ({reqs / (end - start) * 60:.1f}/min)")
print(f"  decode total mean {sum(gen) / len(gen):.0f} tok/s (p90 {pct(gen, .9):.0f}); per stream p50 {pct(per, .5):.1f} tok/s; generated {gtot / 1e3:.0f}k")
print(f"  ITL p50 {hq('inter_token_latency_seconds', .5) * 1000:.0f} ms  p95 {hq('inter_token_latency_seconds', .95) * 1000:.0f} ms  p99 {hq('inter_token_latency_seconds', .99) * 1000:.0f} ms")
print(f"  TTFT p50 {hq('time_to_first_token_seconds', .5):.2f} s  p95 {hq('time_to_first_token_seconds', .95):.2f} s  p99 {hq('time_to_first_token_seconds', .99):.2f} s")
print(f"  prompt tokens: computed {comp / 1e6:.2f}M ({comp / (end - start):.0f}/s), from cache {hit / 1e6:.2f}M -> computed share {comp / (comp + hit) * 100:.0f}%")
print(f"  block hit rate {one(f'sum(increase(vllm:prefix_cache_hits_total{{{V}}}[{W}]))/sum(increase(vllm:prefix_cache_queries_total{{{V}}}[{W}]))') * 100:.0f}%  "
      f"mean computed per request {one(f'sum(increase(vllm:request_prefill_kv_computed_tokens_sum{{{V}}}[{W}]))/sum(increase(vllm:request_prefill_kv_computed_tokens_count{{{V}}}[{W}]))'):.0f}  "
      f"prompt p50/p90 {hq('request_prompt_tokens', .5):.0f}/{hq('request_prompt_tokens', .9):.0f}")
print(f"  real prefill speed {one(f'sum(increase(vllm:request_prefill_kv_computed_tokens_sum{{{V}}}[{W}]))/sum(increase(vllm:request_prefill_time_seconds_sum{{{V}}}[{W}]))'):.0f} tok/s per request; "
      f"KV pool mean {one(f'avg_over_time((max(vllm:kv_cache_usage_perc{{{V}}})*100)[{W}:5s])'):.0f}% max {one(f'max_over_time((max(vllm:kv_cache_usage_perc{{{V}}})*100)[{W}:5s])'):.0f}%; "
      f"preemptions {one(f'sum(increase(vllm:num_preemptions_total{{{V}}}[{W}]))'):.0f}")
pc = q(f'avg_over_time((max by (gpu) (nvml_clocks_event_reason_active{{{G},reason="sw_power_cap"}}))[{W}:1s])')
vr = q(f'max_over_time((max by (gpu) (nvml_memory_used_bytes{{{G}}}))[{W}:1s])/2^20')
sm = q(f'avg_over_time((max by (gpu) (nvml_clock_sm_hz{{{G}}}))[{W}:1s])/1e6')
pw = q(f'avg_over_time((max by (gpu) (nvml_power_draw_watts{{{G}}}))[{W}:1s])')
print("  power-capped " + str({x["metric"]["gpu"]: f'{float(x["value"][1]) * 100:.0f}%' for x in pc}) +
      "  VRAM peak MiB " + str({x["metric"]["gpu"]: round(float(x["value"][1])) for x in vr}))
print("  avg SM MHz " + str({x["metric"]["gpu"]: round(float(x["value"][1])) for x in sm}) + "  avg W " + str({x["metric"]["gpu"]: round(float(x["value"][1])) for x in pw}))
