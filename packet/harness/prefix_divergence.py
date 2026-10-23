#!/usr/bin/env python3
"""Where do the agents' prompts stop being reusable? Reads a tcpdump pcap of requests to vLLM :8000 (plain HTTP),
rebuilds each POST /v1/messages body and reports, per request: session, sizes, and the first character where the
request differs from (a) the previous request of the same session and (b) the previous request of any other
session. Prints lengths, offsets and short hashes only — never prompt text.
usage: prefix_divergence.py capture.pcap [port]"""
import collections, hashlib, json, struct, sys

PCAP, PORT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def packets(path):
    """Yields (ts, frame) with frames normalised to a 14-byte Ethernet-style header (Linux cooked captures from
    `tcpdump -i any`: SLL or SLL2)."""
    with open(path, "rb") as f:
        ghdr = f.read(24)
        endian = "<" if ghdr[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        linktype = struct.unpack(endian + "I", ghdr[20:24])[0]
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                return
            ts_s, ts_us, incl, _ = struct.unpack(endian + "IIII", hdr)
            frame = f.read(incl)
            if linktype == 113:  # LINUX_SLL
                frame = b"\0" * 12 + frame[14:]
            elif linktype == 276:  # LINUX_SLL2: 20-byte header, ethertype in bytes 0-1
                frame = b"\0" * 12 + frame[0:2] + frame[20:]
            yield ts_s + ts_us / 1e6, frame


flows = collections.defaultdict(dict)  # (src ip, src port) -> {seq: payload}
first_ts = {}
for ts, frame in packets(PCAP):
    if len(frame) < 34 or frame[12:14] != b"\x08\x00":
        continue
    ihl = (frame[14] & 0x0F) * 4
    ip = frame[14:14 + ihl]
    if ip[9] != 6:
        continue
    tcp = frame[14 + ihl:]
    sport, dport, seq = struct.unpack("!HHI", tcp[:8])
    if dport != PORT:
        continue
    payload = tcp[(tcp[12] >> 4) * 4:]
    total = struct.unpack("!H", ip[2:4])[0]
    payload = payload[:max(0, total - ihl - (tcp[12] >> 4) * 4)]
    if payload:
        key = (".".join(map(str, ip[12:16])), sport)
        flows[key].setdefault(seq, (ts, payload))


def stream(chunks):
    data, ts_at, expect = bytearray(), [], None
    for seq in sorted(chunks):
        ts, p = chunks[seq]
        if expect is not None and seq < expect:  # retransmit / overlap
            p = p[expect - seq:]
            if not p:
                continue
        ts_at.append((len(data), ts))
        data += p
        expect = seq + len(chunks[seq][1])
    return bytes(data), ts_at


requests = []
for key, chunks in flows.items():
    data, ts_at = stream(chunks)
    pos = 0
    while True:
        h_end = data.find(b"\r\n\r\n", pos)
        if h_end < 0:
            break
        head = data[pos:h_end].decode("latin1")
        length = 0
        for line in head.split("\r\n")[1:]:
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1])
        body = data[h_end + 4:h_end + 4 + length]
        ts = next((t for off, t in reversed(ts_at) if off <= pos), ts_at[0][1])
        line0 = head.split("\r\n", 1)[0]
        if line0.startswith("POST /v1/messages") and "count_tokens" not in line0 and len(body) == length:
            try:
                requests.append((ts, key, json.loads(body)))
            except ValueError:
                pass
        pos = h_end + 4 + length


def render(r):
    """Text in prompt order (system, tools, messages) — the order the chat template lays out tokens."""
    def blocks(c):
        if isinstance(c, str):
            return c
        return "".join(b.get("text", "") or json.dumps(b.get("input", b.get("content", "")), sort_keys=True) for b in c)
    sysblocks = r.get("system", "")
    if isinstance(sysblocks, list):  # vLLM's Anthropic layer drops Claude Code's billing-header block (per-request hash)
        sysblocks = [b for b in sysblocks if not (b.get("type") == "text" and (b.get("text") or "").startswith("x-anthropic-billing-header"))]
    sysm = blocks(sysblocks)
    tools = json.dumps(r.get("tools", []), sort_keys=True)
    msgs = "".join(f"<{m['role']}>" + blocks(m["content"]) for m in r.get("messages", []))
    return sysm, tools, msgs


def common(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:8]
requests.sort(key=lambda x: x[0])
last_by_session, last_any = {}, None
rows = []
for ts, key, r in requests:
    sysm, tools, msgs = render(r)
    full = sysm + tools + msgs
    sess = (r.get("metadata") or {}).get("user_id", "?")[-12:]
    prev = last_by_session.get(sess)
    same = common(prev, full) if prev else None
    other = max((common(v, full) for s, v in last_by_session.items() if s != sess), default=0)
    rows.append((ts, sess, len(sysm), len(tools), len(r.get("messages", [])), len(full), same, other, h(sysm), h(tools), r.get("model")))
    last_by_session[sess] = full

print(f"requests: {len(rows)}  sessions: {len(set(r[1] for r in rows))}  window: {rows[-1][0] - rows[0][0]:.0f} s" if rows else "no requests")
print("system-prompt variants:", collections.Counter(r[8] for r in rows).most_common(6))
print("tool-list variants:   ", collections.Counter(r[9] for r in rows).most_common(4))
print("models:", collections.Counter(r[10] for r in rows).most_common())
print("\n t(s)  session      sys_chars tool_chars msgs total_chars  same-session reuse      best other-session reuse")
t0 = rows[0][0] if rows else 0
for ts, sess, ls, lt, nm, lf, same, other, hs, ht, _ in rows:
    s = "new session" if same is None else f"{same:>8} ({same / lf * 100:5.1f}%)"
    print(f"{ts - t0:6.1f} {sess:12} {ls:9} {lt:10} {nm:4} {lf:11}  {s:24} {other:>8} ({other / lf * 100:5.1f}%)")
follow = [r for r in rows if r[6] is not None]
new = [r for r in rows if r[6] is None]
if follow:
    print(f"\nfollow-ups: {len(follow)}; median reusable share {sorted(r[6] / r[5] for r in follow)[len(follow) // 2] * 100:.1f}%; "
          f"breaks before 90%: {sum(1 for r in follow if r[6] / r[5] < 0.9)}")
if new:
    print(f"new sessions: {len(new)}; median shared-with-other-session prefix {sorted(r[7] for r in new)[len(new) // 2]} chars "
          f"of median {sorted(r[5] for r in new)[len(new) // 2]}")
