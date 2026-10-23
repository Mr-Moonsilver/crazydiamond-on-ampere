#!/usr/bin/env python3
"""Per-request prefix-cache result from a two-way capture of vLLM :8000 (server run with
--enable-prompt-tokens-details): pairs each POST /v1/messages with its response on the same connection and reads
input_tokens / cache_read_input_tokens from the response. Classifies each request against the previous request
of the same session (seen in the capture) and prints counts and sizes only — never prompt text.
usage: cache_by_request.py capture.pcap [port]"""
import collections, json, re, struct, sys

PCAP, PORT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def packets(path):
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
            if linktype == 113:
                frame = frame[14:]
            elif linktype == 276:
                frame = frame[20:]
            else:
                frame = frame[14:]
            yield ts_s + ts_us / 1e6, frame


flows = collections.defaultdict(dict)  # (client, port, direction) -> {seq: (ts, payload)}
for ts, ip in packets(PCAP):
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 6:
        continue
    ihl = (ip[0] & 0x0F) * 4
    total = struct.unpack("!H", ip[2:4])[0]
    tcp = ip[ihl:total]
    sport, dport, seq = struct.unpack("!HHI", tcp[:8])
    payload = tcp[(tcp[12] >> 4) * 4:]
    if not payload:
        continue
    src, dst = ".".join(map(str, ip[12:16])), ".".join(map(str, ip[16:20]))
    if dport == PORT:
        flows[(src, sport, "req")].setdefault(seq, (ts, payload))
    elif sport == PORT:
        flows[(dst, dport, "resp")].setdefault(seq, (ts, payload))


def stream(chunks):
    """Reassembled bytes plus (offset, timestamp) of every segment, so each request gets its own send time."""
    data, expect, at = bytearray(), None, []
    for seq in sorted(chunks):
        ts, p = chunks[seq]
        if expect is not None and seq < expect:
            p = p[expect - seq:]
            if not p:
                continue
        at.append((len(data), ts))
        data += p
        expect = seq + len(chunks[seq][1])
    return bytes(data), at


rows = []
for (cli, port, d) in list(flows):
    if d != "req":
        continue
    req, at = stream(flows[(cli, port, "req")])
    resp, _ = stream(flows.get((cli, port, "resp"), {}))
    bodies, pos = [], 0  # one entry per HTTP request on the connection, so responses pair up in order
    while (h := req.find(b"\r\n\r\n", pos)) >= 0:
        head = req[pos:h].decode("latin1")
        n = next((int(l.split(":", 1)[1]) for l in head.split("\r\n") if l.lower().startswith("content-length:")), 0)
        ts = next((t for off, t in reversed(at) if off <= pos), at[0][1])
        body = None
        if head.startswith("POST /v1/messages") and "count_tokens" not in head.split("\r\n")[0]:
            try:
                body = json.loads(req[h + 4:h + 4 + n])
            except ValueError:
                pass
        bodies.append((ts, body))
        pos = h + 4 + n
    # one usage per response: the final event's input_tokens / cache_read_input_tokens pair in each response's events
    usages = []
    for part in re.split(rb"HTTP/1\.[01] \d{3}", resp)[1:]:
        it = re.findall(rb'"input_tokens":\s*(\d+)', part)
        cr = re.findall(rb'"cache_read_input_tokens":\s*(\d+)', part)
        usages.append((int(it[-1]) if it else None, int(cr[-1]) if cr else None))
    for (t0, b), u in zip(bodies, usages):
        if b:
            rows.append((t0, (b.get("metadata") or {}).get("user_id", "?")[-12:], len(b.get("messages", [])),
                         len(json.dumps(b.get("tools", []))), u[0], u[1], b))

rows.sort()
seen = {}
cls = collections.Counter(); comp = collections.Counter()
for t, sess, nm, tl, it, cr, _ in rows:
    if it is None or cr is None:
        continue
    kind = "follow-up" if sess in seen else ("first-msgs" if nm <= 2 else "first-in-capture")
    seen[sess] = 1
    big = it > 20000
    cls[(kind, big)] += 1; comp[kind] += it
    if big:
        print(f"  big miss: session {sess} msgs {nm} tool_chars {tl} computed {it} cached {cr} ({kind})")
print("requests with usage:", sum(cls.values()), " of", len(rows))
for k in ("follow-up", "first-in-capture", "first-msgs"):
    n = cls[(k, False)] + cls[(k, True)]
    if n:
        print(f"{k:17} n={n:3}  >20k computed: {cls[(k, True)]:3}  computed total {comp[k] / 1e3:.0f}k")
