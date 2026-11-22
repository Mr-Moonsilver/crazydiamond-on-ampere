"""Draw the approach figure for the crazydiamond-on-ampere README (light + dark SVG).

Same model weights and same KV cache in every row, always in the same place (weights left, KV cache right).
Only the GPU memory limit moves. Anything past the limit is hatched: it lives outside GPU memory.
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')

THEMES = {
    'light': dict(bg='#ffffff', ink='#1b2330', muted='#5b6677', line='#c9d1dc', frame='#1b2330',
                  gpu='#eef3f9', gpu_edge='#8fa3bf', weights='#3f5f8f', kv='#e0a02a',
                  ours='#fbf6ea', ours_edge='#e0a02a', tag='#e7ecf2', tag_ink='#334155'),
    'dark': dict(bg='#0d1117', ink='#e6edf3', muted='#9aa7b8', line='#30363d', frame='#e6edf3',
                 gpu='#151d29', gpu_edge='#4d607c', weights='#6f93c8', kv='#f0b443',
                 ours='#1d1910', ours_edge='#f0b443', tag='#21262d', tag_ink='#c9d1d9'),
}

W, H = 1000, 680
X0 = 340                 # bars start here
WW, KW = 320, 200        # model weights and KV cache: the same size in every row
WX, KX = X0, X0 + WW     # weights span WX..KX, KV span KX..KX+KW
END = KX + KW
FONT = "-apple-system, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"

ROWS = [
    dict(title='Model bigger than the GPUs', tag='Offloading engines, e.g. Strata', sub='few or small GPUs',
         limit=X0 + 230, ours=False,
         note=['Everything past the limit lives in CPU RAM or on disk.',
               'Weights are streamed in for every token, so each token waits on the transfer.']),
    dict(title='Model just fits', tag='crazydiamond-on-ampere (this repo)', sub='4 × RTX 3090 · 8 agents × 200k',
         limit=KX + 22, ours=True,
         note=['The weights fill the GPUs; only a sliver is left for the KV cache.',
               'Layer weights stay on the GPUs; attention history lives in host RAM, fetched row by row.']),
    dict(title='Room for model and context', tag='Stock engines, e.g. vLLM', sub='more or bigger GPUs',
         limit=END + 70, ours=False,
         note=['Weights and the whole KV cache fit on the GPUs, with room to spare.',
               'The easy case, but it needs more or bigger GPUs.']),
]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def seg(a, t, x0, x1, limit, y, h, color, pid):
    """One segment: solid inside GPU memory, hatched past the limit."""
    if x0 < limit:
        a(f'<rect x="{x0}" y="{y}" width="{min(x1, limit) - x0}" height="{h}" fill="{t[color]}"/>')
    if x1 > limit:
        s = max(x0, limit)
        a(f'<rect x="{s}" y="{y}" width="{x1 - s}" height="{h}" fill="url(#{pid})" stroke="{t[color]}" stroke-width="1.5"/>')


def svg(t):
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="{FONT}" role="img" aria-labelledby="t d">')
    a('<title id="t">Same model, same context, different GPU memory</title>')
    a('<desc id="d">Three rows with the same model weights and the same KV cache. In the first, the GPU memory limit '
      'falls inside the weights (offloading engines). In the second, the weights just fit and most of the KV cache '
      'lives in host RAM (this engine). In the third, weights and KV cache fit with room to spare (stock engines).</desc>')
    a('<defs>')
    for pid, col in (('hw', t['weights']), ('hk', t['kv'])):
        a(f'<pattern id="{pid}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          f'<rect width="8" height="8" fill="{t["bg"]}"/><rect width="3.5" height="8" fill="{col}"/></pattern>')
    a('</defs>')
    a(f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>')
    a(f'<text x="32" y="46" font-size="22" font-weight="700" fill="{t["ink"]}">Same model, same context, different GPU memory</text>')
    a(f'<text x="32" y="72" font-size="14" fill="{t["muted"]}">Every setup needs the same model weights and the same KV cache, '
      'the working memory that holds each</text>')
    a(f'<text x="32" y="92" font-size="14" fill="{t["muted"]}">request\'s context. Only the GPU memory limit moves. '
      'Hatched parts live outside GPU memory.</text>')
    # column headers, aligned with every row
    a(f'<text x="{WX + WW / 2}" y="134" font-size="12" font-weight="600" letter-spacing="0.6" text-anchor="middle" '
      f'fill="{t["muted"]}">MODEL WEIGHTS</text>')
    a(f'<text x="{KX + KW / 2}" y="134" font-size="12" font-weight="600" letter-spacing="0.6" text-anchor="middle" '
      f'fill="{t["muted"]}">KV CACHE (CONTEXT)</text>')
    for x in (WX, KX, END):
        a(f'<line x1="{x}" y1="140" x2="{x}" y2="146" stroke="{t["line"]}" stroke-width="1.5"/>')
    a(f'<line x1="{WX}" y1="143" x2="{END}" y2="143" stroke="{t["line"]}" stroke-width="1.5"/>')
    for i, r in enumerate(ROWS):
        y = 160 + i * 150
        if r['ours']:
            a(f'<rect x="18" y="{y - 8}" width="{W - 36}" height="140" rx="10" fill="{t["ours"]}" '
              f'stroke="{t["ours_edge"]}" stroke-width="1.5"/>')
        a(f'<text x="32" y="{y + 24}" font-size="16" font-weight="700" fill="{t["ink"]}">{esc(r["title"])}</text>')
        tw = 6.6 * len(r['tag']) + 20
        a(f'<rect x="32" y="{y + 34}" width="{tw:.0f}" height="22" rx="11" fill="{t["tag"]}"/>')
        a(f'<text x="41" y="{y + 49}" font-size="12" font-weight="600" fill="{t["tag_ink"]}">{esc(r["tag"])}</text>')
        a(f'<text x="34" y="{y + 76}" font-size="12" fill="{t["muted"]}">{esc(r["sub"])}</text>')
        by, bh = y + 26, 38
        L = r['limit']
        # GPU memory: a tinted box up to the limit
        a(f'<rect x="{X0 - 6}" y="{by - 8}" width="{L - X0 + 6}" height="{bh + 16}" rx="6" fill="{t["gpu"]}" '
          f'stroke="{t["gpu_edge"]}" stroke-width="1.5"/>')
        seg(a, t, WX, KX, L, by, bh, 'weights', 'hw')
        seg(a, t, KX, END, L, by, bh, 'kv', 'hk')
        a(f'<line x1="{L}" y1="{by - 14}" x2="{L}" y2="{by + bh + 10}" stroke="{t["frame"]}" stroke-width="2" stroke-dasharray="4 3"/>')
        a(f'<text x="{L}" y="{by - 18}" font-size="11" font-weight="600" text-anchor="middle" fill="{t["ink"]}">GPU limit</text>')
        if L > END:
            a(f'<text x="{(END + L) / 2}" y="{by + bh / 2 + 4}" font-size="11" text-anchor="middle" fill="{t["muted"]}">free</text>')
        for j, line in enumerate(r['note']):
            a(f'<text x="{X0}" y="{by + bh + 30 + j * 18}" font-size="13" fill="{t["ink" if j == 0 else "muted"]}">'
              f'{esc(line)}</text>')
    # legend
    ly = H - 24
    items = [('model weights', f'fill="{t["weights"]}"'), ('KV cache', f'fill="{t["kv"]}"'),
             ('GPU memory', f'fill="{t["gpu"]}" stroke="{t["gpu_edge"]}"'),
             ('hatched: outside GPU memory (CPU RAM, host RAM or disk)', 'fill="url(#hw)"')]
    lx = 32
    for label, attrs in items:
        a(f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" {attrs}/>')
        a(f'<text x="{lx + 20}" y="{ly}" font-size="12" fill="{t["muted"]}">{esc(label)}</text>')
        lx += 20 + 6.6 * len(label) + 24
    a('</svg>')
    return '\n'.join(o) + '\n'


OUT.mkdir(parents=True, exist_ok=True)
for name, t in THEMES.items():
    (OUT / f'approach-{name}.svg').write_text(svg(t))
print('ok', OUT)
