"""Draw the approach figure for the crazydiamond-on-ampere README (light + dark SVG).

Same model weights and same KV cache in every row, always in the same place (weights left, KV cache right).
The shaded box is GPU memory; hatched parts live outside it, in CPU / host RAM.
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')

THEMES = {
    'light': dict(bg='#ffffff', ink='#1b2330', muted='#5b6677', line='#c9d1dc',
                  gpu='#e9f0f8', gpu_edge='#7f95b3', weights='#3f5f8f', kv='#e0a02a',
                  ours='#fbf6ea', ours_edge='#e0a02a', tag='#e7ecf2', tag_ink='#334155'),
    'dark': dict(bg='#0d1117', ink='#e6edf3', muted='#9aa7b8', line='#30363d',
                 gpu='#16202e', gpu_edge='#5a6f8f', weights='#6f93c8', kv='#f0b443',
                 ours='#1d1910', ours_edge='#f0b443', tag='#21262d', tag_ink='#c9d1d9'),
}

W, H = 1000, 672
X0 = 340                 # bars start here
WW, KW = 320, 200        # model weights and KV cache: the same size in every row
KX = X0 + WW             # weights span X0..KX, KV cache spans KX..END
END = KX + KW
FONT = "-apple-system, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"

# Each row: segments (x0, x1, kind, on_gpu) and the GPU memory box (x0, x1).
ROWS = [
    dict(title='Model bigger than the GPUs', tag='Weight-offloading engines', sub='few or small GPUs', ours=False,
         segs=[(X0, X0 + 240, 'w', False), (X0 + 240, KX, 'w', True), (KX, END, 'k', True)],
         gpu=(X0 + 240, END),
         note=['Most of the weights live in CPU RAM; the KV cache stays on the GPU.',
               'Weights are streamed in for every token, so each token waits on the transfer.']),
    dict(title='Model just fits', tag='This repo', sub='4 × RTX 3090 · 8 agents × 200k', ours=True,
         segs=[(X0, KX, 'w', True), (KX, KX + 24, 'k', True), (KX + 24, END, 'k', False)],
         gpu=(X0, KX + 24),
         note=['The weights fill the GPUs, so the KV cache moves to host RAM.',
               'Each decode step fetches only the attention rows it needs over PCIe.']),
    dict(title='Room for model and context', tag='Stock engines, e.g. vLLM', sub='more or bigger GPUs', ours=False,
         segs=[(X0, KX, 'w', True), (KX, END, 'k', True)],
         gpu=(X0, END + 70),
         note=['Weights and the whole KV cache fit on the GPUs, with room to spare.',
               'The easy case, but it needs more or bigger GPUs.']),
]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def svg(t):
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="{FONT}" role="img" aria-labelledby="t d">')
    a('<title id="t">Same model, same context, different GPU memory</title>')
    a('<desc id="d">Three rows with the same model weights and the same KV cache. Weight-offloading engines keep '
      'most weights in CPU RAM and the KV cache on the GPU. This engine keeps all weights on the GPUs and moves the '
      'KV cache to host RAM, fetching only the rows attention needs. Stock engines need GPUs big enough for both.</desc>')
    a('<defs>')
    for pid, col in (('hw', t['weights']), ('hk', t['kv'])):
        a(f'<pattern id="{pid}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          f'<rect width="8" height="8" fill="{t["bg"]}"/><rect width="3.5" height="8" fill="{col}"/></pattern>')
    a('</defs>')
    a(f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>')
    a(f'<text x="32" y="46" font-size="22" font-weight="700" fill="{t["ink"]}">Same model, same context, different GPU memory</text>')
    a(f'<text x="32" y="72" font-size="14" fill="{t["muted"]}">Every setup needs the same model weights and the same KV cache '
      '(the working memory that holds each</text>')
    a(f'<text x="32" y="92" font-size="14" fill="{t["muted"]}">request\'s context). The shaded box is GPU memory; '
      'hatched parts live outside it, in CPU or host RAM.</text>')
    a(f'<text x="{X0 + WW / 2}" y="134" font-size="12" font-weight="600" letter-spacing="0.6" text-anchor="middle" '
      f'fill="{t["muted"]}">MODEL WEIGHTS</text>')
    a(f'<text x="{KX + KW / 2}" y="134" font-size="12" font-weight="600" letter-spacing="0.6" text-anchor="middle" '
      f'fill="{t["muted"]}">KV CACHE (CONTEXT)</text>')
    for x in (X0, KX, END):
        a(f'<line x1="{x}" y1="140" x2="{x}" y2="146" stroke="{t["line"]}" stroke-width="1.5"/>')
    a(f'<line x1="{X0}" y1="143" x2="{END}" y2="143" stroke="{t["line"]}" stroke-width="1.5"/>')
    for i, r in enumerate(ROWS):
        y = 162 + i * 160
        if r['ours']:
            a(f'<rect x="18" y="{y - 8}" width="{W - 36}" height="148" rx="10" fill="{t["ours"]}" '
              f'stroke="{t["ours_edge"]}" stroke-width="1.5"/>')
        a(f'<text x="32" y="{y + 30}" font-size="16" font-weight="700" fill="{t["ink"]}">{esc(r["title"])}</text>')
        tw = 6.6 * len(r['tag']) + 20
        a(f'<rect x="32" y="{y + 40}" width="{tw:.0f}" height="22" rx="11" fill="{t["tag"]}"/>')
        a(f'<text x="41" y="{y + 55}" font-size="12" font-weight="600" fill="{t["tag_ink"]}">{esc(r["tag"])}</text>')
        a(f'<text x="34" y="{y + 82}" font-size="12" fill="{t["muted"]}">{esc(r["sub"])}</text>')
        by, bh = y + 30, 38
        g0, g1 = r['gpu']
        a(f'<text x="{g0}" y="{by - 14}" font-size="11" font-weight="600" fill="{t["muted"]}">GPU memory</text>')
        a(f'<rect x="{g0 - 6}" y="{by - 8}" width="{g1 - g0 + 12}" height="{bh + 16}" rx="6" fill="{t["gpu"]}" '
          f'stroke="{t["gpu_edge"]}" stroke-width="1.5"/>')
        for x0, x1, kind, on_gpu in r['segs']:
            col = t['weights'] if kind == 'w' else t['kv']
            if on_gpu:
                a(f'<rect x="{x0}" y="{by}" width="{x1 - x0}" height="{bh}" fill="{col}"/>')
            else:
                pid = 'hw' if kind == 'w' else 'hk'
                a(f'<rect x="{x0}" y="{by}" width="{x1 - x0}" height="{bh}" fill="url(#{pid})" stroke="{col}" stroke-width="1.5"/>')
        if g1 > END:
            a(f'<text x="{(END + g1) / 2}" y="{by + bh / 2 + 4}" font-size="11" text-anchor="middle" fill="{t["muted"]}">free</text>')
        for j, line in enumerate(r['note']):
            a(f'<text x="{X0}" y="{by + bh + 30 + j * 18}" font-size="13" fill="{t["ink" if j == 0 else "muted"]}">'
              f'{esc(line)}</text>')
    ly = H - 24
    items = [('model weights', f'fill="{t["weights"]}"'), ('KV cache', f'fill="{t["kv"]}"'),
             ('GPU memory', f'fill="{t["gpu"]}" stroke="{t["gpu_edge"]}"'),
             ('hatched: in CPU / host RAM', 'fill="url(#hk)"')]
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
