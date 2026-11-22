"""Draw the 'where the memory goes' figure for the crazydiamond-on-ampere README (light + dark SVG)."""
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')

THEMES = {
    'light': dict(bg='#ffffff', ink='#1b2330', muted='#5b6677', line='#c9d1dc', frame='#1b2330',
                  weights='#3f5f8f', kv='#e0a02a', free='#eef2f6', hatch='#3f5f8f', ours='#f6efe0',
                  ours_edge='#e0a02a', tag='#e7ecf2', tag_ink='#334155'),
    'dark': dict(bg='#0d1117', ink='#e6edf3', muted='#9aa7b8', line='#30363d', frame='#e6edf3',
                 weights='#6f93c8', kv='#f0b443', free='#161b22', hatch='#6f93c8', ours='#1f1a10',
                 ours_edge='#f0b443', tag='#21262d', tag_ink='#c9d1d9'),
}

W, H = 1000, 560
X0, X1 = 330, 790          # GPU memory frame: 100 % = 460 px
SCALE = (X1 - X0) / 100.0
FONT = "-apple-system, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"

ROWS = [
    dict(title='Model bigger than the GPUs', tag='Offloading engines, e.g. Strata',
         kv=10, weights=90, free=0, spill=34, ours=False,
         note=['Part of the weights live outside the GPUs and are fetched as needed.',
               'Runs models that do not fit, but every token waits on the transfer.']),
    dict(title='Model just fits', tag='crazydiamond-on-ampere (this repo)',
         kv=9, weights=88, free=3, spill=0, kvspill=26, ours=True, sub='up to 8 agents × 200k tokens',
         note=['Weights fill the GPUs, leaving only a sliver for context.',
               'Layer weights stay on GPU; context history lives in host RAM.']),
    dict(title='Room for model and context', tag='Stock engines, e.g. vLLM',
         kv=35, weights=50, free=15, spill=0, ours=False,
         note=['Plenty of memory for weights and a large KV cache.',
               'The easy case, but it needs more or bigger GPUs.']),
]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def svg(t):
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="{FONT}" role="img" aria-labelledby="t d">')
    a('<title id="t">Where the GPU memory goes</title>')
    a('<desc id="d">Three bars comparing GPU memory use: a model too big for the GPUs (offloading engines), '
      'a model that just fits, with its attention history kept in host RAM (this engine), and a model with room for a large '
      'KV cache (stock engines).</desc>')
    a('<defs><pattern id="spill" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
      f'<rect width="8" height="8" fill="{t["bg"]}"/><rect width="3.5" height="8" fill="{t["hatch"]}"/></pattern>'
      '<pattern id="spillkv" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
      f'<rect width="8" height="8" fill="{t["bg"]}"/><rect width="3.5" height="8" fill="{t["kv"]}"/></pattern></defs>')
    a(f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>')
    # heading
    a(f'<text x="32" y="46" font-size="22" font-weight="700" fill="{t["ink"]}">Where the GPU memory goes</text>')
    a(f'<text x="32" y="72" font-size="14" fill="{t["muted"]}">GPU memory holds the model weights plus the KV cache, '
      'the working memory each request needs for its context.</text>')
    # frame header
    a(f'<text x="{X0}" y="108" font-size="12" font-weight="600" letter-spacing="0.6" fill="{t["muted"]}">GPU MEMORY</text>')
    a(f'<text x="{X1 + 12}" y="108" font-size="12" font-weight="600" letter-spacing="0.6" fill="{t["muted"]}">BEYOND THE GPUs</text>')
    top, bottom = 118, 118 + 3 * 132 - 18
    for i, r in enumerate(ROWS):
        y = 126 + i * 132
        if r['ours']:
            a(f'<rect x="18" y="{y - 10}" width="{W - 36}" height="122" rx="10" fill="{t["ours"]}" '
              f'stroke="{t["ours_edge"]}" stroke-width="1.5"/>')
        # left labels
        a(f'<text x="32" y="{y + 22}" font-size="16" font-weight="700" fill="{t["ink"]}">{esc(r["title"])}</text>')
        tw = 6.6 * len(r['tag']) + 20
        a(f'<rect x="32" y="{y + 32}" width="{tw:.0f}" height="22" rx="11" fill="{t["tag"]}"/>')
        a(f'<text x="41" y="{y + 47}" font-size="12" font-weight="600" fill="{t["tag_ink"]}">{esc(r["tag"])}</text>')
        if r.get('sub'):
            a(f'<text x="34" y="{y + 76}" font-size="12" fill="{t["muted"]}">{esc(r["sub"])}</text>')
        # bar
        by, bh = y + 6, 40
        a(f'<rect x="{X0}" y="{by}" width="{X1 - X0}" height="{bh}" rx="4" fill="{t["free"]}" stroke="{t["line"]}"/>')
        x = X0
        segs = [('kv', r['kv']), ('weights', r['weights'])]
        for name, pct in segs:
            w = pct * SCALE
            a(f'<rect x="{x:.1f}" y="{by}" width="{w:.1f}" height="{bh}" fill="{t[name]}"/>')
            if name == 'weights':
                a(f'<text x="{x + w / 2:.1f}" y="{by + 25}" font-size="13" font-weight="600" text-anchor="middle" '
                  f'fill="#ffffff">model weights</text>')
            x += w
        kvw = r['kv'] * SCALE
        if kvw >= 60:
            a(f'<text x="{X0 + kvw / 2:.1f}" y="{by + 25}" font-size="13" font-weight="600" text-anchor="middle" '
              f'fill="#1b2330">KV cache</text>')
        else:
            a(f'<text x="{X0 + kvw / 2:.1f}" y="{by - 6}" font-size="11" font-weight="600" text-anchor="middle" '
              f'fill="{t["muted"]}">KV</text>')
        if r['spill']:
            sw = r['spill'] * SCALE
            a(f'<rect x="{X1}" y="{by}" width="{sw:.1f}" height="{bh}" fill="url(#spill)" stroke="{t["hatch"]}"/>')
            a(f'<text x="{X1 + 10}" y="{by + bh + 22}" font-size="12" fill="{t["muted"]}">held in CPU RAM / disk,</text>')
            a(f'<text x="{X1 + 10}" y="{by + bh + 40}" font-size="12" fill="{t["muted"]}">streamed in over PCIe</text>')
        if r.get('kvspill'):
            sw = r['kvspill'] * SCALE
            a(f'<rect x="{X1}" y="{by}" width="{sw:.1f}" height="{bh}" fill="url(#spillkv)" stroke="{t["kv"]}"/>')
            a(f'<text x="{X1 + 10}" y="{by + bh + 22}" font-size="12" fill="{t["muted"]}">attention history in host RAM,</text>')
            a(f'<text x="{X1 + 10}" y="{by + bh + 40}" font-size="12" fill="{t["muted"]}">only needed rows fetched</text>')
        # note
        for j, line in enumerate(r['note']):
            a(f'<text x="{X0}" y="{by + bh + 22 + j * 18}" font-size="13" fill="{t["ink" if j == 0 else "muted"]}">'
              f'{esc(line)}</text>')
    a(f'<line x1="{X1}" y1="{top}" x2="{X1}" y2="{bottom}" stroke="{t["frame"]}" stroke-width="2" stroke-dasharray="5 4"/>')
    a(f'<text x="{X1 - 6}" y="{bottom + 16}" font-size="11" text-anchor="end" fill="{t["muted"]}">GPU memory limit</text>')
    # legend
    ly = H - 22
    lx = 32
    for label, fill in (('model weights', t['weights']), ('KV cache (context)', t['kv']),
                        ('free', t['free']), ('weights outside GPU memory', 'url(#spill)'),
                        ('context outside GPU memory', 'url(#spillkv)')):
        a(f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" fill="{fill}" stroke="{t["line"]}"/>')
        a(f'<text x="{lx + 20}" y="{ly}" font-size="12" fill="{t["muted"]}">{label}</text>')
        lx += 20 + 6.6 * len(label) + 22
    a('</svg>')
    return '\n'.join(o) + '\n'


OUT.mkdir(parents=True, exist_ok=True)
for name, t in THEMES.items():
    (OUT / f'approach-{name}.svg').write_text(svg(t))
print('ok', OUT)
