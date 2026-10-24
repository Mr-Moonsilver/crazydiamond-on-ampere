#!/usr/bin/env python3
"""Generate the image test set for the vision qualification (expected answers in manifest.json).
screenshot: terminal-like text with a small 13 px font and a code; table: 4 x 4 grid, ask one cell; diagram: three
labelled boxes with arrows, ask the order; isolation: 8 images with different codes (same layout); big: 4000 x 3000
with a code, to check the pixel bound is enforced (token count). Natural photos: penguin.jpg, zebra.jpg (from macOS).
usage: make_vision_assets.py OUTDIR"""
import json, os, random, sys
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
rnd = random.Random(20261024)
FONT = '/System/Library/Fonts/Menlo.ttc'


def font(px):
    return ImageFont.truetype(FONT, px)


man = {}
# screenshot with small text
code = 'ZK-' + ''.join(rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))
img = Image.new('RGB', (1280, 720), (30, 30, 36)); d = ImageDraw.Draw(img)
lines = [f'$ ./deploy --env staging --build {rnd.randint(1000, 9999)}'] + [f'[{i:02d}:{rnd.randint(10, 59)}] step {i} ok' for i in range(18)]
lines.insert(9, f'[09:41] release token: {code}')
for i, l in enumerate(lines):
    d.text((24, 20 + i * 18), l, font=font(13), fill=(200, 220, 200))
img.save(os.path.join(out, 'screenshot.png')); man['screenshot.png'] = {'q': 'What is the release token shown in this terminal screenshot? Answer with the token only.', 'a': code}
# table
img = Image.new('RGB', (900, 420), 'white'); d = ImageDraw.Draw(img)
cols, rows = ['Region', 'Q1', 'Q2', 'Q3'], ['North', 'South', 'East', 'West']
vals = {(r, c): rnd.randint(100, 999) for r in rows for c in cols[1:]}
for j, c in enumerate(cols):
    d.text((30 + j * 210, 30), c, font=font(26), fill='black')
for i, r in enumerate(rows):
    d.text((30, 100 + i * 75), r, font=font(26), fill='black')
    for j, c in enumerate(cols[1:], 1):
        d.text((30 + j * 210, 100 + i * 75), str(vals[(r, c)]), font=font(26), fill='black')
for i in range(6):
    d.line((20, 80 + i * 75, 880, 80 + i * 75), fill='gray')
img.save(os.path.join(out, 'table.png'))
man['table.png'] = {'q': 'In this table, what is the value for region East in column Q2? Answer with the number only.', 'a': str(vals[('East', 'Q2')]),
                    'followup_q': 'And the value for region North in column Q3? Answer with the number only.', 'followup_a': str(vals[('North', 'Q3')])}
# diagram
img = Image.new('RGB', (1000, 360), 'white'); d = ImageDraw.Draw(img)
labels = rnd.sample(['Parser', 'Router', 'Cache', 'Scheduler', 'Encoder', 'Writer'], 3)
for i, lab in enumerate(labels):
    x = 60 + i * 320
    d.rectangle((x, 120, x + 220, 220), outline='black', width=4); d.text((x + 30, 155), lab, font=font(30), fill='black')
    if i < 2:
        d.line((x + 220, 170, x + 320, 170), fill='black', width=4); d.polygon([(x + 320, 170), (x + 300, 158), (x + 300, 182)], fill='black')
img.save(os.path.join(out, 'diagram.png'))
man['diagram.png'] = {'q': 'This diagram shows three boxes connected by arrows. List the box labels in the order the arrows point, separated by " -> ".', 'a': ' -> '.join(labels)}
# isolation set
iso = []
for k in range(8):
    c = 'ISO-' + ''.join(rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(5))
    img = Image.new('RGB', (800, 300), (250, 250, 245)); d = ImageDraw.Draw(img)
    d.text((40, 120), f'Private image code: {c}', font=font(36), fill=(20, 20, 20))
    name = f'iso{k}.png'; img.save(os.path.join(out, name)); iso.append({'file': name, 'code': c})
man['isolation'] = iso
# big image (pixel bound check)
c = 'BIG-' + ''.join(rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(5))
img = Image.new('RGB', (4000, 3000), (240, 240, 255)); d = ImageDraw.Draw(img)
d.text((400, 1400), f'Code: {c}', font=font(160), fill='black')
img.save(os.path.join(out, 'big.png')); man['big.png'] = {'q': 'What code is written in this image? Answer with the code only.', 'a': c}
man['penguin.jpg'] = {'q': 'What animal is shown in this photo? Answer with one word.', 'a': 'penguin'}
man['zebra.jpg'] = {'q': 'What animal is shown in this photo? Answer with one word.', 'a': 'zebra'}
json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
print('wrote', sorted(os.listdir(out)))
