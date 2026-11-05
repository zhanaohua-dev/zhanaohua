#!/usr/bin/env python3
"""拆「使用场景」四个可动卡片：透明人像素材 + 框内白板（两层叠放）。

输出 img/
  scene-card-N.webp        透明素材（框线 + 人物 + 竖牌），框内纸面已挖空
  scene-card-white-N.webp  框内白板（纯白，alpha = 拱形框内区域）
"""
import json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from collections import deque

SRC = 'img/05-scene.webp'
W, H = 2560, 1440

CARDS = [
    dict(box=(95, 550, 585, 1175), drop=[]),
    dict(box=(600, 490, 1200, 1175), drop=[(860, 489, 1200, 546)]),
    dict(box=(1235, 520, 1975, 1175), drop=[]),
    dict(box=(1995, 555, 2530, 1215), drop=[]),
]
LINE_BAND = (1159, 1172)

img = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32)


def morph(a, r, kind='close'):
    im = Image.fromarray((a * 255).astype(np.uint8))
    if kind == 'open':
        im = im.filter(ImageFilter.MinFilter(2 * r + 1)).filter(ImageFilter.MaxFilter(2 * r + 1))
    else:
        im = im.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.MinFilter(2 * r + 1))
    return np.asarray(im) > 127


def dilate(a, r):
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * r + 1))) > 127


def label(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    areas = {}
    cur = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if lab[sy, sx]:
            continue
        cur += 1
        q = deque([(sy, sx)])
        lab[sy, sx] = cur
        n = 0
        while q:
            y, x = q.popleft()
            n += 1
            if y > 0 and mask[y - 1, x] and not lab[y - 1, x]:
                lab[y - 1, x] = cur; q.append((y - 1, x))
            if y < h - 1 and mask[y + 1, x] and not lab[y + 1, x]:
                lab[y + 1, x] = cur; q.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not lab[y, x - 1]:
                lab[y, x - 1] = cur; q.append((y, x - 1))
            if x < w - 1 and mask[y, x + 1] and not lab[y, x + 1]:
                lab[y, x + 1] = cur; q.append((y, x + 1))
        areas[cur] = n
    return lab, areas


def reachable(passable, starts):
    a = ((~passable) * 255).astype(np.uint8)
    imf = Image.fromarray(np.dstack([a] * 3), 'RGB')
    for s in starts:
        if imf.getpixel(s)[0] == 0:
            ImageDraw.floodfill(imf, s, (128, 128, 128), thresh=0)
    return np.asarray(imf)[..., 0] == 128


meta = []
for i, c in enumerate(CARDS, 1):
    x0, y0, x1, y1 = c['box']
    roi = img[y0:y1, x0:x1]
    hh, ww = roi.shape[:2]

    med = np.asarray(Image.fromarray(roi.astype(np.uint8)).filter(ImageFilter.MedianFilter(31))).astype(np.float32)
    ds = np.abs(roi - med).max(axis=2)
    ds = np.asarray(Image.fromarray(ds.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)

    # ── 人像素材 ──
    passable = morph(ds < 12, 3, 'open')
    edges = [(x, 1) for x in range(0, ww, 2)] + [(x, hh - 2) for x in range(0, ww, 2)] + \
            [(1, y) for y in range(0, hh, 2)] + [(ww - 2, y) for y in range(0, hh, 2)]
    fg = ~reachable(passable, edges)

    lab, areas = label(~fg)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    fg = fg | np.isin(lab, [k for k in areas if k not in border and areas[k] < 6000])
    fg = morph(fg, 4, 'close')

    lab, areas = label(fg)
    fg = np.isin(lab, [k for k, a in areas.items() if a >= 400])

    # 只留贴着主体（人物+框）的块，扫掉远处纸纹噪点
    lab, areas = label(fg)
    if areas:
        main = max(areas, key=areas.get)
        near = dilate(lab == main, 80)
        keep = [k for k in areas if k == main or ((lab == k) & near).any()]
        fg = np.isin(lab, keep)

    for (dx0, dy0, dx1, dy1) in c['drop']:
        fg[dy0:dy1, dx0:dx1] = False

    # ── 拱形框内白板：颜色≈纸、面积成片、不贴边界的连通块 ──
    paperish = morph(ds < 6, 2, 'open')
    lab, areas = label(paperish)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    ids = [k for k, a in areas.items() if k not in border and a > 5000]
    ins = morph(np.isin(lab, ids), 2, 'close')
    # 推向框线内沿：贴着框线的那几像素 ds 偏大，单独补一层
    # 沿着框内纸面向框线推，遇到棕红框线就停
    R, G, Bl = roi[..., 0], roi[..., 1], roi[..., 2]
    frame_line = morph((R - Bl > 52) & (R < 232), 2, 'close')
    blocker = dilate(frame_line, 3)
    for _ in range(4):
        ins = ins | (dilate(ins, 5) & (ds < 28) & ~blocker)
    ins = morph(ins, 2, 'close')
    ins = ins & ~blocker

    # 素材里把框内挖空
    fg = fg & ~ins

    ly0, ly1 = LINE_BAND
    yy0, yy1 = max(ly0 - y0, 0), min(ly1 - y0, fg.shape[0])
    if yy1 > yy0:
        fg[yy0:yy1, :] = False
        ins[yy0:yy1, :] = False

    alpha = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)
    Image.fromarray(np.dstack([roi, alpha]).astype(np.uint8), 'RGBA').save(f'/tmp/scene-card-{i}.png')

    wa = np.asarray(Image.fromarray((ins * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))).astype(np.float32)
    white = np.dstack([np.full(roi.shape, 255, np.float32), wa]).astype(np.uint8)
    Image.fromarray(white, 'RGBA').save(f'/tmp/scene-card-white-{i}.png')

    meta.append(dict(n=i, left=x0, top=y0, w=x1 - x0, h=y1 - y0))
    print(i, 'fg', int(fg.sum()), 'ins', int(ins.sum()))

json.dump(meta, open('/tmp/scene-meta.json', 'w'), indent=1)
for m in meta:
    print(f'card{m["n"]}: {m["left"]/W*100:.3f}% {m["top"]/H*100:.3f}% {m["w"]/W*100:.3f}% {m["h"]/H*100:.3f}%')
