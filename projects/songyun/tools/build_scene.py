#!/usr/bin/env python3
"""拆「使用场景」四个可动卡片：透明人像素材 + 框内白板（两层叠放）。

用法：cd projects/songyun && python3 tools/build_scene.py

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


def box_blur(a, r):
    a = a.astype(np.float32)
    k = 2 * r + 1
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * a.ndim
            pad[axis] = (r, r)
            ap = np.pad(a, pad, mode='edge')
            c = np.cumsum(ap, axis=axis)
            zero_shape = list(c.shape); zero_shape[axis] = 1
            c = np.concatenate([np.zeros(zero_shape, np.float32), c], axis=axis)
            hi = [slice(None)] * a.ndim; hi[axis] = slice(k, None)
            lo = [slice(None)] * a.ndim; lo[axis] = slice(0, -k)
            a = (c[tuple(hi)] - c[tuple(lo)]) / k
    return a


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
    # 笔触之间漏掉的浅色小块一律补进素材（宁可多盖，不留白点）
    fg = fg | np.isin(lab, [k for k, a in areas.items() if a < 3000])
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
    # 收敛成「拱形框内那片纸」：在框内矩形范围里，把所有成片的纸面都算上
    ins = morph(ins, 6, 'open')
    ins = morph(ins, 7, 'close')
    ins = ins & ~blocker

    if ins.any():
        rows = np.where(ins.any(axis=1))[0]
        cols = np.where(ins.any(axis=0))[0]
        bb = np.zeros_like(ins)
        bb[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
        cand = morph((ds < 8) & bb & ~blocker, 2, 'open')
        lab, areas = label(cand)
        ids = [k for k in areas if (ins & (lab == k)).any()]
        ins = np.isin(lab, ids)
        ins = morph(ins, 4, 'close')
        ins = ins & ~blocker
        # 逐行补满：架空里人物挡出来的缺口也并进来，提亮时框内是一整片
        for y in np.where(ins.any(axis=1))[0]:
            xs = np.where(ins[y])[0]
            ins[y, xs.min():xs.max() + 1] = True

    # 框内纸面并入素材：提亮时框里自然发白，不用再加一层白
    fg = fg | ins

    ly0, ly1 = LINE_BAND
    yy0, yy1 = max(ly0 - y0, 0), min(ly1 - y0, fg.shape[0])
    if yy1 > yy0:
        fg[yy0:yy1, :] = False
        ins[yy0:yy1, :] = False

    # 轮廓收 1px：把沿边那一圈"纸色"裁掉，提亮时才不会浮出白边
    fg_in = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 127
    alpha_fg = np.asarray(Image.fromarray((fg_in * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32)
    # 框内纸面那一块边缘要柔，提亮后是宣纸渐亮而不是一条白边
    # 边缘硬化一点：半透明带越窄，提亮后人物周围越不会有灰晕
    a8 = np.asarray(Image.fromarray((fg_in * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32)
    alpha = np.where(a8 > 140, 255.0, 0.0).astype(np.float32)
    alpha = np.asarray(Image.fromarray(alpha.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.55))).astype(np.float32)

    # 边缘那一两像素原来的颜色是纸色，提亮后会浮出一圈白边；
    # 把「明显是纸色」的像素换成邻近实色的颜色（alpha 不变，默认态看不出差别）
    core = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 127
    if core.any():
        w = core.astype(np.float32)
        num = box_blur(roi * w[..., None], 4)
        den = box_blur(w, 4)
        ok = (~core) & (den > 0.10)          # 连 fg 外那一圈半透明边一起换色，提亮才不会出光晕
        fill = num / np.maximum(den, 1e-3)[..., None]
        roi = np.where(ok[..., None], fill, roi)

    Image.fromarray(np.dstack([roi, alpha]).astype(np.uint8), 'RGBA').save(
        f'img/scene-card-{i}.webp', 'WEBP', lossless=True, quality=100, method=6)

    meta.append(dict(n=i, left=x0, top=y0, w=x1 - x0, h=y1 - y0))
    print(i, 'fg', int(fg.sum()), 'ins', int(ins.sum()))

for m in meta:
    print(f'card{m["n"]}: {m["left"]/W*100:.3f}% {m["top"]/H*100:.3f}% {m["w"]/W*100:.3f}% {m["h"]/H*100:.3f}%')
