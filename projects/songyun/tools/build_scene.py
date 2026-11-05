#!/usr/bin/env python3
"""拆「使用场景」四个可动卡片：透明人像素材 + 框内白板遮罩。

输出 /tmp/
  scene-card-N.png   透明素材（框线 + 人物 + 竖牌），框内纸面已挖空
  scene-white-N.png  框内白板遮罩（白色，alpha = 框内区域）
定位写进 scene-boxes.json，供 HTML 用。
"""
import json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from collections import deque

SRC = 'img/05-scene.webp'
W, H = 2560, 1440

CARDS = [
    dict(box=(95, 550, 585, 1175), drop=[], seed=(355, 700)),
    dict(box=(600, 490, 1200, 1175), drop=[(860, 489, 1200, 546)], seed=(900, 660)),
    dict(box=(1235, 520, 1975, 1175), drop=[], seed=(1875, 620)),
    dict(box=(1995, 555, 2530, 1215), drop=[], seed=(2200, 660)),
]
LINE_BAND = (1159, 1172)

img = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32)


def blur(a, r):
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), 'F')) if False else \
            np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
    return np.dstack([blur(a[..., k], r) for k in range(a.shape[2])])


def label(mask):
    """4-连通标记，返回 labels(int32) 与 每块面积 dict"""
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


def morph(a, r):
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.MinFilter(2 * r + 1))
    return np.asarray(im) > 127


def fg_mask(roi, drops):
    med = np.asarray(Image.fromarray(roi.astype(np.uint8)).filter(ImageFilter.MedianFilter(31))).astype(np.float32)
    d = np.abs(roi - med).max(axis=2)
    d = np.asarray(Image.fromarray(d.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32)
    seed = (d > 14).astype(np.uint8) * 255
    seed = np.asarray(Image.fromarray(seed).filter(ImageFilter.MedianFilter(3)))

    rgb = np.dstack([seed] * 3)
    imf = Image.fromarray(rgb, 'RGB')
    px = imf.load(); w, h = imf.size
    starts = [(x, 1) for x in range(0, w, 2)] + [(x, h - 2) for x in range(0, w, 2)] + \
             [(1, y) for y in range(0, h, 2)] + [(w - 2, y) for y in range(0, h, 2)]
    for s in starts:
        if px[s][0] == 0:
            ImageDraw.floodfill(imf, s, (128, 128, 128), thresh=0)
    fg = np.asarray(imf)[..., 0] != 128

    for (dx0, dy0, dx1, dy1) in drops:
        fg[dy0:dy1, dx0:dx1] = False

    # 填补被前景包住的空洞
    lab, areas = label(~fg)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = np.isin(lab, [k for k in areas if k not in border])
    fg = fg | holes

    # 面积过滤：删掉小噪点
    lab, areas = label(fg)
    keep = [k for k, a in areas.items() if a >= 400]
    fg = np.isin(lab, keep)

    return morph(fg, 4)


def inside_mask(roi, seed_xy, box):
    """框内纸面区域：从种子点 flood 纸色，框线/人物当屏障"""
    med = np.asarray(Image.fromarray(roi.astype(np.uint8)).filter(ImageFilter.MedianFilter(51))).astype(np.float32)
    d = np.abs(roi - med).max(axis=2)
    luma = roi.mean(axis=2)
    passable = ((d < 9) & (luma > 195)).astype(np.uint8) * 255
    passable = np.asarray(Image.fromarray(passable).filter(ImageFilter.MedianFilter(3)))
    rgb = np.dstack([passable] * 3)
    imf = Image.fromarray(rgb, 'RGB')
    sx, sy = seed_xy
    ImageDraw.floodfill(imf, (sx, sy), (128, 128, 128), thresh=0)
    m = np.asarray(imf)[..., 0] == 128
    lab, areas = label(m)
    if not areas:
        return m
    # 只保留与种子同块
    keep = lab[sy, sx]
    m = lab == keep
    return morph(m, 2)


boxes = []
for i, c in enumerate(CARDS, 1):
    x0, y0, x1, y1 = c['box']
    roi = img[y0:y1, x0:x1]
    fg = fg_mask(roi, c['drop'])
    ins = inside_mask(roi, (c['seed'][0] - x0, c['seed'][1] - y0), c['box'])

    # 线带挖空
    ly0, ly1 = LINE_BAND
    yy0, yy1 = max(ly0 - y0, 0), min(ly1 - y0, fg.shape[0])
    if yy1 > yy0:
        fg[yy0:yy1, :] = False

    alpha = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)
    Image.fromarray(np.dstack([roi, alpha]).astype(np.uint8), 'RGBA').save(f'/tmp/scene-card-{i}.png')

    white = np.zeros(roi.shape, np.float32) + 255
    wa = np.asarray(Image.fromarray(((~ins) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)
    Image.fromarray(np.dstack([white, 255 - wa]).astype(np.uint8), 'RGBA').save(f'/tmp/scene-white-{i}.png')

    boxes.append(dict(n=i, box=[x0, y0, x1 - x0, y1 - y0]))
    print(i, 'fg', int(fg.sum()), 'inside', int(ins.sum()))

json.dump(boxes, open('/tmp/scene-boxes.json', 'w'), indent=1)
print('done')
