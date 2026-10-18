#!/usr/bin/env python3
"""西梅田 禅園 焼物PRリール (1080x1920 / 30fps / 30s) を生成する。

usage: python3 make_reel.py --fonts DIR [--out reel.mp4] [--still T ...]
  DIR には Mincho-500.ttf / Mincho-700.ttf (Shippori Mincho) と
  Cormorant-500.ttf (Cormorant Garamond) を置く。
"""
import argparse
import math
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 30.0
NFRAMES = int(DUR * FPS)
ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "..", "images")
SRC_H = 2880                  # 読み込み時に縮小する高さ
THUMB = 600.0                 # 構図指定は 900x600 基準の座標で書く

TEXT = (0.965, 0.937, 0.894)
GOLD = (0.82, 0.69, 0.47)
SHIKON = (0.27, 0.12, 0.33)   # 紫紺

# --- ショット定義 ------------------------------------------------------------
# cam: (開始中心x, y, 開始高さ, 終了中心x, y, 終了高さ, 開始角度, 終了角度)  900x600基準
SHOTS = [
    dict(img="9月_調理風景0011.jpg", t0=0.0, t1=5.4,
         cam=(415, 300, 560, 520, 340, 470, 0, 0), fire=0.0),
    dict(img="9月_調理風景0021.jpg", t0=4.8, t1=10.2,
         cam=(450, 300, 600, 455, 285, 520, 0, 0), fire=1.0),
    dict(img="9月_調理風景0040.jpg", t0=9.4, t1=14.6,
         cam=(510, 295, 520, 520, 300, 450, 0, 0), fire=1.0),
    dict(img="9月_調理風景0044.jpg", t0=14.0, t1=19.4,
         cam=(590, 310, 600, 592, 290, 520, 0, 0), fire=0.8),
    dict(img="紫紺_焼物0009.jpg", t0=18.8, t1=25.2,
         cam=(452, 315, 600, 452, 315, 545, 0, -4), fire=0.0, plate=True),
    dict(img="紫紺_焼物0002修.jpg", t0=24.4, t1=30.0,
         cam=(455, 320, 600, 455, 320, 540, 0, 0), fire=0.0, endcard=True),
]
# (開始, 終了, 種類)  隣り合うショット間のトランジション
TRANS = [
    (4.8, 5.4, "heat"),
    (9.4, 10.2, "glow"),
    (14.0, 14.6, "whip"),
    (18.8, 19.5, "brush"),
    (24.4, 25.2, "dissolve"),
]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_cam(x):
    x = min(max(x, 0.0), 1.0)
    return 0.6 * x + 0.4 * (0.5 - 0.5 * math.cos(math.pi * x))


# --- 画像 -------------------------------------------------------------------
_cache = {}


def load(name, blur=False):
    key = (name, blur)
    if key not in _cache:
        im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
        im = im.resize((round(im.width * SRC_H / im.height), SRC_H), Image.LANCZOS)
        if blur:
            small = im.resize((im.width // 8, im.height // 8), Image.LANCZOS)
            small = small.filter(ImageFilter.GaussianBlur(10))
            im = small.resize(im.size, Image.BICUBIC)
        _cache[key] = im
    return _cache[key]


def shot_frame(shot, t):
    p = ease_cam((t - shot["t0"]) / (shot["t1"] - shot["t0"]))
    cx0, cy0, h0, cx1, cy1, h1, a0, a1 = shot["cam"]
    src = load(shot["img"], blur=shot.get("endcard", False))
    s = src.height / THUMB
    cx = (cx0 + (cx1 - cx0) * p) * s
    cy = (cy0 + (cy1 - cy0) * p) * s
    k = (h0 + (h1 - h0) * p) * s / H
    th = math.radians(a0 + (a1 - a0) * p)
    a, b = k * math.cos(th), -k * math.sin(th)
    d, e = k * math.sin(th), k * math.cos(th)
    c = cx - a * W / 2 - b * H / 2
    f = cy - d * W / 2 - e * H / 2
    if shot.get("plate"):
        return plate_frame(shot, src, p, a, b, d, e, cx, cy)
    out = src.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    arr = np.asarray(out, dtype=np.float32) / 255.0
    if shot.get("endcard"):
        arr = arr * 0.34 * np.array([0.95, 0.9, 1.05], np.float32)
    return arr


PLATE_Y = 860   # 皿の中心を置く画面上の高さ


_plate_bg = {}


def plate_bg(name):
    """皿の左側の卓 (赤) だけを引き伸ばしてぼかした背景"""
    if name not in _plate_bg:
        src = load(name, blur=True)
        strip = src.crop((0, 0, int(src.width * 0.2), src.height)).resize((W, H), Image.BICUBIC)
        _plate_bg[name] = np.asarray(strip.filter(ImageFilter.GaussianBlur(30)), np.float32) / 255.0
    return _plate_bg[name]


def plate_frame(shot, src, p, a, b, d, e, cx, cy):
    """皿全体を見せるため、横長写真を正方形で上寄せに置き、上下はぼかした卓で埋める。"""
    # 横幅いっぱい=正方形 (高さ600基準で縦全体) を 1080px に
    k = src.height / W * (1 - 0.08 * p)
    th = math.atan2(d, a)
    a2, b2 = k * math.cos(th), -k * math.sin(th)
    d2, e2 = k * math.sin(th), k * math.cos(th)
    c2 = cx - a2 * W / 2 - b2 * PLATE_Y
    f2 = cy - d2 * W / 2 - e2 * PLATE_Y
    fg = np.asarray(src.transform((W, H), Image.AFFINE, (a2, b2, c2, d2, e2, f2), resample=Image.BICUBIC),
                    dtype=np.float32) / 255.0
    bg = plate_bg(shot["img"])
    half = W / 2 * (1 - 0.08 * p) ** -1 * 0.88
    dist = np.abs(yy[:, :1] - PLATE_Y)
    m = np.clip((half - dist) / 140, 0, 1)[..., None]
    return fg * m + bg * 0.9 * (1 - m)


# --- 共通レイヤー -------------------------------------------------------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r2 = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
VIGNETTE = np.clip(1 - 0.38 * _r2, 0.45, 1)[..., None]


def _lowfreq_noise(seed, scale):
    rng = np.random.default_rng(seed)
    small = rng.random((H // scale + 2, W // scale + 2)).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((W + scale * 2, H + scale * 2), Image.BICUBIC)
    n = np.asarray(im, dtype=np.float32)[:H, :W] / 255.0
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(scale / 3)),
                   dtype=np.float32) / 255.0
    return (n - n.min()) / (n.max() - n.min())


HEAT_NOISE = _lowfreq_noise(7, 60)
GRAIN = [np.random.default_rng(100 + i).normal(0, 0.014, (H, W, 1)).astype(np.float32) for i in range(6)]


def grade(arr, t):
    # 暖色寄りのトーンとビネット
    arr = arr * np.array([1.03, 1.0, 0.95], np.float32)
    arr = arr * VIGNETTE
    arr = arr + GRAIN[int(t * FPS) % len(GRAIN)]
    return arr


# --- 火の粉 -----------------------------------------------------------------
def _glow_sprite(r):
    g = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d2 = (g[0] ** 2 + g[1] ** 2) / (r * r)
    a = np.exp(-d2 * 9) + 0.35 * np.exp(-d2 * 2.2)
    return a[..., None] * np.array([1.0, 0.55, 0.16], np.float32)


EMBER_SPRITES = {r: _glow_sprite(r) for r in (6, 9, 13)}
_rng = np.random.default_rng(3)
EMBERS = [dict(t0=_rng.uniform(4.0, 19.5), life=_rng.uniform(1.4, 2.8),
               x=_rng.uniform(80, W - 80), y=_rng.uniform(1250, 1950),
               vy=_rng.uniform(170, 360), sway=_rng.uniform(10, 40), ph=_rng.uniform(0, 6.28),
               r=int(_rng.choice([6, 9, 13], p=[0.5, 0.35, 0.15])), fl=_rng.uniform(6, 14))
          for _ in range(150)]


def add_embers(arr, t, amount):
    if amount <= 0:
        return arr
    for e in EMBERS:
        age = t - e["t0"]
        if not 0 <= age <= e["life"]:
            continue
        life = age / e["life"]
        alpha = math.sin(math.pi * life) * (0.65 + 0.35 * math.sin(e["fl"] * t + e["ph"])) * amount
        x = int(e["x"] + e["sway"] * math.sin(2.1 * age + e["ph"]))
        y = int(e["y"] - e["vy"] * age)
        spr = EMBER_SPRITES[e["r"]]
        r = e["r"]
        y0, y1, x0, x1 = y - r, y + r + 1, x - r, x + r + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        arr[y0:y1, x0:x1] += spr * alpha
    return arr


# --- テキスト ---------------------------------------------------------------
class Fonts:
    def __init__(self, d):
        self.dir = d
        self._c = {}

    def get(self, name, size):
        key = (name, size)
        if key not in self._c:
            self._c[key] = ImageFont.truetype(os.path.join(self.dir, name), size)
        return self._c[key]


FONTS = None


def _sprite_from_img(img, shadow):
    """RGBA(白文字) → (色付き前景 premultiplied, alpha) の numpy ペア。影を含む。"""
    a = np.asarray(img.split()[3], dtype=np.float32) / 255.0
    sh = np.asarray(img.split()[3].filter(ImageFilter.GaussianBlur(shadow)), dtype=np.float32) / 255.0
    return a, np.clip(sh * 1.6, 0, 1) * 0.55


def render_vertical_char(ch, size, font):
    pad = size
    img = Image.new("RGBA", (size + pad * 2, size + pad * 2), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    ox, oy = pad, pad
    if ch in "、。":
        ox, oy = pad + int(size * 0.6), pad - int(size * 0.62)
    dr.text((ox, oy), ch, font=font, fill=(255, 255, 255, 255))
    return _sprite_from_img(img, size * 0.18), pad


def render_line(text, font, tracking):
    size = font.size
    widths = [font.getlength(c) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    pad = size
    img = Image.new("RGBA", (int(total) + pad * 2, int(size * 1.5) + pad * 2), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    x = pad
    for c, w in zip(text, widths):
        dr.text((x, pad), c, font=font, fill=(255, 255, 255, 255))
        x += w + tracking
    return _sprite_from_img(img, size * 0.2), pad, total


def blit(arr, sprite, x, y, color, alpha):
    if alpha <= 0.003:
        return
    a, sh = sprite
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    ys0, xs0 = max(0, -y), max(0, -x)
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(H, y + h), min(W, x + w)
    if y1 <= y0 or x1 <= x0:
        return
    a = a[ys0:ys0 + (y1 - y0), xs0:xs0 + (x1 - x0), None] * alpha
    sh = sh[ys0:ys0 + (y1 - y0), xs0:xs0 + (x1 - x0), None] * alpha
    reg = arr[y0:y1, x0:x1]
    reg *= (1 - sh)
    reg[:] = reg * (1 - a) + np.array(color, np.float32) * a


class VText:
    """縦書きテロップ。columns は右から左の順。"""

    def __init__(self, columns, x_right, y_top, size, t_in, t_out, font="Mincho-500.ttf",
                 color=TEXT, stagger=0.07, col_delay=0.35, backdrop=0.4):
        f = FONTS.get(font, size)
        self.items = []
        lead, gap = size * 1.14, size * 1.55
        delay = 0.0
        for ci, col in enumerate(columns):
            x = x_right - size - ci * gap
            for i, ch in enumerate(col):
                spr, pad = render_vertical_char(ch, size, f)
                self.items.append((spr, x - pad, y_top + i * lead - pad, t_in + delay + i * stagger))
            delay += len(col) * stagger + col_delay
        self.size, self.t_out, self.color = size, t_out, color
        self.t_in = t_in
        # 文字の背後をうっすら暗くして可読性を確保
        x0 = x_right - size - (len(columns) - 1) * gap - size * 0.6
        x1 = x_right + size * 0.6
        y1 = y_top + max(len(c) for c in columns) * lead + size * 0.3
        box = Image.new("L", (W, H), 0)
        ImageDraw.Draw(box).rectangle([x0, y_top - size * 0.5, x1, y1], fill=255)
        box = box.filter(ImageFilter.GaussianBlur(size * 1.1))
        self.back = (np.asarray(box, np.float32) / 255.0 * backdrop)[..., None]

    def draw(self, arr, t):
        out = 1 - smooth((t - self.t_out) / 0.5)
        if out <= 0:
            return
        k = smooth((t - self.t_in + 0.2) / 0.6) * out
        if k > 0 and self.back.max() > 0:
            arr *= 1 - self.back * k
        for spr, x, y, ts in self.items:
            p = smooth((t - ts) / 0.55)
            if p <= 0:
                continue
            blit(arr, spr, x, y - (1 - p) * self.size * 0.35, self.color, p * out)


class HLine:
    def __init__(self, text, cx, y, size, t_in, t_out, font="Mincho-500.ttf", color=TEXT, tracking=0.0,
                 rise=18):
        f = FONTS.get(font, size)
        self.spr, self.pad, self.w = render_line(text, f, tracking * size)
        self.x, self.y = cx - self.w / 2 - self.pad, y - self.pad
        self.t_in, self.t_out, self.color, self.rise = t_in, t_out, color, rise

    def draw(self, arr, t):
        p = smooth((t - self.t_in) / 0.7) * (1 - smooth((t - self.t_out) / 0.5))
        blit(arr, self.spr, self.x, self.y + (1 - smooth((t - self.t_in) / 0.7)) * self.rise, self.color, p)


class Rule:
    """中央から左右に伸びる細線"""

    def __init__(self, cx, y, half, t_in, t_out, color=GOLD, thick=2):
        self.cx, self.y, self.half, self.t_in, self.t_out, self.color, self.thick = cx, y, half, t_in, t_out, color, thick

    def draw(self, arr, t):
        p = smooth((t - self.t_in) / 0.8)
        o = 1 - smooth((t - self.t_out) / 0.5)
        if p <= 0 or o <= 0:
            return
        hw = int(self.half * p)
        reg = arr[self.y:self.y + self.thick, self.cx - hw:self.cx + hw]
        reg[:] = reg * (1 - 0.9 * o) + np.array(self.color, np.float32) * 0.9 * o


class Gradient:
    def __init__(self, y0, y1, strength, t_in, t_out, tint=(0, 0, 0)):
        ramp = np.clip((np.arange(H, dtype=np.float32) - y0) / (y1 - y0), 0, 1)
        self.mask = (ramp ** 1.4 * strength)[:, None, None]
        self.tint = np.array(tint, np.float32)
        self.t_in, self.t_out = t_in, t_out

    def draw(self, arr, t):
        p = smooth((t - self.t_in) / 0.8) * (1 - smooth((t - self.t_out) / 0.5))
        if p > 0:
            m = self.mask * p
            arr[:] = arr * (1 - m) + self.tint * m


def build_overlays():
    ov = []
    # 0: 仕込み
    ov.append(Gradient(560, 0, 0.55, 0.0, 4.6))
    ov.append(HLine("西梅田 禅園", W / 2, 262, 32, 0.3, 4.6, tracking=0.5, color=GOLD))
    ov.append(VText(["一切れに、", "心を通す。"], 965, 400, 74, 0.7, 4.5, backdrop=0.55))
    # 1: 炭火
    ov.append(VText(["炭火と、", "向き合う。"], 300, 360, 74, 5.6, 9.3))
    # 2: 焼き色
    ov.append(VText(["焼き加減は、", "職人の眼が決める。"], 965, 330, 70, 10.5, 13.8))
    # 3: 柚庵地の艶
    ov.append(VText(["柚庵地を重ね、", "艶をまとう。"], 300, 360, 72, 15.0, 18.6))
    # 4: 商品カード
    ov.append(Gradient(1300, 1640, 0.6, 19.4, 24.6))
    ov.append(HLine("紫紺コース　焼物", W / 2, 1400, 36, 19.9, 24.6, color=GOLD, tracking=0.3))
    ov.append(Rule(W // 2, 1466, 150, 20.2, 24.6))
    ov.append(HLine("秋鮭柚庵焼き", W / 2, 1490, 84, 20.3, 24.6, font="Mincho-700.ttf", tracking=0.12))
    ov.append(HLine("朴葉ソース　舞茸", W / 2, 1610, 44, 20.8, 24.6, tracking=0.2))
    ov.append(HLine("※画像はイメージです", W / 2, 1690, 24, 21.2, 24.6, tracking=0.1, color=(0.85, 0.82, 0.78)))
    # 5: エンドカード
    ov.append(HLine("懐石・割烹", W / 2, 700, 38, 25.3, 99, tracking=0.6, color=GOLD))
    ov.append(HLine("西梅田 禅園", W / 2, 775, 104, 25.6, 99, font="Mincho-700.ttf", tracking=0.22))
    ov.append(HLine("ZEN-EN  NISHI-UMEDA", W / 2, 925, 38, 26.0, 99, font="Cormorant-500.ttf",
                    tracking=0.25, color=GOLD))
    ov.append(Rule(W // 2, 1010, 170, 26.3, 99))
    ov.append(HLine("旬の恵みと匠の技を、大切なひとときに。", W / 2, 1060, 36, 26.6, 99, tracking=0.08))
    ov.append(HLine("大阪・西梅田　ハービスPLAZA B2F", W / 2, 1150, 32, 27.0, 99, tracking=0.08,
                    color=(0.85, 0.83, 0.79)))
    ov.append(HLine("ご予約はプロフィールのリンクから", W / 2, 1250, 36, 27.4, 99, tracking=0.1, color=GOLD))
    return ov


OVERLAYS = None


# --- トランジション -----------------------------------------------------------
def tr_heat(a, b, p):
    v = 0.62 * (yy / H)[..., None] + 0.38 * HEAT_NOISE[..., None]
    edge = 1.25 * (1 - smooth(p)) - 0.1
    m = np.clip((v - edge) / 0.09, 0, 1)
    glow = np.exp(-((v - edge) / 0.08) ** 2) * math.sin(math.pi * p)
    out = a * (1 - m) + b * m
    return out + glow * np.array([1.0, 0.42, 0.1], np.float32) * 0.38


def tr_glow(a, b, p):
    s = smooth(p)
    out = a * (1 - s) + b * s
    small = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((W // 6, H // 6), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    bloom = np.asarray(small, np.float32) / 255.0
    k = math.sin(math.pi * p) * 0.55
    return out + bloom * bloom * np.array([1.0, 0.7, 0.4], np.float32) * k


def tr_whip(a, b, p):
    s = 0.5 - 0.5 * math.cos(math.pi * p)
    speed = math.sin(math.pi * p)
    stack = np.concatenate([a, b], axis=0)
    off = s * H
    acc = np.zeros_like(a)
    n = 9
    for i in range(n):
        o = int(np.clip(off + (i - (n - 1) / 2) * speed * 26, 0, H))
        acc += stack[o:o + H]
    return acc / n


def tr_circle(a, b, p, center=(W / 2, H / 2)):
    s = smooth(p)
    r = s * math.hypot(W, H) * 0.62
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)[..., None]
    m = np.clip((r - d) / 160 + 0.5, 0, 1)
    return a * (1 - m) + b * m


BRUSH_NOISE = _lowfreq_noise(21, 24)


def tr_brush(a, b, p):
    """筆で掃くような横方向のワイプ"""
    v = 0.78 * (xx / W)[..., None] + 0.22 * BRUSH_NOISE[..., None]
    edge = smooth(p) * 1.3 - 0.15
    m = np.clip((edge - v) / 0.06, 0, 1)
    return a * (1 - m) + b * m


def tr_dissolve(a, b, p):
    s = smooth(p)
    return a * (1 - s) + b * s


TR_FN = dict(heat=tr_heat, glow=tr_glow, whip=tr_whip, circle=tr_circle, brush=tr_brush, dissolve=tr_dissolve)


# --- フレーム合成 -------------------------------------------------------------
def render(t):
    active = [i for i, s in enumerate(SHOTS) if s["t0"] <= t < s["t1"] or (i == len(SHOTS) - 1 and t >= s["t0"])]
    if len(active) == 1:
        arr = shot_frame(SHOTS[active[0]], t)
    else:
        i = active[0]
        t0, t1, kind = TRANS[i]
        arr = TR_FN[kind](shot_frame(SHOTS[i], t), shot_frame(SHOTS[i + 1], t), (t - t0) / (t1 - t0))
    fire = 0.0
    for s in SHOTS:
        w = smooth((t - s["t0"]) / 0.6) * (1 - smooth((t - s["t1"] + 0.6) / 0.6))
        fire = max(fire, s["fire"] * w)
    if fire > 0:
        flick = 1 + 0.025 * fire * math.sin(23 * t) * math.sin(7.3 * t + 1)
        arr = arr * flick
    arr = add_embers(arr.copy(), t, fire)
    for o in OVERLAYS:
        o.draw(arr, t)
    arr = grade(arr, t)
    # 開始・終了のフェード
    arr = arr * smooth(t / 0.35) * (1 - 0.6 * smooth((t - (DUR - 0.5)) / 0.5))
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def init_worker(fonts_dir):
    global FONTS, OVERLAYS
    FONTS = Fonts(fonts_dir)
    OVERLAYS = build_overlays()


def frame_bytes(i):
    return render(i / FPS).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "zenen_yakimono_reel.mp4"))
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--still-dir", default=".")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--music", default=os.path.join(ROOT, "Morning_Mist_in_the_Garden.mp3"),
                    help="BGM (冒頭30秒を使用)。空文字なら無音")
    args = ap.parse_args()

    if args.still:
        init_worker(args.fonts)
        for t in args.still:
            Image.fromarray(render(t)).save(os.path.join(args.still_dir, f"still_{t:05.2f}.jpg"), quality=88)
        return

    if args.music:
        audio_in = ["-i", args.music]
        afilter = ["-af", f"atrim=0:{DUR},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.3,"
                          f"afade=t=out:st={DUR - 2.5}:d=2.5,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"]
    else:
        audio_in = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        afilter = []
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", *audio_in, "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "12M", "-bufsize", "24M",
           "-pix_fmt", "yuv420p", "-profile:v", "high", *afilter, "-c:a", "aac", "-b:a", "192k", "-t", str(DUR),
           "-movflags", "+faststart", "-shortest", args.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.jobs, initializer=init_worker, initargs=(args.fonts,)) as pool:
        for i, buf in enumerate(pool.imap(frame_bytes, range(NFRAMES), chunksize=4)):
            ff.stdin.write(buf)
            if i % 90 == 0:
                print(f"frame {i}/{NFRAMES}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
