import cv2, numpy as np, random, time, os, json
from collections import deque

# ============================== 1. CONFIG ==============================
W, H = 640, 480
R = 14                                                    # ball radius
GRAVITY, BOUNCE, MAXV = 700, 0.65, 650                    # gravity, bounciness, speed limit (px/s)
K5 = np.ones((5, 5), np.uint8)
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
DISK = xx ** 2 + yy ** 2 <= R * R                         # ball footprint used for collision checks
ZERO = np.zeros(2)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GAME_WIN, MASK_WIN, CTRL_WIN = "GUIDANCE", "Mask", "Controls"
BOSS_R = 46                                               # Groudon hit radius
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                         # e.g. running inside a notebook / Spyder cell
    HERE = os.getcwd()
HSV_FILE = os.path.join(HERE, "hsv_config.json")          # slider values are saved here between runs

# --- HSV settings (edited live from the "Controls" window) ---
HSV_KEYS = [("H min", 179), ("H max", 179), ("S min", 255), ("S max", 255), ("V min", 255), ("V max", 255)]
HSV = {"H min": 10, "H max": 35, "S min": 60, "S max": 255, "V min": 80, "V max": 255}   # default = yellow

def load_hsv():
    try:
        with open(HSV_FILE) as f:
            HSV.update({k: int(v) for k, v in json.load(f).items() if k in HSV})
    except Exception:
        pass

def save_hsv():
    try:
        with open(HSV_FILE, "w") as f:
            json.dump(HSV, f)
    except Exception:
        pass

def hsv_range():
    return (np.array([HSV["H min"], HSV["S min"], HSV["V min"]], np.uint8),
            np.array([HSV["H max"], HSV["S max"], HSV["V max"]], np.uint8))

def open_settings():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(CTRL_WIN, W + 50, 300)
    for k, mx in HSV_KEYS:
        cv2.createTrackbar(k, CTRL_WIN, HSV[k], mx, lambda x: None)
    info = np.full((46, 460, 3), 30, np.uint8)
    cv2.putText(info, "H: 0-179   S, V: 0-255   |  white in 'Mask' = detected", (8, 28), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(CTRL_WIN, info)

def read_settings():
    for k, _ in HSV_KEYS:
        HSV[k] = cv2.getTrackbarPos(k, CTRL_WIN)

def close_settings():
    save_hsv()
    try:
        cv2.destroyWindow(CTRL_WIN)
    except cv2.error:
        pass

# --- stages for "Fixed Stage" mode (drawn with shapes only, no images) ---
STAGES = [
    dict(name="ROCK FARMING", time=60, start=(62, 150), kind="rock", goal_r=26,
         goals=[(588, 90), (588, 255), (582, 408)],                                   # 3 rocks, top -> bottom on the right
         rects=[(0, 440, W, H), (20, 190, 112, 212), (290, 250, 334, 440), (196, 0, 240, 130),
                (520, 128, 636, 146), (520, 300, 636, 318)],
         lines=[], hazards=[], boxes=[(62, 150, "START", False)],
         movers=[dict(c=(430, 330), a=0.0, l=48, amp=40, f=0.30, ph=0.0, axis=1, th=10)]),
    dict(name="THERESA", time=30, start=(520, 45), kind="box", goal_r=34,
         goals=[(66, 402)],                                                           # blue box, bottom-left
         rects=[(30, 52, 330, 92), (590, 190, 632, 350), (430, 350, 522, 440), (0, 440, W, H)],
         lines=[((622, 115), (400, 165)), ((420, 255), (172, 305)), ((178, 338), (70, 372))],   # slides
         hazards=[(250, 418, 22), (380, 418, 22)],                                    # red zones: touch = back to start
         boxes=[(520, 45, "START", False), (66, 402, "GOAL", True)],
         movers=[dict(c=(340, 212), a=0.0, l=36, amp=55, f=0.35, ph=0.0, axis=0, th=10)]),
]

# ============================== 2. IMAGE PROCESSING ==============================
def find_glove(frame):
    """Camera frame -> (cleaned mask, final mask). Final mask is empty when the blob is too small to be the glove."""
    lo, hi = hsv_range()
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)             # 1. blur + convert to HSV
    raw = cv2.inRange(hsv, lo, hi)                                                        # 2. keep pixels inside the HSV range
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, K5, iterations=2)                       # 3. remove noise
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, K5, iterations=2)                    # 4. fill holes
    return clean, (clean if cv2.countNonZero(clean) > 800 else np.zeros_like(clean))      # 5. too small = not the glove

def mask_view(frame, clean, final):
    """Content of the 'Mask' window: what is detected + what the camera sees through the mask."""
    area = cv2.countNonZero(clean)
    m = cv2.cvtColor(cv2.resize(clean, (320, 240)), cv2.COLOR_GRAY2BGR)
    seen = cv2.resize(cv2.bitwise_and(frame, frame, mask=clean), (320, 240))
    ok = final.any()
    cv2.putText(m, f"mask  area={area}px", (6, 20), FONT, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(m, "DETECTED" if ok else "TOO SMALL (<800px)", (6, 40), FONT, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(seen, "camera x mask", (6, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([m, seen])

# ============================== 3. SPRITES (png if available, placeholder otherwise) ==============================
_raw, _cache = {}, {}

def find_asset(name):
    for d in (HERE, os.path.join(HERE, "assets"), os.path.join(HERE, "images"), os.path.join(HERE, "pokemons")):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    return None

def sprite(name, size, fallback):
    """BGRA sprite fitted into a size x size box. Missing file -> procedural placeholder."""
    key = (name, size)
    if key not in _cache:
        if name not in _raw:
            f = find_asset(name)
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED) if f else None
            if img is None:
                if not name.startswith("_"):
                    print(f"[!] {name} not found (put it next to bounce.py or in ./assets / ./pokemons) -> using placeholder")
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            _raw[name] = img
        img = _raw[name]
        if img is None:
            _cache[key] = fallback(name, size)
        else:
            s = size / max(img.shape[:2])
            _cache[key] = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return _cache[key]

def ph_pokeball(name, size):
    s = np.zeros((size, size, 4), np.uint8)
    c, r = size // 2, size // 2 - 1
    cv2.circle(s, (c, c), r, (255, 255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(s, (c, c), (r, r), 0, 180, 360, (40, 40, 230, 255), -1, cv2.LINE_AA)
    cv2.line(s, (c - r, c), (c + r, c), (20, 20, 20, 255), max(2, size // 12), cv2.LINE_AA)
    cv2.circle(s, (c, c), max(3, size // 6), (255, 255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(s, (c, c), max(3, size // 6), (20, 20, 20, 255), max(1, size // 16), cv2.LINE_AA)
    cv2.circle(s, (c, c), r, (20, 20, 20, 255), max(1, size // 14), cv2.LINE_AA)
    return s

def ph_monster(name, size):
    rnd = random.Random(name)
    col = (rnd.randint(60, 255), rnd.randint(60, 255), rnd.randint(60, 255), 255)
    s = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    cv2.circle(s, (c, c), c - 2, col, -1, cv2.LINE_AA)
    cv2.circle(s, (c, c), c - 2, (20, 20, 20, 255), 2, cv2.LINE_AA)
    for dx in (-1, 1):
        cv2.circle(s, (c + dx * size // 6, c - size // 8), max(3, size // 10), (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(s, (c + dx * size // 6, c - size // 8), max(2, size // 20), (0, 0, 0, 255), -1, cv2.LINE_AA)
    t = name.replace(".png", "").upper()
    sc = size / 200
    w = cv2.getTextSize(t, FONT, sc, 1)[0][0]
    cv2.putText(s, t, (c - w // 2, c + size // 3), FONT, sc, (0, 0, 0, 255), 1, cv2.LINE_AA)
    return s

def ph_groudon(name, size):
    s = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    for i in range(8):                                    # spikes
        a = i * np.pi / 4
        p1 = (int(c + c * 0.98 * np.cos(a - 0.2)), int(c + c * 0.98 * np.sin(a - 0.2)))
        p2 = (int(c + c * 0.98 * np.cos(a + 0.2)), int(c + c * 0.98 * np.sin(a + 0.2)))
        cv2.fillPoly(s, [np.array([p1, p2, (int(c + c * 0.55 * np.cos(a)), int(c + c * 0.55 * np.sin(a)))], np.int32)], (30, 30, 120, 255), cv2.LINE_AA)
    cv2.circle(s, (c, c), int(c * 0.75), (40, 40, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(s, (c, c), int(c * 0.75), (15, 15, 80, 255), 3, cv2.LINE_AA)
    for dx in (-1, 1):
        cv2.ellipse(s, (c + dx * size // 6, c - size // 10), (size // 10, size // 16), 0, 0, 360, (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(s, (c + dx * size // 6, c - size // 10), size // 28, (0, 0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(s, "GROUDON", (c - size // 3, c + size // 4), FONT, size / 230, (255, 255, 255, 255), 2, cv2.LINE_AA)
    return s

def ph_rock(name, size):
    rnd = random.Random(name)
    s = np.zeros((size, size, 4), np.uint8)
    c, n = size / 2, 9
    pts = np.array([(c + c * rnd.uniform(.78, .98) * np.cos(2 * np.pi * i / n + rnd.uniform(-.2, .2)),
                     c + c * rnd.uniform(.78, .98) * np.sin(2 * np.pi * i / n + rnd.uniform(-.2, .2))) for i in range(n)], np.int32)
    cv2.fillPoly(s, [pts], (70, 95, 125, 255), cv2.LINE_AA)
    inner = ((pts - c) * 0.55 + c - [c * 0.12, c * 0.15]).astype(np.int32)
    cv2.fillPoly(s, [inner], (105, 135, 170, 255), cv2.LINE_AA)
    cv2.polylines(s, [pts], True, (25, 35, 45, 255), 2, cv2.LINE_AA)
    cv2.line(s, (int(c * 0.9), int(c * 0.8)), (int(c * 1.2), int(c * 1.3)), (30, 40, 55, 255), 2, cv2.LINE_AA)
    return s

def overlay(img, spr, cx, cy, alpha=1.0):
    """Alpha-blend a BGRA sprite centered at (cx, cy)."""
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    a0, b0, a1, b1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if a0 >= a1 or b0 >= b1:
        return
    s = spr[b0 - y0:b1 - y0, a0 - x0:a1 - x0]
    a = s[:, :, 3:4].astype(np.float32) / 255 * alpha
    roi = img[b0:b1, a0:a1]
    roi[:] = (roi * (1 - a) + s[:, :, :3] * a).astype(np.uint8)

def rot(s, deg):
    h, w = s.shape[:2]
    return cv2.warpAffine(s, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

# ============================== 4. LEVELS & PHYSICS ==============================
def draw_bar(m, cx, cy, a, l, th=10):
    dx, dy = int(l * np.cos(a)), int(l * np.sin(a))
    cv2.line(m, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 255, th)

def mover_center(m, off):
    return (m["c"][0] + off, m["c"][1]) if m["axis"] == 0 else (m["c"][0], m["c"][1] + off)

def movers_masks(movers, t):
    """Moving bars -> list of (mask, velocity) for this instant."""
    out = []
    for m in movers:
        ph = 2 * np.pi * m["f"] * t + m["ph"]
        off, vel = m["amp"] * np.sin(ph), m["amp"] * 2 * np.pi * m["f"] * np.cos(ph)
        cx, cy = mover_center(m, off)
        mk = np.zeros((H, W), np.uint8)
        draw_bar(mk, cx, cy, m["a"], m["l"], m["th"])
        out.append((mk > 0, np.array([vel, 0.0]) if m["axis"] == 0 else np.array([0.0, vel])))
    return out

def make_level(wave, keep):
    """Random bars + bumpers + moving bars (all get harder with the wave). `keep` = rects that must stay free."""
    n_bars, n_bump, n_mov = 2 + wave // 2, 1 + wave // 3, max(0, (wave - 3) // 2)
    for _ in range(300):
        bars, bumps, env = (np.zeros((H, W), np.uint8) for _ in range(3))
        movers = []
        for _ in range(n_bars):
            cx = random.randint(80, W - 80)
            cy = random.randint(90, 220) if wave == 10 else random.randint(150, H - 110)
            draw_bar(bars, cx, cy, np.radians(random.uniform(-35, 35)), random.randint(45, 80))
        for _ in range(n_bump):
            by_pos = random.randint(90, 220) if wave == 10 else random.randint(150, H - 110)
            cv2.circle(bumps, (random.randint(80, W - 80), by_pos), 22, 255, -1)
        for _ in range(n_mov):
            my_pos = random.randint(100, 200) if wave == 10 else random.randint(170, H - 130)
            m = dict(c=(random.randint(120, W - 120), my_pos), a=np.radians(random.choice([0, 0, random.uniform(-30, 30)])),
                     l=random.randint(40, 65), amp=random.randint(35, 70), f=random.uniform(0.25, 0.45 + 0.03 * wave),
                     ph=random.uniform(0, 6.28), axis=random.randint(0, 1), th=10)
            movers.append(m)
            for off in np.linspace(-m["amp"], m["amp"], 9):                              # area swept by the moving bar
                draw_bar(env, *mover_center(m, off), m["a"], m["l"], m["th"])
        allm = bars | bumps | env
        if all(not allm[max(y0, 0):y1, max(x0, 0):x1].any() for x0, y0, x1, y1 in keep):
            return bars > 0, bumps > 0, movers
    for x0, y0, x1, y1 in keep:                                                          # fallback: just carve the zone out
        bars[max(y0, 0):y1, max(x0, 0):x1] = 0
        bumps[max(y0, 0):y1, max(x0, 0):x1] = 0
    return bars > 0, bumps > 0, []

def collide(p, v, solid, vs, e=BOUNCE):
    """Push the ball out of `solid` (bool mask) and bounce it. vs = velocity of that object, e = bounciness."""
    x, y = int(p[0]), int(p[1])
    if x < R or y < R or x >= W - R or y >= H - R:
        return
    hit = solid[y - R:y + R + 1, x - R:x + R + 1] & DISK                                  # solid pixels touching the ball
    if hit.sum() < 4:
        return
    ys, xs = np.nonzero(hit)
    n = -np.array([xs.mean() - R, ys.mean() - R])                                         # normal: from object toward ball
    d = np.hypot(*n)
    n = n / d if d > 1e-3 else np.array([0.0, -1.0])
    rel = v - vs                                                                          # velocity relative to the object
    vn = rel @ n
    if vn < 0:                                                                            # moving into it -> bounce
        v[:] = vs + rel - (1 + e) * vn * n
    p += n * 2

def new_ball():
    return np.array([random.uniform(100, W - 100), 30.0]), np.zeros(2)

# --- stage art (drawn once with shapes) ---
def stripes(img, x0, y0, x1, y1):
    roi = img[y0:y1, x0:x1]
    h = y1 - y0
    for i in range(-h, x1 - x0 + h, 16):
        cv2.fillPoly(roi, [np.array([(i, h), (i + 8, h), (i + 8 + h, 0), (i + h, 0)], np.int32)], (40, 200, 230), cv2.LINE_AA)

def blue_box(bg, cx, cy, text, warn):
    x0, y0, x1, y1 = cx - 42, cy - 30, cx + 42, cy + 30
    ov = bg.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (255, 160, 60), -1)
    bg[:] = cv2.addWeighted(ov, 0.3, bg, 0.7, 0)
    cv2.rectangle(bg, (x0, y0), (x1, y1), (255, 170, 70), 2)
    if warn:
        cv2.polylines(bg, [np.array([(cx, cy - 16), (cx - 16, cy + 12), (cx + 16, cy + 12)], np.int32)], True, (255, 200, 120), 2, cv2.LINE_AA)
        cv2.putText(bg, "!", (cx - 4, cy + 8), FONT, 0.5, (255, 200, 120), 2, cv2.LINE_AA)
    cv2.putText(bg, text, (x0 + 2, y0 - 6), FONT, 0.4, (255, 200, 120), 1, cv2.LINE_AA)

def build_stage(st):
    """Stage dict -> (solid mask, background picture)."""
    solid = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in st["rects"]:
        cv2.rectangle(solid, (x0, y0), (x1, y1), 255, -1)
    for a, b in st["lines"]:
        cv2.line(solid, a, b, 255, 12)
    ys, xs = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    chk = ((xs // 40 + ys // 40) % 2) == 0                                                # floor tiles
    bg[chk], bg[~chk] = (60, 64, 68), (50, 54, 58)
    bg[(xs % 40 == 0) | (ys % 40 == 0)] = (74, 78, 82)
    for x0, y0, x1, y1 in st["rects"]:                                                    # raised blocks
        cv2.rectangle(bg, (x0, y0), (x1, y1), (80, 84, 90), -1)
        cv2.rectangle(bg, (x0, y0), (x1, min(y0 + 9, y1)), (125, 130, 136), -1)
        cv2.rectangle(bg, (x0, y0), (x1, y1), (28, 28, 32), 2)
        if y1 >= H:
            stripes(bg, x0, y0 + 10, x1, y0 + 18)
    for a, b in st["lines"]:                                                              # slides
        cv2.line(bg, a, b, (28, 28, 32), 17, cv2.LINE_AA)
        cv2.line(bg, a, b, (96, 101, 107), 12, cv2.LINE_AA)
        cv2.line(bg, (a[0], a[1] - 3), (b[0], b[1] - 3), (145, 150, 156), 2, cv2.LINE_AA)
    for cx, cy, t, warn in st["boxes"]:
        blue_box(bg, cx, cy, t, warn)
    d = np.hypot((xs - W / 2) / (W / 2), (ys - H / 2) / (H / 2))                          # vignette
    bg = (bg * np.clip(1 - 0.3 * d ** 2, 0.5, 1)[..., None]).astype(np.uint8)
    return solid > 0, bg

_stage_cache = {}

# ============================== 5. GAME ==============================
def label(img, s, y, size, color, thick=2):
    w = cv2.getTextSize(s, FONT, size, thick)[0][0]
    if w > W - 30:                                                                        # shrink long text to fit
        size *= (W - 30) / w
        w = cv2.getTextSize(s, FONT, size, thick)[0][0]
    cv2.putText(img, s, ((W - w) // 2, y), FONT, size, color, thick, cv2.LINE_AA)

def draw_button(img, txt, c, r, prog):
    cv2.circle(img, c, r, (255, 230, 0), 3, cv2.LINE_AA)
    if prog > 0:
        cv2.ellipse(img, c, (r + 8, r + 8), -90, 0, int(360 * min(prog, 1.0)), (90, 255, 120), 5, cv2.LINE_AA)
    lines = txt.split("\n")
    sz = min(0.8, 1.5 * r / max(cv2.getTextSize(l, FONT, 1, 2)[0][0] for l in lines))
    lh = int(44 * sz)
    y0 = c[1] - (len(lines) - 1) * lh // 2 + int(8 * sz)
    for i, l in enumerate(lines):
        w = cv2.getTextSize(l, FONT, sz, 2)[0][0]
        cv2.putText(img, l, (c[0] - w // 2, y0 + i * lh), FONT, sz, (255, 255, 255), 2, cv2.LINE_AA)

_bm = {}
def overlap(mask, c, r):
    """How many mask pixels lie inside the circle (c, r)."""
    if (c, r) not in _bm:
        x0, y0, x1, y1 = max(c[0] - r, 0), max(c[1] - r, 0), min(c[0] + r + 1, W), min(c[1] + r + 1, H)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        _bm[(c, r)] = (x0, y0, x1, y1, (gx - c[0]) ** 2 + (gy - c[1]) ** 2 <= r * r)
    x0, y0, x1, y1, d = _bm[(c, r)]
    return int((mask[y0:y1, x0:x1] & d).sum())


class Game:
    def __init__(self):
        self.state, self.mode, self.stage_i = "menu", None, 0     # menu / modes / stages / play / over / win
        self.hold, self.msg = {}, ""
        self.t, self.ang, self.still, self.spawn_hold = 0.0, 0.0, 0.0, 0.0
        self.trail = deque(maxlen=14)
        self.p, self.v = new_ball()
        self.bars_m = self.bump_m = np.zeros((H, W), bool)
        self.movers, self.mov_cache, self.fx = [], [], []
        self.boulders, self.spawn_q = [], []
        self.wave, self.lives, self.current_pokemon_file = 1, 3, "Bagon.png"
        self.tr, self.target = 34, (320, 240)
        self.boss_hp, self.boss_base, self.boss_flash, self.atk_timer, self.shake = 3, (320, 200), 0.0, 0.0, 0.0
        self.banner, self.banner_t = "", 0.0
        self.stage, self.solid, self.bg, self.got, self.time_left = None, None, None, [], 0.0

    # ---------------- menus ----------------
    def buttons(self):
        y = 300
        if self.state == "menu":
            return [("START", (215, y), 55, "modes"), ("SETTINGS", (425, y), 55, "settings")]
        if self.state == "modes":
            return [("RANDOM\nMODE", (130, y), 62, "rand"), ("FIXED\nSTAGE", (320, y), 62, "stages"), ("BACK", (510, y), 62, "menu")]
        if self.state == "stages":
            return [("ROCK\nFARMING", (130, y), 62, "s0"), ("THERESA", (320, y), 62, "s1"), ("BACK", (510, y), 62, "modes")]
        return [("RETRY", (215, y), 55, "retry"), ("MENU", (425, y), 55, "menu")]      # over / win

    def act(self, a):
        self.hold.clear()
        if a in ("menu", "modes", "stages"):
            self.state = a
        elif a == "rand":
            self.start_rand()
        elif a[0] == "s" and a[1:].isdigit():
            self.start_stage(int(a[1:]))
        elif a == "retry":
            self.restart()

    def restart(self):
        if self.mode == "rand":
            self.start_rand()
        elif self.mode == "stage":
            self.start_stage(self.stage_i)

    def finish(self, state, msg):
        self.state, self.msg = state, msg
        self.hold.clear()

    # ---------------- starting games ----------------
    def start_rand(self):
        self.mode, self.state = "rand", "play"
        self.wave, self.lives = 1, 3
        self.boss_hp, self.boss_flash, self.atk_timer, self.shake = 3, 0.0, 0.0, 0.0
        self.boulders, self.spawn_q, self.fx = [], [], []
        self.ang = 0.0
        self.banner, self.banner_t = "WAVE 1", 1.5
        self.new_wave()

    def new_wave(self):
        self.t = 0.0
        pokemon_list = ["Bagon.png", "Beldum.png", "Lotad.png", "Mudkip.png", "Quagsire.png", "Spheal.png", "Torchic.png", "Trapinch.png", "Treecko.png"]
        if self.wave < 10:
            self.current_pokemon_file = random.choice(pokemon_list)     # randomize file from Pokemons folder every wave
            self.tr = max(20, 34 - 2 * (self.wave - 1))               # target shrinks every wave
            self.target = (random.randint(80, W - 80), random.randint(160, H - 100))
            m, (tx, ty) = self.tr + R + 8, self.target
            keep = [(tx - m, ty - m, tx + m, ty + m)]
        else:                                                         # boss placed lower, roams left-right around boss_base
            self.boss_base = (random.randint(210, 430), random.randint(380, 430))
            m, (bx, by) = BOSS_R + R + 8, self.boss_base
            keep = [(bx - 90 - m, by - m, bx + 90 + m, by + m)]
        self.bars_m, self.bump_m, self.movers = make_level(self.wave, keep)
        self.respawn()

    def start_stage(self, i):
        st = STAGES[i]
        if i not in _stage_cache:
            _stage_cache[i] = build_stage(st)
        self.mode, self.state, self.stage_i, self.stage = "stage", "play", i, st
        self.solid, self.bg = _stage_cache[i]
        self.bars_m, self.bump_m = self.solid, np.zeros((H, W), bool)
        self.movers, self.fx, self.t = st["movers"], [], 0.0
        self.got, self.time_left = [False] * len(st["goals"]), float(st["time"])
        self.shake, self.ang = 0.0, 0.0
        self.banner, self.banner_t = st["name"], 1.5
        self.respawn()

    def respawn(self):
        if self.mode == "stage":
            self.p, self.v = np.array(self.stage["start"], float), np.zeros(2)
        else:
            self.p, self.v = new_ball()
        self.trail.clear()
        self.still, self.spawn_hold = 0.0, 0.8                        # ball waits a moment so you can get ready

    def lose_life(self):
        self.lives -= 1
        if self.lives <= 0:
            self.finish("over", f"GAME OVER  -  WAVE {self.wave}")
        else:
            self.respawn()

    # ---------------- per-frame update ----------------
    def boss_pos(self):
        return self.boss_base[0] + 90 * np.sin(0.7 * self.t), self.boss_base[1]

    def update_boss(self, dt):
        self.atk_timer += dt
        if self.atk_timer >= 10:                                      # attack every 10 s: shake + falling boulders
            self.atk_timer, self.shake = 0.0, 2.0
            self.spawn_q = [[random.uniform(0.2, 1.8), random.randint(30, W - 30), random.choice([18, 22, 26, 30])] for _ in range(8)]
        self.shake = max(0.0, self.shake - dt)
        for q in self.spawn_q:
            q[0] -= dt
        for _, x, r in [q for q in self.spawn_q if q[0] <= 0]:
            self.boulders.append({"x": x, "y": -r, "vy": 150.0, "r": r})
        self.spawn_q = [q for q in self.spawn_q if q[0] > 0]
        for b in self.boulders:
            b["vy"] += 650 * dt
            b["y"] += b["vy"] * dt
        self.boulders = [b for b in self.boulders if b["y"] < H + 60]

    def update(self, dt, solid_y, vp):
        self.t += dt
        self.fx = [[x, y, a + dt] for x, y, a in self.fx if a < 0.6]
        self.banner_t -= dt
        self.boss_flash -= dt
        boss = self.mode == "rand" and self.wave == 10
        if self.mode == "stage":
            self.time_left -= dt
            if self.time_left <= 0:
                return self.finish("over", "TIME UP")
        if boss:
            self.update_boss(dt)
        self.mov_cache = movers_masks(self.movers, self.t)
        if self.spawn_hold > 0:
            self.spawn_hold -= dt
            return
        p, v = self.p, self.v
        for _ in range(4):                                            # 4 sub-steps per frame so the ball can't tunnel through thin bars
            h = dt / 4
            v[1] += GRAVITY * h
            if self.shake > 0:
                v[0] += random.uniform(-250, 250) * h                 # the earthquake jostles the ball
            sp = np.hypot(*v)
            if sp > MAXV:
                v *= MAXV / sp
            p += v * h
            if p[0] < R or p[0] > W - R:
                v[0] *= -BOUNCE
                p[0] = np.clip(p[0], R, W - R)
            if p[1] < R:
                v[1] *= -BOUNCE
                p[1] = R
            collide(p, v, self.bars_m, ZERO)
            collide(p, v, self.bump_m, ZERO, 1.25)                    # bumpers add energy
            for mk, mv in self.mov_cache:
                collide(p, v, mk, mv)
            collide(p, v, solid_y, vp)                                # the glove paddle
        self.trail.append((int(p[0]), int(p[1])))
        self.ang += v[0] * dt / R
        self.still = self.still + dt if np.hypot(*v) < 25 else 0.0
        self.rules_rand(boss) if self.mode == "rand" else self.rules_stage()

    def rules_rand(self, boss):
        p = self.p
        if boss:
            bx, by = self.boss_pos()
            if self.boss_flash <= 0 and np.hypot(p[0] - bx, p[1] - by) < BOSS_R + R:
                self.boss_hp -= 1
                self.boss_flash = 1.2
                self.fx.append([bx, by, 0.0])
                if self.boss_hp <= 0:
                    return self.finish("win", "GROUDON CAUGHT!")
                return self.new_wave()                                # new layout, Groudon moves, ball respawns
            if any(np.hypot(p[0] - b["x"], p[1] - b["y"]) < b["r"] + R * 0.8 for b in self.boulders):
                return self.lose_life()                               # hit by a boulder
        elif np.hypot(*(p - self.target)) < self.tr + R:              # pokemon reached -> next wave
            self.fx.append([*self.target, 0.0])
            self.wave += 1
            self.banner, self.banner_t = (f"WAVE {self.wave}", 1.5) if self.wave < 10 else ("FINAL BOSS: GROUDON", 2.5)
            if self.wave == 10:
                self.lives, self.boss_hp, self.atk_timer = 3, 3, 0.0  # lives refilled for the boss
            return self.new_wave()
        if p[1] > H + R:
            return self.lose_life()
        if self.still > 3:                                            # stuck -> free respawn
            self.respawn()

    def rules_stage(self):
        p, st = self.p, self.stage
        for i, (gx, gy) in enumerate(st["goals"]):
            if not self.got[i] and np.hypot(p[0] - gx, p[1] - gy) < st["goal_r"] + R * 0.6:
                self.got[i] = True
                self.fx.append([gx, gy, 0.0])
        if all(self.got):
            return self.finish("win", f"{st['name']} CLEAR  -  {st['time'] - self.time_left:.1f}s")
        for hx, hy, hr in st["hazards"]:
            if np.hypot(p[0] - hx, p[1] - hy) < hr + R * 0.5:         # red zone -> back to start
                self.fx.append([hx, hy, 0.0])
                return self.respawn()
        if p[1] > H + R:
            self.respawn()

    # ---------------- drawing ----------------
    def draw_ball(self, img):
        cx, cy = int(self.p[0]), int(self.p[1])
        for i, q in enumerate(self.trail):
            cv2.circle(img, q, 2 + i // 3, (255, 255, 255), -1, cv2.LINE_AA)
        glow = np.zeros_like(img)
        cv2.circle(glow, (cx, cy), R + 6, (255, 255, 255), -1, cv2.LINE_AA)
        img[:] = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 8))
        if self.mode == "rand":
            overlay(img, rot(sprite("pokeball.png", 2 * R + 6, ph_pokeball), -np.degrees(self.ang)), cx, cy)
        else:
            cv2.circle(img, (cx, cy), R, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_scene(self, frame, solid_y):
        playing = self.state == "play"
        stage = playing and self.mode == "stage"
        img = cv2.addWeighted(self.bg, 0.85, frame, 0.15, 0) if stage else (frame * 0.35).astype(np.uint8)
        neon = np.zeros_like(img)
        if playing:
            if not stage:
                neon[self.bars_m] = (255, 230, 0)
                neon[self.bump_m] = (255, 80, 220)
            for mk, _ in self.mov_cache:
                neon[mk] = (120, 255, 0)
        neon[solid_y] = (0, 220, 255)
        img = cv2.add(img, cv2.add(cv2.GaussianBlur(neon, (0, 0), 7), neon // 2))        # glow = blur + add
        if not playing:
            return img
        pulse = int(3 * np.sin(time.time() * 5))
        if stage:
            st = self.stage
            for hx, hy, hr in st["hazards"]:
                cv2.circle(img, (hx, hy), hr + pulse, (60, 60, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (hx, hy), 5, (60, 60, 255), -1, cv2.LINE_AA)
            for (gx, gy), got in zip(st["goals"], self.got):
                if st["kind"] == "rock":
                    overlay(img, sprite("orirock.png", 2 * st["goal_r"] + 6, ph_rock), gx, gy, 0.35 if got else 1.0)
                col = (90, 255, 120) if got else ((0, 165, 255) if st["kind"] == "rock" else (255, 170, 70))
                cv2.circle(img, (gx, gy), st["goal_r"] + 6 + pulse, col, 3, cv2.LINE_AA)
                if got:
                    cv2.polylines(img, [np.array([(gx - 10, gy), (gx - 3, gy + 8), (gx + 12, gy - 9)], np.int32)], False, (90, 255, 120), 4, cv2.LINE_AA)
        elif self.wave < 10:
            tx, ty = self.target
            cv2.circle(img, (tx, ty), int(self.tr * 1.25) + pulse, (90, 255, 120), 3, cv2.LINE_AA)
            overlay(img, sprite(self.current_pokemon_file, int(self.tr * 2.2), ph_monster), tx, ty)
        else:
            bx, by = self.boss_pos()
            if self.boss_flash <= 0 or int(self.boss_flash * 12) % 2 == 0:
                overlay(img, sprite("groudon.png", int(BOSS_R * 2.6), ph_groudon), int(bx), int(by))
            cv2.circle(img, (int(bx), int(by)), BOSS_R + pulse, (60, 60, 255), 2, cv2.LINE_AA)
            for i, b in enumerate(self.boulders):
                overlay(img, sprite(f"_rock{b['r'] % 4}", 2 * b["r"], ph_rock), int(b["x"]), int(b["y"]))
        for x, y, a in self.fx:
            cv2.circle(img, (int(x), int(y)), int(10 + a * 120), (255, 255, 255), 2, cv2.LINE_AA)
        self.draw_ball(img)
        if self.shake > 0:                                                                # screen shake
            a = 14 * min(1.0, self.shake)
            img = cv2.warpAffine(img, np.float32([[1, 0, random.uniform(-a, a)], [0, 1, random.uniform(-a, a)]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        return img

    def draw_hud(self, img):
        if self.mode == "rand":
            cv2.putText(img, f"WAVE {self.wave}/10   LIVES {self.lives}", (10, 28), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if self.wave == 10:
                label(img, "GROUDON", 24, 0.6, (60, 60, 255), 2)
                for i in range(3):
                    x = W // 2 - 95 + i * 65
                    cv2.rectangle(img, (x, 32), (x + 60, 44), (40, 40, 220) if i < self.boss_hp else (60, 60, 60), -1)
                    cv2.rectangle(img, (x, 32), (x + 60, 44), (255, 255, 255), 1)
                if self.shake > 0:
                    label(img, "GROUDON USES EARTHQUAKE!", 90, 0.8, (60, 60, 255), 2)
                elif self.atk_timer > 8 and int(time.time() * 6) % 2:
                    label(img, "! WARNING !", 90, 0.9, (60, 60, 255), 2)
                    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)
        else:
            st = self.stage
            cv2.putText(img, st["name"], (10, 28), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if st["kind"] == "rock":
                cv2.putText(img, f"ROCKS {sum(self.got)}/{len(self.got)}", (10, 54), FONT, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            tl = max(0.0, self.time_left)
            col = (60, 60, 255) if tl < 10 else (255, 255, 255)
            s = f"TIME {tl:04.1f}"
            cv2.putText(img, s, (W - 10 - cv2.getTextSize(s, FONT, 0.75, 2)[0][0], 28), FONT, 0.75, col, 2, cv2.LINE_AA)
            cv2.rectangle(img, (W - 170, 38), (W - 10, 46), (60, 60, 60), -1)
            cv2.rectangle(img, (W - 170, 38), (W - 170 + int(160 * tl / st["time"]), 46), col, -1)
        if self.banner_t > 0:
            label(img, self.banner, H // 2 - 60, 1.1, (255, 255, 255), 3)
        if self.spawn_hold > 0:
            label(img, "READY...", H - 30, 0.7, (255, 255, 255), 2)

    def draw_menu(self, img):
        titles = {"menu": "GUIDANCE", "modes": "SELECT MODE", "stages": "SELECT STAGE", "over": self.msg, "win": self.msg}
        col = (60, 60, 255) if self.state == "over" else (90, 255, 120) if self.state == "win" else (255, 230, 0)
        label(img, titles[self.state], 110, 1.1, col, 3)
        for txt, c, r, a in self.buttons():
            draw_button(img, txt, c, r, self.hold.get(a, 0.0))
        label(img, "hold the glove over a button for 1s  (or press SPACE / 1 / 2 / 3)   |   S = HSV settings", H - 30, 0.5, (200, 200, 200), 1)

# ============================== 6. MAIN LOOP ==============================
def main():
    load_hsv()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(GAME_WIN)
    cv2.namedWindow(MASK_WIN)                                 # game + mask windows stay open for the whole run
    cv2.moveWindow(GAME_WIN, 20, 20)
    cv2.moveWindow(MASK_WIN, W + 50, 20)
    game = Game()
    last, pc, vp = time.time(), None, np.zeros(2)
    settings, s_age = False, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not found")
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))
        dt, last = max(min(time.time() - last, 0.05), 1e-3), time.time()

        # --- HSV sliders -> detection settings ---
        if settings:
            s_age += 1
            closed = False
            try:
                if s_age > 5 and cv2.getWindowProperty(CTRL_WIN, cv2.WND_PROP_VISIBLE) < 1:
                    closed = True                             # user closed the Controls window with [X]
                else:
                    read_settings()
            except cv2.error:
                closed = True
            if closed:
                settings = False
                close_settings()

        # --- image processing: find the glove + its velocity ---
        clean, yellow = find_glove(frame)
        cv2.imshow(MASK_WIN, mask_view(frame, clean, yellow))
        solid_y = yellow > 0
        if solid_y.any():
            m = cv2.moments(yellow, True)
            c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            vp = np.clip(0.6 * vp + 0.4 * (c - pc) / dt, -900, 900) if pc is not None else np.zeros(2)
            pc = c                                            # velocity = how far the glove's center moved between frames
        else:
            pc, vp = None, np.zeros(2)

        # --- game ---
        action = None
        if game.state == "play":
            game.update(dt, solid_y, vp)
        else:                                                 # menus: press a button by holding the glove over it for 1 second
            best, bo = None, 0
            for _, c, r, a in game.buttons():
                ov = overlap(solid_y, c, r)
                if ov > 300 and ov > bo:
                    best, bo = a, ov
            for _, _, _, a in game.buttons():
                game.hold[a] = game.hold.get(a, 0.0) + dt if a == best else 0.0
            if best and game.hold[best] > 1.0:
                action = best

        # --- draw ---
        img = game.draw_scene(frame, solid_y)
        if game.state == "play":
            game.draw_hud(img)
        else:
            game.draw_menu(img)
        cv2.imshow(GAME_WIN, img)

        # --- keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('s'):
            action = "settings"
        elif key == ord('m'):
            game.state, game.mode = "menu", None
            game.hold.clear()
        elif key == ord('r'):
            game.restart()
        elif key == ord('b') and game.state == "play":        # manual ball respawn (if it gets stuck)
            game.respawn()
        elif game.state != "play" and key in (32, ord('1'), ord('2'), ord('3')):
            btns = game.buttons()
            idx = 0 if key == 32 else key - ord('1')
            if idx < len(btns):
                action = btns[idx][3]

        if action == "settings":
            if settings:
                settings = False
                close_settings()
            else:
                settings, s_age = True, 0
                open_settings()
            game.hold.clear()
        elif action:
            game.act(action)

    save_hsv()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()