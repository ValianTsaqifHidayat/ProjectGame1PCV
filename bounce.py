import cv2, numpy as np, random, time
from collections import deque
 
# ============================== 1. CONFIG ==============================
W, H = 640, 480
R = 12                                                    # ball radius
GRAVITY, BOUNCE, MAXV = 700, 0.65, 650                    # gravity, bounciness, speed limit (px/s)
Y_LO, Y_HI = np.array([10, 60, 80], np.uint8), np.array([35, 255, 255], np.uint8)   # yellow range (HSV)
K5 = np.ones((5, 5), np.uint8)
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
DISK = xx ** 2 + yy ** 2 <= R * R                         # ball footprint used for collision checks
BTN, BR = (W // 2, H // 2 + 40), 50                       # START button (center, radius)
BTN_MASK = np.zeros((H, W), np.uint8)
cv2.circle(BTN_MASK, BTN, BR, 1, -1)
BTN_MASK = BTN_MASK > 0
PIPE = "Pipeline: camera | raw yellow | cleaned yellow"
 
# ============================== 2. IMAGE PROCESSING ==============================
def find_yellow(frame):
    """Camera frame -> (raw yellow mask, cleaned yellow mask)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)             # 1. blur + convert to HSV
    raw = cv2.inRange(hsv, Y_LO, Y_HI)                                                    # 2. keep yellow pixels
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, K5, iterations=2)                       # 3. remove noise
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, K5, iterations=2)                    # 4. fill holes
    return raw, (clean if clean.sum() > 255 * 800 else np.zeros_like(clean))              # 5. too small = not the glove
 
# ============================== 3. GAME LOGIC ==============================
def make_level(score, target, tr):
    """Random bars + bumpers (as masks). Keeps the target area free."""
    while True:
        bars, bumps = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
        for _ in range(2 + min(score // 2, 3)):
            cx, cy = random.randint(80, W - 80), random.randint(150, H - 110)
            a, l = np.radians(random.uniform(-35, 35)), random.randint(45, 80)
            dx, dy = int(l * np.cos(a)), int(l * np.sin(a))
            cv2.line(bars, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 10)
        for _ in range(1 + min(score // 3, 2)):
            cv2.circle(bumps, (random.randint(80, W - 80), random.randint(150, H - 110)), 22, 255, -1)
        m = tr + R + 5
        if not (bars | bumps)[target[1] - m:target[1] + m, target[0] - m:target[0] + m].any():
            return bars > 0, bumps > 0
 
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
 
def new_target(tr):
    return (random.randint(80, W - 80), random.randint(160, H - 100)), tr
 
def label(img, s, y, size, color, thick=2):
    w = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0][0]
    cv2.putText(img, s, ((W - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
 
# ============================== 4. MAIN LOOP ==============================
cap = cv2.VideoCapture(0)
state = "menu"                                            # menu -> play -> over
p, v = new_ball()
target, tr = new_target(34)
bars_m = bump_m = np.zeros((H, W), bool)
score, lives, still, hold = 0, 3, 0.0, 0.0
trail, last, pc, vp, show_pipe = deque(maxlen=14), time.time(), None, np.zeros(2), False
 
while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera not found")
        break
    frame = cv2.resize(cv2.flip(frame, 1), (W, H))
    dt, last = max(min(time.time() - last, 0.05), 1e-3), time.time()
 
    # --- image processing: find the glove + its velocity ---
    raw, yellow = find_yellow(frame)
    solid_y = yellow > 0
    if solid_y.any():
        m = cv2.moments(yellow, True)
        c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        vp = np.clip(0.6 * vp + 0.4 * (c - pc) / dt, -900, 900) if pc is not None else np.zeros(2)
        pc = c                                            # velocity = how far the glove's center moved between frames
    else:
        pc, vp = None, np.zeros(2)
 
    go = False
    if state == "play":
        # --- physics (4 sub-steps per frame so the ball can't tunnel through thin bars) ---
        for _ in range(4):
            h = dt / 4
            v[1] += GRAVITY * h
            sp = np.hypot(*v)
            if sp > MAXV:
                v *= MAXV / sp
            p += v * h
            if p[0] < R or p[0] > W - R:
                v[0] *= -BOUNCE; p[0] = np.clip(p[0], R, W - R)
            if p[1] < R:
                v[1] *= -BOUNCE; p[1] = R
            collide(p, v, bars_m, np.zeros(2))
            collide(p, v, bump_m, np.zeros(2), 1.25)      # bumpers add energy
            collide(p, v, solid_y, vp)                    # the glove paddle
        trail.append((int(p[0]), int(p[1])))
        still = still + dt if np.hypot(*v) < 25 else 0.0
        if np.hypot(*(p - target)) < tr + R:              # target hit -> next level
            score += 1
            target, tr = new_target(max(18, tr - 2))
            bars_m, bump_m = make_level(score, target, tr)
            p, v = new_ball(); trail.clear()
        elif p[1] > H + R or still > 3:                   # fell off the screen (or stuck)
            lives -= 0 if still > 3 else 1
            p, v = new_ball(); trail.clear(); still = 0.0
            state = "play" if lives > 0 else "over"
    else:
        # --- menu / game over: the START button is pressed by holding the glove over it for 1 second ---
        hold = hold + dt if (solid_y & BTN_MASK).sum() > 300 else 0.0
        go = hold > 1.0
 
    # --- draw ---
    img = (frame * 0.35).astype(np.uint8)                 # dimmed camera as background
    neon = np.zeros_like(img)
    neon[bars_m] = (255, 230, 0)
    neon[bump_m] = (255, 80, 220)
    neon[solid_y] = (0, 220, 255)
    img = cv2.add(img, cv2.add(cv2.GaussianBlur(neon, (0, 0), 7), neon // 2))   # glow = blur + add
    if state == "play":
        for i, q in enumerate(trail):
            cv2.circle(img, q, 2 + i // 3, (255, 255, 255), -1, cv2.LINE_AA)
        r = tr + int(4 * np.sin(time.time() * 5))         # pulsing target
        cv2.circle(img, target, r, (90, 255, 120), 3, cv2.LINE_AA)
        cv2.circle(img, target, r // 3, (90, 255, 120), -1, cv2.LINE_AA)
        glow = np.zeros_like(img)
        cv2.circle(glow, (int(p[0]), int(p[1])), R + 6, (255, 255, 255), -1, cv2.LINE_AA)
        img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 8))
        cv2.circle(img, (int(p[0]), int(p[1])), R, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, f"SCORE {score}   LIVES {lives}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        label(img, "BOUNCE ON IT" if state == "menu" else f"GAME OVER  -  SCORE {score}", 110, 1.1, (255, 230, 0), 3)
        cv2.circle(img, BTN, BR, (255, 230, 0), 3, cv2.LINE_AA)
        cv2.ellipse(img, BTN, (BR + 8, BR + 8), -90, 0, int(360 * min(hold, 1.0)), (90, 255, 120), 5, cv2.LINE_AA)
        label(img, "START" if state == "menu" else "RETRY", BTN[1] + 8, 0.8, (255, 255, 255))
        label(img, "hold the yellow glove over the button (or press SPACE)", H - 30, 0.55, (200, 200, 200), 1)
    cv2.imshow("BOUNCE ON IT", img)
 
    if show_pipe:                                         # shows each processing step (nice for demos/reports)
        panels = [cv2.resize(frame, (320, 240))] + [cv2.resize(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), (320, 240)) for m in (raw, yellow)]
        cv2.imshow(PIPE, np.hstack(panels))
 
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('p'):
        show_pipe = not show_pipe
        if not show_pipe:
            cv2.destroyWindow(PIPE)
    go = go or key == ord('r') or (key == 32 and state != "play")
    if go:                                                # (re)start the game
        p, v = new_ball(); trail.clear()
        score, lives, still, hold = 0, 3, 0.0, 0.0
        target, tr = new_target(34)
        bars_m, bump_m = make_level(0, target, tr)
        state = "play"
 
cap.release()
cv2.destroyAllWindows()