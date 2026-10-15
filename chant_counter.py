"""
Silent chant counter - counts each time your upper and lower lips come apart.

Setup (once):   pip install opencv-python mediapipe
Run:            python chant_counter.py

It first LEARNS YOUR NUMBERS (about 11 seconds):
  step 1: keep your mouth CLOSED and nod / move your head a bit
  step 2: CHANT normally 5-8 times
Then it counts only when the lip gap is clearly bigger than anything it saw
with your mouth closed - so nodding/head movement with closed lips never counts.

Keys (click the small window first):  r = reset count,  c = learn numbers again,  q = quit
Longer mantra:  python chant_counter.py --per 8   (mouth openings in one full round)
"""
import argparse, math, os, time, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
MODEL = "face_landmarker.task"

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--per", type=int, default=1, help="mouth openings in ONE full mantra")
ap.add_argument("--min-gap", type=float, default=0.30,
                help="COMPULSORY minimum seconds between two counted chants (default 0.30)")
ap.add_argument("--closed-secs", type=float, default=5)
ap.add_argument("--chant-secs", type=float, default=7)
ap.add_argument("--sensitivity", type=float, default=0.45,
                help="0-1: where between 'closed' and 'chanting' the count triggers (lower = more sensitive)")
args = ap.parse_args()

if not os.path.exists(MODEL):
    print("Downloading face model (one time)...")
    urllib.request.urlretrieve(URL, MODEL)

lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO, num_faces=1))

def dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def valley_width(prof):
    """Width (in pixel rows, sub-pixel) of the dark band between the lips along a vertical brightness profile.
    Closed lips = thin dark line (~1-2 px). Lips apart = a thicker dark band."""
    p = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
    if len(p) < 16:
        return 0.0
    i = int(np.argmin(p[3:-3])) + 3
    top = np.percentile(p[:max(i - 4, 2)], 70) if i > 6 else p.max()
    bot = np.percentile(p[i + 4:], 70) if len(p) - i > 8 else p.max()
    ref = 0.5 * (top + bot)
    if ref - p[i] < 2:
        return 0.0
    th = p[i] + 0.5 * (ref - p[i])
    a = i
    while a > 0 and p[a] < th:
        a -= 1
    b = i
    while b < len(p) - 1 and p[b] < th:
        b += 1
    return float(b - a)

# (upper inner lip, lower inner lip) pairs at three places along the mouth, averaged for stability
PAIRS = [(13, 14), (82, 87), (312, 317)]

def lip_gap(p, w, h, gray=None):
    """Lip gap from the face-tracking lip points, relative to mouth width."""
    g = sum(dist(p[a], p[b], w, h) for a, b in PAIRS) / len(PAIRS)
    return g / max(dist(p[78], p[308], w, h), 1e-6)

def start_learning():
    return {"phase": "wait", "t": 0.0, "closed": [], "chant": []}

cap = cv2.VideoCapture(args.camera)
t0 = time.time()
opens, is_open, last_t, sm = 0, False, 0.0, None
up = dn = None
st = start_learning()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    now = time.time()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, int((now - t0) * 1000))
    msg, gap = "no face", None
    if res.face_landmarks:
        p = res.face_landmarks[0]
        gap = lip_gap(p, w, h, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        sm = gap if sm is None else 0.5 * sm + 0.5 * gap
        for i in (13, 14, 82, 87, 312, 317):
            cv2.circle(frame, (int(p[i].x * w), int(p[i].y * h)), 2, (0, 165, 255), -1)

        if st["phase"] == "wait":
            st.update(phase="closed", t=now)
        if st["phase"] == "closed":
            st["closed"].append(sm)
            msg = "1/2 CLOSE mouth, nod a little (%ds)" % max(0, args.closed_secs - (now - st["t"]))
            if now - st["t"] >= args.closed_secs:
                st.update(phase="chant", t=now)
        elif st["phase"] == "chant":
            st["chant"].append(sm)
            msg = "2/2 CHANT 5-8 times (%ds)" % max(0, args.chant_secs - (now - st["t"]))
            if now - st["t"] >= args.chant_secs:
                c_hi = float(np.percentile(st["closed"], 99))     # highest gap seen with mouth closed
                o_lvl = float(np.percentile(st["chant"], 92))     # typical 'open' peak while chanting
                if o_lvl < c_hi * 1.15 + 0.004:
                    print("WARNING: chanting is hard to tell apart from closed-mouth movement "
                          "(closed max %.3f, open %.3f). Improve light / sit closer, then press c." % (c_hi, o_lvl))
                up = c_hi + args.sensitivity * max(o_lvl - c_hi, 0.004)
                dn = c_hi + 0.2 * max(o_lvl - c_hi, 0.004)
                print("Learned: closed max %.3f, open %.3f -> count above %.3f" % (c_hi, o_lvl, up))
                st["phase"] = "run"; is_open = False
        elif st["phase"] == "run":
            msg = ""
            if not is_open and sm > up and now - last_t > args.min_gap:
                is_open, last_t, opens = True, now, opens + 1
                if opens % args.per == 0:
                    print("count:", opens // args.per)
            elif is_open and sm < dn:
                is_open = False
    else:
        sm = None

    small = cv2.resize(frame, (320, int(320 * h / w)))
    sh = small.shape[0]
    cv2.putText(small, str(opens // args.per), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 140, 255), 4)
    if gap is not None:
        cv2.putText(small, "gap %.3f" % sm, (10, sh - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    if up is not None and st["phase"] == "run" and sm is not None:
        scale = up * 1.6
        cv2.rectangle(small, (10, sh - 20), (10 + int(300 * min(sm / scale, 1)), sh - 8), (0, 140, 255), -1)
        cv2.line(small, (10 + int(300 * up / scale),) * 1 + (sh - 24,), (10 + int(300 * up / scale), sh - 4), (255, 255, 255), 2)
    if msg:
        cv2.putText(small, msg, (10, sh - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Chant counter (r reset, c relearn, q quit)", small)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        opens = 0
    if k == ord("c"):
        st, up, dn = start_learning(), None, None

print("Final count:", opens // args.per)
cap.release()
cv2.destroyAllWindows()
