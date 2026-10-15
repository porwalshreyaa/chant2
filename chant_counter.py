"""
Silent chant counter - counts each time your mouth opens after being closed.
Uses MediaPipe Face Landmarker (lip landmarks), so eyes / head movement are ignored.

Setup (once):
    pip install opencv-python mediapipe
Run:
    python chant_counter.py
For a longer mantra, set how many mouth openings make one round:
    python chant_counter.py --per 8
Keys (click the small window first): r = reset count, q = quit.
It keeps counting while you work in other windows. Just don't close it.
Tuning: if it misses chants, lower --min-range; if it counts extras, raise it.
"""
import argparse, math, os, time, urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
MODEL = "face_landmarker.task"

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--min-range", type=float, default=0.02,
                help="smallest lip-gap change that counts as a chant")
ap.add_argument("--min-gap", type=float, default=0.12,
                help="minimum seconds between two chants")
ap.add_argument("--per", type=int, default=1,
                help="mouth openings in ONE full mantra (e.g. 8 for an 8-syllable mantra)")
args = ap.parse_args()

if not os.path.exists(MODEL):
    print("Downloading face model (one time)...")
    urllib.request.urlretrieve(URL, MODEL)

lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO, num_faces=1))

cap = cv2.VideoCapture(args.camera)
opens, is_open, last_t = 0, False, 0.0
lo = hi = sm = None
t0 = time.time()

def dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, int((time.time() - t0) * 1000))
    level = 0.0
    if res.face_landmarks:
        p = res.face_landmarks[0]
        # inner-lip gap (13,14) divided by mouth width (78,308)
        mar = dist(p[13], p[14], w, h) / max(dist(p[78], p[308], w, h), 1e-6)
        sm = mar if sm is None else 0.5 * sm + 0.5 * mar
        if lo is None:
            lo = hi = sm
        # adaptive floor / ceiling that slowly forget
        lo = min(sm, lo + (hi - lo) * 0.003)
        hi = max(sm, hi - (hi - lo) * 0.003)
        rng = hi - lo
        level = (sm - lo) / rng if rng > 0 else 0.0
        now = time.time()
        if rng >= args.min_range:
            if not is_open and level > 0.6 and now - last_t > args.min_gap:
                is_open, last_t, opens = True, now, opens + 1
                if opens % args.per == 0:
                    print("count:", opens // args.per)
            elif is_open and level < 0.35:
                is_open = False
        for i in (13, 14, 78, 308):
            cv2.circle(frame, (int(p[i].x * w), int(p[i].y * h)), 2, (0, 165, 255), -1)

    small = cv2.resize(frame, (320, int(320 * h / w)))
    count = opens // args.per
    cv2.putText(small, str(count), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 140, 255), 4)
    cv2.rectangle(small, (10, small.shape[0] - 20), (10 + int(300 * min(level, 1)), small.shape[0] - 8),
                  (0, 140, 255), -1)
    cv2.imshow("Chant counter (r reset, q quit)", small)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        opens = 0

print("Final count:", opens // args.per)
cap.release()
cv2.destroyAllWindows()
