import json, threading, time
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

CALIB_FRAMES = 12      # shuru me seedha dekho (baseline)
YAW_THRESHOLD = 0.16
PITCH_DOWN = 0.14
PITCH_UP = 0.16
YOLO_CONF = 0.45
COOLDOWN = 8
FLAG_SCORE = 10

# code: (hold seconds, weight, message)
RULES = {
    "NO_FACE":         (3.0, 2, "Face camera me nahi dikh raha"),
    "MULTIPLE_FACES":  (1.5, 4, "Ek se zyada log frame me detect hue"),
    "LOOKING_AWAY":    (2.5, 1, "Baar-baar side me dekh raha hai"),
    "LOOKING_DOWN":    (3.0, 1, "Lambe time se neeche dekh raha hai"),
    "PHONE":           (1.0, 5, "Mobile phone detect hua"),
    "BOOK":            (2.0, 3, "Book / notes detect hue"),
    "TAB_SWITCH":      (0, 2, "Tab / window switch kiya"),
    "COPY_PASTE":      (0, 2, "Copy / paste try kiya"),
    "FULLSCREEN_EXIT": (0, 2, "Fullscreen se bahar nikla"),
    "NOISE":           (0, 1, "Background me lagataar awaaz"),
}
LOG_DIR = Path(__file__).parent / "logs"


class Analyzer:
    def __init__(self, use_yolo=True, static=False, yolo_weights="yolov8n.pt"):
        import mediapipe as mp
        self._lock = threading.Lock()
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static, max_num_faces=3,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.yolo = None
        if use_yolo:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights)   # pehli baar auto-download

    @staticmethod
    def _head_ratios(lm):
        nose, left, right, top, chin = lm[1], lm[234], lm[454], lm[10], lm[152]
        yaw = (nose.x - left.x) / ((right.x - left.x) or 1e-6)
        pitch = (nose.y - top.y) / ((chin.y - top.y) or 1e-6)
        return yaw, pitch

    def analyze(self, frame_bgr):
        info = {"faces": 0, "yaw": 0.5, "pitch": 0.58, "phone": 0, "book": 0, "persons": 0}
        with self._lock:
            res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            faces = res.multi_face_landmarks or []
            info["faces"] = len(faces)
            if faces:
                info["yaw"], info["pitch"] = self._head_ratios(faces[0].landmark)
            if self.yolo is not None:
                out = self.yolo.predict(frame_bgr, conf=YOLO_CONF, classes=[0, 67, 73],
                                        verbose=False, imgsz=480)[0]
                for c in out.boxes.cls.tolist():
                    c = int(c)
                    if c == 67: info["phone"] += 1
                    elif c == 73: info["book"] += 1
                    elif c == 0: info["persons"] += 1
        return info


class Session:
    def __init__(self, sid, name=""):
        self.sid, self.name = sid, name
        self.started = time.time()
        self.violations, self.score = [], 0
        self._since, self._last_fired, self._calib = {}, {}, []
        self.base_yaw, self.base_pitch = 0.5, 0.58

    @property
    def calibrating(self): return len(self._calib) < CALIB_FRAMES
    @property
    def flagged(self): return self.score >= FLAG_SCORE

    def evaluate(self, info, now=None):
        now = now or time.time()
        active = set()
        if info["faces"] == 0:
            active.add("NO_FACE")
        else:
            if info["faces"] > 1 or info["persons"] > 1:
                active.add("MULTIPLE_FACES")
            if self.calibrating and info["faces"] == 1:
                self._calib.append((info["yaw"], info["pitch"]))
                if not self.calibrating:
                    self.base_yaw = float(np.median([c[0] for c in self._calib]))
                    self.base_pitch = float(np.median([c[1] for c in self._calib]))
            elif not self.calibrating:
                dy, dp = info["yaw"] - self.base_yaw, info["pitch"] - self.base_pitch
                if abs(dy) > YAW_THRESHOLD: active.add("LOOKING_AWAY")
                elif dp > PITCH_DOWN: active.add("LOOKING_DOWN")
                elif dp < -PITCH_UP: active.add("LOOKING_AWAY")
        if info["phone"]: active.add("PHONE")
        if info["book"]: active.add("BOOK")
        return self._apply(active, now)

    def _apply(self, active, now):
        new = []
        for code in list(self._since):
            if code not in active:
                del self._since[code]
        for code in active:
            self._since.setdefault(code, now)
            if (now - self._since[code] >= RULES[code][0]
                    and now - self._last_fired.get(code, 0) >= COOLDOWN):
                new.append(self.add_violation(code, now=now))
        return sorted(active), new

    def add_violation(self, code, detail="", now=None):
        now = now or time.time()
        _, weight, msg = RULES[code]
        v = {"time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
             "elapsed_sec": int(now - self.started), "code": code,
             "message": msg + (f" ({detail})" if detail else ""), "weight": weight}
        self._last_fired[code] = now
        self.violations.append(v)
        self.score += weight
        self.save()
        return v

    def report(self):
        counts = {}
        for v in self.violations:
            counts[v["code"]] = counts.get(v["code"], 0) + 1
        return {"session_id": self.sid, "candidate": self.name,
                "duration_sec": int(time.time() - self.started),
                "score": self.score, "flag_threshold": FLAG_SCORE,
                "flagged_for_review": self.flagged, "summary": counts,
                "violations": self.violations,
                "note": "Automated signals only. Final decision human reviewer ka."}

    def save(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            (LOG_DIR / f"{self.sid}.json").write_text(json.dumps(self.report(), indent=2))
        except OSError:
            pass