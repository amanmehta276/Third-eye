import json, time
import cv2
from proctor_core import Analyzer, Session

analyzer = Analyzer(use_yolo=True)
session = Session("local-test", "Aman")
cap = cv2.VideoCapture(0)
banner, banner_until = "", 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    info = analyzer.analyze(frame)
    active, new = session.evaluate(info)
    if new:
        banner, banner_until = new[-1]["message"], time.time() + 4

    h, w = frame.shape[:2]
    status = "Calibrating... seedha dekho" if session.calibrating else "Monitoring"
    cv2.putText(frame, f"{status} | Score: {session.score}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if active:
        cv2.putText(frame, ", ".join(active), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if time.time() < banner_until:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "WARNING: " + banner, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if session.flagged:
        cv2.putText(frame, "FLAGGED FOR REVIEW", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("AI Proctor (q = quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(json.dumps(session.report(), indent=2))