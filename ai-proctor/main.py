import asyncio, re
from pathlib import Path

import cv2, numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from proctor_core import Analyzer, Session, RULES

app = FastAPI(title="AI Proctor")
analyzer = Analyzer(use_yolo=True, static=True)   # static=True: multi-session safe
sessions: dict[str, Session] = {}
SID_RE = re.compile(r"^[\w-]{1,64}$")


def get_session(sid: str, name: str = "") -> Session:
    if not SID_RE.match(sid):
        raise HTTPException(400, "bad session id")
    if sid not in sessions:
        sessions[sid] = Session(sid, name)
    return sessions[sid]


def payload(s: Session, active=(), new=()):
    return {"calibrating": s.calibrating, "active": list(active),
            "new": list(new), "score": s.score, "flagged": s.flagged}


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    if not SID_RE.match(sid):
        await ws.close(code=1008)
        return
    await ws.accept()
    s = get_session(sid, ws.query_params.get("name", ""))
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            info = await loop.run_in_executor(None, analyzer.analyze, frame)
            active, new = s.evaluate(info)
            await ws.send_json(payload(s, active, new))
    except WebSocketDisconnect:
        pass


class Event(BaseModel):
    session_id: str
    type: str
    detail: str = ""


@app.post("/api/event")
def browser_event(e: Event):
    if e.type not in RULES:
        raise HTTPException(400, "unknown event type")
    s = get_session(e.session_id)
    v = s.add_violation(e.type, e.detail)
    return payload(s, new=[v])


@app.get("/api/report/{sid}")
def report(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "session not found")
    return sessions[sid].report()


# frontend last me mount hota hai
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="ui")