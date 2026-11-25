from fastapi import FastAPI

app=FastAPI()
@app.get("/api/attentionlevel")
def attention():
    return {"attentionlevel":"0"}

import random

@app.get("/api/results")
def results():
    windows = []
    for i in range(30):                 # 30 windows × 10 seconds = 5 minutes of lecture
        windows.append({
            "windowIndex": i,
            "startSec": i * 10,
            "attentionLevel": round(random.uniform(0.2, 0.9), 2),
            "text": f"fake transcript sentence number {i}"
        })
    return {"windows": windows}

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")