# api_simple.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, joblib, numpy as np, os
from datetime import datetime

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OSRM = "http://router.project-osrm.org"
MODEL = "traffic_speed_model.pkl"   # put your model here (optional)
FALLBACK_MPS = 10.0                 # fallback speed if model missing (m/s)

model = None
if os.path.exists(MODEL):
    try:
        model = joblib.load(MODEL)
        print("Loaded model")
    except Exception as e:
        print("Model load failed, using fallback:", e)
        model = None

def route_from_osrm(src, dst):
    # src/dst: [lat, lon]
    coords = f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
    url = f"{OSRM}/route/v1/driving/{coords}?overview=full&geometries=geojson&alternatives=false"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    if not j.get("routes"): raise ValueError("No route")
    rt = j["routes"][0]
    dist = float(rt["distance"])              # meters
    geom = rt["geometry"]["coordinates"]      # [[lon,lat],...]
    path = [[lat, lon] for lon, lat in geom] # convert to [lat,lon]
    return dist, path

def predict_speed_mps(length_m):
    # simple features: length, hour, weekday
    if model is None:
        return FALLBACK_MPS
    X = np.array([[float(length_m), float(datetime.utcnow().hour), float(datetime.utcnow().weekday())]])
    try:
        pred = float(model.predict(X).reshape(-1)[0])
        # heuristic: if prediction looks like km/h, convert
        if pred > 40: pred = pred / 3.6
        return max(pred, 0.1)
    except Exception:
        return FALLBACK_MPS

class Req(BaseModel):
    src: list
    dst: list

@app.post("/route")
def route(req: Req):
    try:
        dist, path = route_from_osrm(req.src, req.dst)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    speed = predict_speed_mps(dist)               # m/s
    eta_s = dist / speed if speed>0 else dist/FALLBACK_MPS
    return {"path": path, "distance_m": dist, "predicted_speed_mps": speed, "travel_time_seconds": eta_s}
