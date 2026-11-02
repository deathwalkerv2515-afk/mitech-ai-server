# =============================================================================
# MiTech Biometric AI Face Recognition & Anti-Spoofing Engine
# File: face_engine.py (Render Production Ready)
# =============================================================================

import io
import json
import requests
import numpy as np
from PIL import Image
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiTech Biometric AI Engine")

# CORS setup para makakonekta ang Kiosk at Web Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PERMANENT IN-MEMORY CACHE (Compatible sa lahat ng Python versions) ────────
# URL -> list of face encodings
# Hindi na buburahin ang cache kada scan upang maging 0 milliseconds ang matching
# at hindi mag-timeout sa pag-download ng picture mula sa Supabase.
PHOTO_URL_CACHE = {}


def get_ear(eye):
    """Kalkulahin ang Eye Aspect Ratio (EAR)."""
    a = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    b = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    c = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return float((a + b) / (2.0 * c))


@app.get("/")
def root():
    return {
        "status": "MiTech AI Engine Live",
        "cached_photos": len(PHOTO_URL_CACHE),
        "engine": "FastAPI + face_recognition"
    }


@app.post("/clear-cache")
def clear_cache():
    """Tawagin kapag may binago, in-update, o binurang larawan ng empleyado."""
    PHOTO_URL_CACHE.clear()
    return {
        "status": "success",
        "message": "Permanent photo cache cleared successfully."
    }


@app.post("/verify-face")
async def verify_face(
    image1: UploadFile = File(...),         # Live Webcam Frame 1
    image2: UploadFile = File(None),        # Live Webcam Frame 2
    candidates: str = Form(...)             # JSON list ng mga empleyado para sa branch na ito
):
    try:
        # 1. I-parse ang candidate list na ipinasa ng Kiosk Terminal
        try:
            employee_list = json.loads(candidates)
        except Exception:
            return {"matched": False, "message": "Invalid candidates JSON payload."}

        if not employee_list:
            return {
                "matched": False,
                "message": "ACCESS DENIED: Walang rehistradong empleyado sa branch na ito."
            }

        # 2. Basahin at i-proseso ang Live Frame 1
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((800, 800))
        np1 = np.array(pil1)

        # HOG face detector na may 1x upsample para mabilis at accurate kahit medyo malayo sa camera
        face_locations1 = face_recognition.face_locations(np1, number_of_times_to_upsample=1, model="hog")
        if not face_locations1:
            face_locations1 = face_recognition.face_locations(np1, number_of_times_to_upsample=0, model="hog")
            if not face_locations1:
                return {
                    "matched": False,
                    "message": "Walang mukhang nakita. Tumingin nang diretso sa bilog."
                }

        live_encodings = face_recognition.face_encodings(np1, face_locations1)
        if not live_encodings:
            return {
                "matched": False,
                "message": "Hindi ma-proseso ang mukha. Siguraduhing maliwanag ang paligid."
            }

        live_face = live_encodings[0]

        # 3. Liveness & Anti-Still-Photo Spoofing Check (Crash-proof)
        if image2 is not None:
            try:
                c2 = await image2.read()
                pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
                pil2.thumbnail((800, 800))

                # Tiyaking parehas ang sukat bago ikumpara upang maiwasan ang dimension error
                if pil1.size == pil2.size:
                    np2 = np.array(pil2)
                    pixel_diff = float(np.mean(np.abs(np1.astype(float) - np2.astype(float))))
                    if pixel_diff < 0.45:
                        return {
                            "matched": False,
                            "message": "ACCESS DENIED: Peke o litrato sa cellphone ang nakatapat."
                        }
            except Exception as spoof_err:
                print(f"Anti-spoofing warning: {spoof_err}")

        # 4. Kunin ang mga Facial Encodings ng Candidates gamit ang Permanent URL Cache
        candidate_encodings = {}  # emp_id -> list of encodings
        candidate_meta = {}       # emp_id -> emp dict

        for emp in employee_list:
            emp_id = emp.get("employee_id")
            if not emp_id:
                continue

            candidate_meta[emp_id] = emp
            candidate_encodings[emp_id] = []

            photos = emp.get("face_photos") or []
            if not photos and emp.get("photo_url"):
                photos = [{"angle": "front", "url": emp.get("photo_url")}]

            photo_urls = [p.get("url") if isinstance(p, dict) else p for p in photos if p]

            for url in photo_urls:
                if not url or not isinstance(url, str) or not url.startswith("http"):
                    continue

                # KUNG NASA CACHE NA: Instant 0ms retrieval! Walang download delay!
                if url in PHOTO_URL_CACHE:
                    candidate_encodings[emp_id].extend(PHOTO_URL_CACHE[url])
                else:
                    try:
                        res = requests.get(url, timeout=5.0)
                        if res.status_code == 200:
                            ref_pil = Image.open(io.BytesIO(res.content)).convert("RGB")
                            ref_pil.thumbnail((640, 640))
                            encs = face_recognition.face_encodings(np.array(ref_pil))
                            PHOTO_URL_CACHE[url] = encs
                            candidate_encodings[emp_id].extend(encs)
                    except Exception as err:
                        print(f"Warning: Failed to fetch image {url} for {emp_id}: {err}")

        # 5. Euclidean Distance Matching (Threshold: 0.585)
        best_match = None
        min_distance = 1.0
        MATCH_THRESHOLD = 0.585

        for emp_id, enc_list in candidate_encodings.items():
            if not enc_list:
                continue

            distances = face_recognition.face_distance(enc_list, live_face)
            if len(distances) == 0:
                continue

            dist = float(np.min(distances))
            if dist < min_distance:
                min_distance = dist
                best_match = candidate_meta[emp_id]

        if best_match and min_distance <= MATCH_THRESHOLD:
            confidence = round((1.0 - min_distance) * 100, 1)
            return {
                "matched": True,
                "employee_id": best_match["employee_id"],
                "full_name": best_match["full_name"],
                "department": best_match.get("department", "Staff"),
                "position": best_match.get("position", "Member"),
                "salary_type": best_match.get("salary_type", "Fixed Monthly"),
                "branch_code": best_match.get("branch_code", "MAIN"),
                "confidence": f"{confidence}%",
                "distance": float(min_distance)
            }
        else:
            return {
                "matched": False,
                "message": "ACCESS DENIED: Hindi rehistrado ang mukha sa branch na ito."
            }

    except Exception as e:
        return {"matched": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
