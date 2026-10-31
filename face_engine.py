import io
import json
import requests
import numpy as np
from PIL import Image
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiTech Anti-Spoofing Cloud AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWN_FACE_ENCODINGS = {}
KNOWN_EMPLOYEES_META = {}

def get_ear(eye):
    """Kalkulahin ang Eye Aspect Ratio (EAR) para malaman kung nakapikit o nakadilat."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

@app.get("/")
def root():
    return {"status": "MiTech AI Engine Live", "cached": len(KNOWN_FACE_ENCODINGS)}

@app.post("/verify-face")
async def verify_face(
    image1: UploadFile = File(...),         # Frame 1 (Nakaharap)
    image2: UploadFile = File(None),        # Frame 2 (Kumurap)
    candidates: str = Form(...)
):
    try:
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((480, 480))
        np1 = np.array(pil1)

        live_encodings = face_recognition.face_encodings(np1)
        if not live_encodings:
            return {"matched": False, "message": "Walang mukhang nakita sa camera."}

        # ── 1. STRICT ANTI-SPOOFING (Bawal ang Cellphone / Picture) ──
        if image2 is not None:
            c2 = await image2.read()
            pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
            pil2.thumbnail((480, 480))
            np2 = np.array(pil2)

            lm1 = face_recognition.face_landmarks(np1)
            lm2 = face_recognition.face_landmarks(np2)

            if lm1 and lm2:
                ear1 = (get_ear(lm1[0]['left_eye']) + get_ear(lm1[0]['right_eye'])) / 2.0
                ear2 = (get_ear(lm2[0]['left_eye']) + get_ear(lm2[0]['right_eye'])) / 2.0

                # Kapag litrato sa cellphone: parehong bukas ang mata (walang kurap)
                # Ang totoong kurap ay bumababa ang EAR sa < 0.20 o may malaking bawas (> 0.035)
                has_blinked = (ear2 < 0.20) or (abs(ear1 - ear2) >= 0.035)

                if not has_blinked:
                    return {
                        "matched": False,
                        "message": "PEKE / LITRATO DETECTED: Bawal ang litrato sa cellphone o papel! Paki-kurap ang iyong mga mata."
                    }
        else:
            return {
                "matched": False,
                "message": "LIVENESS ERROR: Kailangan ng motion frames para kumpirmahing tao ka."
            }

        live_face = live_encodings[0]
        employee_list = json.loads(candidates)

        # ── 2. Memory Caching para Mabilis ──
        for emp in employee_list:
            emp_id = emp["employee_id"]
            if emp_id not in KNOWN_FACE_ENCODINGS:
                KNOWN_FACE_ENCODINGS[emp_id] = []
                KNOWN_EMPLOYEES_META[emp_id] = emp

                photos = emp.get("face_photos") or []
                if not photos and emp.get("photo_url"):
                    photos = [{"angle": "front", "url": emp.get("photo_url")}]

                for p in photos:
                    url = p.get("url") if isinstance(p, dict) else p
                    if not url: continue
                    try:
                        res = requests.get(url, timeout=3)
                        if res.status_code == 200:
                            ref_pil = Image.open(io.BytesIO(res.content)).convert("RGB")
                            ref_pil.thumbnail((480, 480))
                            encs = face_recognition.face_encodings(np.array(ref_pil))
                            if encs:
                                KNOWN_FACE_ENCODINGS[emp_id].append(encs[0])
                    except Exception:
                        pass

        # ── 3. Face Distance Matching ──
        best_match = None
        min_distance = 1.0
        MATCH_THRESHOLD = 0.50

        for emp_id, enc_list in KNOWN_FACE_ENCODINGS.items():
            if not enc_list: continue
            dist = float(np.min(face_recognition.face_distance(enc_list, live_face)))
            if dist < min_distance:
                min_distance = dist
                best_match = KNOWN_EMPLOYEES_META.get(emp_id)

        if best_match and min_distance <= MATCH_THRESHOLD:
            confidence = round((1.0 - min_distance) * 100, 1)
            return {
                "matched": True,
                "employee_id": best_match["employee_id"],
                "full_name": best_match["full_name"],
                "department": best_match.get("department", "Staff"),
                "position": best_match.get("position", "Member"),
                "salary_type": best_match.get("salary_type", "Fixed Monthly"),
                "confidence": f"{confidence}%",
                "distance": float(min_distance)
            }
        else:
            return {
                "matched": False,
                "message": "ACCESS DENIED: Hindi rehistrado ang mukha sa kumpanya."
            }

    except Exception as e:
        return {"matched": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)