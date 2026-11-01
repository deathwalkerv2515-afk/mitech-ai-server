import io
import json
import requests
import numpy as np
from PIL import Image
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiTech Anti-Spoofing Cloud AI Engine (Fast & High-Accuracy)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent Memory Cache para sa mabilis na pag-match (Hindi na magda-download paulit-ulit)
KNOWN_FACE_ENCODINGS = {}
KNOWN_EMPLOYEES_META = {}

def get_ear(eye):
    """Kalkulahin ang Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

@app.get("/")
def root():
    return {"status": "MiTech AI Engine Live", "cached": len(KNOWN_FACE_ENCODINGS)}

@app.post("/verify-face")
async def verify_face(
    image1: UploadFile = File(...),         # Frame 1
    image2: UploadFile = File(None),        # Frame 2
    candidates: str = Form(...)
):
    try:
        # 1. Mabilis na i-load at i-downscale ang Frame 1 (360px para 3x faster sa CPU)
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((360, 360))
        np1 = np.array(pil1)

        # Hanapin ang mukha
        face_locations1 = face_recognition.face_locations(np1, model="hog")
        if not face_locations1:
            return {"matched": False, "message": "Walang mukhang nakita sa camera. Pakiayos ang pwesto."}

        live_encodings = face_recognition.face_encodings(np1, face_locations1)
        if not live_encodings:
            return {"matched": False, "message": "Hindi ma-proseso ang mukha. Siguraduhing maliwanag."}

        # ── 2. SMART ANTI-SPOOFING (Passive Micro-Movement + Dynamic Liveness) ──
        if image2 is not None:
            c2 = await image2.read()
            pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
            pil2.thumbnail((360, 360))
            np2 = np.array(pil2)

            lm1 = face_recognition.face_landmarks(np1, face_locations1)
            face_locations2 = face_recognition.face_locations(np2, model="hog")
            lm2 = face_recognition.face_landmarks(np2, face_locations2) if face_locations2 else None

            if lm1 and lm2:
                # A. Suriin ang Eye Aspect Ratio (Blink or Eye Fluctuation)
                ear1 = (get_ear(lm1[0]['left_eye']) + get_ear(lm1[0]['right_eye'])) / 2.0
                ear2 = (get_ear(lm2[0]['left_eye']) + get_ear(lm2[0]['right_eye'])) / 2.0
                has_eye_action = (abs(ear1 - ear2) >= 0.015) or (ear1 < 0.18) or (ear2 < 0.18)

                # B. Suriin ang Natural Human Micro-Movement (3D Non-Rigid Shift)
                # Ang picture sa cellphone/papel kapag nakatapat ay 100% frozen (0.0 pixel jitter)
                nose1 = np.array(lm1[0]['nose_tip'])
                nose2 = np.array(lm2[0]['nose_tip'])
                chin1 = np.array(lm1[0]['chin'])
                chin2 = np.array(lm2[0]['chin'])

                motion_diff = float(np.mean(np.abs(nose1 - nose2)) + np.mean(np.abs(chin1 - chin2)))

                # Kung 100% static (litrato sa stand/cellphone na walang anumang natural human movement)
                is_completely_frozen = (motion_diff < 0.12) and not has_eye_action

                # Kung masyadong malikot o inalog ang papel
                is_excessive_shake = motion_diff > 35.0

                if is_completely_frozen:
                    return {
                        "matched": False,
                        "message": "PEKE / STATIC PICTURE: Nakitang litrato o hindi gumagalaw. Paki-harap ang mukha nang natural."
                    }
                elif is_excessive_shake:
                    return {
                        "matched": False,
                        "message": "MASYADONG MALIKOT: Manatiling nakaharap nang maayos sa camera."
                    }

        live_face = live_encodings[0]
        employee_list = json.loads(candidates)

        # ── 3. MEMORY CACHE (Pre-warmed Encodings para Instant ang Tugon) ──
        for emp in employee_list:
            emp_id = emp["employee_id"]
            if emp_id not in KNOWN_FACE_ENCODINGS:
                KNOWN_FACE_ENCODINGS[emp_id] = []
                KNOWN_EMPLOYEES_META[emp_id] = emp

                # Kumuha ng mga anggulo (front, left, right, up, down)
                photos = emp.get("face_photos") or []
                if not photos and emp.get("photo_url"):
                    photos = [{"angle": "front", "url": emp.get("photo_url")}]

                for p in photos:
                    url = p.get("url") if isinstance(p, dict) else p
                    if not url:
                        continue
                    try:
                        res = requests.get(url, timeout=2.5)
                        if res.status_code == 200:
                            ref_pil = Image.open(io.BytesIO(res.content)).convert("RGB")
                            ref_pil.thumbnail((320, 320))
                            encs = face_recognition.face_encodings(np.array(ref_pil))
                            if encs:
                                KNOWN_FACE_ENCODINGS[emp_id].append(encs[0])
                    except Exception:
                        pass

        # ── 4. HIGH-ACCURACY FACE DISTANCE MATCHING ──
        best_match = None
        min_distance = 1.0
        # 0.52 ang standard benchmark para sa dlib 128D: mataas ang accuracy kahit may bintana/backlight sa likod
        MATCH_THRESHOLD = 0.52

        for emp_id, enc_list in KNOWN_FACE_ENCODINGS.items():
            if not enc_list:
                continue
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
