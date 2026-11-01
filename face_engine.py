import io
import json
import requests
import numpy as np
from PIL import Image
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiTech Anti-Spoofing & 3D Liveness AI Engine")

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
    """Kalkulahin ang Eye Aspect Ratio (EAR) para sa mata."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return float((A + B) / (2.0 * C))

def get_mar(lm):
    """Kalkulahin ang Mouth Aspect Ratio (MAR) para sa bibig."""
    top = np.array(lm['top_lip'][9])
    bot = np.array(lm['bottom_lip'][9])
    left = np.array(lm['top_lip'][0])
    right = np.array(lm['top_lip'][6])
    return float(np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6))

def get_relative_features(lm):
    """Kalkulahin ang 3D Relative Ratios ng mukha."""
    left_eye = np.mean(lm['left_eye'], axis=0)
    right_eye = np.mean(lm['right_eye'], axis=0)
    nose = np.mean(lm['nose_tip'], axis=0)
    chin = np.array(lm['chin'][8])
    mouth = np.mean(lm['top_lip'] + lm['bottom_lip'], axis=0)

    eye_dist = float(np.linalg.norm(left_eye - right_eye) + 1e-6)

    d_nose_left = float(np.linalg.norm(nose - left_eye) / eye_dist)
    d_nose_right = float(np.linalg.norm(nose - right_eye) / eye_dist)
    d_nose_chin = float(np.linalg.norm(nose - chin) / eye_dist)
    d_mouth_chin = float(np.linalg.norm(mouth - chin) / eye_dist)

    return np.array([d_nose_left, d_nose_right, d_nose_chin, d_mouth_chin])

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
        # 1. I-load at i-scale ang Frame 1 (360px para mabilis)
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((360, 360))
        np1 = np.array(pil1)

        face_locations1 = face_recognition.face_locations(np1, model="hog")
        if not face_locations1:
            return {"matched": False, "message": "Walang mukhang nakita sa camera. Paki-harap ang totoong mukha."}

        live_encodings = face_recognition.face_encodings(np1, face_locations1)
        if not live_encodings:
            return {"matched": False, "message": "Hindi malinaw ang mukha. Siguraduhing maliwanag."}

        # ── 2. BULLETPROOF 3D LIVENESS & PHONE SCREEN DETECTION ──
        if image2 is not None:
            c2 = await image2.read()
            pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
            pil2.thumbnail((360, 360))
            np2 = np.array(pil2)

            lm1 = face_recognition.face_landmarks(np1, face_locations1)
            face_locations2 = face_recognition.face_locations(np2, model="hog")
            lm2 = face_recognition.face_landmarks(np2, face_locations2) if face_locations2 else None

            if lm1 and lm2:
                # A. Pagsuri sa pagkurap o paggalaw ng mata
                ear1 = (get_ear(lm1[0]['left_eye']) + get_ear(lm1[0]['right_eye'])) / 2.0
                ear2 = (get_ear(lm2[0]['left_eye']) + get_ear(lm2[0]['right_eye'])) / 2.0
                ear_change = abs(ear1 - ear2)

                # B. Pagsuri sa bibig
                mar1 = get_mar(lm1[0])
                mar2 = get_mar(lm2[0])
                mar_change = abs(mar1 - mar2)

                # C. 3D Non-Rigid Facial Deformation (Huli ang Cellphone/Papel dito!)
                # Sa cellphone o picture, 100% FLAT ito. Kahit iwagayway mo ang phone, 0.0 ang deformation!
                feat1 = get_relative_features(lm1[0])
                feat2 = get_relative_features(lm2[0])
                internal_deformation = float(np.sum(np.abs(feat1 - feat2)))

                # D. Pagsuri kung inalog ang cellphone (Rigid Translation)
                nose1 = np.array(lm1[0]['nose_tip'][2])
                nose2 = np.array(lm2[0]['nose_tip'][2])
                global_shift = float(np.linalg.norm(nose1 - nose2))

                # E. Specular Screen Glare (Reflection sa salamin ng cellphone)
                gray1 = np.mean(np1, axis=2)
                specular_glare = float(np.sum(gray1 > 248) / gray1.size)

                # ── MGA PATAKARAN SA PAG-HARANG NG PEKE ──
                # 1. Kung static picture sa cellphone (walang kurap, walang galaw ang bibig, at 0 deformation)
                if ear_change < 0.009 and mar_change < 0.009 and internal_deformation < 0.012:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Cellphone Screen o Litrato ang nakita! Bawal ang litrato."
                    }

                # 2. Kung inalog ang cellphone sa kamay para magkunwaring gumagalaw (flat rigid movement)
                if global_shift > 1.2 and internal_deformation < 0.011 and ear_change < 0.008:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Bawal iwagayway ang litrato o cellphone sa camera!"
                    }

                # 3. Kung may matinding reflection ng screen ng cellphone
                if specular_glare > 0.14 and internal_deformation < 0.018:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Cellphone reflection detected. Ilapit ang totoong mukha."
                    }
        else:
            return {"matched": False, "message": "Liveness check requires 2 frames."}

        live_face = live_encodings[0]
        employee_list = json.loads(candidates)

        # ── 3. PRE-WARMED CACHE & EMBEDDINGS MATCHING ──
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

        # ── 4. MATCHING ──
        best_match = None
        min_distance = 1.0
        MATCH_THRESHOLD = 0.50

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
