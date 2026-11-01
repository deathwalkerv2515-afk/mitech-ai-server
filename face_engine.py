import io
import json
import requests
import numpy as np
from PIL import Image
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiTech Anti-Spoofing & Liveness AI Engine")

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
    """Kalkulahin ang Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return float((A + B) / (2.0 * C))

def crop_normalized_region(img_np, box, size=(64, 64)):
    """I-crop at i-normalize ang mata para sa pixel correlation test."""
    top, right, bottom, left = box
    top = max(0, int(top))
    left = max(0, int(left))
    bottom = min(img_np.shape[0], int(bottom))
    right = min(img_np.shape[1], int(right))
    if top >= bottom or left >= right:
        return np.zeros(size, dtype=np.float32)
    crop = img_np[top:bottom, left:right]
    pil_crop = Image.fromarray(crop).convert("L").resize(size)
    arr = np.array(pil_crop, dtype=np.float32)
    std = float(np.std(arr))
    if std < 1e-4:
        return arr - np.mean(arr)
    return (arr - np.mean(arr)) / std

def compute_correlation(arr1, arr2):
    """Kalkulahin ang pagkakapareho ng pixels sa pagitan ng dalawang frames."""
    if arr1.shape != arr2.shape:
        return 0.0
    return float(np.mean(arr1 * arr2))

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
        # 1. I-load ang Frame 1
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((400, 400))
        np1 = np.array(pil1)

        face_locations1 = face_recognition.face_locations(np1, model="hog")
        if not face_locations1:
            return {"matched": False, "message": "Walang mukhang nakita sa camera. Pakiayos ang pwesto."}

        live_encodings = face_recognition.face_encodings(np1, face_locations1)
        if not live_encodings:
            return {"matched": False, "message": "Hindi ma-proseso ang mukha. Siguraduhing maliwanag."}

        # ── 2. ANTI-SPOOFING (Bawal ang Cellphone Screen at Static Picture) ──
        if image2 is not None:
            c2 = await image2.read()
            pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
            pil2.thumbnail((400, 400))
            np2 = np.array(pil2)

            face_locations2 = face_recognition.face_locations(np2, model="hog")
            if not face_locations2:
                return {"matched": False, "message": "Nawala ang mukha sa pangalawang frame. Manatiling nakaharap."}

            lm1 = face_recognition.face_landmarks(np1, face_locations1)
            lm2 = face_recognition.face_landmarks(np2, face_locations2)

            if lm1 and lm2:
                # A. Pagsuri sa Pagkurap (Blink Analysis)
                ear1 = (get_ear(lm1[0]['left_eye']) + get_ear(lm1[0]['right_eye'])) / 2.0
                ear2 = (get_ear(lm2[0]['left_eye']) + get_ear(lm2[0]['right_eye'])) / 2.0
                ear_change = abs(ear1 - ear2)

                # Ang totoong kurap ay nagbabago ng EAR (> 0.022) o pumipikit (< 0.185)
                has_blinked = (ear_change >= 0.022) or (ear1 < 0.185 or ear2 < 0.185)

                # B. Pagsuri sa Mata gamit ang Pixel Correlation (HULING-HULI ANG LITRATO DITO!)
                top1, right1, bottom1, left1 = face_locations1[0]
                top2, right2, bottom2, left2 = face_locations2[0]
                h1, w1 = bottom1 - top1, right1 - left1
                h2, w2 = bottom2 - top2, right2 - left2

                eye_box1 = (top1 + 0.18 * h1, left1 + 0.1 * w1, top1 + 0.52 * h1, right1 - 0.1 * w1)
                eye_box2 = (top2 + 0.18 * h2, left2 + 0.1 * w2, top2 + 0.52 * h2, right2 - 0.1 * w2)
                eye_crop1 = crop_normalized_region(np1, eye_box1)
                eye_crop2 = crop_normalized_region(np2, eye_box2)
                eye_corr = compute_correlation(eye_crop1, eye_crop2)

                # C. Pagsuri sa Buong Mukha (Face Rigidness)
                face_crop1 = crop_normalized_region(np1, face_locations1[0])
                face_crop2 = crop_normalized_region(np2, face_locations2[0])
                face_corr = compute_correlation(face_crop1, face_crop2)

                # D. Specular Glare ng Screen ng Cellphone
                gray1 = np.mean(np1[top1:bottom1, left1:right1], axis=2)
                glare_ratio = float(np.sum(gray1 > 248) / (gray1.size + 1e-6))

                # ── RULES: HARANGIN ANG CELLPHONE AT LITRATO ──
                # 1. Kapag litrato sa cellphone: Walang kurap at 100% identical ang pixels ng mata (> 0.93)
                if not has_blinked and eye_corr > 0.930 and face_corr > 0.940:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Cellphone Screen o Litrato ang nakita! Bawal ang litrato sa camera."
                    }

                # 2. Kapag walang anumang pagkurap at may reflection ng salamin ng cellphone
                if not has_blinked and glare_ratio > 0.08 and eye_corr > 0.910:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Cellphone Reflection detected. Bawal ang litrato sa cellphone."
                    }

                # 3. Kapag hindi kumurap ang tao sa camera
                if not has_blinked and ear_change < 0.012:
                    return {
                        "matched": False,
                        "message": "ACCESS DENIED: Walang kurap na nakita. Paki-kurap ang iyong mga mata habang nag-i-scan."
                    }
        else:
            return {"matched": False, "message": "Liveness check requires 2 frames."}

        live_face = live_encodings[0]
        employee_list = json.loads(candidates)

        # ── 3. PRE-WARMED CACHE AT PAG-MATCH ──
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

        # ── 4. VERIFY EMPLOYEE IDENTITY ──
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
