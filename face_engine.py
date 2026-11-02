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

# Dynamic in-memory cache
KNOWN_FACE_ENCODINGS = {}
KNOWN_EMPLOYEES_META = {}
CACHED_PHOTO_HASHES = {}

def get_ear(eye):
    """Kalkulahin ang Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return float((A + B) / (2.0 * C))

@app.get("/")
def root():
    return {
        "status": "MiTech AI Engine Live",
        "cached_employees": len(KNOWN_FACE_ENCODINGS)
    }

@app.post("/clear-cache")
def clear_cache():
    """Tawagin ito kapag may binura o binagong empleyado para mag-refresh agad."""
    KNOWN_FACE_ENCODINGS.clear()
    KNOWN_EMPLOYEES_META.clear()
    CACHED_PHOTO_HASHES.clear()
    return {"status": "Cache Cleared Successfully"}

@app.post("/verify-face")
async def verify_face(
    image1: UploadFile = File(...),         # Frame 1
    image2: UploadFile = File(None),        # Frame 2
    candidates: str = Form(...)             # JSON list ng empleyado para sa branch na ito
):
    try:
        # 1. I-parse ang mga kandidatong empleyado para sa kasalukuyang branch
        try:
            employee_list = json.loads(candidates)
        except Exception:
            return {"matched": False, "message": "Invalid candidates JSON payload."}

        if not employee_list:
            return {
                "matched": False,
                "message": "ACCESS DENIED: Walang rehistradong empleyado sa branch na ito."
            }

        # Kuhanin ang listahan ng legal employee IDs sa active branch
        current_branch_emp_ids = {emp["employee_id"] for emp in employee_list}

        # Tanggalin sa cache ang mga mukha na hindi na kabilang sa branch o na-delete na
        cached_keys = list(KNOWN_FACE_ENCODINGS.keys())
        for c_id in cached_keys:
            if c_id not in current_branch_emp_ids:
                KNOWN_FACE_ENCODINGS.pop(c_id, None)
                KNOWN_EMPLOYEES_META.pop(c_id, None)
                CACHED_PHOTO_HASHES.pop(c_id, None)

        # 2. I-load ang Live Frame 1 na may malinaw na resolution (800x800)
        c1 = await image1.read()
        pil1 = Image.open(io.BytesIO(c1)).convert("RGB")
        pil1.thumbnail((800, 800))
        np1 = np.array(pil1)

        face_locations1 = face_recognition.face_locations(np1, model="hog")
        if not face_locations1:
            return {"matched": False, "message": "Walang mukhang nakita. Tumingin nang diretso sa bilog."}

        live_encodings = face_recognition.face_encodings(np1, face_locations1)
        if not live_encodings:
            return {"matched": False, "message": "Hindi ma-proseso ang mukha. Siguraduhing maliwanag ang paligid."}

        # 3. Mabilis at maaasahang Liveness Check
        if image2 is not None:
            c2 = await image2.read()
            pil2 = Image.open(io.BytesIO(c2)).convert("RGB")
            pil2.thumbnail((800, 800))
            np2 = np.array(pil2)

            face_locations2 = face_recognition.face_locations(np2, model="hog")
            if face_locations2:
                lm1 = face_recognition.face_landmarks(np1, face_locations1)
                lm2 = face_recognition.face_landmarks(np2, face_locations2)

                if lm1 and lm2:
                    ear1 = (get_ear(lm1[0]['left_eye']) + get_ear(lm1[0]['right_eye'])) / 2.0
                    ear2 = (get_ear(lm2[0]['left_eye']) + get_ear(lm2[0]['right_eye'])) / 2.0
                    ear_change = abs(ear1 - ear2)
                    has_blinked = (ear_change >= 0.015) or (ear1 < 0.19 or ear2 < 0.19)

                    # Harangan kapag 100% static na printout o picture
                    if not has_blinked and ear_change < 0.005:
                        return {
                            "matched": False,
                            "message": "ACCESS DENIED: Paki-kurap ang iyong mga mata habang nag-i-scan."
                        }

        live_face = live_encodings[0]

        # 4. I-encode lamang ang mga empleyado ng active branch
        for emp in employee_list:
            emp_id = emp["employee_id"]
            photos = emp.get("face_photos") or []
            if not photos and emp.get("photo_url"):
                photos = [{"angle": "front", "url": emp.get("photo_url")}]

            photo_urls = [p.get("url") if isinstance(p, dict) else p for p in photos if p]
            current_hash = ",".join(filter(None, photo_urls))

            # Kapag bago o binago ang picture, i-download at i-encode
            if emp_id not in KNOWN_FACE_ENCODINGS or CACHED_PHOTO_HASHES.get(emp_id) != current_hash:
                KNOWN_FACE_ENCODINGS[emp_id] = []
                KNOWN_EMPLOYEES_META[emp_id] = emp
                CACHED_PHOTO_HASHES[emp_id] = current_hash

                for url in photo_urls:
                    if not url:
                        continue
                    try:
                        res = requests.get(url, timeout=3.0)
                        if res.status_code == 200:
                            ref_pil = Image.open(io.BytesIO(res.content)).convert("RGB")
                            ref_pil.thumbnail((640, 640))
                            encs = face_recognition.face_encodings(np.array(ref_pil))
                            if encs:
                                KNOWN_FACE_ENCODINGS[emp_id].append(encs[0])
                    except Exception as err:
                        print(f"Error fetching photo for {emp_id}: {err}")

        # 5. Isolate matching: Sa active branch candidates LAMANG maghahanap
        best_match = None
        min_distance = 1.0
        MATCH_THRESHOLD = 0.56  # Sakto para sa normal na webcam na may anino

        for emp_id in current_branch_emp_ids:
            enc_list = KNOWN_FACE_ENCODINGS.get(emp_id, [])
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
                "message": "ACCESS DENIED: Hindi rehistrado ang mukha sa branch na ito."
            }

    except Exception as e:
        return {"matched": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
