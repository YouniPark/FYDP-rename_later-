"""Enroll faces from headshot videos using the same enroll_face() path as image upload."""

import sys, os, json, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend-new"))

from app.face_service.settings import face_service_settings
from app.face_service.embedding_store import ArcFaceEmbeddingStore
from app.face_service.arcface_recognizer import ArcFaceRuntimeRecognizer

MAX_FRAMES = 20
FRAME_SKIP = 5

# rm backend-new/data/arcface_embeddings.pkl 
# python enroll_from_video.py 

# --- Load model + store ---
store = ArcFaceEmbeddingStore(face_service_settings.arcface_embedding_store_path)
recognizer = ArcFaceRuntimeRecognizer(face_service_settings, store)

# --- Load people.json ---
people_json = os.path.join(os.path.dirname(__file__), "WebServer", "database", "PeopleDatabase", "people.json")
with open(people_json) as f:
    people = json.load(f)

headshots_dir = os.path.join(os.path.dirname(__file__), "WebServer", "database", "PeopleDatabase")

for person in people:
    face_id = str(person["id"])
    name = person.get("name", face_id)
    headshot = person.get("headshot")
    image_path = person.get("image")

    # Remove old embeddings for this person
    store.remove(face_id)

    enrolled = 0

    # --- Enroll from headshot video ---
    if headshot:
        video_path = os.path.join(headshots_dir, headshot)
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            print(f"\n[{name}] (id={face_id}) enrolling from video: {headshot}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % FRAME_SKIP != 0:
                    continue
                ok = recognizer.enroll_face(face_id, frame)
                if ok:
                    enrolled += 1
                    print(f"  Frame {frame_idx}: enrolled [{enrolled}/{MAX_FRAMES}]")
                if enrolled >= MAX_FRAMES:
                    break
            cap.release()
        else:
            print(f"\n[{name}] (id={face_id}) WARNING: video not found: {video_path}")

    # --- Also enroll from the still image ---
    if image_path:
        img_full_path = os.path.join(headshots_dir, image_path)
        if os.path.exists(img_full_path):
            img = cv2.imread(img_full_path)
            if img is not None:
                ok = recognizer.enroll_face(face_id, img)
                if ok:
                    enrolled += 1
                    print(f"  Image: enrolled [{enrolled}]")
        else:
            print(f"  WARNING: image not found: {img_full_path}")

    print(f"  Total embeddings for {name}: {enrolled}")

print(f"\nDone! Store has: {list(store.get_all().keys())}")