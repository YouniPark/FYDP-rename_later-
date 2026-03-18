"""
This program is responsible for deep learning feature extractor to generate a 128-D vector describing a face
"""

import os
import cv2
import pickle

class FaceEmbedder:

    def __init__(self, confidence_threshold=0.7):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.dataset_dir = os.path.join(BASE_DIR, "dataset")

        self.confidence_threshold = confidence_threshold

        protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
        modelPath = os.path.join(BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
        embedderPath = os.path.join(BASE_DIR, "face_embedding_model/openface_nn4.small2.v1.t7")

        # Load detector
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        if self.detector.empty():
            raise ValueError(f"Face detector model failed to load. Check paths:\n{protoPath}\n{modelPath}")

        # Load embedder
        self.embedder = cv2.dnn.readNetFromTorch(embedderPath)
        if self.embedder.empty():
            raise ValueError(f"Face embedder model failed to load. Check path:\n{embedderPath}")

        # Store embeddings
        self.X = []
        self.y = []

        # ------------------ Process dataset and save embeddings ------------------

    def process_dataset(self):

        print("[INFO] Processing video dataset...")

        dataset_dir = self.dataset_dir

        for person in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person)

            if not os.path.isdir(person_dir):
                continue

            print(f"[INFO] Processing person: {person}")

            for file in os.listdir(person_dir):
                file_path = os.path.join(person_dir, file)
                
                if file.lower().endswith((".mp4", ".mov")):
                    frames = self.extract_frames(file_path)

                elif file.lower().endswith((".jpg", ".png", ".jpeg")):
                    frames = [cv2.imread(file_path)]

                else:
                    continue

                for frame in frames:
                    faces = self.detect_faces(frame)

                    for face in faces:
                        embedding = self.get_embedding(face)
                        self.X.append(embedding)
                        self.y.append(person)

        print(f"[INFO] Extracted {len(self.X)} embeddings")

    def save_embeddings(self, output_file: str):

        # Save embeddings
        print(f"[INFO] Saving {len(self.X)} embeddings to {output_file}...")

        with open(output_file, "wb") as f:
            pickle.dump({"embeddings": self.X, "labels": self.y}, f)

        print("[INFO] Done!")

    def extract_frames(self, video_path, step=10):

        cap = cv2.VideoCapture(video_path)
        frames = []
        fid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % step == 0:
                frames.append(frame)
            fid += 1

        cap.release()
        return frames
    
    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123), swapRB=False)
        self.detector.setInput(blob)
        detections = self.detector.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > self.confidence_threshold:
                box = detections[0,0,i,3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")

                # Clip coordinates to image size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                face = frame[y1:y2, x1:x2]
                faces.append(face)

        return faces
    
    # ------------------ Get embedding for a face image ------------------

    def get_embedding(self, face):

        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
        self.embedder.setInput(face_blob)
        vec = self.embedder.forward()

        return vec.flatten()

 