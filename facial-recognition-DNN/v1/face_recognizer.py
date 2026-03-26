import os
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder


class FaceRecognizer:
    """
    A wrapper class for face recognition using a pre-trained classifier
    and embeddings. Supports predicting on single embeddings and live video.
    """

    def __init__(self, model_file_name: str, classifier_type="knn"):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_file = os.path.join(BASE_DIR, model_file_name)
        self.model = None
        self.classifier_type = classifier_type.lower()
        self.label_encoder = LabelEncoder()
        self.conf_threshold = 0.6  # threshold for unknown vs. known faces

    # ------------------ Load model ------------------ #
    def load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found: {self.model_file}")

        with open(self.model_file, "rb") as f:
            self.model, self.label_encoder = pickle.load(f)
        print(f"[INFO] Model loaded from {self.model_file}")

    # ------------------ Predict ------------------ #
    def predict(self, X, threshold=0.5):
        """
        Predict the class of face embeddings.
        If probability is below threshold, label as 'Unknown'.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X).max(axis=1)

        results = []
        for pred, prob in zip(preds, probs):
            name = self.label_encoder.inverse_transform([pred])[0]
            if prob < threshold:
                name = "Unknown"
            results.append({"name": name, "probability": prob})

        return results

    # ------------------ Live recognition ------------------ #
    def live_recognition(self, detector, embedder, threshold=0.5):
        """
        Run live recognition from webcam (or any camera index).
        detector: OpenCV DNN face detector
        embedder: Pretrained face embedding model (e.g., FaceNet)
        """
        cap = cv2.VideoCapture(0)
        print("[INFO] Starting live recognition. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False)
            detector.setInput(blob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype("int")

                    face = frame[y1:y2, x1:x2]

                    # Generate embedding
                    face_blob = cv2.dnn.blobFromImage(
                        face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                    )
                    embedder.setInput(face_blob)
                    embedding = embedder.forward().flatten()

                    result = self.predict([embedding], threshold=threshold)[0]

                    # Draw box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{result['name']} ({result['probability']*100:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()