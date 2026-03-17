import cv2
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import os

class FaceTrainer:
    def __init__(self, embedding_file_name: str, classifier_type="knn"):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.embeddings_file = os.path.join(BASE_DIR, embedding_file_name)
        
        self.classifier_type = classifier_type.lower()
        self.model = None
        self.label_encoder = LabelEncoder()

        # Will be loaded after calling load_embeddings()
        self.X = None
        self.y = None
        self.y_encoded = None
        
    def load_embeddings(self):
        print("[INFO] Loading embeddings...")
        with open(self.embeddings_file, "rb") as f:
            data = pickle.load(f)
        self.X = data["embeddings"]
        self.y = data["labels"]

    def encode_labels(self):
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"[INFO] Encoded {len(self.label_encoder.classes_)} unique labels.")
    
    # ------------------ Train classifier ------------------
    def train(self):
        X = self.X
        y = self.y
        y_encoded = self.label_encoder.fit_transform(y)

        if self.classifier_type == "svm":
            self.model = SVC(kernel="linear", probability=True)

        elif self.classifier_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=3)

        elif self.classifier_type == "logreg":
            self.model = LogisticRegression(max_iter=500)
            
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        self.model.fit(X, y_encoded)
        print(f"[INFO] {self.classifier_type.upper()} model trained on {len(X)} samples.")

    # ------------------ Save model ------------------
    def save_model(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump((self.model, self.label_encoder), f)
        print(f"[INFO] Model saved to {file_path}")
