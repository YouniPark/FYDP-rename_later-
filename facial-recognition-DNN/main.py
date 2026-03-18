from face_recognizer import FaceRecognizer
from train_model import FaceTrainer
from extract_embedding import FaceEmbedder

import cv2
import os

embedder = FaceEmbedder()
embedder.process_dataset()
embedder.save_embeddings("embeddings.pkl")

trainer = FaceTrainer(embedding_file_name="embeddings.pkl", classifier_type="knn")
trainer.load_embeddings()
trainer.train()
trainer.save_model("face_recognizer.pkl")

recognizer = FaceRecognizer(model_file_name="face_recognizer.pkl", classifier_type="knn")
recognizer.load_model()
recognizer.live_recognition(detector=embedder.detector, embedder=embedder.embedder)

