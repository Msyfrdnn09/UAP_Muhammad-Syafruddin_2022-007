import json
import numpy as np
import tensorflow as tf
from PIL import Image


class ImagePredictor:
    def __init__(self, model_path, config_path, labels_path):
        # Native Keras 3 format (.keras) → AMAN
        self.model = tf.keras.models.load_model(model_path)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.img_size = tuple(self.config["img_size"])

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB")
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image: Image.Image):
        img = self.preprocess(image)
        preds = self.model.predict(img, verbose=0)[0]

        idx = int(np.argmax(preds))

        return {
            "label": self.class_names[idx],
            "confidence": float(preds[idx]),
            "all_predictions": {
                self.class_names[i]: float(preds[i])
                for i in range(len(self.class_names))
            }
        }
