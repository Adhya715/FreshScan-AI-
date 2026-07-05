<<<<<<< HEAD
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "freshscan_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten"
]
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = image / 255.0

    image = tf.expand_dims(image, axis=0)

    return image

def predict(image_path):
    image = preprocess_image(image_path)

    predictions = model.predict(image)

    predicted_class = tf.argmax(predictions[0]).numpy()

    confidence = float(tf.reduce_max(predictions[0]).numpy())

    class_name = CLASS_NAMES[predicted_class]

    fruit, condition = class_name.split("_")

    return fruit, condition, confidence

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip('"')

    fruit, condition, confidence = predict(image_path)

    print(f"\nFruit: {fruit}")
    print(f"Condition: {condition}")
=======
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "freshscan_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten"
]
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = image / 255.0

    image = tf.expand_dims(image, axis=0)

    return image

def predict(image_path):
    image = preprocess_image(image_path)

    predictions = model.predict(image)

    predicted_class = tf.argmax(predictions[0]).numpy()

    confidence = float(tf.reduce_max(predictions[0]).numpy())

    class_name = CLASS_NAMES[predicted_class]

    fruit, condition = class_name.split("_")

    return fruit, condition, confidence

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip('"')

    fruit, condition, confidence = predict(image_path)

    print(f"\nFruit: {fruit}")
    print(f"Condition: {condition}")
>>>>>>> b691b143c65ce7a3c65f7c006142b96a3bf3d63b
    print(f"Confidence: {confidence:.2%}")