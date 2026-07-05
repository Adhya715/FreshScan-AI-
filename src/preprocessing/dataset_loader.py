from pathlib import Path
import tensorflow as tf
import os

# project root = two levels up from this file (src/preprocessing -> src -> FreshScan-AI)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "Fruit Freshness Dataset" / "data" / "raw"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

print("Current working directory:", os.getcwd())
print("Dataset exists:", DATASET_PATH.exists())

def get_image_paths():
    dataset = []

    for fruit_folder in DATASET_PATH.iterdir():

        if not fruit_folder.is_dir():
            continue

        fruit_name = fruit_folder.name

        for freshness_folder in fruit_folder.iterdir():

            if not freshness_folder.is_dir():
                continue

            freshness = freshness_folder.name
            valid_extensions = {".jpg", ".jpeg", ".png"}

            for image_path in freshness_folder.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
                    dataset.append({
                        "image_path": image_path,
                        "fruit": fruit_name,
                        "freshness": freshness,
                        "filename": image_path.name
                    })
    return dataset
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label

def create_tf_dataset():
    dataset = get_image_paths()

    image_paths = [str(item["image_path"]) for item in dataset]

    labels = [
        CLASS_TO_INDEX[f"{item['fruit']}_{item['freshness']}"]
        for item in dataset
    ]

    tf_dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    tf_dataset = tf_dataset.shuffle(buffer_size=len(image_paths))

    tf_dataset = tf_dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    tf_dataset = tf_dataset.batch(BATCH_SIZE)

    tf_dataset = tf_dataset.prefetch(AUTOTUNE)

    total_batches = tf.data.experimental.cardinality(tf_dataset).numpy()

    print(f"Total batches: {total_batches}")

    train_size = int(0.7 * total_batches)
    val_size = int(0.15 * total_batches)

    train_dataset = tf_dataset.take(train_size)

    validation_dataset = (
        tf_dataset
        .skip(train_size)
        .take(val_size)
    )

    test_dataset = (
        tf_dataset
        .skip(train_size + val_size)
    )

    return train_dataset, validation_dataset, test_dataset

if __name__ == "__main__":

    train_dataset, validation_dataset, test_dataset = create_tf_dataset()

    print("Training batches:",
          tf.data.experimental.cardinality(train_dataset).numpy())

    print("Validation batches:",
          tf.data.experimental.cardinality(validation_dataset).numpy())

    print("Test batches:",
          tf.data.experimental.cardinality(test_dataset).numpy())