import os
import tensorflow as tf
import mlflow

BASE_DIR = "dataset_preprocessing"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("1. Memuat dataset...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    print("2. Membangun model (Versi Ringan untuk CI/CD)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("3. Melatih model (Hanya 1 Epoch dan 2 Step)...")
    model.fit(train_generator, epochs=1, steps_per_epoch=2)

    print("4. Menyimpan model ke format MLflow...")
    mlflow.keras.save_model(model, "saved_model")
    print("Proses Selesai!")

if __name__ == "__main__":
    main()