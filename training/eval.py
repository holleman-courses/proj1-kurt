import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#CHANGE THE DIRECTORY IN DATA_DIR TO SUIT YOUR SYSTEM
MODEL_PATH = "open_hand_model.h5"
DATA_DIR   = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\preliminary_dataset"
IMG_SIZE   = (96, 96)
BATCH_SIZE = 1   # one image at a time; with only 11 images this is fine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load the trained model
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}\n")

    # 2) Set up a test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    # 3) Evaluate on the preliminary set
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss:     {loss:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}%\n")

    # 4) Inspect individual predictions
    y_true      = test_gen.classes
    y_prob      = model.predict(test_gen, verbose=1).flatten()
    y_pred      = (y_prob > 0.5).astype(int)

    print("True labels:      ", y_true)
    print("Predicted labels: ", y_pred)
    print("Predicted probs:  ", np.round(y_prob, 3))

if __name__ == "__main__":
    main()