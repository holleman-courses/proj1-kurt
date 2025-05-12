import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# 1) CONFIGURATION — EDIT THIS
# ─────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\processed_dataset"
IMG_SIZE     = (96, 96)
BATCH_SIZE   = 64
EPOCHS       = 48

TFLITE_FILE  = "open_hand_model.tflite"
C_HEADER     = "open_hand_model_data.h"

# ─────────────────────────────────────────────────────────────
# 2) DATA GENERATORS (with validation split)
# ─────────────────────────────────────────────────────────────
def get_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8,1.2),
        horizontal_flip=True,
    )
    train_gen = datagen.flow_from_directory(
        
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        seed=42
    )

    
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        seed=42
    )
    print("Train classes:", train_gen.class_indices)
    print(" # train samples:", train_gen.samples)
    print("Validation classes:", val_gen.class_indices)
    print(" # val samples:", val_gen.samples)

    return train_gen, val_gen

# ─────────────────────────────────────────────────────────────
# 3) MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
def build_model():
        #FUCKING DOG SHIT MODEL
    # model = keras.Sequential([
    #     # Layer 1
    #     keras.layers.InputLayer(input_shape=(*IMG_SIZE, 1)),
    #     keras.layers.Conv2D(16, 3, padding="same", activation=None),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Activation("relu"),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Dropout(0.4),

    #     # Layer 2
    #     keras.layers.Conv2D(32, 3, padding="same", activation=None),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Activation("relu"),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Dropout(0.4),

    #     # Layer 3
    #     keras.layers.Conv2D(64, 3, padding="same", activation=None),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Activation("relu"),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Dropout(0.4),

    #     # keras.layers.Conv2D(128, 3, padding="same", activation=None),
    #     # keras.layers.BatchNormalization(),
    #     # keras.layers.Activation("relu"),
    #     # keras.layers.MaxPool2D(),
    #     # keras.layers.Dropout(0.4),

    #     # Tiny classifier head
    #     keras.layers.GlobalAveragePooling2D(),
    #     keras.layers.Dropout(0.3),
    #     keras.layers.Dense(16, activation="relu"),      # small bottleneck
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(1, activation="sigmoid")    # single‐score output 
    # ])
        #new model getting over 90% accuracy smh.
    model =  keras.Sequential([
        keras.layers.InputLayer(input_shape=(*IMG_SIZE,1)),
        keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])  


    

    model.compile(
        optimizer= optimizers.Adam(learning_rate=1e-3),     #adjust learning rate here
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─────────────────────────────────────────────────────────────
# 4) TRAINING
# ─────────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen, class_weight):
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
    )
    #model.save("open_hand_model.h5") #COMMENTED OUT BECAUSE I ALREADY HAVE THE MODEL
    return model, history


'''
# ─────────────────────────────────────────────────────────────
# 5) TFLITE QUANTIZATION
# ─────────────────────────────────────────────────────────────
def representative_data_gen():
    for images, _ in train_gen.take(100):
        yield [tf.cast(images * 255.0, tf.uint8)]

def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(TFLITE_FILE, "wb").write(tflite_model)
    print(f"→ Wrote {TFLITE_FILE}")
    return tflite_model

# ─────────────────────────────────────────────────────────────
# 6) DUMP C-HEADER
# ─────────────────────────────────────────────────────────────
def write_c_header(tflite_bytes):
    with open(C_HEADER, "w") as f:
        f.write("#ifndef OPEN_HAND_MODEL_DATA_H_\n#define OPEN_HAND_MODEL_DATA_H_\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("const uint8_t open_hand_model_data[] = {\n")
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0: f.write("    ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11: f.write("\n")
        f.write(f"\n}};\n\nconst size_t open_hand_model_data_len = {len(tflite_bytes)};\n\n")
        f.write("#endif  // OPEN_HAND_MODEL_DATA_H_\n")
    print(f"→ Wrote {C_HEADER}")
'''



# ─────────────────────────────────────────────────────────────
# 7) MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_gen, val_gen = get_generators()
    train_counts = np.bincount(train_gen.classes)
    val_counts   = np.bincount(val_gen.classes)
    print("Train distribution: hands=", train_counts[0], "non_hands=", train_counts[1])
    print("Val   distribution: hands=", val_counts[0],   "non_hands=", val_counts[1])

    
    total = train_counts.sum()
    class_weight = {
        0: total / (2 * train_counts[0]),    # weight for “hands”
        1: total / (2 * train_counts[1])     # weight for “non_hands”
    }

    model = build_model()
    model.summary()
    model, history = train_model(model, train_gen, val_gen, class_weight)

    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"],     label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\test_set",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False                           # keep order for metrics
    )


    # after training:
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_acc:.3f},  loss: {test_loss:.4f}")



    # 1) Get model probabilities on the entire test set (no shuffling!)
    probs_test = model.predict(test_gen, verbose=0).flatten()
    y_true     = test_gen.classes                         # ground‑truth labels
    y_pred     = (probs_test > 0.5).astype(np.uint8)      # threshold @ 0.5

    # 2) Confusion‑matrix elements
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # 3) Rates
    false_rejection_rate = FN / (TP + FN) if (TP + FN) else 0
    false_positive_rate  = FP / (FP + TN) if (FP + TN) else 0

    # 4) Print summary
    print("\n── Confusion‑matrix summary ──")
    print(f"TP: {TP:5d}   FP: {FP:5d}")
    print(f"FN: {FN:5d}   TN: {TN:5d}")
    print(f"False Rejection Rate (FRR): {false_rejection_rate:.4f}")
    print(f"False Positive  Rate (FPR): {false_positive_rate :.4f}")
    # # grab one validation batch
    # x_val, y_val = next(val_gen)
    # # raw model outputs
    # probs = model.predict(x_val).flatten()
    # print("First 10 probs:", np.round(probs[:10], 3))
    # print("First 10 true labels:", y_val[:10])
    # # how many of this batch it calls “hand” vs “non_hands”
    # print("Predicted hands:", np.sum(probs > 0.5),
    #     "  non_hands:", np.sum(probs <= 0.5))



    #tflite_bytes = convert_to_tflite(model)
    #write_c_header(tflite_bytes)
