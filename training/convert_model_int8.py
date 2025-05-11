#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────
# 1) Configuration
# ─────────────────────────────────────────────────────────────
H5_MODEL         = "open_hand_model.h5"
TFLITE_MODEL     = "open_hand_model_int8.tflite"
C_HEADER         = "open_hand_model_data.h"
ARRAY_NAME       = "open_hand_model_data"

# Directory with a few hundred training images, organized as:
#   <DATA_DIR>/hands/...   (96×96 grayscale)
#   <DATA_DIR>/non_hands/...
# Used for calibration of activations.
DATA_DIR         = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\processed_dataset"
IMG_SIZE         = (96, 96)
REP_CALIB_SAMPLES = 100

# ─────────────────────────────────────────────────────────────
# 2) Load the Keras model
# ─────────────────────────────────────────────────────────────
print(f"Loading Keras model from {H5_MODEL}...")
model = keras.models.load_model(H5_MODEL)

# ─────────────────────────────────────────────────────────────
# 3) Set up representative dataset generator
# ─────────────────────────────────────────────────────────────
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
rep_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=True,
    seed=42
)

def representative_data_gen():
    count = 0
    for x in rep_gen:
        # x has shape (1, 96,96,1), dtype float32 in [0,1]
        yield [x]
        count += 1
        if count >= REP_CALIB_SAMPLES:
            break

# ─────────────────────────────────────────────────────────────
# 4) Convert to fully‐quantized int8 TFLite
# ─────────────────────────────────────────────────────────────
print("Converting to fully‐integer int8 TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Make sure input and output tensors are SIGNED int8
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# ─────────────────────────────────────────────────────────────
# 5) Save the quantized .tflite file
# ─────────────────────────────────────────────────────────────
with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_quant_model)
print(f"Wrote quantized TFLite model to {TFLITE_MODEL} ({os.path.getsize(TFLITE_MODEL)} bytes)")

# ─────────────────────────────────────────────────────────────
# 6) Dump to a C‐header array
# ─────────────────────────────────────────────────────────────
print(f"Writing C header to {C_HEADER}...")
with open(C_HEADER, "w") as f:
    f.write(f"#ifndef {ARRAY_NAME.upper()}_H_\n")
    f.write(f"#define {ARRAY_NAME.upper()}_H_\n\n")
    f.write("#include <cstdint>\n\n")
    f.write(f"const uint8_t {ARRAY_NAME}[] = {{\n")

    for i, byte in enumerate(tflite_quant_model):
        if i % 16 == 0:
            f.write("  ")
        f.write(f"0x{byte:02x}, ")
        if i % 16 == 15:
            f.write("\n")

    f.write(f"\n}};\n\n")
    f.write(f"const size_t {ARRAY_NAME}_len = {len(tflite_quant_model)};\n\n")
    f.write(f"#endif  // {ARRAY_NAME.upper()}_H_\n")

print("Done.")