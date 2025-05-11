import os
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────
# 1) Configuration
# ─────────────────────────────────────────────────────────────
H5_MODEL     = "open_hand_model.h5"
TFLITE_MODEL = "open_hand_model.tflite"
C_HEADER     = "open_hand_model_data.h"
ARRAY_NAME   = "open_hand_model_data"

# ─────────────────────────────────────────────────────────────
# 2) Load the Keras model
# ─────────────────────────────────────────────────────────────
print(f"Loading Keras model from {H5_MODEL}...")
model = keras.models.load_model(H5_MODEL)

# ─────────────────────────────────────────────────────────────
# 3) Convert to TFLite (dynamic‐range quantization)
# ─────────────────────────────────────────────────────────────
print("Converting to TFLite with dynamic‐range quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# For full integer quantization you'd add a representative_dataset, 
# but dynamic-range works without one.
tflite_model = converter.convert()

# ─────────────────────────────────────────────────────────────
# 4) Save the .tflite file
# ─────────────────────────────────────────────────────────────
with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)
print(f"Wrote TFLite model to {TFLITE_MODEL} ({len(tflite_model)} bytes)")

# ─────────────────────────────────────────────────────────────
# 5) Dump to a C‐header array
# ─────────────────────────────────────────────────────────────
print(f"Writing C header to {C_HEADER}...")
with open(C_HEADER, "w") as f:
    f.write(f"#ifndef {ARRAY_NAME.upper()}_H_\n")
    f.write(f"#define {ARRAY_NAME.upper()}_H_\n\n")
    f.write("#include <cstdint>\n\n")
    f.write(f"const uint8_t {ARRAY_NAME}[] = {{\n")

    # write 16 bytes per line
    for i, byte in enumerate(tflite_model):
        if i % 16 == 0:
            f.write("  ")
        f.write(f"0x{byte:02x}, ")
        if i % 16 == 15:
            f.write("\n")

    f.write(f"\n}};\n\n")
    f.write(f"const size_t {ARRAY_NAME}_len = {len(tflite_model)};\n\n")
    f.write(f"#endif  // {ARRAY_NAME.upper()}_H_\n")

print("Done.")