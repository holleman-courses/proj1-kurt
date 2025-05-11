import serial
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PORT       = "COM3"
BAUD       = 115200
FRAME_SIZE = 96*96
MODEL_PATH = "open_hand_model_int8.tflite"


# ─────────────────────────────────────────────────────────────
# 1) Capture the raw frame from Serial (as before)
# ─────────────────────────────────────────────────────────────
ser = serial.Serial(PORT, BAUD, timeout=5)
print("Waiting for FRAME_START...")
while True:
    line = ser.readline().decode("ascii", errors="ignore").strip()
    if line == "FRAME_START":
        break
raw = ser.read(FRAME_SIZE)
print(list(raw[:20])) 

img = np.frombuffer(raw, dtype=np.uint8).reshape(96,96)



# Show the captured image
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
'''
# ─────────────────────────────────────────────────────────────
# 2) Prepare the quantized input for the INT8 model
# ─────────────────────────────────────────────────────────────
# Normalize to [0,1] exactly like your training
normalized = img.astype(np.float32) / 255.0

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input details for scale & zero_point
input_details  = interpreter.get_input_details()[0]
scale, zero_pt = input_details['quantization']
print(f"Input quant params: scale={scale}, zp={zero_pt}")

# Quantize your normalized data into int8
q_input = np.round(normalized / scale + zero_pt).astype(np.int8)

# Reshape to [1,96,96,1]
q_input = q_input.reshape((1,96,96,1))

# ─────────────────────────────────────────────────────────────
# 3) Run inference
# ─────────────────────────────────────────────────────────────
interpreter.set_tensor(input_details['index'], q_input)
interpreter.invoke()

# Get and dequantize the output
output_details = interpreter.get_output_details()[0]
q_out = interpreter.get_tensor(output_details['index'])[0][0]
out_scale, out_zp = output_details['quantization']
confidence = (q_out - out_zp) * out_scale

print("Raw INT8 output:", int(q_out))
print(f"Dequantized confidence: {confidence:.3f}")

'''