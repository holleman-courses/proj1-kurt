#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras

# Path to your saved Keras model
MODEL_PATH = "open_hand_model.h5"

def main():
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    # Print the model architecture summary
    print("\nModel summary:\n")
    model.summary()

if __name__ == "__main__":
    main()