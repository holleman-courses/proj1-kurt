import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import optimizers


#you gotta change this to the path with the 10 images required
DATA_DIR     = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\preliminary_dataset"
IMG_SIZE     = (96, 96)
BATCH_SIZE   = 64
EPOCHS       = 5


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




def build_model():
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

def train_model(model, train_gen, val_gen, class_weight):
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
    )
    model.save("open_hand_model_LIMITED.h5")
    return model

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
    model = train_model(model, train_gen, val_gen, class_weight)

    # grab one validation batch
    x_val, y_val = next(val_gen)
    # raw model outputs
    probs = model.predict(x_val).flatten()
    print("First 10 probs:", np.round(probs[:10], 3))
    print("First 10 true labels:", y_val[:10])
    # how many of this batch it calls “hand” vs “non_hands”
    print("Predicted hands:", np.sum(probs > 0.5),
        "  non_hands:", np.sum(probs <= 0.5))
