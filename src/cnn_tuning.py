import os, math, tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# ========= 1) PATHS & BASIC SETTINGS =========
DATA_ROOT = "/Users/misty/Desktop/extracurriculars/SUDATAxSPDSC mentoring/Interstellar/archive"  # <-- put YOUR root
IMG_SIZE   = (128, 128)     # you can raise to (224,224) if you have GPU time
BATCH_SIZE = 32
COLOR_MODE = "rgb"          # change to "grayscale" if your images are grayscale
CHANNELS   = 3 if COLOR_MODE=="rgb" else 1
SEED       = 42

# ========= 2) LOAD DATASETS FROM FOLDERS =========
# Folder layout must be: root/class_a/*.jpg, root/class_b/*.png, ...
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_ROOT,
    validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode=COLOR_MODE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_ROOT,
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode=COLOR_MODE
)

num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Performance niceties
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# Optional: separate test split (here we carve from val for a quick demo)
test_ds = val_ds.take( math.ceil(0.25 * tf.data.experimental.cardinality(val_ds).numpy()) )
val_ds  = val_ds.skip(  math.ceil(0.25 * tf.data.experimental.cardinality(val_ds).numpy()) )

# ========= 3) DATA AUGMENTATION PIPELINE (helps generalization) =========
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")

# ========= 4) MODEL FACTORY WITH TUNABLE HYPERPARAMS =========
def build_model(hp: kt.HyperParameters):
    inputs = keras.Input(shape=(*IMG_SIZE, CHANNELS))

    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Tune number of conv blocks (2–4)
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=4, step=1)
    filters0   = hp.Choice("filters0", values=[32, 64, 96, 128])

    f = filters0
    for b in range(num_blocks):
        x = layers.Conv2D(f, kernel_size=hp.Choice(f"k{b}", values=[3,5]), padding="same", activation="relu")(x)
        x = layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)
        # optionally grow filters each block
        f = hp.Choice(f"grow{b}", values=[f, f*2])

    x = layers.Dropout(hp.Float("dropout_cnn", 0.0, 0.4, step=0.1))(x)
    x = layers.Flatten()(x)

    # Tune dense layer size
    dense_units = hp.Int("dense_units", min_value=64, max_value=512, step=64)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(hp.Float("dropout_dense", 0.0, 0.5, step=0.1))(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ========= 5) SET UP THE TUNER (RandomSearch is simple & fast) =========
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=12,                # try more for better results if you have time
    executions_per_trial=1,
    overwrite=True,
    directory="kt_logs",
    project_name="plant_disease_cnn"
)

tuner.search_space_summary()

# ========= 6) RUN THE SEARCH =========
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=15,                    # light training for search; keep modest
    callbacks=[early_stop]
)

# ========= 7) RETRIEVE & TRAIN THE BEST MODEL LONGER =========
best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest hyperparameters:")
for k in best_hp.values:
    print(f" - {k}: {best_hp.get(k)}")

best_model = tuner.hypermodel.build(best_hp)

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,                    # now train longer on the best config
    callbacks=[early_stop]
)

# ========= 8) EVALUATE & SAVE =========
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}")

best_model.save("best_cnn_tuned.keras")   # SavedModel/Keras format