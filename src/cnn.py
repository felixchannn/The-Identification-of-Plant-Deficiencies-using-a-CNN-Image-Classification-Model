import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras/TensorFlow for CNN training
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# Evaluation utilities
try:
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

DATA_ROOT = "./archive-2"

# Image/generator/model config for K-Fold training
TARGET_SIZE = (227, 227)
COLOR_MODE = "rgb"  # "grayscale" or "rgb"
BATCH_SIZE = 32
EPOCHS = 10
MODEL_ARCH = "tutorial"  # "tutorial" uses 4x Conv2D stack; "simple" uses build_keras_cnn

# Dataset size controls
TRAIN_CLASS_FILTER = ["eggplant__K", "cucumber__K", "tomato__K"]  # [] means use all classes
MAX_SAMPLES_PER_CLASS = None

def load_image_as_array(path, size=(28, 28)):
    """Load and normalize image to grayscale and resize it."""
    from PIL import Image
    img = Image.open(path).convert("L")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def build_dataset_from_folders(root_dir, size=(28, 28), class_filter=None, per_class_limit=None):
    """
    Loads all images under root_dir, returning (X, y, class_names).
    - X: numpy array of shape (N, 28, 28, 1)
    - y: integer class indices of shape (N,)
    - class_names: sorted list of class folder names, index-aligned with y
    """
    class_names_all = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    if class_filter:
        allowed = set(class_filter)
        class_names = [d for d in class_names_all if d in allowed]
    else:
        class_names = class_names_all
    
    class_name_to_index = {name: idx for idx, name in enumerate(class_names)}

    images = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        count_for_class = 0
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_dir, fname)
            try:
                img_arr = load_image_as_array(img_path, size=size)
            except Exception:
                continue
            images.append(img_arr)
            labels.append(class_name_to_index[class_name])
            count_for_class += 1
            if per_class_limit is not None and count_for_class >= per_class_limit:
                break

    if not images:
        raise RuntimeError(f"No images found under {root_dir}")

    X = np.stack(images, axis=0)  # (N, 28, 28)
    X = X[..., np.newaxis]        # (N, 28, 28, 1)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names

def build_dataframe_from_folders(root_dir, class_filter=None, per_class_limit=None):
    """
    Builds a DataFrame with columns ["path", "class"] from a directory tree
    where each class is a subfolder containing images.
    """
    records = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_filter and class_name not in set(class_filter):
            continue
        count_for_class = 0
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            records.append({
                "path": os.path.join(class_dir, fname),
                "class": class_name,
            })
            count_for_class += 1
            if per_class_limit is not None and count_for_class >= per_class_limit:
                break
    return pd.DataFrame.from_records(records, columns=["path", "class"])

def build_keras_cnn(input_shape, num_classes):
    """Simple CNN architecture."""
    model = Sequential()
    # First conv block
    model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Second conv block
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    # Classifier
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_tutorial_cnn(input_shape, num_classes):
    """Tutorial-style CNN architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    if not KERAS_AVAILABLE:
        raise RuntimeError("Keras/TensorFlow is not available. Please install tensorflow to run CNN training.")
    
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for StratifiedKFold. Please install scikit-learn.")

    # Configure class filtering
    class_filter = TRAIN_CLASS_FILTER if TRAIN_CLASS_FILTER else None

    print("Building dataset...")
    X, y, class_names = build_dataset_from_folders(
        DATA_ROOT,
        size=(28, 28),
        class_filter=class_filter,
        per_class_limit=MAX_SAMPLES_PER_CLASS,
    )
    num_classes = len(class_names)
    print(f"Total samples: {X.shape[0]}, Classes: {num_classes}")
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier. Increase TRAIN_CLASS_FILTER to include another class.")

    # Stratified K-Fold Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    last_fold_artifacts = {}

    channels = 1 if COLOR_MODE == "grayscale" else 3
    batch_size = BATCH_SIZE
    epochs = EPOCHS

    # Build a master DataFrame of all samples for convenience/analysis
    df_all = build_dataframe_from_folders(
        DATA_ROOT,
        class_filter=class_filter,
        per_class_limit=MAX_SAMPLES_PER_CLASS,
    )
    print(f"DataFrame built: {len(df_all)} rows, columns={list(df_all.columns)}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Per-fold train/val dataframes
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        val_df = df_all.iloc[val_idx].reset_index(drop=True)
        print(f"Fold {fold_idx}: train_df={len(train_df)} rows, val_df={len(val_df)} rows")

        # Generators will handle labels; to_categorical here ensures class count availability
        _ = to_categorical(y_train, num_classes=num_classes)
        _ = to_categorical(y_val, num_classes=num_classes)

        if MODEL_ARCH == "tutorial":
            model = build_tutorial_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], channels), num_classes=num_classes)
        else:
            model = build_keras_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], channels), num_classes=num_classes)
        model.summary()

        # ImageDataGenerators and dataframe-based generators
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            horizontal_flip=True,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=None,
            x_col="path",
            y_col="class",
            class_mode="categorical",
            classes=class_names,
            target_size=TARGET_SIZE,
            color_mode=COLOR_MODE,
            batch_size=batch_size,
            shuffle=True,
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=None,
            x_col="path",
            y_col="class",
            class_mode="categorical",
            classes=class_names,
            target_size=TARGET_SIZE,
            color_mode=COLOR_MODE,
            batch_size=batch_size,
            shuffle=False,
        )

        steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
        validation_steps = int(np.ceil(len(val_df) / batch_size))

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
        )

        # Plot training vs validation accuracy for this fold
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.subplots(figsize=(10, 5))
            plt.subplot(1, 1, 1)
            plt.plot(history.history['accuracy'], label='accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.5, 1])
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(f'accuracy_curve_fold_{fold_idx}.png', dpi=200, bbox_inches='tight')
            plt.show()

        val_loss, val_acc = model.evaluate(val_generator, verbose=0, steps=validation_steps)
        fold_accuracies.append(val_acc)
        print(f"Fold {fold_idx} validation accuracy: {val_acc:.4f}")

        # Save artifacts from the last fold for optional reporting/predictions
        if fold_idx == n_splits:
            last_fold_artifacts = {
                'model': model,
                'val_generator': val_generator,
                'validation_steps': validation_steps,
                'history': history,
            }

    # Aggregate metrics across folds
    fold_accuracies = np.array(fold_accuracies, dtype=np.float32)
    print(f"\nK-Fold results over {n_splits} folds:")
    print(f"  Accuracies: {fold_accuracies}")
    print(f"  Mean accuracy: {fold_accuracies.mean():.4f}  Std: {fold_accuracies.std():.4f}")

    # plot last fold loss curve and confusion matrix
    model = last_fold_artifacts.get('model')
    val_generator = last_fold_artifacts.get('val_generator')
    validation_steps = last_fold_artifacts.get('validation_steps')
    history = last_fold_artifacts.get('history')

    if history is not None and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], color='b', label='validation loss')
        plt.title('Validation Loss (last fold)')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    if SKLEARN_AVAILABLE and model is not None and val_generator is not None:
        val_pred_probs = model.predict(val_generator, verbose=0, steps=validation_steps)
        y_val_pred = np.argmax(val_pred_probs, axis=1)
        y_val_true = val_generator.classes[:len(y_val_pred)]
        cm = confusion_matrix(y_true=y_val_true, y_pred=y_val_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        tbl = ax.table(cellText=cm, rowLabels=class_names, colLabels=class_names, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        plt.title('Confusion Matrix (last fold)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_table.png', dpi=200, bbox_inches='tight')
        plt.show()

        print('Confusion matrix (last fold validation):')
        print(cm)
