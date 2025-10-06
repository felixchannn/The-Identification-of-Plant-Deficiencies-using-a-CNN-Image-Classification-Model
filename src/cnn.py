import os
import numpy as np
from PIL import Image
from conv import Conv3x3
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Optional: Keras/TensorFlow for training a CNN similar to your tutorial
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# Optional: evaluation utilities
try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Optional: seaborn heatmap for nicer confusion matrix visualization
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

DATA_ROOT = "./archive-2"

# Mode switch:
#   - "demo": run the hand-rolled Conv3x3 forward pass on selected images (existing behavior)
#   - "train": build a dataset from all class folders and train a Keras CNN
MODE = "train"  # change to "train" to use the Keras model

# Selection controls: choose by class folder and either index or count
SELECT_CLASS = "eggplant__N"
SELECT_INDEX = 0     # used when selecting a single sample by index
SELECT_COUNT = 30     # when >0, selects up to this many samples of SELECT_CLASS

def list_image_paths_and_labels(root_dir):
    samples = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), class_name))
    return samples


def load_image_as_array(path, size=(28, 28)): #normalise the image to grayscale and resize it to 28x28
    img = Image.open(path).convert("L")
    img = img.resize(size) #resize the image to 28x28
    arr = np.array(img, dtype=np.float32) / 255.0 #div by 255 to normalise the image to grayscale
    return arr

def build_dataset_from_folders(root_dir, size=(28, 28)):
    """
    Loads all images under root_dir, returning (X, y, class_names).
    - X: numpy array of shape (N, 28, 28, 1)
    - y: integer class indices of shape (N,)
    - class_names: sorted list of class folder names, index-aligned with y
    """
    class_names = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    class_name_to_index = {name: idx for idx, name in enumerate(class_names)}

    images = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
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

    if not images:
        raise RuntimeError(f"No images found under {root_dir}")

    X = np.stack(images, axis=0)  # (N, 28, 28)
    X = X[..., np.newaxis]        # (N, 28, 28, 1)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names

def simple_train_val_split(num_samples, train_ratio=0.8, shuffle=True, seed=42):
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    split = int(num_samples * train_ratio)
    return indices[:split], indices[split:]

def build_keras_cnn(input_shape, num_classes):
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

if __name__ == "__main__":
    if MODE == "train":
        if not KERAS_AVAILABLE:
            raise RuntimeError("Keras/TensorFlow is not available. Please install tensorflow to use MODE='train'.")

        print("Building dataset...")
        X, y, class_names = build_dataset_from_folders(DATA_ROOT, size=(28, 28))
        num_classes = len(class_names)
        print(f"Total samples: {X.shape[0]}, Classes: {num_classes}")

        train_idx, val_idx = simple_train_val_split(X.shape[0], train_ratio=0.8)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)

        model = build_keras_cnn(input_shape=(28, 28, 1), num_classes=num_classes)
        model.summary()

        # Data augmentation to reduce overfitting
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=5,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
        )
        datagen.fit(X_train)

        print("Training model with data augmentation...")
        batch_size = 250
        epochs = 10
        steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))
        history = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
        )

        val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"Validation accuracy: {val_acc:.4f}")

        # Plot validation loss over epochs
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], color='b', label='validation loss')
            plt.title('Test Loss')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        # Confusion matrix on validation set
        if SKLEARN_AVAILABLE:
            val_pred_probs = model.predict(X_val, batch_size=250, verbose=0)
            y_val_pred = np.argmax(val_pred_probs, axis=1)
            cm = confusion_matrix(y_true=y_val, y_pred=y_val_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            tbl = ax.table(cellText=cm, rowLabels=class_names, colLabels=class_names, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.5)
            plt.title('Confusion Matrix')
            # Save for headless environments and also show interactively if available
            plt.tight_layout()
            plt.savefig('confusion_matrix_table.png', dpi=200, bbox_inches='tight')
            plt.show()
            
            print('Confusion matrix (validation):')
            print(cm)

            # Seaborn heatmap (prettier)
            if SEABORN_AVAILABLE:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix (Heatmap)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig('confusion_matrix_heatmap.png', dpi=200, bbox_inches='tight')
                plt.show()

            # Alternative confusion matrix using one-hot labels and a different style
            if SEABORN_AVAILABLE:
                Y_pred = model.predict(X_val, batch_size=250, verbose=0)
                Y_pred_classes = np.argmax(Y_pred, axis=1)
                Y_true = np.argmax(y_val_cat, axis=1)
                confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
                f, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap='Greens',
                            linecolor='gray', fmt='.1f', ax=ax,
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig('confusion_matrix_heatmap_greens.png', dpi=200, bbox_inches='tight')
                plt.show()
        else:
            print('sklearn not available; skipping confusion matrix.')

        # Optional: run inference on the same selected-image controls used in demo mode
        try:
            samples_all = list_image_paths_and_labels(DATA_ROOT)
            class_samples = [s for s in samples_all if s[1] == SELECT_CLASS]
            if class_samples:
                if SELECT_COUNT and SELECT_COUNT > 0:
                    selected = class_samples[:SELECT_COUNT]
                else:
                    index = max(0, min(SELECT_INDEX, len(class_samples) - 1))
                    selected = [class_samples[index]]

                print(f"\nRunning predictions on {len(selected)} selected sample(s) from '{SELECT_CLASS}'")
                for i, (image_path, true_label_name) in enumerate(selected, start=1):
                    img_arr = load_image_as_array(image_path, size=(28, 28))
                    img_arr = img_arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)
                    probs = model.predict(img_arr, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_name = class_names[pred_idx]
                    pred_conf = float(probs[pred_idx])
                    print(f"[{i}/{len(selected)}] {image_path}")
                    print(f"  True label: {true_label_name}")
                    print(f"  Predicted: {pred_name} (p={pred_conf:.3f})")
            else:
                print(f"No samples found for SELECT_CLASS='{SELECT_CLASS}' to run predictions.")
        except Exception as e:
            print(f"Skipping selected-sample predictions due to error: {e}")

    else:
        samples = list_image_paths_and_labels(DATA_ROOT)
        if not samples:
            raise RuntimeError(f"No images found under {DATA_ROOT}")

        print(f"Total images found: {len(samples)}")

        # Select image(s) by class
        if not SELECT_CLASS:
            raise RuntimeError("SELECT_CLASS must be set to a folder name under archive-2")

        class_samples = [s for s in samples if s[1] == SELECT_CLASS]
        if not class_samples:
            raise RuntimeError(f"No images found for class '{SELECT_CLASS}' under {DATA_ROOT}")

        if SELECT_COUNT and SELECT_COUNT > 0:
            selected = class_samples[:SELECT_COUNT]
            print(f"Selected {len(selected)} samples from class '{SELECT_CLASS}'")
        else:
            index = max(0, min(SELECT_INDEX, len(class_samples) - 1))
            selected = [class_samples[index]]

        conv = Conv3x3(8)
        for i, (image_path, label) in enumerate(selected, start=1):
            image = load_image_as_array(image_path, size=(28, 28))
            plt.imshow(image, cmap='gray')
            plt.title(f"{label}")
            plt.axis('off')
            plt.show()

            output = conv.forward(image)
            print(f"[{i}/{len(selected)}] Sample: {image_path}")
            print(f"Label: {label}")
            print(f"Conv output shape: {output.shape}")  # (26, 26, 8)
            fig, ax = plt.subplots(figsize=(8, 6))
            
        