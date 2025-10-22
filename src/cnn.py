# =============================================================================
# IMPORTS AND DEPENDENCY MANAGEMENT
# =============================================================================
# Core libraries for data manipulation and visualization
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- near the top of src/cnn.py ---
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "archive-2"   # <— makes ./src/cnn.py robust
FILTER_SIZE_DEFAULT = 32                 # try 8, 16, 32, 64 ...

# Keras/TensorFlow imports for CNN training with graceful fallback
# This allows the script to run even if TensorFlow is not installed
try:
    from tensorflow.keras.utils import to_categorical  # Convert labels to one-hot encoding
    from tensorflow.keras.models import Sequential      # Sequential model architecture
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D  # CNN layers
    from tensorflow.keras.optimizers import Adam       # Adam optimizer
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# Scikit-learn imports for evaluation metrics and cross-validation
try:
    from sklearn.metrics import confusion_matrix        # Confusion matrix for classification evaluation
    from sklearn.model_selection import StratifiedKFold  # K-fold cross-validation maintaining class balance
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# Dataset configuration
DATA_ROOT = "./archive-2"  # Root directory containing class folders with images

# Model and training hyperparameters
TARGET_SIZE = (227, 227)  # Input image dimensions (width, height)
COLOR_MODE = "rgb"        # Color channels: "rgb" for 3 channels, "grayscale" for 1 channel
BATCH_SIZE = 32           # Number of samples processed in each training batch
EPOCHS = 10              # Number of complete passes through the training dataset
MODEL_ARCH = "tutorial"   # CNN architecture choice: "tutorial" (4 Conv2D layers) or "simple" (2 Conv2D layers)

# Dataset filtering and size controls
TRAIN_CLASS_FILTER = ["K", "N_K", "N", "healthy"]  # Specific classes to train on (empty list = all classes)
MAX_SAMPLES_PER_CLASS = None  # Limit samples per class (None = no limit)

# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def load_image_as_array(path, size=(28, 28)):
    """
    Load and preprocess an image file.
    
    Args:
        path (str): Path to the image file
        size (tuple): Target dimensions (width, height) for resizing
        
    Returns:
        np.array: Normalized grayscale image array with values in [0, 1]
    """
    from PIL import Image
    # Load image and convert to grayscale (L mode)
    img = Image.open(path).convert("L")
    # Resize to target dimensions
    img = img.resize(size)
    # Convert to numpy array and normalize pixel values to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def build_dataset_from_folders(root_dir, size=(28, 28), class_filter=None, per_class_limit=None):
    """
    Build a complete dataset from a directory structure where each subdirectory represents a class.
    
    Directory structure expected:
    root_dir/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class2/
        ├── image3.jpg
        └── image4.jpg
    
    Args:
        root_dir (str): Root directory containing class subdirectories
        size (tuple): Target image dimensions for resizing
        class_filter (list): Optional list of class names to include (None = all classes)
        per_class_limit (int): Optional limit on number of samples per class (None = no limit)
        
    Returns:
        tuple: (X, y, class_names)
            - X: numpy array of shape (N, height, width, 1) containing normalized images
            - y: numpy array of shape (N,) containing integer class labels
            - class_names: list of class names in sorted order, index-aligned with y
    """
    # Discover all class directories
    class_names_all = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Apply class filtering if specified
    if class_filter:
        allowed = set(class_filter)
        class_names = [d for d in class_names_all if d in allowed]
    else:
        class_names = class_names_all
    
    # Create mapping from class names to integer indices
    class_name_to_index = {name: idx for idx, name in enumerate(class_names)}

    # Load images and labels
    images = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        count_for_class = 0
        
        # Process all image files in the class directory
        for fname in sorted(os.listdir(class_dir)):
            # Skip non-image files
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            img_path = os.path.join(class_dir, fname)
            try:
                # Load and preprocess the image
                img_arr = load_image_as_array(img_path, size=size)
            except Exception:
                # Skip corrupted or unreadable images
                continue
            
            images.append(img_arr)
            labels.append(class_name_to_index[class_name])
            count_for_class += 1
            
            # Stop if we've reached the per-class limit
            if per_class_limit is not None and count_for_class >= per_class_limit:
                break

    if not images:
        raise RuntimeError(f"No images found under {root_dir}")

    # Convert lists to numpy arrays with proper shapes
    X = np.stack(images, axis=0)  # Stack images: (N, height, width)
    X = X[..., np.newaxis]        # Add channel dimension: (N, height, width, 1)
    y = np.array(labels, dtype=np.int64)  # Convert labels to numpy array
    return X, y, class_names

def build_dataframe_from_folders(root_dir, class_filter=None, per_class_limit=None):
    """
    Build a pandas DataFrame containing file paths and class labels.
    
    This function creates a DataFrame suitable for use with Keras ImageDataGenerator,
    which can efficiently load images on-demand during training.
    
    Args:
        root_dir (str): Root directory containing class subdirectories
        class_filter (list): Optional list of class names to include
        per_class_limit (int): Optional limit on number of samples per class
        
    Returns:
        pd.DataFrame: DataFrame with columns ["path", "class"] containing file paths and labels
    """
    records = []
    
    # Iterate through all class directories
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Apply class filtering
        if class_filter and class_name not in set(class_filter):
            continue
        
        count_for_class = 0
        
        # Process all image files in the class directory
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Add record with file path and class label
            records.append({
                "path": os.path.join(class_dir, fname),
                "class": class_name,
            })
            count_for_class += 1
            
            # Stop if we've reached the per-class limit
            if per_class_limit is not None and count_for_class >= per_class_limit:
                break
    
    return pd.DataFrame.from_records(records, columns=["path", "class"])

# =============================================================================
# CNN MODEL ARCHITECTURES
# =============================================================================

def build_keras_cnn(input_shape, num_classes):
    """
    Build a simple CNN architecture with two convolutional blocks.
    
    Architecture:
    - Conv2D(8 filters, 5x5 kernel) -> MaxPool2D -> Dropout(0.25)
    - Conv2D(16 filters, 3x3 kernel) -> MaxPool2D -> Dropout(0.25)
    - Flatten -> Dense(256) -> Dropout(0.5) -> Dense(num_classes)
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model ready for training
    """
    model = Sequential()
    
    # First convolutional block: Extract low-level features
    model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))  # Reduce spatial dimensions
    model.add(Dropout(0.25))  # Prevent overfitting
    
    # Second convolutional block: Extract higher-level features
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # Further reduce dimensions
    model.add(Dropout(0.25))  # Prevent overfitting
    
    # Classification head: Convert features to class predictions
    model.add(Flatten())  # Convert 2D feature maps to 1D vector
    model.add(Dense(256, activation='relu'))  # Fully connected layer
    model.add(Dropout(0.5))  # High dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with class probabilities
    
    # Compile model with Adam optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_tutorial_cnn(input_shape, num_classes):
    """
    Build a deeper CNN architecture with four convolutional blocks.
    
    Architecture:
    - Conv2D(32, 3x3) -> MaxPool2D -> Conv2D(32, 3x3) -> MaxPool2D
    - Conv2D(32, 3x3) -> MaxPool2D -> Conv2D(32, 3x3) -> MaxPool2D
    - Flatten -> Dense(128) -> Dense(num_classes)
    
    This architecture is deeper than the simple CNN and may capture more complex patterns,
    but requires more computational resources and data to train effectively.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model ready for training
    """
    model = Sequential([
        # First convolutional block: Extract basic features
        Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D((2, 2)),
        
        # Second convolutional block: Extract more complex features
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        
        # Third convolutional block: Extract high-level features
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        
        # Fourth convolutional block: Final feature extraction
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        
        # Classification head
        Flatten(),  # Convert 2D feature maps to 1D vector
        Dense(128, activation='relu'),  # Fully connected layer
        Dense(num_classes, activation='softmax'),  # Output layer with class probabilities
    ])
    
    # Compile model with default Adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =============================================================================
# METRIC HELPERS (Precision, Recall, F1, MCC) – pure NumPy + optional sklearn
# =============================================================================
import math

def _safe_div(n, d):
    return float(n) / float(d) if d != 0 else 0.0

def confusion_matrix_np(y_true, y_pred, labels):
    """Numpy fallback if sklearn isn't available."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def per_class_precision_recall_f1_from_cm(cm: np.ndarray):
    """Rows=true labels, Cols=predicted labels."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)     # true counts per class
    pred_cnt = cm.sum(axis=0).astype(np.float64)    # predicted counts per class

    precision = np.array([_safe_div(tp[i], pred_cnt[i]) for i in range(len(tp))], dtype=np.float64)
    recall    = np.array([_safe_div(tp[i], support[i])  for i in range(len(tp))], dtype=np.float64)
    f1 = np.array([
        _safe_div(2.0 * precision[i] * recall[i], precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
        for i in range(len(tp))
    ], dtype=np.float64)
    return precision, recall, f1, support

def f1_micro_from_cm(cm: np.ndarray) -> float:
    """Micro-F1 = overall accuracy for single-label multiclass."""
    correct = float(np.trace(cm))
    total = float(cm.sum())
    return _safe_div(correct, total)

def f1_macro_from_vec(f1_vec: np.ndarray, support: np.ndarray) -> float:
    """Unweighted mean over classes that actually appear."""
    mask = support > 0
    return float(f1_vec[mask].mean()) if np.any(mask) else 0.0

def f1_weighted_from_vec(f1_vec: np.ndarray, support: np.ndarray) -> float:
    """Support-weighted mean over classes."""
    tot = float(support.sum())
    return float((f1_vec * support).sum() / tot) if tot > 0 else 0.0

def mcc_from_cm(cm: np.ndarray) -> float:
    """
    Multiclass MCC (Gorodkin/Jurman form).
    cm: KxK confusion matrix
    """
    cm = cm.astype(np.float64)
    t_k = cm.sum(axis=1)          # true counts per class (rows)
    p_k = cm.sum(axis=0)          # predicted counts per class (cols)
    c = np.trace(cm)              # correct predictions
    s = cm.sum()                  # total
    sum_pk_tk = float(np.dot(p_k, t_k))
    num = c * s - sum_pk_tk
    den1 = s**2 - np.dot(p_k, p_k)
    den2 = s**2 - np.dot(t_k, t_k)
    den = math.sqrt(den1 * den2) if den1 > 0 and den2 > 0 else 0.0
    return _safe_div(num, den)

# =============================================================================
# MAIN EXECUTION: K-FOLD CROSS-VALIDATION TRAINING
# =============================================================================

if __name__ == "__main__":
    # Check dependencies before proceeding
    if not KERAS_AVAILABLE:
        raise RuntimeError("Keras/TensorFlow is not available. Please install tensorflow to run CNN training.")
    
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for StratifiedKFold. Please install scikit-learn.")

    # =============================================================================
    # DATASET PREPARATION
    # =============================================================================
    # Configure class filtering based on global settings
    class_filter = TRAIN_CLASS_FILTER if TRAIN_CLASS_FILTER else None

    print("Building dataset...")
    # Load all images into memory for stratified splitting
    X, y, class_names = build_dataset_from_folders(
        DATA_ROOT,
        size=(28, 28),  # Note: This will be resized to TARGET_SIZE by ImageDataGenerator
        class_filter=class_filter,
        per_class_limit=MAX_SAMPLES_PER_CLASS,
    )
    num_classes = len(class_names)
    print(f"Total samples: {X.shape[0]}, Classes: {num_classes}")
    
    # Validate dataset requirements
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier. Increase TRAIN_CLASS_FILTER to include another class.")

    # =============================================================================
    # K-FOLD CROSS-VALIDATION SETUP
    # =============================================================================
    # Configure stratified K-fold cross-validation
    # StratifiedKFold ensures each fold maintains the same class distribution as the original dataset
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []  # Store accuracy for each fold
    last_fold_artifacts = {}  # Store artifacts from the last fold for final evaluation

    # Determine number of channels based on color mode
    channels = 1 if COLOR_MODE == "grayscale" else 3
    batch_size = BATCH_SIZE
    epochs = EPOCHS

    # Build a master DataFrame for efficient data loading during training
    # This DataFrame will be used by ImageDataGenerator for on-demand image loading
    df_all = build_dataframe_from_folders(
        DATA_ROOT,
        class_filter=class_filter,
        per_class_limit=MAX_SAMPLES_PER_CLASS,
    )
    print(f"DataFrame built: {len(df_all)} rows, columns={list(df_all.columns)}")

    # =============================================================================
    # K-FOLD TRAINING LOOP
    # =============================================================================
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")
        
        # Split data into training and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create DataFrames for this fold's train/validation data
        # These will be used by ImageDataGenerator for efficient data loading
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        val_df = df_all.iloc[val_idx].reset_index(drop=True)
        print(f"Fold {fold_idx}: train_df={len(train_df)} rows, val_df={len(val_df)} rows")

        # Ensure categorical encoding is available (ImageDataGenerator handles this, but we check here)
        _ = to_categorical(y_train, num_classes=num_classes)
        _ = to_categorical(y_val, num_classes=num_classes)

        # =============================================================================
        # MODEL CREATION
        # =============================================================================
        # Build CNN model based on architecture choice
        if MODEL_ARCH == "tutorial":
            model = build_tutorial_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], channels), num_classes=num_classes)
        else:
            model = build_keras_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], channels), num_classes=num_classes)
        model.summary()  # Print model architecture summary

        # =============================================================================
        # DATA AUGMENTATION AND GENERATORS
        # =============================================================================
        # Create data generators with augmentation for training and without for validation
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,        # Normalize pixel values to [0, 1]
            horizontal_flip=True,     # Randomly flip images horizontally
            rotation_range=5,         # Random rotation up to 5 degrees
            width_shift_range=0.1,    # Random horizontal shift up to 10%
            height_shift_range=0.1,   # Random vertical shift up to 10%
            zoom_range=0.1,           # Random zoom up to 10%
        )
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only normalization for validation

        # Create generators that load images on-demand from DataFrames
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=None,           # Paths in DataFrame are already absolute
            x_col="path",            # Column containing image file paths
            y_col="class",           # Column containing class labels
            class_mode="categorical", # Convert labels to one-hot encoding
            classes=class_names,     # List of class names in order
            target_size=TARGET_SIZE,  # Resize images to this size
            color_mode=COLOR_MODE,   # Number of color channels
            batch_size=batch_size,   # Number of images per batch
            shuffle=True,            # Shuffle training data
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
            shuffle=False,           # Don't shuffle validation data
        )

        # Calculate steps per epoch based on dataset size and batch size
        steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
        validation_steps = int(np.ceil(len(val_df) / batch_size))

        # =============================================================================
        # MODEL TRAINING
        # =============================================================================
        print(f"Starting training for fold {fold_idx}...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,  # Show progress during training
        )

        # =============================================================================
        # FOLD EVALUATION AND VISUALIZATION
        # =============================================================================
        # Plot training vs validation accuracy curve for this fold
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.subplots(figsize=(10, 5))
            plt.subplot(1, 1, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.5, 1])  # Focus on accuracy range 50%-100%
            plt.legend(loc='lower right')
            plt.title(f'Training Progress - Fold {fold_idx}')
            plt.tight_layout()
            plt.savefig(f'accuracy_curve_fold_{fold_idx}.png', dpi=200, bbox_inches='tight')
            plt.show()

        # Evaluate model performance on validation set
        val_loss, val_acc = model.evaluate(val_generator, verbose=0, steps=validation_steps)
        fold_accuracies.append(val_acc)
        print(f"Fold {fold_idx} validation accuracy: {val_acc:.4f}")

        # Save artifacts from the last fold for final evaluation and visualization
        if fold_idx == n_splits:
            last_fold_artifacts = {
                'model': model,              # Trained model
                'val_generator': val_generator,  # Validation data generator
                'validation_steps': validation_steps,  # Number of validation steps
                'history': history,           # Training history
            }

    # =============================================================================
    # FINAL RESULTS AND EVALUATION
    # =============================================================================
    # Calculate and display aggregate metrics across all folds
    fold_accuracies = np.array(fold_accuracies, dtype=np.float32)
    print(f"\nK-Fold Cross-Validation Results over {n_splits} folds:")
    print(f"  Individual fold accuracies: {fold_accuracies}")
    print(f"  Mean accuracy: {fold_accuracies.mean():.4f}  Std: {fold_accuracies.std():.4f}")
    print(f"  Min accuracy: {fold_accuracies.min():.4f}  Max accuracy: {fold_accuracies.max():.4f}")

    # =============================================================================
    # FINAL VISUALIZATION AND ANALYSIS
    # =============================================================================
    # Extract artifacts from the last fold for detailed analysis
    model = last_fold_artifacts.get('model')
    val_generator = last_fold_artifacts.get('val_generator')
    validation_steps = last_fold_artifacts.get('validation_steps')
    history = last_fold_artifacts.get('history')

    # Plot validation loss curve from the last fold
    if history is not None and 'val_loss' in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['val_loss'], color='b', label='Validation Loss')
        plt.title('Validation Loss Curve (Last Fold)')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Generate and visualize confusion matrix for the last fold
    if SKLEARN_AVAILABLE and model is not None and val_generator is not None:
        print("\nGenerating confusion matrix for final evaluation...")
        
        # Make predictions on validation set
        val_pred_probs = model.predict(val_generator, verbose=0, steps=validation_steps)
        y_val_pred = np.argmax(val_pred_probs, axis=1)  # Convert probabilities to class predictions
        y_val_true = val_generator.classes[:len(y_val_pred)]  # Get true labels
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true=y_val_true, y_pred=y_val_pred)

        # Create confusion matrix visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create table with confusion matrix values
        tbl = ax.table(
            cellText=cm, 
            rowLabels=class_names, 
            colLabels=class_names, 
            loc='center',
            cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        
        plt.title('Confusion Matrix (Last Fold Validation Set)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_table.png', dpi=200, bbox_inches='tight')
        plt.show()

        # Print confusion matrix for detailed analysis
        print('\nConfusion Matrix (Last Fold Validation):')
        print('Rows = True Labels, Columns = Predicted Labels')
        print(f'Class names: {class_names}')
        print(cm)
        
        # Calculate and display per-class metrics
        print('\nPer-Class Performance Analysis:')
        for i, class_name in enumerate(class_names):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f'  {class_name}:')
            print(f'    Precision: {precision:.4f}')
            print(f'    Recall: {recall:.4f}')
            print(f'    F1-Score: {f1_score:.4f}')
