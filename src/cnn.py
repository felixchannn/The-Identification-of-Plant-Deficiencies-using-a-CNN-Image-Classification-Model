import os
import numpy as np
from PIL import Image
from conv import Conv3x3
import matplotlib.pyplot as plt

DATA_ROOT = "/Users/matt/Documents/usyd_yr2_sem2/mentorship program/Interstellar/archive-2"

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

if __name__ == "__main__":
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
        #plt.imshow(image, cmap='gray')
        #plt.title(f"{label}")
        #plt.axis('off')
        #plt.show()
        output = conv.forward(image)
        print(f"[{i}/{len(selected)}] Sample: {image_path}")
        print(f"Label: {label}")
        print(f"Conv output shape: {output.shape}")  # (26, 26, 8)
        