# this will be the page for the initial framework
import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from conv import Conv3x3

DATA_ROOT = "/Users/misty/Desktop/extracurriculars/SUDATAxSPDSC mentoring/Interstellar/archive"

# Selection controls: choose by class folder and either index or count
SELECT_CLASS = "eggplant__N"
SELECT_INDEX = 0     # used when selecting a single sample by index
SELECT_COUNT = 30     # when >0, selects up to this many samples of SELECT_CLASS

conv = Conv3x3(32) # 8 filters of size 3x3

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


def load_image_as_array(path, size=(128, 128)): #normalise the image to grayscale and resize it to 28x28
    img = Image.open(path).convert("RGB") # CHANGED TO RGB FROM L
    img = img.resize(size, resample=Image.LANCZOS) #resize the image to 28x28 # high quality sampling filter something like bicubic
    arr = np.array(img, dtype=np.float32) / 255.0 #div by 255 to normalise the image to grayscale
    return arr

def maxpool2x2(x): # this is a pooling layer. throws away weaker details 
    H, W, C = x.shape # height, width, channels
    H2, W2 = H // 2, W // 2 # new height and width after pooling
    pooled = np.zeros((H2, W2, C)) # initialize the pooled output
    for i in range(H2): # iterate over the new height
        for j in range(W2): # iterate over the new width
            patch = x[2*i:2*i+2, 2*j:2*j+2, :] # 2x2 patch
            pooled[i, j] = patch.max(axis=(0, 1)) # max value in the patch for each channel
    return pooled # return the pooled output 


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
        
    for i, (image_path, label) in enumerate(selected, start=1):
        image = load_image_as_array(image_path, size=(128, 128))
        image_gray = image.mean(axis=2)  # shape: (96,96)
        plt.imshow(image)
        plt.title(f"{label}")
        plt.axis('off')
        plt.show()
        output_before = conv.forward(image_gray)
        print("Before pooling:", output_before.shape)
        output = np.maximum(0, output_before)   # ReLU
        output = maxpool2x2(output)             # 2×2 pooling
        print("After pooling:", output.shape)
        print(f"[{i}/{len(selected)}] Sample: {image_path}")
        print(f"Label: {label}")
        
