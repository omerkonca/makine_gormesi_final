import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize


data_path = r"C:\Users\omerk\Documents\GitHub\makine_görmesi_final\BraTS2020_ValidationData"


# Görüntülerin ve maskelerin dosya yolu
image_dir = os.path.join(data_path, "images")
mask_dir = os.path.join(data_path, "masks")

# Örnek bir görüntü ve maskeyi yükleyip görselleşmesi
example_image = imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
example_mask = imread(os.path.join(mask_dir, os.listdir(mask_dir)[0]))

# Görüntüleme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Brain MRI")
plt.imshow(example_image, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(example_mask, cmap="gray")
plt.show()
