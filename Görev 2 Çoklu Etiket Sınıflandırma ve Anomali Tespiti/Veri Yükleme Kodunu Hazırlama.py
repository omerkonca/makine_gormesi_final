import os
import cv2
import numpy as np

# Veri seti dizinini burada belirtin
image_dir = 'C:\\Users\\omerk\\path_to_your_image_directory'  # Görüntülerin bulunduğu dizin
label_dir = 'C:\\Users\\omerk\\path_to_your_label_directory'  # Etiketlerin bulunduğu dizin

# Görüntüleri yüklemek için fonksiyon
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Görüntüleri gri tonlamalı yükleyin
        if img is not None:
            images.append(img)
    return np.array(images)

# Etiketleri yüklemek için fonksiyon
def load_labels(label_dir):
    labels = []
    for filename in os.listdir(label_dir):
        label_path = os.path.join(label_dir, filename)
        with open(label_path, 'r') as f:
            labels.append(f.read().strip())  # Etiketleri uygun şekilde okuyun
    return np.array(labels)

# Görüntüleri ve etiketleri yükleyelim
images = load_images(image_dir)
labels = load_labels(label_dir)

# Görüntüleri normalize edelim
images = images / 255.0  # 0-1 arası normalize etme

# Etiketleri ikili sınıflara dönüştürelim
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Veri boyutlarını kontrol edelim
print(f"Loaded {len(images)} images.")
print(f"Loaded {len(labels)} labels.")









import matplotlib.pyplot as plt

# İlk birkaç görseli gösterelim
for i in range(5):
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.show()








from sklearn.model_selection import train_test_split

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Veri boyutlarını kontrol edelim
print(f"Training set size: {X_train.shape}, {y_train.shape}")
print(f"Test set size: {X_test.shape}, {y_test.shape}")









from sklearn.model_selection import train_test_split

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Veri boyutlarını kontrol edelim
print(f"Training set size: {X_train.shape}, {y_train.shape}")
print(f"Test set size: {X_test.shape}, {y_test.shape}")
