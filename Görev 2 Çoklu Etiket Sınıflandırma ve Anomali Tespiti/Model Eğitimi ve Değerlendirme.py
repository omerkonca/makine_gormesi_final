import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

# Veri seti yolu
image_dir = 'path_to_your_image_directory'
label_dir = 'path_to_your_label_directory'

# Görüntüleri yükleme
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # veya cv2.IMREAD_COLOR
        images.append(img)
    return np.array(images)

# Etiketleri yükleme (örnek etiket yapısına göre)
def load_labels(label_dir):
    labels = []
    for filename in os.listdir(label_dir):
        label_path = os.path.join(label_dir, filename)
        with open(label_path, 'r') as f:
            labels.append(f.read().strip())  # Etiketleri uygun şekilde okuyun
    return np.array(labels)

# Görüntüleri ve etiketleri yükle
images = load_images(image_dir)
labels = load_labels(label_dir)

# Görüntüleri normalize etme
images = images / 255.0

# Etiketleri ikili sınıflandırma için dönüştürme
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Görüntü Augmentasyonu
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(lb.classes_), activation='sigmoid')  # Çoklu etiketler için sigmoid
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Değerlendirme
model.evaluate(X_test, y_test)
