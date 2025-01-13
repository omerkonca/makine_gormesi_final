import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Görüntüleri ve etiketleri yükle
def load_data(image_dir, labels_file):
    images = []
    labels = []
    
    with open(labels_file, 'r') as file:
        label_map = {line.split()[0]: line.split()[1] for line in file.readlines()}
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))  # Resmi 128x128 boyutuna getir
            images.append(image)
            label = label_map.get(filename, None)
            if label:
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Etiketleri sayısal verilere dönüştür
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # One-hot encoding etiketleri
    labels = to_categorical(labels, num_classes=len(label_encoder.classes_))
    
    return images, labels







image_dir = 'C:/path/to/your/images'
labels_file = 'C:/path/to/your/labels.txt'

X, y = load_data(image_dir, labels_file)

# Verileri eğitim ve test setlerine ayıralım
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri uygun formatta yeniden şekillendirelim
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)  # (örneğin 128x128 boyutunda gri tonlama görüntüler)
X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)









# Modeli eğitelim
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Eğitim sonuçlarını görselleştirelim
import matplotlib.pyplot as plt

# Eğitim doğruluğu ve kaybı
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

# Eğitim kaybı ve doğrulama kaybı
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()










# Test setinde modelin doğruluğunu ölçelim
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy}")

# Modeli kaydedelim
model.save('multi_label_model.h5')
