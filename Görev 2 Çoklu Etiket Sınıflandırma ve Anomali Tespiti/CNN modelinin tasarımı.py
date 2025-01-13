import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Modeli kurma
model = Sequential([
    # 1. Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    
    # 2. Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3. Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten ve Fully Connected Layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Overfitting'i engellemek için Dropout ekliyoruz
    
    # Çıkış Layer'ı: Çoklu etiketler için softmax activation
    Dense(y_train.shape[1], activation='softmax')
])

# Modeli derleyelim
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin özetini görüntüleyelim
model.summary()
