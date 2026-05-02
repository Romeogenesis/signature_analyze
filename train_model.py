import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import os

# Отключаем лишние логи
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*60)
print("🚀 ЗАПУСК ОБУЧЕНИЯ (БЕЗ LAMBDA СЛОЕВ)")
print("="*60)

# Проверка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU обнаружено: {len(gpus)}")
else:
    print("⚠️ Обучение на CPU")

# ============================================================
# 1. ГЕНЕРАТОР ДАННЫХ
# ============================================================
def generate_signature_image(seed=None):
    if seed is not None:
        random.seed(seed)
    
    img = np.ones((150, 150), dtype=np.uint8) * 255
    start_x = random.randint(20, 40)
    start_y = random.randint(60, 100)
    num_points = random.randint(5, 9)
    thickness = random.randint(2, 4)
    
    points = [(start_x, start_y)]
    for _ in range(num_points - 1):
        nx = random.randint(50, 130)
        ny = random.randint(30, 120)
        points.append((nx, ny))
    
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], 0, thickness)
    
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return np.clip(img, 0, 255)

def create_dataset(n_pairs=5000):
    print(f"\n🎨 Генерация {n_pairs} пар...")
    X1_list, X2_list, y_list = [], [], []
    
    for i in tqdm(range(n_pairs), desc="Генерация"):
        base_seed = random.randint(0, 100000)
        img1 = generate_signature_image(base_seed)
        
        if i % 2 == 0:
            # Положительная пара (тот же человек)
            img2 = generate_signature_image(base_seed)
            shift_x, shift_y = random.randint(-3, 3), random.randint(-3, 3)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img2 = cv2.warpAffine(img2, M, (150, 150), borderMode=cv2.BORDER_REPLICATE)
            noise = np.random.normal(0, 15, img2.shape).astype(np.uint8)
            img2 = cv2.add(img2, noise)
            img2 = np.clip(img2, 0, 255)
            label = 1
        else:
            # Отрицательная пара (разные люди)
            new_seed = random.randint(0, 100000)
            while abs(new_seed - base_seed) < 1000:
                new_seed = random.randint(0, 100000)
            img2 = generate_signature_image(new_seed)
            label = 0
        
        X1_list.append(np.expand_dims(img1.astype(np.float32) / 255.0, axis=-1))
        X2_list.append(np.expand_dims(img2.astype(np.float32) / 255.0, axis=-1))
        y_list.append(label)
    
    return np.array(X1_list), np.array(X2_list), np.array(y_list)

# ============================================================
# 2. МОДЕЛЬ (БЕЗ LAMBDA)
# ============================================================
def create_model():
    input_img = layers.Input(shape=(150, 150, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    encoded = layers.Dense(128, activation='sigmoid')(x)
    
    encoder = Model(input_img, encoded, name='encoder')
    
    # Сиамская часть
    input_a = layers.Input(shape=(150, 150, 1))
    input_b = layers.Input(shape=(150, 150, 1))
    
    feat_a = encoder(input_a)
    feat_b = encoder(input_b)
    
    # ВМЕСТО LAMBDA: Используем Subtract для разницы, затем Lambda только для abs (безопасно)
    # Или лучше: просто возведем в квадрат разницу, это тоже работает для метрики расстояния
    # Но самый надежный способ без lambda функций - использовать слой Subtract и потом Square
    
    diff = layers.Subtract()([feat_a, feat_b])
    # Абсолютное значение можно эмулировать или просто использовать квадрат разницы (Euclidean distance squared)
    # Для бинарной классификации квадрат разницы часто работает даже лучше
    merged = layers.Multiply()([diff, diff]) # (a-b)^2
    
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, encoder

# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================
def main():
    X1, X2, y = create_dataset(n_pairs=6000)
    
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {len(X1_train)}, Val: {len(X1_val)}")
    
    print("\n🏗️ Построение модели...")
    model, encoder = create_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('signature_model.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    print("\n🔥 Обучение...")
    model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n💾 Сохранение...")
    model.save('signature_model.keras')
    encoder.save('signature_encoder.keras')
    
    print("\n✅ ГОТОВО! Модель сохранена в signature_model.keras")

if __name__ == "__main__":
    main()