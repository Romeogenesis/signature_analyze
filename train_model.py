import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import os

# Отключаем лишние логи TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*60)
print("🚀 ЗАПУСК ОБУЧЕНИЯ НЕЙРОСЕТИ (СИНТЕТИЧЕСКИЕ ДАННЫЕ)")
print("="*60)

# Проверка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU обнаружено: {len(gpus)} устр-в")
else:
    print("⚠️ GPU не найдено, обучение на CPU (может быть медленно)")

# ============================================================
# 1. ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ПОДПИСЕЙ
# ============================================================
def generate_signature_image(seed=None):
    """Генерирует изображение подписи программно."""
    if seed is not None:
        random.seed(seed)
    
    img = np.ones((150, 150), dtype=np.uint8) * 255
    
    # Параметры стиля "человека"
    start_x = random.randint(20, 40)
    start_y = random.randint(60, 100)
    num_points = random.randint(5, 9)
    thickness = random.randint(2, 4)
    
    points = [(start_x, start_y)]
    for _ in range(num_points - 1):
        nx = random.randint(50, 130)
        ny = random.randint(30, 120)
        points.append((nx, ny))
    
    # Рисуем линии
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], 0, thickness)
    
    # Добавляем немного естественного шума
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    img = np.clip(img, 0, 255)
    
    return img

def create_dataset(n_pairs=5000):
    """Создает пары изображений: (img1, img2, label)."""
    print(f"\n🎨 Генерация {n_pairs} пар синтетических подписей...")
    
    X1_list = []
    X2_list = []
    y_list = []
    
    for i in tqdm(range(n_pairs), desc="Генерация"):
        # Создаем базовую подпись с уникальным seed
        base_seed = random.randint(0, 100000)
        img1 = generate_signature_image(base_seed)
        
        if i % 2 == 0:
            # ПОЛОЖИТЕЛЬНАЯ ПАРА (тот же человек)
            # Берем тот же seed или очень близкий + вариации
            img2 = generate_signature_image(base_seed) 
            
            # Аугментация для второй подписи того же человека
            # 1. Небольшой сдвиг
            shift_x = random.randint(-3, 3)
            shift_y = random.randint(-3, 3)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img2 = cv2.warpAffine(img2, M, (150, 150), borderMode=cv2.BORDER_REPLICATE)
            
            # 2. Шум
            noise = np.random.normal(0, 15, img2.shape).astype(np.uint8)
            img2 = cv2.add(img2, noise)
            img2 = np.clip(img2, 0, 255)
            
            # 3. Легкое размытие иногда
            if random.random() > 0.7:
                img2 = cv2.GaussianBlur(img2, (3, 3), 0)
                
            label = 1
        else:
            # ОТРИЦАТЕЛЬНАЯ ПАРА (разные люди)
            new_seed = random.randint(0, 100000)
            while abs(new_seed - base_seed) < 1000: # Гарантируем различие
                new_seed = random.randint(0, 100000)
            
            img2 = generate_signature_image(new_seed)
            label = 0
        
        # Предобработка
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0
        
        X1_list.append(np.expand_dims(img1_f, axis=-1))
        X2_list.append(np.expand_dims(img2_f, axis=-1))
        y_list.append(label)
    
    return np.array(X1_list), np.array(X2_list), np.array(y_list)

# ============================================================
# 2. АРХИТЕКТУРА МОДЕЛИ (SIAMESE NETWORK)
# ============================================================
def create_model():
    """Создает и компилирует Siamese сеть."""
    
    # Энкодер (общий для обоих входов)
    input_img = layers.Input(shape=(150, 150, 1), name='input_img')
    
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
    input_a = layers.Input(shape=(150, 150, 1), name='input_a')
    input_b = layers.Input(shape=(150, 150, 1), name='input_b')
    
    feat_a = encoder(input_a)
    feat_b = encoder(input_b)
    
    # Вычисляем расстояние между признаками
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name='abs_diff')([feat_a, feat_b])
    
    # Классификатор
    x = layers.Dense(128, activation='relu')(distance)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, encoder

# ============================================================
# 3. ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# ============================================================
def main():
    # 1. Генерация данных
    X1, X2, y = create_dataset(n_pairs=6000)
    
    # 2. Разделение на Train/Val
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Размер выборки:")
    print(f"   Train: {len(X1_train)} пар")
    print(f"   Val:   {len(X1_val)} пар")
    
    # 3. Создание модели
    print("\n🏗️ Построение архитектуры нейросети...")
    model, encoder = create_model()
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'signature_model.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # 5. Обучение
    print("\n🔥 НАЧАЛО ОБУЧЕНИЯ...")
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Сохранение
    print("\n💾 Сохранение модели...")
    model.save('signature_model.keras')
    encoder.save('signature_encoder.keras')
    
    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("Файлы сохранены:")
    print("   - signature_model.keras (полная модель)")
    print("   - signature_encoder.keras (энкодер)")
    print("="*60)
    
    # 7. Тестирование
    print("\n🧪 Быстрый тест модели...")
    test_seed = 99999
    t1 = generate_signature_image(test_seed)
    t2 = generate_signature_image(test_seed) # Та же подпись
    
    t1_p = np.expand_dims(t1.astype(np.float32)/255.0, axis=(0, -1))
    t2_p = np.expand_dims(t2.astype(np.float32)/255.0, axis=(0, -1))
    
    pred = model.predict([t1_p, t2_p], verbose=0)[0][0]
    print(f"   Сходство двух одинаковых подписей: {pred:.2%}")
    print(f"   Вердикт: {'✅ СОВПАДАЮТ' if pred > 0.5 else '❌ РАЗНЫЕ'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()