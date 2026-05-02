import keras
from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import os

class SignatureVerifier:
    """Сервис для верификации подписей с использованием обученной Siamese-сети (Keras)"""
    
    def __init__(self, model_path='signature_model.keras', img_size=(150, 150)):
        self.img_size = img_size
        self.model = None
        
        # Загрузка модели если файл существует
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Модель не найдена по пути {model_path}")
            print("Запустите обучение: python train_model.py")
    
    def load_model(self, model_path):
        """Загрузка обученной модели Keras"""
        self.model = load_model(model_path)
        print(f"Модель загружена: {model_path}")
    
    def preprocess_image(self, img_path):
        """Предобработка изображения для подачи в модель"""
        # Чтение изображения
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Попытка загрузить через PIL если cv2 не смог
            try:
                pil_img = Image.open(img_path).convert('L')
                img = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Не удалось загрузить изображение: {img_path}, ошибка: {e}")
        
        # Изменение размера
        img = cv2.resize(img, self.img_size)
        
        # Нормализация
        img = img.astype(np.float32) / 255.0
        
        # Преобразование в тензор (B, H, W, C) для Keras/TensorFlow
        img = np.expand_dims(img, axis=-1)  # Добавляем канал
        img = np.expand_dims(img, axis=0)   # Добавляем батч
        
        return img
    
    def verify(self, img_path1, img_path2):
        """
        Сравнение двух подписей.
        
        Возвращает:
            dict с ключами:
                - similarity: процент схожести (0-100)
                - is_match: True/False
                - confidence: уровень уверенности ('high', 'medium', 'low')
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена. Запустите обучение сначала.")
        
        # Предобработка изображений
        img1 = self.preprocess_image(img_path1)
        img2 = self.preprocess_image(img_path2)
        
        # Предсказание
        prediction = self.model.predict([img1, img2], verbose=0)[0][0]
        
        # prediction - это вероятность совпадения (0-1)
        similarity_percent = float(prediction) * 100
        
        # Определение порога
        threshold = 0.5  # Сигмоида > 0.5 считаем совпадением
        is_match = float(prediction) > threshold
        
        # Уровень уверенности
        if float(prediction) > 0.8 or float(prediction) < 0.2:
            confidence = 'high'
        elif float(prediction) > 0.6 or float(prediction) < 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'similarity': round(similarity_percent, 2),
            'is_match': is_match,
            'confidence': confidence,
            'raw_prediction': float(prediction)
        }


# Для совместимости с веб-приложением
def verify_signatures(file1_path, file2_path, model_path='signature_model.keras'):
    """Удобная функция для вызова из веб-приложения"""
    verifier = SignatureVerifier(model_path=model_path)
    return verifier.verify(file1_path, file2_path)
