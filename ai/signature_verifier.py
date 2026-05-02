import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import os

class SignatureVerifier:
    """Сервис для верификации подписей с использованием обученной Siamese-сети"""
    
    def __init__(self, model_path='ai/models/siamese_model.pth', img_size=(144, 144)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.model = None
        
        # Загрузка модели если файл существует
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Модель не найдена по пути {model_path}")
            print("Запустите обучение: python ai/train_siamese.py")
    
    def load_model(self, model_path):
        """Загрузка обученной модели"""
        from siamese_network import SiameseNetwork
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Модель загружена с {self.device}")
    
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
        img = (img - img.mean()) / (img.std() + 1e-7)
        
        # Преобразование в тензор (B, C, H, W)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.to(self.device)
        
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
        with torch.no_grad():
            enc1, enc2 = self.model(img1, img2)
            similarity = F.cosine_similarity(enc1, enc2).item()
        
        # Конвертация схожести (-1 до 1) в проценты (0 до 100)
        similarity_percent = ((similarity + 1) / 2) * 100
        
        # Определение порога (настраивается экспериментально)
        threshold = 0.7  # Косинусное сходство > 0.7 считаем совпадением
        is_match = similarity > threshold
        
        # Уровень уверенности
        abs_sim = abs(similarity)
        if abs_sim > 0.8:
            confidence = 'high'
        elif abs_sim > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'similarity': round(similarity_percent, 2),
            'is_match': is_match,
            'confidence': confidence,
            'raw_similarity': similarity
        }


# Для совместимости с веб-приложением
def verify_signatures(file1_path, file2_path, model_path='ai/models/siamese_model.pth'):
    """Удобная функция для вызова из веб-приложения"""
    verifier = SignatureVerifier(model_path=model_path)
    return verifier.verify(file1_path, file2_path)
