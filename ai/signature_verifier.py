"""
Сервис для использования обученной Siamese сети в веб-приложении.
Загружает модель и предоставляет метод для сравнения двух подписей.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path


class SiameseNetwork(nn.Module):
    """
    Siamese сеть (та же архитектура, что и при обучении).
    """
    
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        # Берём предобученную ResNet18 и удаляем последние слои
        resnet = models.resnet18(weights=None)  # веса загрузим отдельно
        
        # Удаляем полностью связанные слои и пулинг
        modules = list(resnet.children())[:-2]  # Убираем avgpool и fc
        
        self.feature_extractor = nn.Sequential(*modules)
        
        # Адаптивный пулинг для фиксированного размера
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полностью связанные слои для эмбеддинга
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward_once(self, x):
        """Прямой проход через сеть для одного изображения"""
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = self.embedding(x)
        # Нормализуем эмбеддинг
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x1, x2):
        """Прямой проход для пары изображений"""
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        return embedding1, embedding2


class SignatureVerifier:
    """
    Сервис для верификации подписей с использованием обученной Siamese сети.
    """
    
    def __init__(self, model_path='models/signature_siamese.pth', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = None
        self.image_size = 224
        
        # Проверка наличия модели
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена по пути: {self.model_path}\n"
                "Сначала запустите обучение: python ai/train_siamese.py"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка обученной модели"""
        print(f"Загрузка модели из {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Получаем параметры из чекпоинта
        self.embedding_dim = checkpoint.get('embedding_dim', 128)
        self.image_size = checkpoint.get('image_size', 224)
        
        # Инициализируем модель
        self.model = SiameseNetwork(embedding_dim=self.embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Модель загружена успешно! (embedding_dim={self.embedding_dim}, image_size={self.image_size})")
    
    def _preprocess_image(self, image_path):
        """Предобработка изображения"""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        img = Image.open(image_path).convert('L')  # Чёрно-белое
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def verify_signatures(self, image1_path, image2_path):
        """
        Сравнение двух подписей.
        
        Args:
            image1_path: Путь к первому изображению подписи
            image2_path: Путь ко второму изображению подписи
            
        Returns:
            dict: Результат сравнения с полями:
                - similarity: Процент схожести (0-100)
                - is_match: Булево значение (True/False)
                - distance: Евклидово расстояние между эмбеддингами
                - confidence: Уровень уверенности (high/medium/low)
        """
        # Предобработка изображений
        img1 = self._preprocess_image(image1_path)
        img2 = self._preprocess_image(image2_path)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embedding1, embedding2 = self.model(img1, img2)
            
            # Вычисление расстояния
            distance = torch.nn.functional.pairwise_distance(embedding1, embedding2).item()
        
        # Преобразование расстояния в процент схожести
        # При обучении margin=1.0, поэтому нормализуем относительно этого значения
        # distance=0 -> similarity=100%, distance>=1.0 -> similarity=0%
        similarity = max(0, (1.0 - distance)) * 100
        
        # Определение порога совпадения (можно настроить)
        threshold = 0.5  # 50% схожести
        is_match = similarity >= threshold * 100
        
        # Определение уровня уверенности
        if distance < 0.3:
            confidence = "high"
        elif distance < 0.7:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'similarity': round(similarity, 2),
            'is_match': is_match,
            'distance': round(distance, 4),
            'confidence': confidence,
            'threshold': threshold * 100
        }
    
    def verify_signatures_from_bytes(self, image1_bytes, image2_bytes):
        """
        Сравнение двух подписей из байтов (для загрузки через веб-форму).
        
        Args:
            image1_bytes: Байты первого изображения
            image2_bytes: Байты второго изображения
            
        Returns:
            dict: Результат сравнения (см. verify_signatures)
        """
        from io import BytesIO
        
        # Загрузка изображений из байтов
        img1 = Image.open(BytesIO(image1_bytes)).convert('L')
        img2 = Image.open(BytesIO(image2_bytes)).convert('L')
        
        # Трансформация
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        img1_tensor = transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = transform(img2).unsqueeze(0).to(self.device)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embedding1, embedding2 = self.model(img1_tensor, img2_tensor)
            
            # Вычисление расстояния
            distance = torch.nn.functional.pairwise_distance(embedding1, embedding2).item()
        
        # Преобразование в процент схожести
        similarity = max(0, (1.0 - distance)) * 100
        
        # Порог и уверенность
        threshold = 0.5
        is_match = similarity >= threshold * 100
        
        if distance < 0.3:
            confidence = "high"
        elif distance < 0.7:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'similarity': round(similarity, 2),
            'is_match': is_match,
            'distance': round(distance, 4),
            'confidence': confidence,
            'threshold': threshold * 100
        }


# Singleton instance
_verifier_instance = None


def get_verifier(model_path='models/signature_siamese.pth'):
    """Получить экземпляр верификатора (singleton)"""
    global _verifier_instance
    
    if _verifier_instance is None:
        _verifier_instance = SignatureVerifier(model_path=model_path)
    
    return _verifier_instance


if __name__ == '__main__':
    # Тестирование верификатора
    print("Тестирование SignatureVerifier...")
    
    try:
        verifier = get_verifier()
        print("✓ Верификатор успешно инициализирован!")
        
        # Если есть тестовые изображения, можно проверить
        test_img1 = Path('data/val/person_000/sig_000.png')
        test_img2 = Path('data/val/person_000/sig_001.png')
        test_img3 = Path('data/val/person_001/sig_000.png')
        
        if test_img1.exists() and test_img2.exists():
            print(f"\nТест 1: Сравнение подписей одного человека")
            result1 = verifier.verify_signatures(str(test_img1), str(test_img2))
            print(f"  Схожесть: {result1['similarity']}%")
            print(f"  Совпадение: {result1['is_match']}")
            print(f"  Уверенность: {result1['confidence']}")
        
        if test_img1.exists() and test_img3.exists():
            print(f"\nТест 2: Сравнение подписей разных людей")
            result2 = verifier.verify_signatures(str(test_img1), str(test_img3))
            print(f"  Схожесть: {result2['similarity']}%")
            print(f"  Совпадение: {result2['is_match']}")
            print(f"  Уверенность: {result2['confidence']}")
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print("Сначала запустите обучение модели: python ai/train_siamese.py")
