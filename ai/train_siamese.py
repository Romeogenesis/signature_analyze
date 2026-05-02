"""
Siamese Network для сравнения подписей.
Использует архитектуру на основе CNN с контрастивной функцией потерь.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class SignatureDataset(Dataset):
    """
    Датасет для обучения Siamese сети.
    Ожидает структуру папок:
    data/train/
        person_001/
            sig_001.png
            sig_002.png
        person_002/
            sig_001.png
            ...
    """
    
    def __init__(self, root_dir, transform=None, pairs_count=1000):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pairs_count = pairs_count
        
        # Собираем все изображения по людям
        self.persons = {}
        for person_folder in self.root_dir.iterdir():
            if person_folder.is_dir():
                person_id = person_folder.name
                images = list(person_folder.glob('*.png')) + \
                         list(person_folder.glob('*.jpg')) + \
                         list(person_folder.glob('*.jpeg'))
                if images:
                    self.persons[person_id] = images
        
        self.person_ids = list(self.persons.keys())
        
    def __len__(self):
        return self.pairs_count
    
    def __getitem__(self, idx):
        # Генерируем пару: 50% одинаковые подписи, 50% разные
        if np.random.random() < 0.5:
            # Одинаковая подпись (от одного человека)
            person_id = np.random.choice(self.person_ids)
            images = self.persons[person_id]
            
            if len(images) >= 2:
                img1_path, img2_path = np.random.choice(images, 2, replace=False)
            else:
                # Если только одно изображение, используем его дважды с небольшим аугментированием
                img1_path = images[0]
                img2_path = images[0]
            
            label = 1.0  # Одинаковые
        else:
            # Разные подписи (от разных людей)
            person1_id, person2_id = np.random.choice(self.person_ids, 2, replace=False)
            img1_path = np.random.choice(self.persons[person1_id])
            img2_path = np.random.choice(self.persons[person2_id])
            label = 0.0  # Разные
        
        # Загружаем изображения
        img1 = Image.open(img1_path).convert('L')  # Чёрно-белое
        img2 = Image.open(img2_path).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class SiameseNetwork(nn.Module):
    """
    Siamese сеть на основе предобученной ResNet18.
    """
    
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        # Берём предобученную ResNet18 и удаляем последние слои
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
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


class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для Siamese сети.
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        # Евклидово расстояние между эмбеддингами
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        
        # Loss = label * distance^2 + (1 - label) * max(0, margin - distance)^2
        loss = label * (distance ** 2) + \
               (1 - label) * torch.clamp(self.margin - distance, min=0.0) ** 2
        
        return loss.mean()


def train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_save_path='models/signature_siamese.pth',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    embedding_dim=128,
    image_size=224
):
    """
    Обучение Siamese сети для сравнения подписей.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Трансформации для аугментации данных
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Для одноканального изображения
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Создаём датасеты
    print("Загрузка тренировочных данных...")
    train_dataset = SignatureDataset(train_dir, transform=train_transform, pairs_count=5000)
    print(f"Количество людей в тренировочном наборе: {len(train_dataset.person_ids)}")
    
    print("Загрузкa валидационных данных...")
    val_dataset = SignatureDataset(val_dir, transform=val_transform, pairs_count=1000)
    print(f"Количество людей в валидационном наборе: {len(val_dataset.person_ids)}")
    
    # Создаём dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Инициализируем модель
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Трекинг метрик
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nНачало обучения на {epochs} эпох...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Тренировка
        model.train()
        running_train_loss = 0.0
        
        for img1, img2, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embedding1, embedding2 = model(img1, img2)
            loss = criterion(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                embedding1, embedding2 = model(img1, img2)
                loss = criterion(embedding1, embedding2, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Шаг планировщика
        scheduler.step(avg_val_loss)
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'embedding_dim': embedding_dim,
                'image_size': image_size
            }, model_save_path)
            print(f"✓ Лучшая модель сохранена! Val Loss: {avg_val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
    
    # Сохраняем историю обучения
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs': epochs
    }
    
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Построение графиков
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.diff(train_losses), label='Train Δ')
    plt.plot(np.diff(val_losses), label='Val Δ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Обучение завершено!")
    print(f"📊 Лучший validation loss: {best_val_loss:.4f}")
    print(f"💾 Модель сохранена в: {model_save_path}")
    print(f"📈 График обучения сохранён в: models/training_curves.png")
    
    return model, history


if __name__ == '__main__':
    # Проверка наличия данных
    train_path = Path('data/train')
    val_path = Path('data/val')
    
    if not train_path.exists() or not any(train_path.iterdir()):
        print("⚠️  Предупреждение: Папка data/train пуста или не существует!")
        print("Создам тестовые данные для демонстрации...")
        
        # Создадим фейковые данные для теста
        from PIL import Image, ImageDraw, ImageFont
        
        def create_fake_signature(text, seed, output_path):
            np.random.seed(seed)
            img = Image.new('RGB', (300, 150), color='white')
            draw = ImageDraw.Draw(img)
            
            # Рисуем "подпись" как случайные линии
            points = []
            for i in range(50):
                x = np.random.randint(20, 280)
                y = np.random.randint(30, 120) + np.sin(i * 0.2) * 20
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill='black', width=2)
            
            # Добавим текст
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuScript.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            draw.text((50, 60), text, fill='black', font=font)
            
            # Добавим шум
            pixels = img.load()
            for i in range(img.width):
                for j in range(img.height):
                    if np.random.random() < 0.02:
                        pixels[i, j] = (np.random.randint(0, 50),) * 3
            
            img.save(output_path)
        
        # Создадим структуры папок
        for split in ['train', 'val']:
            split_path = Path(f'data/{split}')
            split_path.mkdir(parents=True, exist_ok=True)
            
            for person_idx in range(10):  # 10 человек
                person_folder = split_path / f'person_{person_idx:03d}'
                person_folder.mkdir(exist_ok=True)
                
                # Создадим 5 подписей для каждого человека
                for sig_idx in range(5):
                    filename = person_folder / f'sig_{sig_idx:03d}.png'
                    create_fake_signature(f'Person {person_idx}', person_idx * 100 + sig_idx, filename)
        
        print("✓ Тестовые данные созданы!")
    
    # Запуск обучения
    model, history = train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_save_path='models/signature_siamese.pth',
        epochs=50,  # Можно увеличить до 100-200 для лучшего качества
        batch_size=32,
        learning_rate=0.001,
        embedding_dim=128,
        image_size=224
    )
