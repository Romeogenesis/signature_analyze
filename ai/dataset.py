import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2

class SignatureDataset(Dataset):
    """
    Датасет для обучения Siamese-сети.
    
    Структура папок должна быть такой:
    data/signatures/
        person_001/
            sig_001.png
            sig_002.png
            ...
        person_002/
            sig_001.png
            sig_002.png
            ...
        ...
    """
    def __init__(self, root_dir, img_size=(144, 144), augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        
        # Собираем все подписи по людям
        self.signatures = []  # (person_id, image_path)
        self.person_ids = {}  # person_name -> person_id
        
        person_idx = 0
        for person_name in sorted(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            
            self.person_ids[person_name] = person_idx
            
            for sig_file in sorted(os.listdir(person_path)):
                if sig_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    sig_path = os.path.join(person_path, sig_file)
                    self.signatures.append((person_idx, sig_path))
            
            person_idx += 1
        
        print(f"Загружено {len(self.signatures)} подписей от {len(self.person_ids)} человек")
    
    def __len__(self):
        return len(self.signatures)
    
    def preprocess_image(self, img_path):
        """Загрузка и предобработка изображения"""
        # Чтение изображения
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        # Изменение размера
        img = cv2.resize(img, self.img_size)
        
        # Аугментация (только для обучения)
        if self.augment:
            # Случайный поворот
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((self.img_size[0]//2, self.img_size[1]//2), angle, 1.0)
            img = cv2.warpAffine(img, M, self.img_size, borderMode=cv2.BORDER_REPLICATE)
            
            # Случайное смещение
            tx = np.random.uniform(-3, 3)
            ty = np.random.uniform(-3, 3)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, self.img_size, borderMode=cv2.BORDER_REPLICATE)
            
            # Добавление шума
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
        
        # Нормализация
        img = img.astype(np.float32) / 255.0
        img = (img - img.mean()) / (img.std() + 1e-7)
        
        # Преобразование в тензор (C, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img
    
    def __getitem__(self, idx):
        person_id1, img_path1 = self.signatures[idx]
        
        # Выбираем вторую подпись
        if np.random.random() > 0.5:
            # Та же личность (positive pair)
            same_person_signatures = [s for s in self.signatures if s[0] == person_id1 and s[1] != img_path1]
            if same_person_signatures:
                person_id2, img_path2 = same_person_signatures[np.random.randint(len(same_person_signatures))]
                label = 1.0  # Одинаковые
            else:
                # Если только одна подпись у человека, берем другую личность
                other_signatures = [s for s in self.signatures if s[0] != person_id1]
                person_id2, img_path2 = other_signatures[np.random.randint(len(other_signatures))]
                label = 0.0  # Разные
        else:
            # Другая личность (negative pair)
            other_signatures = [s for s in self.signatures if s[0] != person_id1]
            person_id2, img_path2 = other_signatures[np.random.randint(len(other_signatures))]
            label = 0.0  # Разные
        
        img1 = self.preprocess_image(img_path1)
        img2 = self.preprocess_image(img_path2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Создание DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
