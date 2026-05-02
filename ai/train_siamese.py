#!/usr/bin/env python3
"""
Скрипт для обучения Siamese-сети для сравнения подписей.

Использование:
    python ai/train_siamese.py --data_dir data/signatures --epochs 50 --batch_size 32

Требования:
    - PyTorch
    - OpenCV (cv2)
    - tqdm (для прогресс-бара)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from siamese_network import SiameseNetwork
from dataset import SignatureDataset, create_data_loader


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss для обучения Siamese-сети.
    
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    где Y=0 для одинаковых пар, Y=1 для разных пар
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Евклидово расстояние между эмбеддингами
        distance = F.pairwise_distance(output1, output2)
        
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        return loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Обучение за одну эпоху"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (img1, img2, labels) in enumerate(pbar):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        output1, output2 = model(img1, img2)
        
        # Вычисление потерь
        loss = criterion(output1, output2, labels)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Вычисление точности (порог 0.5 для косинусного сходства)
        with torch.no_grad():
            similarity = nn.functional.cosine_similarity(output1, output2)
            predictions = (similarity > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Логирование в TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, epoch, writer):
    """Валидация"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Valid {epoch}')
        
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            total_loss += loss.item()
            
            similarity = nn.functional.cosine_similarity(output1, output2)
            predictions = (similarity > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Обучение Siamese-сети для сравнения подписей')
    parser.add_argument('--data_dir', type=str, default='data/signatures',
                        help='Путь к датасету с подписями')
    parser.add_argument('--model_path', type=str, default='ai/models/siamese_model.pth',
                        help='Путь для сохранения модели')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin для contrastive loss')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Размерность эмбеддинга')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Количество воркеров для загрузки данных')
    parser.add_argument('--log_dir', type=str, default='ai/logs',
                        help='Путь для логов TensorBoard')
    
    args = parser.parse_args()
    
    # Проверка наличия данных
    if not os.path.exists(args.data_dir):
        print(f"Ошибка: Датасет не найден по пути {args.data_dir}")
        print("Создайте структуру папок:")
        print("  data/signatures/")
        print("    person_001/")
        print("      sig_001.png")
        print("      sig_002.png")
        print("    person_002/")
        print("      ...")
        return
    
    # Создание директорий
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Загрузка данных
    print("Загрузка датасета...")
    full_dataset = SignatureDataset(args.data_dir, augment=False)
    
    # Разделение на train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Аугментация для тренировочной выборки
    train_dataset.dataset.augment = True
    
    train_loader = create_data_loader(train_dataset, args.batch_size, shuffle=True, 
                                       num_workers=args.num_workers)
    val_loader = create_data_loader(val_dataset, args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)
    
    # Инициализация модели
    print("Инициализация модели...")
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    
    # Loss и оптимизатор
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Обучение
    print("Начало обучения...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.model_path)
            print(f'  -> Сохранена лучшая модель с точностью {val_acc:.2f}%')
    
    writer.close()
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая точность валидации: {best_val_acc:.2f}%")
    print(f"Модель сохранена в: {args.model_path}")
    print(f"Логи TensorBoard: {args.log_dir}")
    print("\nДля запуска веб-приложения выполните:")
    print("  python app.py")


if __name__ == '__main__':
    main()
