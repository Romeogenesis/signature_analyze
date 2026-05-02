import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureEncoder(nn.Module):
    """Энкодер для извлечения признаков из подписи"""
    def __init__(self, embedding_dim=128):
        super(SignatureEncoder, self).__init__()
        
        # Сверточная часть
        self.conv_layers = nn.Sequential(
            # Блок 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Полносвязная часть
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 9 * 9, 512),  # Предполагаем вход 144x144 -> после 4 пулингов 9x9
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_dim),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return F.normalize(x, p=2, dim=1)  # L2 нормализация


class SiameseNetwork(nn.Module):
    """Siamese-сеть для сравнения подписей"""
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = SignatureEncoder(embedding_dim)
    
    def forward(self, x1, x2):
        encoding1 = self.encoder(x1)
        encoding2 = self.encoder(x2)
        return encoding1, encoding2
    
    def predict_similarity(self, x1, x2):
        """Вычисляет схожесть между двумя подписями"""
        with torch.no_grad():
            enc1, enc2 = self.forward(x1, x2)
            # Косинусное сходство
            similarity = F.cosine_similarity(enc1, enc2)
            return similarity.item()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
