import os
from pathlib import Path

# Путь к папке instance
BASE_DIR = Path(__file__).parent
INSTANCE_DIR = BASE_DIR / 'instance'

# Убедимся, что папка instance существует
INSTANCE_DIR.mkdir(exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-very-secret-key'
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{INSTANCE_DIR / 'signatures.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False