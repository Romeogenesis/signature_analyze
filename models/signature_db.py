from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Signature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data = db.Column(db.Text, nullable=False)  # Содержимое подписи (в реальном приложении - путь к файлу или бинарные данные)
    verified = db.Column(db.Boolean, default=False)  # Проверена ли ИИ
    similarity_score = db.Column(db.Float, default=0.0)  # Процент схожести
    
    user = db.relationship('User', backref=db.backref('signatures', lazy=True))

    def __repr__(self):
        return f'<Signature {self.id}>'