from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # В реальности - hash!

    def __repr__(self):
        return f'<User {self.username}>'

class Signature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text, nullable=False)  # Содержимое подписи
    verified = db.Column(db.Boolean, default=False)  # Проверена ли ИИ

    def __repr__(self):
        return f'<Signature {self.id}>'