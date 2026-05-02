import os
import sqlite3
import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

# Настройка приложения
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['MODEL_PATH'] = 'signature_model.keras'

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Глобальная переменная для модели
model = None

def load_model():
    global model
    try:
        print("Загрузка модели...")
        # Загружаем модель. Так как мы убрали Lambda с функцией, safe_mode больше не нужен.
        model = tf.keras.models.load_model(app.config['MODEL_PATH'])
        print("✅ Модель успешно загружена!")
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False

# Загружаем модель при старте
if os.path.exists(app.config['MODEL_PATH']):
    load_model()
else:
    print("⚠️ Файл модели не найден. Запустите сначала train_model.py")

# База данных
DB_NAME = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )''')
    # Создаем админа по умолчанию если нет
    c.execute("SELECT * FROM users WHERE username = ?", ('admin',))
    if not c.fetchone():
        pwd_hash = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ('admin', pwd_hash))
        print("Создан пользователь по умолчанию: admin / admin123")
    conn.commit()
    conn.close()

init_db()

def preprocess_image(file_path):
    """Обработка изображения для модели."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (150, 150))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # Добавляем размерность батча
    return img

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Введите имя и пароль', 'error')
            return redirect(url_for('register'))
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            flash('Пользователь уже существует', 'error')
            return redirect(url_for('register'))
        
        pwd_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
        conn.commit()
        conn.close()
        
        flash('Регистрация успешна! Войдите.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Неверный логин или пароль', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/verify', methods=['POST'])
def verify():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if model is None:
        return jsonify({'error': 'Модель не загружена. Запустите обучение.'}), 500
    
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Файлы не выбраны'}), 400
    
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
    
    file1.save(path1)
    file2.save(path2)
    
    try:
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)
        
        if img1 is None or img2 is None:
            return jsonify({'error': 'Ошибка обработки изображения. Убедитесь, что это картинка.'}), 400
        
        prediction = model.predict([img1, img2], verbose=0)[0][0]
        similarity = float(prediction)
        
        # Удаляем файлы после обработки
        os.remove(path1)
        os.remove(path2)
        
        result_text = "ПОДПИСИ СОВПАДАЮТ" if similarity > 0.5 else "ПОДПИСИ РАЗНЫЕ"
        confidence = "Высокая" if (similarity > 0.8 or similarity < 0.2) else "Средняя"
        
        return jsonify({
            'similarity': round(similarity * 100, 2),
            'result': result_text,
            'confidence': confidence,
            'raw_score': similarity
        })
        
    except Exception as e:
        # Чистим файлы в случае ошибки
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)