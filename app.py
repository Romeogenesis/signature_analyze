from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import base64

from config import Config
from models.signature_db import db, User, Signature

# Импорт сервиса верификации подписей на основе Siamese сети
try:
    from ai.signature_verifier import verify_signatures
    AI_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    AI_AVAILABLE = False
    print(f"⚠️  AI-сервис недоступен: {e}")
    print("Запустите сначала: python ai/train_siamese.py")

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Инициализация базы данных
    db.init_app(app)

    # Создание таблиц и тестового пользователя
    with app.app_context():
        db.create_all()
        # Создаем тестового пользователя если не существует
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()

    @app.route('/')
    def index():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return render_template('dashboard.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            user = User.query.filter_by(username=username).first()
            
            # Проверка пароля с использованием хэша
            if user and user.check_password(password):
                session['user_id'] = user.id
                return redirect(url_for('index'))
            else:
                flash('Неверный логин или пароль.')
        
        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            confirm_password = request.form['confirm_password']

            if password != confirm_password:
                flash('Пароли не совпадают.')
                return render_template('register.html')

            if User.query.filter_by(username=username).first():
                flash('Пользователь с таким именем уже существует.')
                return render_template('register.html')

            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash('Регистрация успешна! Теперь вы можете войти.')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        if request.method == 'POST':
            file1 = request.files.get('signature_file_1')
            file2 = request.files.get('signature_file_2')
            
            if file1 and file2 and file1.filename != '' and file2.filename != '':
                # Сохраняем файлы во временную директорию
                upload_folder = 'uploads'
                os.makedirs(upload_folder, exist_ok=True)
                
                file1_path = os.path.join(upload_folder, f"temp_{session['user_id']}_1.png")
                file2_path = os.path.join(upload_folder, f"temp_{session['user_id']}_2.png")
                
                file1.save(file1_path)
                file2.save(file2_path)
                
                # Читаем данные для сохранения в БД
                file1.seek(0)
                file2.seek(0)
                file1_data = base64.b64encode(file1.read()).decode('utf-8')
                file2_data = base64.b64encode(file2.read()).decode('utf-8')
                
                # Сохраняем подписи в базу
                sig1 = Signature(
                    data=file1_data[:500],  # Сохраняем часть данных для истории
                    verified=True,
                    user_id=session['user_id']
                )
                sig2 = Signature(
                    data=file2_data[:500],
                    verified=True,
                    user_id=session['user_id']
                )
                db.session.add(sig1)
                db.session.add(sig2)
                db.session.commit()

                # Проверяем подписи с помощью ИИ
                result = None
                if AI_AVAILABLE:
                    try:
                        model_path = 'signature_model.keras'
                        result = verify_signatures(file1_path, file2_path, model_path=model_path)
                    except Exception as e:
                        flash(f'Ошибка при проверке нейросетью: {e}')
                        result = None
                
                if result is None:
                    # Резервный эвристический метод если модель не обучена
                    from services.signature_verification import verify_two_signatures
                    result = verify_two_signatures(file1_data, file2_data)
                    # Преобразуем формат результата
                    if 'threshold' in result:
                        result['similarity'] = result.get('similarity', 50.0)
                        result['is_match'] = result['similarity'] >= result.get('threshold', 50.0)
                    if 'confidence' not in result:
                        if result['similarity'] >= 85:
                            result['confidence'] = 'high'
                        elif result['similarity'] >= 70:
                            result['confidence'] = 'medium'
                        else:
                            result['confidence'] = 'low'

                # Удаляем временные файлы
                try:
                    os.remove(file1_path)
                    os.remove(file2_path)
                except:
                    pass

                return render_template('result.html', 
                                       is_match=result['is_match'],
                                       similarity=result['similarity'],
                                       threshold=result.get('threshold', 70.0),
                                       confidence=result['confidence'])
            else:
                flash('Оба файла должны быть выбраны.')

        return render_template('upload.html')

    @app.route('/logout')
    def logout():
        session.pop('user_id', None)
        return redirect(url_for('login'))

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)