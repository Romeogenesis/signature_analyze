from flask import Flask, render_template, request, redirect, url_for, flash, session
from config import Config
from models.signature_db import db, User, Signature
from services.signature_verification import verify_signature_from_db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Инициализация базы данных
    db.init_app(app)

    # Создание таблиц
    with app.app_context():
        db.create_all()

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
            
            # Простая проверка (в реальном мире используйте hash!)
            if user and user.password == password:
                session['user_id'] = user.id
                return redirect(url_for('index'))
            else:
                flash('Неверный логин или пароль.')
        
        return render_template('login.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        if request.method == 'POST':
            file = request.files.get('signature_file')
            if file and file.filename != '':
                # Здесь мы имитируем "анализ" подписи
                # В реальности, файл может быть передан в ИИ-модуль
                signature_data = file.read().decode('utf-8')
                
                # Сохраняем подпись в базу
                new_signature = Signature(data=signature_data, verified=False)
                db.session.add(new_signature)
                db.session.commit()

                # Проверяем подпись (пока без ИИ)
                exists_in_db = verify_signature_from_db(signature_data)

                return render_template('result.html', exists=exists_in_db)
            else:
                flash('Файл не выбран.')

        return render_template('upload.html')

    @app.route('/logout')
    def logout():
        session.pop('user_id', None)
        return redirect(url_for('login'))

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)