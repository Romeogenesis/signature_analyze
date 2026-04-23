from models.signature_db import Signature
from . import ai_service_stub

def verify_signature_from_db(signature_data):
    """
    Проверяет, существует ли подпись в базе данных.
    В будущем будет интегрирована с ИИ.
    """
    # Заглушка: проверяем только точное совпадение
    # !!! ВНИМАНИЕ: Тут нужно подключить базу данных через Flask-SQLAlchemy !!!
    # Используем глобальный объект db из models.signature_db
    from models.signature_db import db
    existing_signature = db.session.execute(
        db.select(Signature).where(Signature.data == signature_data)
    ).scalar_one_or_none()
    
    if existing_signature:
        return True

    # Если не найдено, вызываем ИИ (пока заглушка)
    is_valid = ai_service_stub.analyze_signature_with_ai(signature_data)
    return is_valid