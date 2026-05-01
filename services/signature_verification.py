from services.ai_service_stub import analyze_signature_with_ai

def verify_two_signatures(file1_data, file2_data):
    """
    Сравнивает две подписи с помощью ИИ-алгоритма.
    Возвращает результат сравнения с процентом схожести.
    """
    result = analyze_signature_with_ai(file1_data, file2_data)
    return result