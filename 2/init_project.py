"""
Скрипт инициализации проекта
Создаёт структуру директорий и проверяет зависимости
"""
import sys
from pathlib import Path

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR,
    DATABASE_PATH, CARD_RARITIES, CARD_TYPES
)
from utils.logger import setup_logger


def check_dependencies():
    """Проверка установленных зависимостей"""
    required_packages = [
        "torch", "numpy", "pandas", "scikit-learn",
        "requests", "fastapi", "uvicorn", "loguru"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing)}")
        print("Установите: pip install -r requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True


def check_api_key():
    """Проверка наличия API ключа"""
    from config import ROYALEAPI_KEY
    
    if not ROYALEAPI_KEY or ROYALEAPI_KEY == "your_api_key_here":
        print("⚠️  API ключ RoyaleAPI не настроен")
        print("Отредактируйте .env файл и укажите ROYALEAPI_KEY")
        return False
    
    print("✅ API ключ настроен")
    return True


def create_directories():
    """Создание необходимых директорий"""
    dirs = [
        DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
        BASE_DIR / "scripts",
        BASE_DIR / "web" / "templates",
        BASE_DIR / "web" / "static"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True)
    
    print(f"✅ Директории созданы в {BASE_DIR}")


def init_database():
    """Инициализация базы данных"""
    try:
        from data.database import Database
        
        db = Database(str(DATABASE_PATH))
        db.create_tables()
        print("✅ База данных инициализирована")
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации БД: {e}")
        return False


def main():
    """Основная функция инициализации"""
    print("=" * 50)
    print("Clash Royale Deck Generator - Инициализация")
    print("=" * 50)
    
    # Настройка логгера
    setup_logger()
    
    # Создание директорий
    create_directories()
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Проверка API ключа
    check_api_key()
    
    # Инициализация БД
    init_database()
    
    print("=" * 50)
    print("✅ Инициализация завершена!")
    print("=" * 50)
    print("\nСледующие шаги:")
    print("1. Запустите ETL Pipeline: python scripts/etl_pipeline.py")
    print("2. Обучите модель: python scripts/train_model.py")
    print("3. Запустите веб-приложение: python scripts/run_web.py")


if __name__ == "__main__":
    main()
