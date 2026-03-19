"""
Утилиты логирования
"""
import sys
from pathlib import Path
from loguru import logger

# Базовый путь проекта
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(log_file: str = "app.log", level: str = "INFO") -> None:
    """
    Настройка логгера для проекта

    Args:
        log_file: Имя файла лога
        level: Уровень логирования
    """
    log_path = LOGS_DIR / log_file

    # Удаление стандартного обработчика
    logger.remove()

    # Консольный вывод
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Файловый вывод
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )

    logger.info("Логгер инициализирован")


# Автоинициализация при импорте
setup_logger()
