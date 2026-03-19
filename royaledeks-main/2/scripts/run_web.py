"""
Скрипт для запуска веб-приложения
"""
import sys
from pathlib import Path

# Добавление корня проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import run_server
from utils.logger import logger


def main():
    """Точка входа"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск веб-приложения")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для прослушивания"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоперезагрузка при изменениях"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Запуск веб-приложения")
    logger.info("=" * 50)
    logger.info(f"Хост: {args.host}")
    logger.info(f"Порт: {args.port}")
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
