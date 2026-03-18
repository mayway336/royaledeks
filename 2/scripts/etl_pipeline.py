"""
Скрипт для запуска ETL Pipeline
Загружает данные и сохраняет в БД
"""
import sys
from pathlib import Path

# Добавление корня проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH, MIN_GAMES_PLAYED
from data.database import Database
from data.parser import ClashRoyaleAPI
from utils.logger import logger


def run_etl():
    """Запуск ETL процесса"""
    logger.info("=" * 50)
    logger.info("Запуск ETL Pipeline")
    logger.info("=" * 50)
    
    # Инициализация БД
    logger.info("Инициализация базы данных...")
    db = Database(str(DATABASE_PATH))
    db.connect()
    db.create_tables()
    
    # Инициализация парсера
    logger.info("Инициализация парсера...")
    api = ClashRoyaleAPI()
    
    try:
        # Загрузка карт
        logger.info("Загрузка карт...")
        cards = api.get_all_cards()
        
        if not cards:
            logger.error("Не удалось загрузить карты")
            return False
        
        # Сохранение карт в БД
        logger.info(f"Сохранение {len(cards)} карт в БД...")
        for card in cards:
            db.insert_card(card)
        
        logger.info(f"✓ Карты сохранены: {len(cards)}")
        
        # Загрузка колод
        logger.info("Загрузка колод...")
        top_decks = api.get_top_decks(limit=500, min_games=MIN_GAMES_PLAYED)
        
        if not top_decks:
            logger.warning("Не удалось загрузить колоды")
            return False
        
        # Сохранение колод в БД
        logger.info(f"Сохранение {len(top_decks)} колод в БД...")
        saved_decks = 0
        
        for deck in top_decks:
            # Конвертация названий карт в ID
            card_ids = []
            for card_name in deck.get('cards', []):
                card_id = db.get_card_id_by_name(card_name)
                if not card_id:
                    # Поиск по частичному совпадению
                    card_id = db.get_card_id_by_partial_name(card_name)
                if card_id:
                    card_ids.append(card_id)
            
            # Только колоды с полным набором карт
            if len(card_ids) == 8:
                db.insert_deck(deck, card_ids)
                saved_decks += 1
            else:
                logger.debug(f"Колода пропущена (найдено {len(card_ids)} из 8 карт)")
        
        logger.info(f"✓ Колоды сохранены: {saved_decks}")
        
        # Статистика
        stats = db.get_stats()
        logger.info("=" * 50)
        logger.info("Статистика БД:")
        logger.info(f"  Всего карт: {stats['total_cards']}")
        logger.info(f"  Всего колод: {stats['total_decks']}")
        logger.info(f"  Средний winrate: {stats.get('avg_win_rate', 0):.2%}")
        logger.info("=" * 50)
        
        logger.info("✓ ETL Pipeline завершён успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка ETL: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.disconnect()


def main():
    """Точка входа"""
    success = run_etl()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
