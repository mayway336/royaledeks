"""
Скрипт для проверки веб-парсера
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.parser import ClashRoyaleAPI
from utils.logger import logger


def check_parser():
    """Проверка работы парсера"""
    print("=" * 50)
    print("Проверка веб-парсера Clash Royale")
    print("=" * 50)
    print()
    
    api = ClashRoyaleAPI()
    
    try:
        # Проверка парсинга карт
        print("Тест 1: Парсинг карт (Fandom Wiki)...")
        cards = api.get_all_cards()
        
        if cards:
            print(f"✓ Успешно! Загружено {len(cards)} карт")
            print(f"  Пример: {cards[0]['name']} (эликсир: {cards[0]['elixir_cost']})")
        else:
            print("⚠ Не удалось загрузить карты (будут созданы карты по умолчанию)")
        
        print()
        
        # Проверка парсинга колод
        print("Тест 2: Парсинг колод...")
        decks = api.get_top_decks(limit=5, min_games=1)
        
        if decks:
            print(f"✓ Успешно! Загружено {len(decks)} колод")
            if decks:
                print(f"  Пример колоды: {len(decks[0].get('cards', []))} карт")
                print(f"  Winrate: {decks[0].get('win_rate', 0):.1%}")
        else:
            print("⚠ Не удалось загрузить колоды (будут созданы тестовые колоды)")
        
        print()
        print("=" * 50)
        print("✓ Проверка завершена!")
        print("=" * 50)
        print()
        print("Теперь вы можете запустить:")
        print("  python scripts/etl_pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Точка входа"""
    success = check_parser()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
