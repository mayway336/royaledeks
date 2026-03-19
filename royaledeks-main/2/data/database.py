"""
Модуль работы с базой данных SQLite
"""
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from config import MIN_GAMES_PLAYED, SEASON_MONTHS_BACK
from utils.logger import logger


class Database:
    """
    Класс для работы с SQLite базой данных
    
    Таблицы:
    - cards: Информация о картах
    - decks: Колоды карт
    - deck_cards: Связь колод и карт (многие-ко-многим)
    """
    
    def __init__(self, db_path: str):
        """
        Инициализация подключения к БД
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        
        logger.info(f"Инициализация БД: {db_path}")
    
    def connect(self) -> None:
        """Подключение к базе данных"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        logger.debug("Подключение к БД установлено")
    
    def disconnect(self) -> None:
        """Отключение от базы данных"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.debug("Подключение к БД закрыто")
    
    def create_tables(self) -> None:
        """
        Создание таблиц базы данных
        """
        if not self.conn:
            self.connect()
        
        # Таблица карт
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                card_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                base_name TEXT,
                elixir_cost REAL NOT NULL,
                rarity TEXT NOT NULL CHECK(rarity IN (
                    'Common', 'Rare', 'Epic', 'Legendary', 'Champion'
                )),
                type TEXT NOT NULL CHECK(type IN (
                    'Troop', 'Spell', 'Building'
                )),
                is_evolveable BOOLEAN NOT NULL DEFAULT 0,
                is_hero BOOLEAN NOT NULL DEFAULT 0,
                is_champion BOOLEAN NOT NULL DEFAULT 0,
                is_evolution BOOLEAN NOT NULL DEFAULT 0,
                icon_url TEXT,
                source TEXT DEFAULT 'fandom',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._ensure_cards_columns()
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_base_name ON cards(base_name)")
        
        # Таблица колод
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS decks (
                deck_id TEXT PRIMARY KEY,
                avg_elixir REAL NOT NULL,
                win_rate REAL NOT NULL,
                games_played INTEGER NOT NULL,
                trophy_limit INTEGER,
                season TEXT,
                source TEXT NOT NULL DEFAULT 'royaleapi',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица связи колод и карт
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS deck_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck_id TEXT NOT NULL,
                card_id INTEGER NOT NULL,
                card_order INTEGER NOT NULL,
                FOREIGN KEY (deck_id) REFERENCES decks(deck_id) ON DELETE CASCADE,
                FOREIGN KEY (card_id) REFERENCES cards(card_id),
                UNIQUE(deck_id, card_id)
            )
        """)
        
        # Индексы для ускорения поиска
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deck_cards_deck_id 
            ON deck_cards(deck_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deck_cards_card_id 
            ON deck_cards(card_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decks_games_played 
            ON decks(games_played)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_decks_timestamp 
            ON decks(timestamp)
        """)
        
        self.conn.commit()
        logger.info("Таблицы БД созданы")

    def _ensure_cards_columns(self) -> None:
        """Добавление новых колонок в существующую таблицу cards (миграция без Alembic)."""
        self.cursor.execute("PRAGMA table_info(cards)")
        existing_cols = {row["name"] for row in self.cursor.fetchall()}
        for name, ddl in [
            ("base_name", "ALTER TABLE cards ADD COLUMN base_name TEXT"),
            ("is_champion", "ALTER TABLE cards ADD COLUMN is_champion BOOLEAN NOT NULL DEFAULT 0"),
            ("is_evolution", "ALTER TABLE cards ADD COLUMN is_evolution BOOLEAN NOT NULL DEFAULT 0"),
            ("source", "ALTER TABLE cards ADD COLUMN source TEXT DEFAULT 'fandom'"),
        ]:
            if name not in existing_cols:
                self.cursor.execute(ddl)
    
    def insert_card(self, card_data: Dict[str, Any]) -> int:
        """
        Вставка или обновление карты
        
        Args:
            card_data: Словарь с данными карты
            
        Returns:
            card_id карты
        """
        if not self.conn:
            self.connect()
        
        self.cursor.execute("""
            INSERT OR REPLACE INTO cards 
            (
                card_id, name, base_name, elixir_cost, rarity, type,
                is_evolveable, is_hero, is_champion, is_evolution,
                icon_url, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            card_data['card_id'],
            card_data['name'],
            card_data.get('base_name', None),
            card_data['elixir_cost'],
            card_data['rarity'],
            card_data['type'],
            card_data.get('is_evolveable', False),
            card_data.get('is_hero', False),
            card_data.get('is_champion', False),
            card_data.get('is_evolution', False),
            card_data.get('icon_url', None),
            card_data.get('source', 'fandom')
        ))
        
        self.conn.commit()
        return card_data['card_id']
    
    def insert_deck(self, deck_data: Dict[str, Any], cards: List[int]) -> str:
        """
        Вставка колоды с картами
        
        Args:
            deck_data: Словарь с данными колоды
            cards: Список card_id карт в порядке
            
        Returns:
            deck_id колоды
        """
        if not self.conn:
            self.connect()
        
        deck_id = deck_data['deck_id']
        
        # Вставка колоды
        self.cursor.execute("""
            INSERT OR REPLACE INTO decks 
            (deck_id, avg_elixir, win_rate, games_played, trophy_limit, season, source, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            deck_id,
            deck_data['avg_elixir'],
            deck_data['win_rate'],
            deck_data['games_played'],
            deck_data.get('trophy_limit', None),
            deck_data.get('season', None),
            deck_data.get('source', 'royaleapi'),
            deck_data.get('timestamp', datetime.now())
        ))
        
        # Удаление старых связей карт
        self.cursor.execute("DELETE FROM deck_cards WHERE deck_id = ?", (deck_id,))
        
        # Вставка связей карт
        for order, card_id in enumerate(cards):
            self.cursor.execute("""
                INSERT INTO deck_cards (deck_id, card_id, card_order)
                VALUES (?, ?, ?)
            """, (deck_id, card_id, order))
        
        self.conn.commit()
        return deck_id
    
    def get_all_cards(self) -> List[Dict[str, Any]]:
        """
        Получение всех карт
        
        Returns:
            Список словарей с данными карт
        """
        if not self.conn:
            self.connect()
        
        self.cursor.execute("SELECT * FROM cards ORDER BY card_id")
        rows = self.cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_card_by_id(self, card_id: int) -> Optional[Dict[str, Any]]:
        """
        Получение карты по ID
        
        Args:
            card_id: ID карты
            
        Returns:
            Словарь с данными карты или None
        """
        if not self.conn:
            self.connect()
        
        self.cursor.execute("SELECT * FROM cards WHERE card_id = ?", (card_id,))
        row = self.cursor.fetchone()
        
        return dict(row) if row else None
    
    def get_card_id_by_name(self, name: str) -> Optional[int]:
        """
        Получение ID карты по имени

        Args:
            name: Имя карты

        Returns:
            ID карты или None
        """
        if not self.conn:
            self.connect()

        self.cursor.execute("SELECT card_id FROM cards WHERE name = ?", (name,))
        row = self.cursor.fetchone()

        return row['card_id'] if row else None

    def get_card_id_by_partial_name(self, name: str) -> Optional[int]:
        """
        Получение ID карты по частичному совпадению имени

        Args:
            name: Часть имени карты

        Returns:
            ID карты или None
        """
        if not self.conn:
            self.connect()

        # Поиск по частичному совпадению
        self.cursor.execute("SELECT card_id FROM cards WHERE name LIKE ?", (f"%{name}%",))
        row = self.cursor.fetchone()

        return row['card_id'] if row else None
    
    def get_filtered_decks(
        self,
        min_games: int = MIN_GAMES_PLAYED,
        months_back: int = SEASON_MONTHS_BACK
    ) -> List[Dict[str, Any]]:
        """
        Получение отфильтрованных колод по критериям
        
        Args:
            min_games: Минимальное количество игр
            months_back: За сколько месяцев загружать данные
            
        Returns:
            Список словарей с данными колод
        """
        if not self.conn:
            self.connect()
        
        cutoff_date = datetime.now() - timedelta(days=30 * months_back)
        
        self.cursor.execute("""
            SELECT d.*, GROUP_CONCAT(dc.card_id || ':' || dc.card_order) as cards_data
            FROM decks d
            JOIN deck_cards dc ON d.deck_id = dc.deck_id
            WHERE d.games_played >= ?
            AND d.timestamp >= ?
            GROUP BY d.deck_id
            ORDER BY d.games_played DESC
        """, (min_games, cutoff_date.isoformat()))
        
        rows = self.cursor.fetchall()
        result = []
        
        for row in rows:
            deck_dict = dict(row)
            # Парсинг карт из строки "card_id:order,card_id:order,..."
            cards_data = deck_dict.pop('cards_data', '')
            cards = []
            if cards_data:
                card_pairs = [pair.split(':') for pair in cards_data.split(',')]
                cards = sorted([(int(cid), int(ord)) for cid, ord in card_pairs], key=lambda x: x[1])
                deck_dict['cards'] = [cid for cid, _ in cards]
            else:
                deck_dict['cards'] = []
            
            result.append(deck_dict)
        
        logger.info(f"Получено {len(result)} колод после фильтрации")
        return result
    
    def get_deck_cards(self, deck_id: str) -> List[int]:
        """
        Получение списка карт колоды
        
        Args:
            deck_id: ID колоды
            
        Returns:
            Список card_id в порядке
        """
        if not self.conn:
            self.connect()
        
        self.cursor.execute("""
            SELECT card_id FROM deck_cards 
            WHERE deck_id = ? 
            ORDER BY card_order
        """, (deck_id,))
        
        rows = self.cursor.fetchall()
        return [row['card_id'] for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики базы данных
        
        Returns:
            Словарь со статистикой
        """
        if not self.conn:
            self.connect()
        
        stats = {}
        
        # Количество карт
        self.cursor.execute("SELECT COUNT(*) as count FROM cards")
        stats['total_cards'] = self.cursor.fetchone()['count']
        
        # Количество колод
        self.cursor.execute("SELECT COUNT(*) as count FROM decks")
        stats['total_decks'] = self.cursor.fetchone()['count']
        
        # Колоды по источникам
        self.cursor.execute("""
            SELECT source, COUNT(*) as count 
            FROM decks 
            GROUP BY source
        """)
        stats['decks_by_source'] = {row['source']: row['count'] for row in self.cursor.fetchall()}
        
        # Средняя винрейт
        self.cursor.execute("SELECT AVG(win_rate) as avg_wr FROM decks")
        stats['avg_win_rate'] = self.cursor.fetchone()['avg_wr']
        
        return stats
    
    def clear_decks(self) -> None:
        """Очистка таблицы колод (перед полной перезагрузкой данных)"""
        if not self.conn:
            self.connect()
        
        self.cursor.execute("DELETE FROM deck_cards")
        self.cursor.execute("DELETE FROM decks")
        self.conn.commit()
        logger.info("Таблица колод очищена")
    
    def __enter__(self):
        """Контекстный менеджер - вход"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход"""
        self.disconnect()
