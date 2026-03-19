"""
Модуль предобработки и векторизации данных
"""
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from config import (
    CARD_RARITIES, CARD_TYPES, SPECIAL_CARD_TYPES,
    MAX_SPECIAL_SLOTS, EMBEDDING_DIM
)
from utils.logger import logger


class CardVocabulary:
    """
    Словарь карт для кодирования/декодирования
    """
    
    def __init__(self):
        self.card_to_idx: Dict[int, int] = {}  # card_id -> index
        self.idx_to_card: Dict[int, int] = {}  # index -> card_id
        self.card_to_name: Dict[int, str] = {}  # card_id -> name
        self.name_to_card: Dict[str, int] = {}  # name -> card_id
        self.size: int = 0
    
    def build(self, cards: List[Dict[str, Any]]) -> None:
        """
        Построение словаря из списка карт
        
        Args:
            cards: Список словарей с данными карт
        """
        for idx, card in enumerate(sorted(cards, key=lambda x: x['card_id'])):
            card_id = card['card_id']
            name = card['name']
            
            self.card_to_idx[card_id] = idx
            self.idx_to_card[idx] = card_id
            self.card_to_name[card_id] = name
            self.name_to_card[name] = card_id
        
        self.size = len(self.card_to_idx)
        logger.info(f"Словарь построен: {self.size} карт")
    
    def card_id_to_idx(self, card_id: int) -> int:
        """Конвертация card_id в индекс словаря"""
        return self.card_to_idx.get(card_id, -1)
    
    def idx_to_card_id(self, idx: int) -> int:
        """Конвертация индекса в card_id"""
        return self.idx_to_card.get(idx, -1)
    
    def save(self, path: str) -> None:
        """Сохранение словаря"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'card_to_idx': self.card_to_idx,
                'idx_to_card': self.idx_to_card,
                'card_to_name': self.card_to_name,
                'name_to_card': self.name_to_card,
                'size': self.size
            }, f)
        logger.info(f"Словарь сохранён: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CardVocabulary':
        """Загрузка словаря"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls()
        vocab.card_to_idx = data['card_to_idx']
        vocab.idx_to_card = data['idx_to_card']
        vocab.card_to_name = data['card_to_name']
        vocab.name_to_card = data['name_to_card']
        vocab.size = data['size']
        
        logger.info(f"Словарь загружен: {path}")
        return vocab


class CardFeatureEncoder:
    """
    Кодировщик признаков карт
    
    Признаки для каждой карты:
    - Нормализованная стоимость эликсира (1 признак)
    - One-Hot вектор редкости (5 признаков)
    - One-Hot вектор типа (3 признака)
    - Бинарные флаги (is_evolveable, is_hero) (2 признаках)
    
    Итого: 1 + 5 + 3 + 2 = 11 дополнительных признаков
    """
    
    def __init__(self, cards: List[Dict[str, Any]]):
        """
        Инициализация кодировщика
        
        Args:
            cards: Список карт для построения маппингов
        """
        self.rarity_to_idx = {r: i for i, r in enumerate(CARD_RARITIES)}
        self.type_to_idx = {t: i for i, t in enumerate(CARD_TYPES)}
        
        # Маппинг card_id -> признаки
        self.card_features: Dict[int, np.ndarray] = {}
        self._build_features(cards)
        
        # Размер вектора признаков
        self.feature_dim = 1 + len(CARD_RARITIES) + len(CARD_TYPES) + 2
        
        logger.info(f"Кодировщик инициализирован, размер признаков: {self.feature_dim}")
    
    def _build_features(self, cards: List[Dict[str, Any]]) -> None:
        """Построение матрицы признаков для всех карт"""
        max_elixir = 10.0  # Максимальная стоимость эликсира
        
        for card in cards:
            card_id = card['card_id']
            
            # Нормализованный эликсир
            elixir_norm = card['elixir_cost'] / max_elixir
            
            # One-Hot редкость
            rarity_onehot = np.zeros(len(CARD_RARITIES))
            rarity_idx = self.rarity_to_idx.get(card['rarity'], 0)
            rarity_onehot[rarity_idx] = 1.0
            
            # One-Hot тип
            type_onehot = np.zeros(len(CARD_TYPES))
            type_idx = self.type_to_idx.get(card['type'], 0)
            type_onehot[type_idx] = 1.0
            
            # Бинарные флаги
            is_evolveable = 1.0 if card.get('is_evolveable', False) else 0.0
            is_hero = 1.0 if card.get('is_hero', False) else 0.0
            
            # Конкатенация всех признаков
            features = np.concatenate([
                np.array([elixir_norm]),
                rarity_onehot,
                type_onehot,
                np.array([is_evolveable, is_hero])
            ])
            
            self.card_features[card_id] = features
    
    def get_features(self, card_id: int) -> np.ndarray:
        """
        Получение вектора признаков карты
        
        Args:
            card_id: ID карты
            
        Returns:
            Вектор признаков
        """
        return self.card_features.get(card_id, np.zeros(self.feature_dim))
    
    def get_feature_dim(self) -> int:
        """Получение размерности вектора признаков"""
        return self.feature_dim


class DataPreprocessor:
    """
    Основной класс предобработки данных
    
    Функции:
    - Сортировка карт в колоде по эликсиру
    - Удаление дубликатов
    - Балансировка по среднему эликсиру
    - Создание последовательностей для обучения
    """
    
    def __init__(self, cards: List[Dict[str, Any]]):
        """
        Инициализация препроцессора
        
        Args:
            cards: Список всех карт
        """
        self.cards = cards
        self.card_info = {c['card_id']: c for c in cards}
        self.vocab = CardVocabulary()
        self.vocab.build(cards)
        self.encoder = CardFeatureEncoder(cards)
        
        # Специальные типы карт
        self.evolveable_cards = set(
            c['card_id'] for c in cards if c.get('is_evolveable', False)
        )
        self.hero_cards = set(
            c['card_id'] for c in cards if c.get('is_hero', False)
        )
        self.champion_cards = set(
            c['card_id'] for c in cards if c['rarity'] == 'Champion'
        )
        
        logger.info("Препроцессор инициализирован")
    
    def sort_deck_by_elixir(self, card_ids: List[int]) -> List[int]:
        """
        Сортировка карт колоды по стоимости эликсира
        
        Args:
            card_ids: Список ID карт
            
        Returns:
            Отсортированный список
        """
        return sorted(
            card_ids,
            key=lambda cid: self.card_info.get(cid, {}).get('elixir_cost', 0)
        )
    
    def remove_duplicate_decks(self, decks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Удаление полных дубликатов колод
        
        Args:
            decks: Список колод
            
        Returns:
            Список без дубликатов
        """
        seen = set()
        unique_decks = []
        
        for deck in decks:
            # Сортировка карт для нормализации представления
            sorted_cards = tuple(sorted(deck['cards']))
            
            if sorted_cards not in seen:
                seen.add(sorted_cards)
                unique_decks.append(deck)
        
        logger.info(f"Удалено дубликатов: {len(decks) - len(unique_decks)}")
        return unique_decks
    
    def balance_by_elixir(
        self,
        decks: List[Dict[str, Any]],
        n_bins: int = 5,
        max_per_bin: int = None
    ) -> List[Dict[str, Any]]:
        """
        Балансировка выборки по среднему эликсиру
        
        Args:
            decks: Список колод
            n_bins: Количество бинов для гистограммы
            max_per_bin: Максимум колод в бине (None = без лимита)
            
        Returns:
            Сбалансированный список колод
        """
        # Распределение по бинам
        elixir_values = [d['avg_elixir'] for d in decks]
        bin_edges = np.linspace(
            min(elixir_values),
            max(elixir_values) + 0.1,
            n_bins + 1
        )
        
        bins = defaultdict(list)
        for deck in decks:
            bin_idx = np.digitize(deck['avg_elixir'], bin_edges[:-1]) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            bins[bin_idx].append(deck)
        
        # Логирование распределения
        logger.info("Распределение по эликсиру:")
        for i in range(n_bins):
            logger.info(f"  Бин {i} ({bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}): {len(bins[i])} колод")
        
        # Балансировка
        if max_per_bin:
            balanced = []
            for i in range(n_bins):
                bins[i] = bins[i][:max_per_bin]
                balanced.extend(bins[i])
            logger.info(f"После балансировки: {len(balanced)} колод")
            return balanced
        
        return decks
    
    def create_training_sequences(
        self,
        decks: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Создание последовательностей для обучения
        
        Для авторегрессионного обучения:
        - Input: последовательность карт [START, card_1, card_2, ..., card_{n-1}]
        - Target: последовательность карт [card_1, card_2, ..., card_n, END]
        
        Args:
            decks: Список колод
            
        Returns:
            (input_sequences, target_sequences, card_features)
        """
        input_seqs = []
        target_seqs = []
        all_features = []
        
        START_TOKEN = self.vocab.size  # Специальный токен начала
        # END_TOKEN не нужен явно, т.к. последовательность фиксированной длины
        
        for deck in decks:
            # Сортировка карт
            sorted_cards = self.sort_deck_by_elixir(deck['cards'])
            
            # Конвертация в индексы словаря
            card_indices = [
                self.vocab.card_id_to_idx(cid)
                for cid in sorted_cards
                if self.vocab.card_id_to_idx(cid) != -1
            ]
            
            if len(card_indices) != 8:
                continue  # Пропуск невалидных колод
            
            # Создание последовательностей
            input_seq = [START_TOKEN] + card_indices[:-1]
            target_seq = card_indices
            
            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            
            # Признаки для каждой карты
            deck_features = [
                self.encoder.get_features(cid)
                for cid in sorted_cards
            ]
            all_features.append(deck_features)
        
        # Конвертация в тензоры
        input_tensor = torch.LongTensor(input_seqs)
        target_tensor = torch.LongTensor(target_seqs)
        features_tensor = torch.FloatTensor(np.array(all_features))
        
        logger.info(f"Создано {len(input_seqs)} обучающих последовательностей")
        logger.info(f"  Input shape: {input_tensor.shape}")
        logger.info(f"  Target shape: {target_tensor.shape}")
        logger.info(f"  Features shape: {features_tensor.shape}")
        
        return input_tensor, target_tensor, features_tensor
    
    def get_card_metadata(self) -> Dict[int, Dict[str, Any]]:
        """
        Получение метаданных всех карт
        
        Returns:
            Словарь card_id -> метаданные
        """
        return self.card_info
    
    def save(self, save_dir: str) -> None:
        """
        Сохранение препроцессора
        
        Args:
            save_dir: Директория для сохранения
        """
        import pickle
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение словаря
        self.vocab.save(str(save_path / 'vocabulary.pkl'))
        
        # Сохранение кодировщика
        with open(save_path / 'encoder.pkl', 'wb') as f:
            pickle.dump({
                'card_features': self.encoder.card_features,
                'feature_dim': self.encoder.feature_dim
            }, f)
        
        # Сохранение метаданных
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'card_info': self.card_info,
                'evolveable_cards': self.evolveable_cards,
                'hero_cards': self.hero_cards,
                'champion_cards': self.champion_cards
            }, f)
        
        logger.info(f"Препроцессор сохранён: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, cards: List[Dict[str, Any]]) -> 'DataPreprocessor':
        """
        Загрузка препроцессора
        
        Args:
            load_dir: Директория загрузки
            cards: Список карт (для инициализации)
            
        Returns:
            Загруженный препроцессор
        """
        import pickle
        from pathlib import Path
        
        load_path = Path(load_dir)
        
        # Создание базового объекта
        preprocessor = cls(cards)
        
        # Загрузка словаря
        preprocessor.vocab = CardVocabulary.load(str(load_path / 'vocabulary.pkl'))
        
        # Загрузка кодировщика
        with open(load_path / 'encoder.pkl', 'rb') as f:
            encoder_data = pickle.load(f)
            preprocessor.encoder.card_features = encoder_data['card_features']
            preprocessor.encoder.feature_dim = encoder_data['feature_dim']
        
        # Загрузка метаданных
        with open(load_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            preprocessor.card_info = metadata['card_info']
            preprocessor.evolveable_cards = metadata['evolveable_cards']
            preprocessor.hero_cards = metadata['hero_cards']
            preprocessor.champion_cards = metadata['champion_cards']
        
        logger.info(f"Препроцессор загружен: {load_dir}")
        return preprocessor
