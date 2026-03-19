"""
Rule Engine - Модуль динамического маскирования и валидации колод

Правила слотов Clash Royale (актуальные):
- 1 слот под Эволюцию (обязательно)
- 1 слот под Героя (обязательно)
- 1 Wild-слот (Эволюция ИЛИ Герой ИЛИ Чемпион)

Итого: максимум 3 специальных слота
Валидные комбинации:
- 1 Evolution + 1 Hero + 1 Champion/Wild
- 2 Evolutions + 1 Hero
- 1 Evolution + 2 Heroes
- и т.д.
"""
import torch
from typing import List, Set, Dict, Optional, Tuple
from collections import Counter

from config import MAX_SPECIAL_SLOTS
from utils.logger import logger


class SlotValidator:
    """
    Валидатор слотов для проверки корректности колоды
    """
    
    def __init__(
        self,
        evolveable_cards: Set[int],
        hero_cards: Set[int],
        champion_cards: Set[int]
    ):
        """
        Инициализация валидатора
        
        Args:
            evolveable_cards: Множество ID карт с эволюцией
            hero_cards: Множество ID карт-героев
            champion_cards: Множество ID карт-чемпионов
        """
        self.evolveable_cards = evolveable_cards
        self.hero_cards = hero_cards
        self.champion_cards = champion_cards
        
        # Чемпионы тоже могут быть в wild слоте
        self.special_cards = evolveable_cards | hero_cards | champion_cards
        
        logger.info(f"SlotValidator инициализирован")
        logger.info(f"  Evolution карт: {len(evolveable_cards)}")
        logger.info(f"  Hero карт: {len(hero_cards)}")
        logger.info(f"  Champion карт: {len(champion_cards)}")
    
    def get_card_type(self, card_id: int) -> Optional[str]:
        """
        Определение типа специальной карты
        
        Args:
            card_id: ID карты
            
        Returns:
            Тип карты или None
        """
        if card_id in self.evolveable_cards:
            return 'evolution'
        if card_id in self.hero_cards:
            return 'hero'
        if card_id in self.champion_cards:
            return 'champion'
        return None
    
    def count_special_cards(self, card_ids: List[int]) -> Dict[str, int]:
        """
        Подсчёт специальных карт в колоде
        
        Args:
            card_ids: Список ID карт
            
        Returns:
            Словарь с количеством каждого типа
        """
        counts = {
            'evolution': 0,
            'hero': 0,
            'champion': 0,
            'total_special': 0
        }
        
        for card_id in card_ids:
            card_type = self.get_card_type(card_id)
            if card_type:
                counts[card_type] += 1
                counts['total_special'] += 1
        
        return counts
    
    def validate_deck(self, card_ids: List[int]) -> Tuple[bool, str]:
        """
        Полная валидация колоды
        
        Args:
            card_ids: Список ID карт (8 штук)
            
        Returns:
            (is_valid, message)
        """
        if len(card_ids) != 8:
            return False, f"Неверное количество карт: {len(card_ids)} (ожидалось 8)"
        
        # Проверка на дубликаты
        if len(set(card_ids)) != 8:
            return False, "Обнаружены дубликаты карт"
        
        # Подсчёт специальных карт
        counts = self.count_special_cards(card_ids)
        
        # Проверка лимита специальных слотов
        if counts['total_special'] > MAX_SPECIAL_SLOTS:
            return False, f"Превышен лимит специальных слотов: {counts['total_special']} > {MAX_SPECIAL_SLOTS}"
        
        # Проверка: не более 2 карт одного типа (эволюции/герои)
        if counts['evolution'] > 2:
            return False, f"Слишком много эволюций: {counts['evolution']} (максимум 2)"
        
        if counts['hero'] > 2:
            return False, f"Слишком много героев: {counts['hero']} (максимум 2)"
        
        return True, "Колода валидна"
    
    def can_add_card(
        self,
        current_cards: List[int],
        new_card_id: int
    ) -> Tuple[bool, str]:
        """
        Проверка возможности добавления карты в текущую колоду
        
        Args:
            current_cards: Текущие карты в колоде
            new_card_id: ID добавляемой карты
            
        Returns:
            (can_add, reason)
        """
        # Проверка на дубликат
        if new_card_id in current_cards:
            return False, "Карта уже присутствует в колоде"
        
        # Проверка лимита специальных карт
        new_type = self.get_card_type(new_card_id)
        if new_type:
            counts = self.count_special_cards(current_cards)
            if counts['total_special'] >= MAX_SPECIAL_SLOTS:
                return False, "Достигнут лимит специальных слотов"
            
            # Дополнительная проверка для эволюций и героев
            if new_type == 'evolution' and counts['evolution'] >= 2:
                return False, "Максимум 2 эволюции в колоде"
            
            if new_type == 'hero' and counts['hero'] >= 2:
                return False, "Максимум 2 героя в колоде"
        
        return True, "OK"


class RuleEngine:
    """
    Движок правил для динамического маскирования во время генерации
    
    На каждом шаге генерации создаёт маску, исключающую:
    - Уже выбранные карты
    - Карты, нарушающие правила слотов
    """
    
    def __init__(
        self,
        vocab_size: int,
        evolveable_cards: Optional[Set[int]] = None,
        hero_cards: Optional[Set[int]] = None,
        champion_cards: Optional[Set[int]] = None
    ):
        """
        Инициализация Rule Engine
        
        Args:
            vocab_size: Размер словаря карт
            evolveable_cards: Множество ID эволюций (опционально)
            hero_cards: Множество ID героев (опционально)
            champion_cards: Множество ID чемпионов (опционально)
        """
        self.vocab_size = vocab_size
        
        # Инициализация валидатора (если переданы данные)
        if evolveable_cards is not None:
            self.validator = SlotValidator(
                evolveable_cards,
                hero_cards or set(),
                champion_cards or set()
            )
        else:
            # Пустые множества по умолчанию
            self.validator = SlotValidator(set(), set(), set())
        
        logger.info(f"RuleEngine инициализирован: vocab_size={vocab_size}")
    
    def set_card_sets(
        self,
        evolveable_cards: Set[int],
        hero_cards: Set[int],
        champion_cards: Set[int]
    ) -> None:
        """
        Установка множеств специальных карт
        
        Args:
            evolveable_cards: Множество ID карт с эволюцией
            hero_cards: Множество ID карт-героев
            champion_cards: Множество ID карт-чемпионов
        """
        self.validator = SlotValidator(
            evolveable_cards, hero_cards, champion_cards
        )
    
    def create_mask(
        self,
        generated_cards: Optional[List[List[int]]] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Создание маски для исключения невалидных карт
        
        Args:
            generated_cards: Сгенерированные карты на текущий момент
                          List[batch_cards] или None для начала генерации
            batch_size: Размер батча
            
        Returns:
            Маска [batch_size, vocab_size] где 1 = разрешено, 0 = запрещено
        """
        # Инициализация маски (все разрешены)
        mask = torch.ones(batch_size, self.vocab_size, dtype=torch.float32)
        
        if generated_cards is None or len(generated_cards) == 0:
            return mask
        
        # Для каждого элемента в батче
        for batch_idx, batch_cards in enumerate(generated_cards):
            if batch_idx >= batch_size:
                break
            
            # Исключение уже выбранных карт
            for card_id in batch_cards:
                if 0 <= card_id < self.vocab_size:
                    mask[batch_idx, card_id] = 0.0
            
            # Проверка лимита специальных карт
            if batch_cards:
                counts = self.validator.count_special_cards(batch_cards)
                
                if counts['total_special'] >= MAX_SPECIAL_SLOTS:
                    # Запрет всех специальных карт
                    for card_id in self.validator.special_cards:
                        if 0 <= card_id < self.vocab_size:
                            mask[batch_idx, card_id] = 0.0
                
                # Дополнительный запрет при достижении лимитов по типам
                if counts['evolution'] >= 2:
                    for card_id in self.validator.evolveable_cards:
                        if 0 <= card_id < self.vocab_size:
                            mask[batch_idx, card_id] = 0.0
                
                if counts['hero'] >= 2:
                    for card_id in self.validator.hero_cards:
                        if 0 <= card_id < self.vocab_size:
                            mask[batch_idx, card_id] = 0.0
        
        return mask
    
    def validate_generated_deck(
        self,
        card_ids: List[int],
        card_id_mapping: Optional[Dict[int, int]] = None
    ) -> Tuple[bool, str]:
        """
        Валидация сгенерированной колоды
        
        Args:
            card_ids: ID сгенерированных карт (индексы словаря)
            card_id_mapping: Маппинг index -> actual_card_id (опционально)
            
        Returns:
            (is_valid, message)
        """
        # Конвертация индексов в card_id если нужно
        if card_id_mapping:
            actual_ids = [card_id_mapping.get(idx, idx) for idx in card_ids]
        else:
            actual_ids = card_ids
        
        return self.validator.validate_deck(actual_ids)
    
    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Применение маски к логитам модели
        
        Args:
            logits: Логиты модели [batch_size, vocab_size]
            mask: Маска валидности [batch_size, vocab_size]
            
        Returns:
            Замаскированные логиты
        """
        # Замена невалидных логитов на -inf
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        return masked_logits
    
    def get_valid_cards_count(self, mask: torch.Tensor) -> List[int]:
        """
        Подсчёт количества валидных карт для каждого элемента батча
        
        Args:
            mask: Маска [batch_size, vocab_size]
            
        Returns:
            Список количеств валидных карт
        """
        return mask.sum(dim=1).tolist()


class DeckValidator:
    """
    Высокоуровневый валидатор для проверки готовых колод
    """
    
    def __init__(self, rule_engine: RuleEngine):
        """
        Инициализация валидатора
        
        Args:
            rule_engine: RuleEngine с настроенными правилами
        """
        self.rule_engine = rule_engine
    
    def validate_batch(
        self,
        decks: List[List[int]],
        card_id_mapping: Optional[Dict[int, int]] = None
    ) -> List[Tuple[bool, str]]:
        """
        Валидация батча колод
        
        Args:
            decks: Список колод (каждая - список ID карт)
            card_id_mapping: Маппинг индексов в card_id
            
        Returns:
            Список (is_valid, message) для каждой колоды
        """
        results = []
        for deck in decks:
            is_valid, message = self.rule_engine.validate_generated_deck(
                deck, card_id_mapping
            )
            results.append((is_valid, message))
        
        return results
    
    def get_statistics(
        self,
        validation_results: List[Tuple[bool, str]]
    ) -> Dict[str, float]:
        """
        Статистика валидации
        
        Args:
            validation_results: Результаты валидации
            
        Returns:
            Словарь со статистикой
        """
        total = len(validation_results)
        valid = sum(1 for is_valid, _ in validation_results if is_valid)
        
        # Подсчёт причин невалидности
        invalid_reasons = Counter(
            message for is_valid, message in validation_results if not is_valid
        )
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'validity_rate': valid / total if total > 0 else 0.0,
            'invalid_reasons': dict(invalid_reasons)
        }


__all__ = ["SlotValidator", "RuleEngine", "DeckValidator"]
