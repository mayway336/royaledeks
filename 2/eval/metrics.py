"""
Модуль оценки качества модели
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import torch.nn.functional as F

from utils.logger import logger


class EvaluationMetrics:
    """
    Метрики для оценки качества генерации колод
    
    Метрики:
    - Validity Rate: Доля валидных колод
    - Diversity: Уникальность генераций
    - Meta-Similarity: Сходство с топ-колодами
    - Card Distribution Similarity: Сходство распределения карт
    """
    
    def __init__(
        self,
        rule_engine,
        vocab_size: int,
        card_id_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Инициализация метрик
        
        Args:
            rule_engine: RuleEngine для валидации
            vocab_size: Размер словаря
            card_id_mapping: Маппинг индекс -> card_id
        """
        self.rule_engine = rule_engine
        self.vocab_size = vocab_size
        self.card_id_mapping = card_id_mapping or {i: i for i in range(vocab_size)}
        
        logger.info("EvaluationMetrics инициализирован")
    
    def compute_validity_rate(
        self,
        generated_decks: List[List[int]]
    ) -> Tuple[float, Dict]:
        """
        Вычисление Validity Rate
        
        Args:
            generated_decks: Список сгенерированных колод
            
        Returns:
            (validity_rate, detailed_stats)
        """
        valid_count = 0
        invalid_reasons = Counter()
        
        for deck in generated_decks:
            is_valid, reason = self.rule_engine.validate_generated_deck(
                deck, self.card_id_mapping
            )
            if is_valid:
                valid_count += 1
            else:
                # Категоризация причин
                if "дубликат" in reason.lower():
                    invalid_reasons['duplicates'] += 1
                elif "лимит" in reason.lower() or "слот" in reason.lower():
                    invalid_reasons['slot_violation'] += 1
                elif "количество" in reason.lower():
                    invalid_reasons['wrong_size'] += 1
                else:
                    invalid_reasons['other'] += 1
        
        validity_rate = valid_count / len(generated_decks) if generated_decks else 0.0
        
        stats = {
            'total': len(generated_decks),
            'valid': valid_count,
            'invalid': len(generated_decks) - valid_count,
            'invalid_reasons': dict(invalid_reasons)
        }
        
        logger.info(f"Validity Rate: {validity_rate:.2%}")
        
        return validity_rate, stats
    
    def compute_diversity(
        self,
        generated_decks: List[List[int]]
    ) -> Tuple[float, Dict]:
        """
        Вычисление Diversity (уникальность колод)
        
        Args:
            generated_decks: Список сгенерированных колод
            
        Returns:
            (diversity_score, detailed_stats)
        """
        # Нормализация колод (сортировка для сравнения)
        normalized_decks = [tuple(sorted(deck)) for deck in generated_decks]
        
        # Подсчёт уникальных
        unique_decks = set(normalized_decks)
        
        diversity = len(unique_decks) / len(generated_decks) if generated_decks else 0.0
        
        # Подсчёт повторений
        deck_counts = Counter(normalized_decks)
        most_common = deck_counts.most_common(5)
        
        stats = {
            'total': len(generated_decks),
            'unique': len(unique_decks),
            'top_repeated': most_common
        }
        
        logger.info(f"Diversity: {diversity:.2%} ({len(unique_decks)}/{len(generated_decks)})")
        
        return diversity, stats
    
    def compute_meta_similarity(
        self,
        generated_decks: List[List[int]],
        meta_decks: List[List[int]]
    ) -> Tuple[float, Dict]:
        """
        Вычисление Meta-Similarity (косинусное сходство распределения карт)
        
        Args:
            generated_decks: Сгенерированные колоды
            meta_decks: Топ колоды из меты
            
        Returns:
            (similarity_score, detailed_stats)
        """
        # Подсчёт частот карт
        gen_card_counts = Counter()
        for deck in generated_decks:
            gen_card_counts.update(deck)
        
        meta_card_counts = Counter()
        for deck in meta_decks:
            meta_card_counts.update(deck)
        
        # Векторизация
        all_cards = set(gen_card_counts.keys()) | set(meta_card_counts.keys())
        
        gen_vector = np.array([gen_card_counts.get(c, 0) for c in all_cards], dtype=float)
        meta_vector = np.array([meta_card_counts.get(c, 0) for c in all_cards], dtype=float)
        
        # Нормализация
        gen_vector = gen_vector / gen_vector.sum() if gen_vector.sum() > 0 else gen_vector
        meta_vector = meta_vector / meta_vector.sum() if meta_vector.sum() > 0 else meta_vector
        
        # Косинусное сходство
        dot_product = np.dot(gen_vector, meta_vector)
        norm_gen = np.linalg.norm(gen_vector)
        norm_meta = np.linalg.norm(meta_vector)
        
        if norm_gen > 0 and norm_meta > 0:
            similarity = dot_product / (norm_gen * norm_meta)
        else:
            similarity = 0.0
        
        stats = {
            'gen_unique_cards': len(gen_card_counts),
            'meta_unique_cards': len(meta_card_counts),
            'common_cards': len(set(gen_card_counts.keys()) & set(meta_card_counts.keys()))
        }
        
        logger.info(f"Meta-Similarity: {similarity:.4f}")
        
        return similarity, stats
    
    def compute_card_distribution(
        self,
        generated_decks: List[List[int]]
    ) -> Dict[int, float]:
        """
        Вычисление распределения частот карт
        
        Args:
            generated_decks: Список колод
            
        Returns:
            Словарь card_id -> частота появления
        """
        card_counts = Counter()
        total_cards = 0
        
        for deck in generated_decks:
            card_counts.update(deck)
            total_cards += len(deck)
        
        distribution = {
            card_id: count / total_cards
            for card_id, count in card_counts.items()
        }
        
        return distribution
    
    def compute_average_elixir_distribution(
        self,
        generated_decks: List[List[int]],
        card_elixir_map: Dict[int, float]
    ) -> Tuple[float, List[float]]:
        """
        Вычисление распределения среднего эликсира
        
        Args:
            generated_decks: Список колод
            card_elixir_map: Маппинг card_id -> эликсир
            
        Returns:
            (mean_avg_elixir, distribution)
        """
        avg_elixirs = []
        
        for deck in generated_decks:
            deck_elixirs = [card_elixir_map.get(cid, 3.0) for cid in deck]
            avg_elixir = sum(deck_elixirs) / len(deck_elixirs) if deck_elixirs else 0
            avg_elixirs.append(avg_elixir)
        
        mean_avg = np.mean(avg_elixirs)
        
        return mean_avg, avg_elixirs
    
    def compute_all_metrics(
        self,
        generated_decks: List[List[int]],
        meta_decks: Optional[List[List[int]]] = None,
        card_elixir_map: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Вычисление всех метрик
        
        Args:
            generated_decks: Сгенерированные колоды
            meta_decks: Топ колоды для сравнения (опционально)
            card_elixir_map: Маппинг эликсира (опционально)
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # Validity Rate
        validity_rate, validity_stats = self.compute_validity_rate(generated_decks)
        metrics['validity_rate'] = validity_rate
        metrics['validity_stats'] = validity_stats
        
        # Diversity
        diversity, diversity_stats = self.compute_diversity(generated_decks)
        metrics['diversity'] = diversity
        metrics['diversity_stats'] = diversity_stats
        
        # Meta-Similarity (если есть meta_decks)
        if meta_decks:
            meta_sim, meta_stats = self.compute_meta_similarity(
                generated_decks, meta_decks
            )
            metrics['meta_similarity'] = meta_sim
            metrics['meta_stats'] = meta_stats
        
        # Average Elixir (если есть map)
        if card_elixir_map:
            mean_elixir, elixir_dist = self.compute_average_elixir_distribution(
                generated_decks, card_elixir_map
            )
            metrics['mean_avg_elixir'] = mean_elixir
            metrics['elixir_distribution'] = elixir_dist
        
        # Card Distribution
        card_dist = self.compute_card_distribution(generated_decks)
        metrics['card_distribution'] = card_dist
        
        return metrics
    
    def compare_models(
        self,
        decks_by_model: Dict[str, List[List[int]]],
        meta_decks: Optional[List[List[int]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Сравнение нескольких моделей
        
        Args:
            decks_by_model: Словарь model_name -> generated_decks
            meta_decks: Топ колоды для сравнения
            
        Returns:
            Словарь model_name -> metrics
        """
        results = {}
        
        for model_name, decks in decks_by_model.items():
            logger.info(f"Оценка модели: {model_name}")
            metrics = self.compute_all_metrics(decks, meta_decks)
            results[model_name] = metrics
        
        return results


def generate_evaluation_report(
    metrics: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Генерация текстового отчёта по метрикам
    
    Args:
        metrics: Словарь с метриками
        save_path: Путь для сохранения отчёта
        
    Returns:
        Текст отчёта
    """
    report = []
    report.append("=" * 60)
    report.append("ОТЧЁТ ПО ОЦЕНКЕ КАЧЕСТВА ГЕНЕРАЦИИ")
    report.append("=" * 60)
    report.append("")
    
    # Основные метрики
    report.append("📊 ОСНОВНЫЕ МЕТРИКИ:")
    report.append(f"  Validity Rate:    {metrics.get('validity_rate', 0):.2%}")
    report.append(f"  Diversity:        {metrics.get('diversity', 0):.2%}")
    if 'meta_similarity' in metrics:
        report.append(f"  Meta-Similarity:  {metrics.get('meta_similarity', 0):.4f}")
    if 'mean_avg_elixir' in metrics:
        report.append(f"  Mean Avg Elixir:  {metrics.get('mean_avg_elixir', 0):.2f}")
    report.append("")
    
    # Статистика валидности
    if 'validity_stats' in metrics:
        stats = metrics['validity_stats']
        report.append("📋 СТАТИСТИКА ВАЛИДНОСТИ:")
        report.append(f"  Всего колод:  {stats.get('total', 0)}")
        report.append(f"  Валидные:     {stats.get('valid', 0)}")
        report.append(f"  Невалидные:   {stats.get('invalid', 0)}")
        if stats.get('invalid_reasons'):
            report.append("  Причины брака:")
            for reason, count in stats['invalid_reasons'].items():
                report.append(f"    - {reason}: {count}")
        report.append("")
    
    # Статистика разнообразия
    if 'diversity_stats' in metrics:
        stats = metrics['diversity_stats']
        report.append("🎲 СТАТИСТИКА РАЗНООБРАЗИЯ:")
        report.append(f"  Уникальных колод: {stats.get('unique', 0)}")
        if stats.get('top_repeated'):
            report.append("  Топ повторяющихся:")
            for deck, count in stats['top_repeated'][:3]:
                report.append(f"    - {deck}: {count} раз")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    logger.info("\n" + report_text)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Отчёт сохранён: {save_path}")
    
    return report_text


__all__ = ["EvaluationMetrics", "generate_evaluation_report"]
