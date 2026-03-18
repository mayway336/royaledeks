"""
Скрипт для оценки качества модели
"""
import sys
import pickle
import torch
from pathlib import Path

# Добавление корня проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH, MODELS_DIR, DATA_DIR
from data.database import Database
from data.preprocessor import DataPreprocessor
from model.transformer import DeckGeneratorModel
from rule_engine.rule_engine import RuleEngine
from eval.metrics import EvaluationMetrics, generate_evaluation_report
from utils.logger import logger


def run_evaluation(num_decks: int = 100):
    """
    Запуск оценки модели
    
    Args:
        num_decks: Количество колод для генерации
    """
    logger.info("=" * 50)
    logger.info("Запуск оценки модели")
    logger.info("=" * 50)
    
    # Загрузка модели и препроцессора
    logger.info("Загрузка модели...")
    
    preprocess_path = DATA_DIR / "preprocessor"
    if not preprocess_path.exists():
        logger.error("Препроцессор не найден! Запустите сначала train_model.py")
        return False
    
    preprocessor = DataPreprocessor.load(str(preprocess_path))
    
    model_path = MODELS_DIR / "best_model.pt"
    if not model_path.exists():
        logger.error("Модель не найдена! Запустите сначала train_model.py")
        return False
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = DeckGeneratorModel(
        vocab_size=preprocessor.vocab.size,
        feature_dim=preprocessor.encoder.feature_dim,
        embedding_dim=128,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_seq_len=8
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Модель загружена (epoch {checkpoint['epoch']})")
    
    # Загрузка метаданных
    with open(preprocess_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Инициализация Rule Engine
    rule_engine = RuleEngine(
        vocab_size=preprocessor.vocab.size,
        evolveable_cards=metadata.get('evolveable_cards', set()),
        hero_cards=metadata.get('hero_cards', set()),
        champion_cards=metadata.get('champion_cards', set())
    )
    
    # Загрузка данных для сравнения
    logger.info("Загрузка данных для сравнения...")
    db = Database(str(DATABASE_PATH))
    db.connect()
    
    meta_decks_data = db.get_filtered_decks()
    meta_decks = [d['cards'] for d in meta_decks_data[:500]]  # Топ 500 для сравнения
    
    # Создание маппинга card_id -> elixir
    cards = db.get_all_cards()
    card_elixir_map = {c['card_id']: c['elixir_cost'] for c in cards}
    card_id_mapping = preprocessor.vocab.idx_to_card
    
    db.disconnect()
    
    # Генерация колод
    logger.info(f"Генерация {num_decks} колод...")
    
    card_features = torch.FloatTensor(
        [preprocessor.encoder.get_features(i) for i in range(preprocessor.vocab.size)]
    ).unsqueeze(0)
    
    generated_indices = []
    
    with torch.no_grad():
        for _ in range(num_decks):
            generated = model.generate(
                card_features=card_features,
                rule_engine=rule_engine,
                temperature=1.0,
                top_k=50
            )
            generated_indices.append(generated[0].tolist())
    
    logger.info(f"Сгенерировано {len(generated_indices)} колод")
    
    # Инициализация метрик
    metrics = EvaluationMetrics(
        rule_engine=rule_engine,
        vocab_size=preprocessor.vocab.size,
        card_id_mapping=card_id_mapping
    )
    
    # Вычисление метрик
    logger.info("Вычисление метрик...")
    all_metrics = metrics.compute_all_metrics(
        generated_decks=generated_indices,
        meta_decks=meta_decks,
        card_elixir_map=card_elixir_map
    )
    
    # Генерация отчёта
    report = generate_evaluation_report(
        all_metrics,
        save_path=str(MODELS_DIR / "evaluation_report.txt")
    )
    
    return True


def main():
    """Точка входа"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка качества модели")
    parser.add_argument(
        "--num-decks",
        type=int,
        default=100,
        help="Количество колод для генерации"
    )
    
    args = parser.parse_args()
    
    success = run_evaluation(num_decks=args.num_decks)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
