"""
Скрипт для обучения модели
"""
import sys
import pickle
from pathlib import Path

# Добавление корня проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATABASE_PATH, DATA_DIR, MODELS_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
from data.database import Database
from data.preprocessor import DataPreprocessor
from model.transformer import DeckGeneratorModel
from train.dataset import create_dataloaders
from train.trainer import Trainer
from utils.logger import logger


def run_training():
    """Запуск процесса обучения"""
    logger.info("=" * 50)
    logger.info("Запуск обучения модели")
    logger.info("=" * 50)
    
    # Загрузка данных из БД
    logger.info("Загрузка данных из БД...")
    db = Database(str(DATABASE_PATH))
    db.connect()
    
    # Проверка наличия данных
    stats = db.get_stats()
    if stats['total_decks'] == 0:
        logger.error("База данных пуста! Запустите сначала etl_pipeline.py")
        db.disconnect()
        return False
    
    logger.info(f"В БД: {stats['total_cards']} карт, {stats['total_decks']} колод")
    
    # Загрузка карт и колод
    cards = db.get_all_cards()
    decks = db.get_filtered_decks()
    
    db.disconnect()
    
    if not cards or not decks:
        logger.error("Не удалось загрузить данные")
        return False
    
    logger.info(f"Загружено {len(decks)} колод после фильтрации")
    
    # Предобработка данных
    logger.info("Предобработка данных...")
    preprocessor = DataPreprocessor(cards)
    
    # Создание обучающих последовательностей
    input_seqs, target_seqs, card_features = preprocessor.create_training_sequences(decks)
    
    if len(input_seqs) == 0:
        logger.error("Не удалось создать обучающие последовательности")
        return False
    
    # Создание dataloaders
    logger.info("Создание DataLoader'ов...")
    train_loader, val_loader, test_loader = create_dataloaders(
        input_seqs, target_seqs, card_features,
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Инициализация модели
    logger.info("Инициализация модели...")
    model = DeckGeneratorModel(
        vocab_size=preprocessor.vocab.size,
        feature_dim=preprocessor.encoder.feature_dim,
        embedding_dim=128,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_seq_len=8
    )
    
    # Логирование архитектуры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Параметров модели: {total_params:,} (trainable: {trainable_params:,})")
    
    # Инициализация тренера
    trainer = Trainer(model, learning_rate=LEARNING_RATE)
    
    # Обучение
    logger.info("Начало обучения...")
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        save_path=str(MODELS_DIR / "best_model.pt")
    )
    
    # Сохранение препроцессора
    logger.info("Сохранение препроцессора...")
    preprocessor.save(str(DATA_DIR / "preprocessor"))
    
    # Сохранение истории обучения
    with open(MODELS_DIR / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)
    
    # Финальная статистика
    logger.info("=" * 50)
    logger.info("Обучение завершено!")
    logger.info(f"Лучший val_loss: {min(history['val_loss']):.4f}")
    logger.info(f"Эпох пройдено: {len(history['val_loss'])}")
    logger.info("=" * 50)
    
    # Тестирование
    logger.info("Тестирование на test выборке...")
    test_loss, test_perplexity = trainer.validate(test_loader, 0)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.2f}")
    
    return True


def main():
    """Точка входа"""
    success = run_training()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
