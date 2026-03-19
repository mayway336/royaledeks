"""
Модуль Dataset и DataLoader для обучения
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np

from config import BATCH_SIZE
from utils.logger import logger


class DeckDataset(Dataset):
    """
    Dataset для обучения генерации колод
    """
    
    def __init__(
        self,
        input_sequences: torch.Tensor,
        target_sequences: torch.Tensor,
        card_features: torch.Tensor
    ):
        """
        Инициализация dataset
        
        Args:
            input_sequences: Входные последовательности [num_samples, seq_len]
            target_sequences: Целевые последовательности [num_samples, seq_len]
            card_features: Признаки карт [num_samples, seq_len, feature_dim]
        """
        assert len(input_sequences) == len(target_sequences), \
            "Длины input и target должны совпадать"
        assert len(input_sequences) == len(card_features), \
            "Длины input и features должны совпадать"
        
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.card_features = card_features
        
        self.num_samples = len(input_sequences)
        
        logger.info(f"Dataset инициализирован: {self.num_samples} образцов")
    
    def __len__(self) -> int:
        """Получение размера dataset"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Получение элемента по индексу
        
        Args:
            idx: Индекс элемента
            
        Returns:
            (input_seq, target_seq, features)
        """
        return (
            self.input_sequences[idx],
            self.target_sequences[idx],
            self.card_features[idx]
        )


class DeckDataLoader:
    """
    Обёртка над DataLoader для удобства
    """
    
    def __init__(
        self,
        dataset: DeckDataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        """
        Инициализация DataLoader
        
        Args:
            dataset: Dataset для загрузки
            batch_size: Размер батча
            shuffle: Перемешивать ли данные
            num_workers: Количество рабочих процессов
            pin_memory: Использовать pinned memory для GPU
        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        self.batch_size = batch_size
        self.num_batches = len(self.dataloader)
        
        logger.info(
            f"DataLoader инициализирован: "
            f"batch_size={batch_size}, num_batches={self.num_batches}"
        )
    
    def __iter__(self):
        """Итератор по батчам"""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Количество батчей"""
        return self.num_batches


def create_dataloaders(
    input_sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    card_features: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[DeckDataLoader, DeckDataLoader, DeckDataLoader]:
    """
    Создание train/val/test dataloaders
    
    Args:
        input_sequences: Входные последовательности
        target_sequences: Целевые последовательности
        card_features: Признаки карт
        batch_size: Размер батча
        train_ratio: Доля train выборки
        val_ratio: Доля validation выборки
        test_ratio: Доля test выборки
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Сумма ratios должна быть равна 1.0"
    
    num_samples = len(input_sequences)
    
    # Индексы для разбиения
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Создание датасетов
    train_dataset = DeckDataset(
        input_sequences[train_idx],
        target_sequences[train_idx],
        card_features[train_idx]
    )
    
    val_dataset = DeckDataset(
        input_sequences[val_idx],
        target_sequences[val_idx],
        card_features[val_idx]
    )
    
    test_dataset = DeckDataset(
        input_sequences[test_idx],
        target_sequences[test_idx],
        card_features[test_idx]
    )
    
    logger.info(f"Разбиение данных:")
    logger.info(f"  Train: {len(train_dataset)} ({100*train_ratio:.0f}%)")
    logger.info(f"  Val:   {len(val_dataset)} ({100*val_ratio:.0f}%)")
    logger.info(f"  Test:  {len(test_dataset)} ({100*test_ratio:.0f}%)")
    
    # Создание dataloaders
    train_loader = DeckDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DeckDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DeckDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
