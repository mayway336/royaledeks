"""
Training - Модуль обучения модели
"""
from .trainer import Trainer
from .dataset import DeckDataset, DeckDataLoader

__all__ = ["Trainer", "DeckDataset", "DeckDataLoader"]
